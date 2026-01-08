import os
import sys
import glob
import random
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from argparse import ArgumentParser
from sklearn.metrics import average_precision_score, roc_curve
from torchvision.transforms import Compose, Resize, ToTensor

# -----------------------------------------------------------------------------
# SETUP PATH & IMPORTS
# -----------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(CURRENT_DIR, "..")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.segmenter import AnomalySegmenter

EOMT_ROOT = os.path.join(PROJECT_ROOT, "eomt")
if EOMT_ROOT not in sys.path:
    sys.path.insert(0, EOMT_ROOT)

from training.lightning_module import LightningModule as EoMTLightningModule

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
SEED = 42
NUM_CLASSES = 19
IGNORE_VEHICLES = {9, 13, 14, 15, 16, 17, 18} # trainIds: car, truck, bus, train, mcycle, bicycle

def fpr_at_95_tpr(scores: np.ndarray, labels: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    idxs = np.where(tpr >= 0.95)[0]
    if len(idxs) == 0:
        return 1.0
    return float(fpr[idxs[0]])

def load_my_model(ckpt_path: str, device: torch.device) -> AnomalySegmenter:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint non trovato: {ckpt_path}")

    model = AnomalySegmenter.load_from_checkpoint(
        ckpt_path,
        map_location="cpu",
        strict=True,
        pretrained_eomt_bin=None,
    )
    model.to(device)
    model.eval()
    return model

def get_msp_score(per_pixel_logits):
    """
    Calcola l'MSP anomaly score: 1 - max(softmax(logits))
    """
    probs = torch.softmax(per_pixel_logits, dim=0) 
    max_prob = probs.max(dim=0).values
    return (1.0 - max_prob).detach().cpu().numpy()

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        nargs="+",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
    )
    parser.add_argument("--loadDir", default="./checkpoints/")
    parser.add_argument("--loadWeights", default="last.ckpt")
    
    parser.add_argument("--img_h", type=int, default=1024)
    parser.add_argument("--img_w", type=int, default=1024)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--lf_ignore_vehicles", action="store_true")

    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    IMG_SIZE = (args.img_h, args.img_w)

    # Range Temperatures
    target_temps = [0.5, 0.75, 1.0, 1.1]
    search_temps = [2.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    all_temps = sorted(list(set(target_temps + search_temps)))

    input_transform = Compose([
        Resize(IMG_SIZE, Image.BILINEAR),
        ToTensor(),
    ])
    target_transform = Compose([
        Resize(IMG_SIZE, Image.NEAREST),
    ])

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device in uso: {device}")

    ckpt_full_path = os.path.join(args.loadDir, args.loadWeights)
    print(f"Caricamento pesi HQT-OE da: {ckpt_full_path}")
    model = load_my_model(ckpt_full_path, device)

    anomaly_scores_dict = {t: [] for t in all_temps}
    ood_gts_list = []

    files = []
    for pattern in args.input:
        files.extend(glob.glob(os.path.expanduser(pattern)))
    print(f"Trovate {len(files)} immagini. Valutazione in corso...")

    # -------------------------------------------------------------------------
    # INFERENCE LOOP
    # -------------------------------------------------------------------------
    for path in files:
        try:
            img_pil = Image.open(path).convert("RGB")
            img_tensor = input_transform(img_pil).unsqueeze(0).to(device)
        except Exception as e:
            continue

        with torch.no_grad():
            mask_logits_layers, class_logits_layers = model.model(img_tensor)
            
            final_mask_logits = mask_logits_layers[-1]
            final_class_logits = class_logits_layers[-1]

            if final_mask_logits.shape[-2:] != IMG_SIZE:
                final_mask_logits = F.interpolate(
                    final_mask_logits, size=IMG_SIZE, mode="bilinear", align_corners=False
                )

            # Loop Temperatures
            for t in all_temps:
                scaled_class_logits = final_class_logits / t
                
                per_pixel = EoMTLightningModule.to_per_pixel_logits_semantic(
                    final_mask_logits, scaled_class_logits
                )[0]

                score_map = get_msp_score(per_pixel)
                anomaly_scores_dict[t].append(score_map)

        # ---------------------------------------------------------------------
        # GROUND TRUTH & MAPPING
        # ---------------------------------------------------------------------
        pathGT = path.replace("images", "labels_masks").replace("webp", "png").replace("jpg", "png")
        
        try:
            gt = np.array(target_transform(Image.open(pathGT)))
            
            if "RoadAnomaly" in pathGT:
                gt = np.where(gt == 2, 1, gt)
            elif "LostFound" in pathGT or "LostAndFound" in pathGT:
                gt = gt.astype(np.uint8)
                if args.lf_ignore_vehicles:
                    probs_std = torch.softmax(per_pixel, dim=0) 
                    pred_sem = probs_std.argmax(dim=0).cpu().numpy()
                    ignore_veh = np.isin(pred_sem, list(IGNORE_VEHICLES))
                    gt = gt.copy()
                    gt[ignore_veh] = 255
            elif "Streethazard" in pathGT:
                gt = np.where(gt == 14, 255, gt)
                gt = np.where(gt < 20, 0, gt)
                gt = np.where(gt == 255, 1, gt)

            if 1 not in np.unique(gt):
                for t in all_temps:
                    anomaly_scores_dict[t].pop()
                continue

            ood_gts_list.append(gt)

        except Exception:
            for t in all_temps:
                if len(anomaly_scores_dict[t]) > len(ood_gts_list):
                    anomaly_scores_dict[t].pop()
            continue

    # -------------------------------------------------------------------------
    # METRICS COMPUTATION & PRINTING
    # -------------------------------------------------------------------------
    if not ood_gts_list:
        print("Nessun dato valido.")
        return

    ood_gts_all = np.array(ood_gts_list)
    ood_mask = (ood_gts_all == 1)
    ind_mask = (ood_gts_all == 0)
    val_label = np.concatenate((np.zeros(ind_mask.sum()), np.ones(ood_mask.sum())))

    print("\n" + "=" * 60)
    print(f"RESULTS FOR HQT-OE (Temperature Scaling)")
    print("=" * 60)
    print(f"{'T':<6} | {'AuPRC':<12} | {'FPR@95':<12}")
    print("-" * 40)

    best_auprc = 0.0
    best_t = 1.0

    for t in all_temps:
        scores_all = np.array(anomaly_scores_dict[t])
        val_out = np.concatenate((scores_all[ind_mask], scores_all[ood_mask]))

        auprc = average_precision_score(val_label, val_out)
        fpr95 = fpr_at_95_tpr(val_out, val_label)

        mark = "*" if t in [0.5, 0.75, 1.1] else " "
        print(f"{mark} {t:<4} | {auprc*100.0:<10.4f}% | {fpr95*100.0:<10.4f}%")

        if auprc > best_auprc:
            best_auprc = auprc
            best_t = t

    print("-" * 60)
    print(f"BEST T FOUND: {best_t} (AuPRC: {best_auprc*100.0:.2f}%)")
    print("=" * 60)

if __name__ == "__main__":
    main()