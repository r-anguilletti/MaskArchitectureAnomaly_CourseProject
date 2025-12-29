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


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(CURRENT_DIR, "..")
EOMT_ROOT = os.path.join(PROJECT_ROOT, "eomt")

if EOMT_ROOT not in sys.path:
    sys.path.insert(0, EOMT_ROOT)

from training.lightning_module import LightningModule as EoMTLightningModule

SEED = 42
NUM_CLASSES = 19


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

@torch.no_grad()
def compute_anomaly_map_eomt_style(
    model: AnomalySegmenter,
    img_tensor: torch.Tensor,
    method: str,
    img_size
):
    """
    IDENTICO al prof:
    - forward EoMT
    - upsample mask logits
    - to_per_pixel_logits_semantic
    - MSP / MaxLogit / MaxEntropy / RbA-proxy
    """
    mask_logits_layers, class_logits_layers = model.model(img_tensor)
    final_mask_logits = mask_logits_layers[-1]      # [B,Q,h,w]
    final_class_logits = class_logits_layers[-1]    # [B,Q,C(+1)]

    if final_mask_logits.shape[-2:] != img_size:
        final_mask_logits = F.interpolate(
            final_mask_logits, size=img_size, mode="bilinear", align_corners=False
        )

    per_pixel_logits = EoMTLightningModule.to_per_pixel_logits_semantic(
        final_mask_logits, final_class_logits
    )  # [B,C,H,W]

    pixel_logits = per_pixel_logits[0]  # [C,H,W]
    probs = F.softmax(pixel_logits, dim=0)

    if method == "msp":
        anomaly = 1.0 - probs.max(dim=0).values

    elif method == "maxlogit":
        anomaly = -pixel_logits.max(dim=0).values

    elif method == "maxentropy":
        eps = 1e-8
        anomaly = -(probs * (probs + eps).log()).sum(dim=0)

    elif method == "rba":
        
        anomaly = -pixel_logits.tanh().sum(dim=0)

    else:
        raise ValueError(f"Metodo anomalia non supportato: {method}")

    return anomaly.unsqueeze(0)  # [1,H,W]


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        nargs="+",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
    )
    parser.add_argument("--ckpt_dir", default="./checkpoints/")
    parser.add_argument("--ckpt_name", default="last.ckpt")
    parser.add_argument("--img_h", type=int, default=1024)
    parser.add_argument("--img_w", type=int, default=1024)
    parser.add_argument("--method", default="msp",
                        choices=["msp", "maxlogit", "maxentropy", "rba"])
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    IMG_SIZE = (args.img_h, args.img_w)

    input_transform = Compose([
        Resize(IMG_SIZE, Image.BILINEAR),
        ToTensor(),
    ])
    target_transform = Compose([
        Resize(IMG_SIZE, Image.NEAREST),
    ])

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device in uso: {device}")

    model = load_my_model(
        os.path.join(args.ckpt_dir, args.ckpt_name),
        device
    )

    anomaly_score_list = []
    ood_gts_list = []

    files = glob.glob(os.path.expanduser(args.input[0]))
    print(f"Trovate {len(files)} immagini")

    for path in files:
        img = input_transform(Image.open(path).convert("RGB"))
        img = img.unsqueeze(0).to(device)

        anomaly_map = compute_anomaly_map_eomt_style(
            model, img, args.method, IMG_SIZE
        )
        anomaly_np = anomaly_map[0].cpu().numpy()

        pathGT = path.replace("images", "labels_masks")
        pathGT = pathGT.replace("webp", "png").replace("jpg", "png")

        try:
            gt = np.array(target_transform(Image.open(pathGT)))
        except FileNotFoundError:
            continue

        if "RoadAnomaly" in pathGT:
            gt = np.where(gt == 2, 1, gt)

        elif "LostAndFound" in pathGT:
            gt = np.where(gt == 0, 255, gt)
            gt = np.where(gt == 1, 0, gt)
            gt = np.where((gt > 1) & (gt < 201), 1, gt)

        elif "Streethazard" in pathGT:
            gt = np.where(gt == 14, 255, gt)
            gt = np.where(gt < 20, 0, gt)
            gt = np.where(gt == 255, 1, gt)

        if 1 not in np.unique(gt):
            continue

        ood_gts_list.append(gt)
        anomaly_score_list.append(anomaly_np)

        del img, anomaly_map
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not ood_gts_list:
        print("Nessun dato valido.")
        return

    ood_gts = np.array(ood_gts_list)
    scores = np.array(anomaly_score_list)

    ood_scores = scores[ood_gts == 1]
    ind_scores = scores[ood_gts == 0]

    all_scores = np.concatenate([ind_scores, ood_scores])
    all_labels = np.concatenate([
        np.zeros(len(ind_scores)),
        np.ones(len(ood_scores))
    ])

    auprc = average_precision_score(all_labels, all_scores)
    fpr95 = fpr_at_95_tpr(all_scores, all_labels)

    dataset = args.input[0].split("/")[-3]

    print("=" * 50)
    result_str = f"[HQT-OE-{args.method}-{dataset}] AUPRC: {auprc*100:.2f}% | FPR@95TPR: {fpr95*100:.2f}%"
    print(result_str)
    print("=" * 50)

    with open("results.txt", "a") as f:
        f.write("\n" + result_str)

if __name__ == "__main__":
    main()