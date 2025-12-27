# eval_my_model.py
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

from models.segmenter import AnomalySegmenter  # <-- il TUO modello

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
SEED = 42
NUM_CLASSES = 19


# -----------------------------------------------------------------------------
# UTILS
# -----------------------------------------------------------------------------
def fpr_at_95_tpr(scores: np.ndarray, labels: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    idxs = np.where(tpr >= 0.95)[0]
    if len(idxs) == 0:
        return 1.0
    return float(fpr[idxs[0]])


def load_my_model(ckpt_path: str, device: torch.device) -> AnomalySegmenter:
    """
    Carica il tuo LightningModule dal .ckpt SENZA ricaricare il .bin.
    IMPORTANTISSIMO per non sovrascrivere i pesi del checkpoint.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint non trovato: {ckpt_path}")

    model = AnomalySegmenter.load_from_checkpoint(
        ckpt_path,
        map_location="cpu",
        strict=True,
        pretrained_eomt_bin=None,   # ✅ BLOCCA reload .bin durante eval
    )

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def compute_anomaly_map_eomt_style(model: AnomalySegmenter, img_tensor: torch.Tensor, method: str, img_size):
    """
    Replica la logica del prof/EoMT:
      - prende mask_logits + class_logits dall'ultimo layer
      - fa upsample mask logits
      - drop no-object se presente
      - seg_probs = einsum(class_probs, mask_probs)
      - anomaly map da seg_probs (MSP) oppure dagli pseudo-logits (per maxlogit/entropy)
    """
    # 1) forward "nativo" del decoder EoMT
    mask_logits_layers, class_logits_layers = model.model(img_tensor)
    mask_logits = mask_logits_layers[-1]     # (B,Q,h,w)
    class_logits = class_logits_layers[-1]   # (B,Q,C+1) o (B,Q,C)

    # 2) upsample masks alla risoluzione immagine
    if mask_logits.shape[-2:] != img_size:
        mask_logits = F.interpolate(mask_logits, size=img_size, mode="bilinear", align_corners=False)

    # 3) drop no-object se presente
    if class_logits.shape[-1] == NUM_CLASSES + 1:
        class_logits = class_logits[..., :NUM_CLASSES]
    else:
        class_logits = class_logits[..., :NUM_CLASSES]

    # 4) probabilistic composition (Mask2Former-style)
    mask_probs = mask_logits.sigmoid()                # (B,Q,H,W)
    class_probs = torch.softmax(class_logits, dim=-1) # (B,Q,C)
    seg_probs = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)  # (B,C,H,W)

    # 5) anomaly map (prof-style scoring)
    if method == "msp":
        msp = seg_probs.max(dim=1).values     # (B,H,W)
        anomaly = 1.0 - msp

    elif method == "maxlogit":
        # per avere qualcosa di coerente, usiamo pseudo-logits = log(seg_probs)
        pseudo_logits = torch.log(seg_probs.clamp_min(1e-6))
        maxlogit = pseudo_logits.max(dim=1).values
        anomaly = -maxlogit

    elif method == "maxentropy":
        eps = 1e-8
        p = seg_probs / (seg_probs.sum(dim=1, keepdim=True).clamp_min(eps))  # normalizza (sicurezza)
        entropy = -(p * (p + eps).log()).sum(dim=1)  # (B,H,W)
        anomaly = entropy

    elif method == "rba":
        msp = seg_probs.max(dim=1).values
        accept_threshold = 0.5
        anomaly = torch.clamp(accept_threshold - msp, min=0) / accept_threshold

    else:
        raise ValueError(f"Metodo anomalia non supportato: {method}")

    return anomaly  # (B,H,W)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        nargs="+",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        help="Path o glob pattern immagini input",
    )
    parser.add_argument("--ckpt_dir", default="./checkpoints/", help="Cartella checkpoint Lightning")
    parser.add_argument("--ckpt_name", default="last.ckpt", help="Nome checkpoint (es: last.ckpt o seg-xx-xxxxxx.ckpt)")

    parser.add_argument("--img_h", type=int, default=1024)
    parser.add_argument("--img_w", type=int, default=1024)

    parser.add_argument("--method", default="msp", choices=["msp", "maxlogit", "maxentropy", "rba"])
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    # Reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

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

    if not os.path.exists("results.txt"):
        open("results.txt", "w").close()

    ckpt_full_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    print(f"Caricamento ckpt da: {ckpt_full_path}")

    try:
        model = load_my_model(ckpt_full_path, device)
        print("✅ Modello caricato con successo.")
        print("✅ NOTA: se vedi ancora [LOAD_EOMT_BIN] allora stai usando un file diverso o non hai salvato la modifica.\n")
    except Exception as e:
        print(f"❌ Errore critico nel caricamento del modello: {e}")
        return

    anomaly_score_list = []
    ood_gts_list = []

    file_list = glob.glob(os.path.expanduser(str(args.input[0])))
    print(f"Trovate {len(file_list)} immagini da elaborare.")

    for path in file_list:
        print(f"Processing: {path}")

        img_pil = Image.open(path).convert("RGB")
        img_tensor = input_transform(img_pil).unsqueeze(0).float().to(device)

        with torch.no_grad():
            anomaly_map_bhw = compute_anomaly_map_eomt_style(
                model,
                img_tensor,
                method=args.method,
                img_size=IMG_SIZE,
            )
            anomaly_np = anomaly_map_bhw[0].detach().cpu().numpy()

        # --- GT path mapping ---
        pathGT = path.replace("images", "labels_masks")
        if "RoadObsticle21" in pathGT:
            pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
            pathGT = pathGT.replace("jpg", "png")
        if "RoadAnomaly" in pathGT:
            pathGT = pathGT.replace("jpg", "png")

        try:
            gt_img = Image.open(pathGT)
        except FileNotFoundError:
            print(f"Warning: GT non trovata per {path}, skip.")
            continue

        gt_img = target_transform(gt_img)
        ood_gts = np.array(gt_img)

        # --- dataset-specific relabeling (come prof) ---
        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts == 2), 1, ood_gts)

        elif "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts == 0), 255, ood_gts)
            ood_gts = np.where((ood_gts == 1), 0, ood_gts)
            ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)

        elif "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts == 14), 255, ood_gts)
            ood_gts = np.where((ood_gts < 20), 0, ood_gts)
            ood_gts = np.where((ood_gts == 255), 1, ood_gts)

        # skip immagini senza OOD (come prof)
        if 1 not in np.unique(ood_gts):
            continue

        ood_gts_list.append(ood_gts)
        anomaly_score_list.append(anomaly_np)

        del img_tensor, anomaly_map_bhw
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not ood_gts_list:
        print("Nessun dato valido raccolto per la valutazione.")
        return

    print("Calcolo metriche in corso...")

    ood_gts_flat = np.array(ood_gts_list)
    anomaly_scores_flat = np.array(anomaly_score_list)

    # ignora 255
    valid_mask = (ood_gts_flat != 255)
    ood_mask = (ood_gts_flat == 1) & valid_mask
    ind_mask = (ood_gts_flat == 0) & valid_mask

    ood_scores = anomaly_scores_flat[ood_mask]
    ind_scores = anomaly_scores_flat[ind_mask]

    all_scores = np.concatenate((ind_scores, ood_scores))
    all_labels = np.concatenate((np.zeros(len(ind_scores)), np.ones(len(ood_scores))))

    auprc = average_precision_score(all_labels, all_scores)
    fpr95 = fpr_at_95_tpr(all_scores, all_labels)

    result_str = f"[MYMODEL-{args.method}] AUPRC: {auprc * 100.0:.2f}% | FPR@95TPR: {fpr95 * 100.0:.2f}%"
    print("\n" + "=" * 50)
    print(result_str)
    print("=" * 50 + "\n")

    with open("results.txt", "a") as f:
        f.write("\n" + result_str)


if __name__ == "__main__":
    main()