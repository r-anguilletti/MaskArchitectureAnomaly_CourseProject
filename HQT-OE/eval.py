import os
import glob
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from argparse import ArgumentParser
from sklearn.metrics import roc_curve, average_precision_score
from torchvision.transforms import v2 as T
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

from models.segmenter import AnomalySegmenter


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
IMG_SIZE = (518, 518)
NUM_CLASSES = 19
IGNORE_LABEL = 255


# ------------------------------------------------------------
# TRANSFORMS
# ------------------------------------------------------------
input_transform = T.Compose([
    T.Resize(IMG_SIZE, interpolation=T.InterpolationMode.BILINEAR),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

target_transform = T.Compose([
    T.Resize(IMG_SIZE, interpolation=T.InterpolationMode.NEAREST),
])


# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------
def fpr_at_95_tpr(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    if len(tpr) == 0:
        return 1.0
    idx = np.where(tpr >= 0.95)[0]
    return float(fpr[idx[0]]) if len(idx) > 0 else 1.0


def compute_anomaly_map(logits, method):
    """
    logits: (C=19, H, W)
    returns: anomaly score map (H, W)
    """
    if method == "energy":
        return -torch.logsumexp(logits, dim=0)

    elif method == "msp":
        probs = F.softmax(logits, dim=0)
        return 1.0 - probs.max(dim=0).values

    elif method == "maxlogit":
        return -logits.max(dim=0).values

    elif method == "entropy":
        probs = F.softmax(logits, dim=0)
        eps = 1e-8
        return -(probs * (probs + eps).log()).sum(dim=0)

    else:
        raise ValueError(f"Unknown method {method}")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    parser = ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True,
                        help="Glob immagini")
    parser.add_argument("--ckpt", required=True,
                        help="Checkpoint del modello")
    parser.add_argument("--method", default="energy",
                        choices=["energy", "msp", "maxlogit", "entropy"])
    parser.add_argument("--smooth_sigma", type=float, default=0.0,
                        help="Gaussian smoothing sigma (0 = off)")
    parser.add_argument("--debug_n", type=int, default=3,
                        help="Numero immagini con debug GT")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n--- Eval config | device={device} | method={args.method} ---\n")

    # -------------------------
    # LOAD MODEL
    # -------------------------
    print("--> Loading model...")
    model = AnomalySegmenter.load_from_checkpoint(args.ckpt)
    model.to(device)
    model.eval()

    # -------------------------
    # COLLECT FILES
    # -------------------------
    files = []
    for p in args.input:
        files.extend(glob.glob(os.path.expanduser(p), recursive=True))
    files = sorted(set(files))

    print(f"Trovate {len(files)} immagini")
    if len(files) == 0:
        return

    anomaly_maps = []
    gt_maps = []

    # -------------------------
    # LOOP
    # -------------------------
    for idx, path in enumerate(tqdm(files, desc="Processing")):
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue

        img_t = input_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img_t)[0]   # (19, H, W)
            anomaly = compute_anomaly_map(logits, args.method)

            if args.smooth_sigma > 0:
                anomaly = torch.from_numpy(
                    gaussian_filter(anomaly.cpu().numpy(), sigma=args.smooth_sigma)
                ).to(device)

            anomaly = anomaly.cpu().numpy()

        # -------------------------
        # GT PATH + MAPPING
        # -------------------------
        pathGT = None

        if "RoadAnomaly21" in path or "RoadObsticle21" in path:
            cand = path.replace("images", "labels_masks")
            cand = os.path.splitext(cand)[0] + ".png"
            if os.path.exists(cand):
                pathGT = cand

        elif "RoadAnomaly" in path:
            cand = path.replace(".jpg", ".labels.png")
            if os.path.exists(cand):
                pathGT = cand

        elif "leftImg8bit" in path:
            cand = path.replace("leftImg8bit", "gtCoarse")
            cand = cand.replace("_leftImg8bit", "_gtCoarse_labelIds")
            cand = os.path.splitext(cand)[0] + ".png"
            if os.path.exists(cand):
                pathGT = cand

        if pathGT is None:
            continue

        gt = Image.open(pathGT)
        gt = target_transform(gt)
        gt = np.array(gt)

        new_gt = np.ones_like(gt) * IGNORE_LABEL

        # RoadAnomaly21 / RoadObsticle21
        if "RoadAnomaly21" in pathGT or "RoadObsticle21" in pathGT:
            new_gt[gt == 0] = 0
            new_gt[gt == 1] = 1

        # Old RoadAnomaly
        elif "RoadAnomaly" in pathGT:
            new_gt[gt == 1] = 0
            new_gt[gt == 2] = 1

        # LostAndFound / Fishyscapes
        else:
            new_gt[gt == 1] = 0
            new_gt[gt > 1] = 1

        if idx < args.debug_n:
            uniq, cnt = np.unique(new_gt, return_counts=True)
            print(f"[DBG GT IMAGE] {pathGT}")
            print(" uniq:", dict(zip(uniq.tolist(), cnt.tolist())))

        anomaly_maps.append(anomaly)
        gt_maps.append(new_gt)

    # -------------------------
    # STACK & FILTER
    # -------------------------
    gt_all = np.stack(gt_maps)
    score_all = np.stack(anomaly_maps)

    valid = gt_all != IGNORE_LABEL
    labels = gt_all[valid]
    scores = score_all[valid]

    print("\n--- DEBUG GT DISTRIBUTION ---")
    uniq, cnt = np.unique(labels, return_counts=True)
    print("GT labels:", dict(zip(uniq.tolist(), cnt.tolist())))
    print("Anomaly %:", 100.0 * (labels == 1).mean())

    print("\n--- DEBUG SCORE STATS ---")
    print("scores min/max/mean:",
          scores.min(), scores.max(), scores.mean())
    print("mean score ID  :", scores[labels == 0].mean())
    print("mean score OOD :", scores[labels == 1].mean())

    # -------------------------
    # METRICS
    # -------------------------
    auprc = average_precision_score(labels, scores)
    fpr95 = fpr_at_95_tpr(scores, labels)

    print("\n--- METRICS ---")
    print(f"[{args.method.upper()}] AuPRC={auprc*100:.2f}% | "
          f"FPR@95={fpr95*100:.2f}%")


if __name__ == "__main__":
    main()