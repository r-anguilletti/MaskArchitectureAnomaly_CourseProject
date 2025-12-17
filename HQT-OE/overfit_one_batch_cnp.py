import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# -------------------------------------------------
# Ensure local imports work in Colab when running via absolute path
# -------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from cnp_zip_dataset import CNPZipDataset
from models.segmenter import AnomalySegmenter

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
ZIP_PATH = "/content/drive/MyDrive/Anomaly_Segmentation/cnp_dataset.zip"
BATCH_SIZE = 1
N_ITERS = 300
LR = 5e-4
MIN_ANOM_PIXELS = 2000   # ensure the chosen one-batch actually contains anomaly
SEED = 0

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -------------------------------------------------
# DATA (pick a batch that actually contains anomalies)
# -------------------------------------------------
ds = CNPZipDataset(ZIP_PATH)

# We search at sample-level (batch_size=1) to guarantee the chosen sample contains anomaly.
dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

x = y = None
for try_i, (xb, yb) in enumerate(dl, start=1):
    anom = int(yb.sum().item())
    if anom >= MIN_ANOM_PIXELS:
        x, y = xb, yb
        print(f"Selected batch #{try_i} with anomaly_pixels={anom}")
        break
    if try_i >= 200:
        # fallback: take whatever we got last
        x, y = xb, yb
        print(f"[WARN] Could not find batch with >= {MIN_ANOM_PIXELS} anomaly pixels in 200 tries. Using anomaly_pixels={anom}")
        break

x = x.to(device)
y = y.to(device)  # [B,H,W] in {0,1}

print("Batch shapes:", x.shape, y.shape)
print("Mask unique:", torch.unique(y).tolist())
pos_per_img = y.view(y.size(0), -1).sum(dim=1)
print("Anomaly pixels per image:", pos_per_img.tolist())
print("Anomaly ratio:", float(pos_per_img.item()) / float(y.numel()))

# -------------------------------------------------
# MODEL
# -------------------------------------------------
model = AnomalySegmenter()  # LightningModule, but we use it like a plain nn.Module here
model = model.to(device)
model.train()

ANOMALY_CH = int(getattr(model, "anomaly_class_idx", 19))
print("Using anomaly channel:", ANOMALY_CH)

# -------------------------------------------------
# OPTIM
# -------------------------------------------------
opt = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR,
    weight_decay=0.0,
)

# -------------------------------------------------
# LOSS
# -------------------------------------------------
EPS = 1e-6

def dice_loss_from_logits(bin_logit, y01):
    """Soft Dice loss for binary segmentation (helps with sparse positives)."""
    y01 = y01.float()
    prob = torch.sigmoid(bin_logit)
    inter = (prob * y01).sum()
    denom = prob.sum() + y01.sum()
    dice = (2.0 * inter + EPS) / (denom + EPS)
    return 1.0 - dice

def loss_fn(logits, y01, debug=False):
    """Binary anomaly loss from multi-class logits with class-imbalance handling.

    We build a binary logit as anomaly-vs-non-anomaly log-odds:
        bin_logit = logit_anom - logsumexp(logits_non_anom)

    Then we use BCEWithLogits with a pos_weight = neg/pos to avoid the trivial
    solution 'predict all normal'.
    """
    if logits.dim() != 4:
        raise RuntimeError(f"Expected logits [B,C,H,W], got {tuple(logits.shape)}")

    C = logits.size(1)
    if ANOMALY_CH < 0 or ANOMALY_CH >= C:
        raise RuntimeError(f"ANOMALY_CH={ANOMALY_CH} out of range for C={C}")

    y01 = y01.float()
    logit_a = logits[:, ANOMALY_CH]  # [B,H,W]

    if ANOMALY_CH == C - 1:
        logits_na = logits[:, :ANOMALY_CH]
    elif ANOMALY_CH == 0:
        logits_na = logits[:, 1:]
    else:
        logits_na = torch.cat([logits[:, :ANOMALY_CH], logits[:, ANOMALY_CH + 1 :]], dim=1)

    logit_na = torch.logsumexp(logits_na, dim=1)  # [B,H,W]
    bin_logit = logit_a - logit_na

    # class imbalance: pos_weight = neg/pos
    pos = y01.sum()
    neg = y01.numel() - pos
    if pos < 1:
        # No positives -> nothing to learn; return zero but keep graph
        return (bin_logit * 0.0).sum()

    pos_weight = (neg / pos).clamp(min=1.0, max=1000.0).detach()

    if debug:
        with torch.no_grad():
            print(f"[debug] pos={int(pos.item())} neg={int(neg.item())} pos_weight={float(pos_weight.item()):.2f}")
            print(f"[debug] bin_logit stats: min={float(bin_logit.min()):.3f} max={float(bin_logit.max()):.3f} mean={float(bin_logit.mean()):.3f}")

    bce = F.binary_cross_entropy_with_logits(bin_logit, y01, pos_weight=pos_weight)
    dsc = dice_loss_from_logits(bin_logit, y01)

    # Weighted sum: BCE stabilizes logits; Dice directly optimizes overlap.
    return bce + 0.5 * dsc

# -------------------------------------------------
# OVERFIT LOOP
# -------------------------------------------------
print("=== START OVERFIT ===")
for it in range(1, N_ITERS + 1):
    opt.zero_grad(set_to_none=True)

    logits = model(x)  # [B,C,H,W]
    loss = loss_fn(logits, y, debug=(it == 1))
    loss.backward()
    opt.step()

    if it % 20 == 0 or it == 1:
        with torch.no_grad():
            C = logits.size(1)
            if ANOMALY_CH == C - 1:
                logits_na = logits[:, :ANOMALY_CH]
            elif ANOMALY_CH == 0:
                logits_na = logits[:, 1:]
            else:
                logits_na = torch.cat([logits[:, :ANOMALY_CH], logits[:, ANOMALY_CH + 1 :]], dim=1)

            lg = logits[:, ANOMALY_CH] - torch.logsumexp(logits_na, dim=1)
            pred = (torch.sigmoid(lg) > 0.5).long()

            if it == 1 or it % 100 == 0:
                print("[debug] pred_pos_pixels=", int(pred.sum().item()), "gt_pos_pixels=", int(y.sum().item()))

            inter = (pred & y).sum().float()
            union = (pred | y).sum().float()
            iou = (inter / (union + 1e-6)).item()

            # Soft IoU (uses probabilities) to see progress even when hard threshold is empty
            prob = torch.sigmoid(lg)
            soft_inter = (prob * y.float()).sum().item()
            soft_union = (prob + y.float() - prob * y.float()).sum().item() + 1e-6
            soft_iou = soft_inter / soft_union

        print(f"it={it:03d} | loss={loss.item():.4f} | IoU={iou:.3f} | softIoU={soft_iou:.3f} | logits={tuple(logits.shape)}")

print("=== DONE ===")