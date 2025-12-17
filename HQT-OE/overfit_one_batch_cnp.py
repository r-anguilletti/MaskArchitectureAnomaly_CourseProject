import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# -------------------------------------------------
# Ensure local imports work
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
LR = 5e-5          # più vicino al tuo training reale (Lightning lr=5e-5)
SEED = 0

MIN_ANOM_PIXELS = 2000

# Energy loss hyperparams (come nel tuo modello)
T = 1.0
M_IN = -7.0
M_OUT = -5.0
W_ENERGY = 1.0     # peso della energy loss (qui è l'unica loss)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -------------------------------------------------
# DATA: pick one sample with enough anomaly pixels
# -------------------------------------------------
ds = CNPZipDataset(ZIP_PATH)
dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

x = y = None
for try_i, (xb, yb) in enumerate(dl, start=1):
    anom = int(yb.sum().item())
    if anom >= MIN_ANOM_PIXELS:
        x, y = xb, yb
        print(f"Selected sample #{try_i} with anomaly_pixels={anom}")
        break
    if try_i >= 300:
        x, y = xb, yb
        print(f"[WARN] Could not find sample with >= {MIN_ANOM_PIXELS} anomaly pixels in 300 tries. Using anomaly_pixels={anom}")
        break

x = x.to(device)
y = y.to(device)  # [1,H,W] in {0,1}

pos = int(y.sum().item())
tot = int(y.numel())
print("Batch shapes:", x.shape, y.shape)
print("Anomaly pixels:", pos, "ratio:", pos / tot)

# -------------------------------------------------
# MODEL
# -------------------------------------------------
model = AnomalySegmenter()
model = model.to(device)
model.train()

ANOMALY_CLASS_IDX = int(getattr(model, "anomaly_class_idx", 19))
IGNORE_IDX = int(getattr(model, "ignore_index", 255))
print("Model anomaly_class_idx:", ANOMALY_CLASS_IDX, "ignore_index:", IGNORE_IDX)

# -------------------------------------------------
# OPTIM (solo parametri trainabili, utile con LoRA)
# -------------------------------------------------
opt = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR,
    weight_decay=0.0,
)

# -------------------------------------------------
# ENERGY helpers
# -------------------------------------------------
def energy_map_from_logits(seg_logits, anomaly_class_idx=19, T=1.0):
    """
    seg_logits: [B,C,H,W]
    energy = -logsumexp(logits_ID / T) over ID classes (0..anomaly_class_idx-1)
    """
    id_logits = seg_logits[:, :anomaly_class_idx]  # [B,19,H,W] if anomaly=19
    return -torch.logsumexp(id_logits / T, dim=1)  # [B,H,W]

def energy_loss(seg_logits, y01):
    """
    y01: [B,H,W] binary mask (1=anomaly, 0=normal)
    """
    E = energy_map_from_logits(seg_logits, anomaly_class_idx=ANOMALY_CLASS_IDX, T=T)

    in_mask = (y01 == 0)
    out_mask = (y01 == 1)

    loss = torch.tensor(0.0, device=seg_logits.device)

    if in_mask.any():
        loss_in = torch.mean(F.relu(E[in_mask] - M_IN) ** 2)
        loss = loss + loss_in
    else:
        loss_in = torch.tensor(0.0, device=seg_logits.device)

    if out_mask.any():
        loss_out = torch.mean(F.relu(M_OUT - E[out_mask]) ** 2)
        loss = loss + loss_out
    else:
        loss_out = torch.tensor(0.0, device=seg_logits.device)

    return loss, loss_in, loss_out, E

# -------------------------------------------------
# OVERFIT LOOP
# -------------------------------------------------
print("=== START ENERGY OVERFIT ===")
for it in range(1, N_ITERS + 1):
    opt.zero_grad(set_to_none=True)

    seg_logits = model(x).float()  # [1,20,H,W]
    loss, loss_in, loss_out, E = energy_loss(seg_logits, y)

    (W_ENERGY * loss).backward()
    opt.step()

    if it % 20 == 0 or it == 1:
        with torch.no_grad():
            in_mask = (y == 0)
            out_mask = (y == 1)

            E_in_mean = float(E[in_mask].mean().item()) if in_mask.any() else float("nan")
            E_out_mean = float(E[out_mask].mean().item()) if out_mask.any() else float("nan")

            # quanto rispetta i margini?
            in_ok = float((E[in_mask] <= M_IN).float().mean().item()) if in_mask.any() else float("nan")
            out_ok = float((E[out_mask] >= M_OUT).float().mean().item()) if out_mask.any() else float("nan")

        print(
            f"it={it:03d} | loss={float(loss.item()):.4f} "
            f"(in={float(loss_in.item()):.4f}, out={float(loss_out.item()):.4f}) | "
            f"E_in_mean={E_in_mean:.3f} E_out_mean={E_out_mean:.3f} | "
            f"in_ok={in_ok:.3f} out_ok={out_ok:.3f}"
        )

print("=== DONE ===")