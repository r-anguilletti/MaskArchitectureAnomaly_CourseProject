# ===== DEBUG FALSE POSITIVES (LostAndFound) - Colab cell (FIXED IMPORTS) =====
import os, sys, glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor

# -----------------------------
# ✅ EDITA QUESTI 3
# -----------------------------
# Metti QUI la cartella che contiene models/ e eomt/
# (quindi NON HQT-OE, ma la root "MaskArchitectureAnomaly_CourseProject")
PROJECT_ROOT = "/content/drive/MyDrive/Anomaly_Segmentation/MaskArchitectureAnomaly_CourseProject"
CKPT_PATH    = "/content/drive/MyDrive/Anomaly_Segmentation/MaskArchitectureAnomaly_CourseProject/HQT-OE/checkpoints/last.ckpt"
IMG_GLOB     = "/content/drive/MyDrive/Anomaly_Segmentation/FS_LostFound_full/images/*.png"
# -----------------------------

METHOD = "msp"
IMG_SIZE = (1024, 1024)  # (H,W)
TOPK = 12
MAX_SCAN = 300
THRESH = 0.50

# ---- PATH SETUP (Colab-safe)
EOMT_ROOT = os.path.join(PROJECT_ROOT, "eomt")

assert os.path.isdir(PROJECT_ROOT), f"PROJECT_ROOT non esiste: {PROJECT_ROOT}"
assert os.path.isdir(os.path.join(EOMT_ROOT, "models")), f"Non trovo {EOMT_ROOT}/models"
assert os.path.isdir(EOMT_ROOT), f"Non trovo {EOMT_ROOT}"

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if EOMT_ROOT not in sys.path:
    sys.path.insert(0, EOMT_ROOT)

print("sys.path[0:3] =", sys.path[:3])
print("PROJECT_ROOT =", PROJECT_ROOT)
print("EOMT_ROOT    =", EOMT_ROOT)

from models.segmenter import AnomalySegmenter
from training.lightning_module import LightningModule as EoMTLightningModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---- transforms
input_tf = Compose([Resize(IMG_SIZE, Image.BILINEAR), ToTensor()])
target_tf = Compose([Resize(IMG_SIZE, Image.NEAREST)])

def map_gt_lostandfound(gt_arr: np.ndarray) -> np.ndarray:
    ood_gts = gt_arr
    ood_gts = np.where((ood_gts == 0), 255, ood_gts)
    ood_gts = np.where((ood_gts == 1), 0, ood_gts)
    ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)
    return ood_gts

@torch.no_grad()
def anomaly_map_from_model(model, img_tensor, method: str):
    mask_logits_layers, class_logits_layers = model.model(img_tensor)
    mask_logits = mask_logits_layers[-1]
    class_logits = class_logits_layers[-1]

    if mask_logits.shape[-2:] != IMG_SIZE:
        mask_logits = F.interpolate(mask_logits, size=IMG_SIZE, mode="bilinear", align_corners=False)

    per_pixel_logits = EoMTLightningModule.to_per_pixel_logits_semantic(mask_logits, class_logits)
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
        raise ValueError("Unknown method")

    return anomaly.detach().cpu().numpy()

# ---- load model
model = AnomalySegmenter.load_from_checkpoint(
    CKPT_PATH, map_location="cpu", strict=True, pretrained_eomt_bin=None
).to(device).eval()
print("Loaded ckpt:", CKPT_PATH)

# ---- scan images
files = sorted(glob.glob(IMG_GLOB))
print("Found images:", len(files))
if MAX_SCAN is not None:
    files = files[:MAX_SCAN]

rows = []
for p in files:
    img_pil = Image.open(p).convert("RGB")
    x = input_tf(img_pil).unsqueeze(0).float().to(device)
    a = anomaly_map_from_model(model, x, METHOD)

    gt_p = p.replace("images", "labels_masks").replace(".jpg", ".png").replace(".webp", ".png")
    if not os.path.exists(gt_p):
        continue

    gt = np.array(target_tf(Image.open(gt_p)))
    gt = map_gt_lostandfound(gt)

    # skip identico al prof
    if 1 not in np.unique(gt):
        continue

    valid = (gt != 255)
    id_mask = (gt == 0) & valid
    ood_mask = (gt == 1) & valid

    fp_mean = float(a[id_mask].mean()) if id_mask.any() else 0.0
    tp_mean = float(a[ood_mask].mean()) if ood_mask.any() else 0.0
    fp_rate = float((a[id_mask] > THRESH).mean()) if id_mask.any() else 0.0
    tp_rate = float((a[ood_mask] > THRESH).mean()) if ood_mask.any() else 0.0

    rows.append((fp_mean, fp_rate, tp_mean, tp_rate, p, gt_p))

rows = sorted(rows, key=lambda t: (t[1], t[0]), reverse=True)
print(f"Kept {len(rows)} images with OOD pixels (after prof-style skip).")

def show_case(img_path, gt_path):
    img = np.array(Image.open(img_path).convert("RGB").resize((IMG_SIZE[1], IMG_SIZE[0])))
    gt = np.array(target_tf(Image.open(gt_path)))
    gt = map_gt_lostandfound(gt)

    x = input_tf(Image.open(img_path).convert("RGB")).unsqueeze(0).float().to(device)
    a = anomaly_map_from_model(model, x, METHOD)

    valid = (gt != 255)
    fp_mask = (a > THRESH) & (gt == 0) & valid

    fig, axs = plt.subplots(1, 4, figsize=(18, 5))
    axs[0].imshow(img); axs[0].set_title("RGB"); axs[0].axis("off")
    axs[1].imshow(a); axs[1].set_title(f"Anomaly map ({METHOD})"); axs[1].axis("off")
    axs[2].imshow((gt == 1).astype(np.uint8)); axs[2].set_title("GT OOD"); axs[2].axis("off")
    axs[3].imshow(fp_mask.astype(np.uint8)); axs[3].set_title(f"FP mask (>{THRESH})"); axs[3].axis("off")
    plt.show()

for r in rows[:TOPK]:
    fp_mean, fp_rate, tp_mean, tp_rate, img_p, gt_p = r
    print(f"\n=== {os.path.basename(img_p)} ===")
    print(f"fp_rate={fp_rate:.3f} fp_mean={fp_mean:.3f} | tp_rate={tp_rate:.3f} tp_mean={tp_mean:.3f}")
    show_case(img_p, gt_p)