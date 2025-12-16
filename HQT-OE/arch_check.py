import torch
from torch.utils.data import DataLoader

from datasets.hybrid_anomaly_dataset import HybridAnomalyDataset
from models.anomaly_segmenter import AnomalySegmenter


# ============================================================
# CONFIG
# ============================================================
CITYSCAPES_ROOT = "/content/Cityscapes_Local"
COCO_ROOT = "/content/COCO_Local"

IMG_SIZE = (518, 518)
BATCH_SIZE = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# DATASET & DATALOADER
# ============================================================
print("📦 Loading dataset...")
dataset = HybridAnomalyDataset(
    cityscapes_root=CITYSCAPES_ROOT,
    coco_root=COCO_ROOT,
    img_size=IMG_SIZE,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,     # IMPORTANT: zero per debug
    pin_memory=True,
)

batch = next(iter(loader))
imgs, masks = batch

print(f"🖼 Image batch shape: {imgs.shape}")
print(f"🧩 Mask batch shape : {masks.shape}")
print(f"🧩 Unique labels    : {torch.unique(masks)}")


# ============================================================
# MODEL
# ============================================================
print("🧠 Initializing model...")
model = AnomalySegmenter(
    img_size=IMG_SIZE,
    num_classes=20,
    anomaly_class_idx=19,
)
model.to(DEVICE)
model.train()

imgs = imgs.to(DEVICE)
masks = masks.to(DEVICE)


# ============================================================
# FORWARD
# ============================================================
print("🚀 Forward pass...")
with torch.autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
    seg_logits = model(imgs)

print(f"🔢 Logits shape: {seg_logits.shape}")

assert seg_logits.shape[:2] == (BATCH_SIZE, 20)
assert seg_logits.shape[-2:] == IMG_SIZE


# ============================================================
# LOSS
# ============================================================
print("📉 Computing losses...")
seg_logits_fp32 = seg_logits.float()

# CE (in-distribution only)
mask_ce = masks.clone()
mask_ce[mask_ce == 19] = 255

loss_ce = model.ce_loss(seg_logits_fp32, mask_ce)
loss_energy = model._energy_loss(seg_logits_fp32, masks)
loss = loss_ce + 0.1 * loss_energy

print(f"✅ CE loss     : {loss_ce.item():.4f}")
print(f"🔥 Energy loss : {loss_energy.item():.4f}")
print(f"📊 Total loss  : {loss.item():.4f}")

assert torch.isfinite(loss), "❌ LOSS IS NaN / INF"


# ============================================================
# BACKWARD
# ============================================================
print("🔄 Backward pass...")
loss.backward()

# Check gradients (LoRA + heads)
grad_ok = False
for name, p in model.named_parameters():
    if p.requires_grad and p.grad is not None:
        grad_ok = True
        print(f"✅ Grad OK: {name} | mean={p.grad.abs().mean():.2e}")
        break

assert grad_ok, "❌ No gradients flowing!"


# ============================================================
# ENERGY MAP CHECK
# ============================================================
print("🧠 Energy map sanity...")
energy_map = -torch.logsumexp(seg_logits_fp32, dim=1)

print(f"Energy stats: min={energy_map.min():.2f}, max={energy_map.max():.2f}")

assert torch.isfinite(energy_map).all()


# ============================================================
# DONE
# ============================================================
print("\n🎉 SANITY CHECK PASSED")
print("✔ Architecture works")
print("✔ Forward / Loss / Backward OK")
print("✔ Dataset ↔ Model coherent")