import os
import time
import random
import numpy as np
import torch
import torch.nn.functional as F

# ------------------------------------------------------------
# (A) Safety / Debug utilities
# ------------------------------------------------------------
def seed_everything(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def tinfo(x: torch.Tensor, name: str):
    if x is None:
        print(f"[{name}] = None")
        return
    with torch.no_grad():
        print(
            f"[{name}] shape={tuple(x.shape)} dtype={x.dtype} device={x.device} "
            f"min={x.min().item():.4g} max={x.max().item():.4g} mean={x.float().mean().item():.4g}"
        )

def mask_stats(mask: torch.Tensor, ignore_index=255):
    # mask: [B,H,W] or [H,W]
    m = mask
    if m.dim() == 3:
        m = m[0]
    uniq = torch.unique(m)
    # show only a few uniques if too many
    uniq_list = uniq.detach().cpu().tolist()
    if len(uniq_list) > 30:
        shown = uniq_list[:30]
        extra = len(uniq_list) - 30
    else:
        shown = uniq_list
        extra = 0

    total = m.numel()
    ignore = (m == ignore_index).sum().item()
    valid = total - ignore
    print(f"[mask] total={total} valid={valid} ignore={ignore} ignore_index={ignore_index}")
    print(f"[mask] unique (first): {shown}" + (f" ... (+{extra} more)" if extra else ""))

def exists(path: str):
    ok = os.path.exists(path)
    print(f"[path] {path} -> {'OK' if ok else 'MISSING'}")
    return ok

# ------------------------------------------------------------
# (B) Import your stuff
# ------------------------------------------------------------
# Qui presumo che tu abbia già:
# - HybridAnomalyDataset
# - il tuo modello Segmenter / simile
# Adatta gli import a come li hai nel progetto.
#
# Esempio:
# from segmenter import Segmenter
# from datasets.hybrid_anomaly_dataset import HybridAnomalyDataset
#
# Nel tuo codice attuale tu usi HybridAnomalyDataset e un model custom. :contentReference[oaicite:1]{index=1}

from models.segmenter import AnomalySegmenter  # <-- se il tuo modello si chiama così, altrimenti cambia
from datasets.hybrid_anomaly import HybridAnomalyDataset  # <-- cambia se path diverso


# ------------------------------------------------------------
# (C) Config
# ------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0

CITYSCAPES_ROOT = "/content/Cityscapes_Local"
COCO_ROOT       = "/content/COCO_Local"

BATCH_SIZE = 2
IMG_SIZE = (512, 1024)   # se il tuo dataset già ridimensiona, puoi ignorare
NUM_WORKERS = 0          # IMPORTANTISSIMO per evitare freeze mentre debugghi
PIN_MEMORY = False

IGNORE_INDEX = 255       # tipico; se il tuo dataset usa altro, cambia
LR = 1e-4
MAX_STEPS = 200
FIND_MIXED_BATCH_TRIES = 200  # niente while True infinito

# ------------------------------------------------------------
# (D) Main
# ------------------------------------------------------------
def main():
    seed_everything(SEED)
    print(f"[env] device={DEVICE} torch={torch.__version__}")
    print(f"[cfg] bs={BATCH_SIZE} num_workers={NUM_WORKERS} pin_memory={PIN_MEMORY}")
    print(f"[cfg] cityscapes={CITYSCAPES_ROOT}")
    print(f"[cfg] coco={COCO_ROOT}")

    # --- quick path checks
    exists(CITYSCAPES_ROOT)
    exists(COCO_ROOT)
    # controllo rapido COCO annotations (spesso è qui che si impalla)
    exists(os.path.join(COCO_ROOT, "annotations"))
    exists(os.path.join(COCO_ROOT, "annotations", "instances_train2017.json"))

    print("\n📦 Building dataset ...")
    t0 = time.time()
    dataset = HybridAnomalyDataset(
        cityscapes_root=CITYSCAPES_ROOT,
        coco_root=COCO_ROOT,
        # aggiungi qui eventuali parametri che usi tu (split, transforms, ecc.)
    )
    print(f"✅ Dataset built in {time.time()-t0:.2f}s  len={len(dataset)}")

    print("\n📦 Building dataloader ...")
    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
        persistent_workers=False,
    )

    # --- sanity: prendo 1 batch e stampo roba
    print("\n🧪 Fetching first batch ...")
    t0 = time.time()
    batch = next(iter(loader))
    print(f"✅ First batch fetched in {time.time()-t0:.2f}s")

    # assumo che dataset ritorni (img, mask) oppure dict; gestisco entrambi
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        imgs, masks = batch[0], batch[1]
    elif isinstance(batch, dict):
        imgs, masks = batch["image"], batch["mask"]
    else:
        raise RuntimeError(f"Batch format non riconosciuto: type={type(batch)}")

    tinfo(imgs, "imgs")
    tinfo(masks, "masks")
    mask_stats(masks, ignore_index=IGNORE_INDEX)

    # ------------------------------------------------------------
    # Cerca un batch "misto" (ID + anomaly) ma con limite tentativi
    # ------------------------------------------------------------
    print("\n🔎 Searching a mixed batch (has_id && has_anom) ...")
    mixed = None
    it = iter(loader)
    for k in range(1, FIND_MIXED_BATCH_TRIES + 1):
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader)
            b = next(it)

        if isinstance(b, (list, tuple)) and len(b) >= 2:
            imgs_k, masks_k = b[0], b[1]
        else:
            imgs_k, masks_k = b["image"], b["mask"]

        # convenzione tua attuale: anomaly == 1, ID == 0 (?)  (adatta se diverso)
        # Nel tuo script stai controllando proprio "has_id" e "has_anom". :contentReference[oaicite:2]{index=2}
        has_id   = (masks_k == 0).any().item()
        has_anom = (masks_k == 1).any().item()

        if k <= 10 or k % 20 == 0:
            # debug progress
            uniq = torch.unique(masks_k).detach().cpu().tolist()
            print(f"  try {k:03d}: has_id={has_id} has_anom={has_anom} uniques={uniq[:15]}{'...' if len(uniq)>15 else ''}")

        if has_id and has_anom:
            mixed = (imgs_k, masks_k)
            print(f"✅ Found mixed batch at try={k}")
            break

    if mixed is None:
        print("\n❌ Non ho trovato un batch misto entro il limite.")
        print("   Questo spiega perché nel tuo script originale poteva sembrare 'bloccato'.")
        print("   Azioni: controlla che il dataset produca davvero label 0/1 e che anomaly non sia sempre assente.")
        return

    imgs, masks = mixed

    # ------------------------------------------------------------
    # Model + one-batch overfit
    # ------------------------------------------------------------
    print("\n🧠 Building model ...")
    model = AnomalySegmenter()  # <-- se serve config/args, mettili qui
    model = model.to(DEVICE)
    model.train()

    optim = torch.optim.AdamW(model.parameters(), lr=LR)

    # move batch to device
    imgs  = imgs.to(DEVICE, non_blocking=False)
    masks = masks.to(DEVICE, non_blocking=False)

    print("\n🚀 Starting overfit on ONE batch ...")
    print("   (Se la loss non scende, il problema è nel pipeline: labels, loss, forward, ecc.)")

    # se hai logits [B,C,H,W], useremo cross entropy
    # se hai output diverso, adatta questa parte
    for step in range(1, MAX_STEPS + 1):
        optim.zero_grad(set_to_none=True)

        out = model(imgs)

        # out può essere dict o tensor, gestiamo
        logits = out["logits"] if isinstance(out, dict) and "logits" in out else out

        if step == 1:
            tinfo(logits, "logits(step1)")
            # controlla match shape
            print(f"[debug] masks shape = {tuple(masks.shape)}  logits shape = {tuple(logits.shape)}")

        # Se masks è [B, H, W] ok; se [B,1,H,W] squeeze
        if masks.dim() == 4 and masks.size(1) == 1:
            masks_ce = masks[:, 0]
        else:
            masks_ce = masks

        # IMPORTANT: masks deve essere long per CE
        masks_ce = masks_ce.long()

        loss = F.cross_entropy(logits, masks_ce, ignore_index=IGNORE_INDEX)

        loss.backward()
        optim.step()

        if step <= 10 or step % 20 == 0:
            print(f"step {step:03d}/{MAX_STEPS}  loss={loss.item():.6f}")

    print("\n✅ Done. Se loss è scesa tanto, training pipeline OK.")
    print("   Se loss non scende: mismatch label/ignore_index, output del modello, o dataset masks.")

if __name__ == "__main__":
    main()
