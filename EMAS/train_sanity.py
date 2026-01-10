# train_clean.py
import os
import zipfile
from pathlib import Path
from io import BytesIO
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from PIL import Image
from torchvision import transforms

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from models.segmenter import AnomalySegmenter
from cnp_zip_dataset import CNPZipDataset


# ---------------------------
# Cityscapes mapping labelIds -> trainIds (0..18) + 255 ignore
# ---------------------------
ID_TO_TRAINID = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
    23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18
}

def labelids_to_trainids(mask_pil: Image.Image) -> torch.Tensor:
    m = np.array(mask_pil, dtype=np.int32)
    out = np.full_like(m, 255, dtype=np.int64)
    for k, v in ID_TO_TRAINID.items():
        out[m == k] = v
    return torch.from_numpy(out).long()


# ---------------------------
# Cityscapes ZIP dataset
# return (img, mask_trainIds, "city")
# ---------------------------
class CityscapesZipLabelIdsToTrainIds(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size=(518, 518),
        zip_left="leftImg8bit_trainvaltest.zip",
        zip_gt="gtFine_trainvaltest.zip",
    ):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size

        self.zleft = zipfile.ZipFile(self.root / zip_left)
        self.zgt   = zipfile.ZipFile(self.root / zip_gt)

        self.img_names = sorted([
            n for n in self.zleft.namelist()
            if n.startswith(f"leftImg8bit/{split}/") and n.endswith("_leftImg8bit.png")
        ])

        self.mask_names = []
        for n in self.img_names:
            city = n.split("/")[-2]
            base = n.split("/")[-1].replace("_leftImg8bit.png", "")
            m = f"gtFine/{split}/{city}/{base}_gtFine_labelIds.png"
            self.mask_names.append(m)

        assert len(self.img_names) == len(self.mask_names) and len(self.img_names) > 0

        self.img_tf = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

        self.mask_tf = transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        with self.zleft.open(self.img_names[idx]) as f:
            img = Image.open(BytesIO(f.read())).convert("RGB")
            img.load()
        img_t = self.img_tf(img)

        with self.zgt.open(self.mask_names[idx]) as f:
            m = Image.open(BytesIO(f.read())).convert("L")
            m.load()
        m = self.mask_tf(m)
        mask_t = labelids_to_trainids(m)

        return img_t, mask_t, 0

class CityscapesFolderLabelIdsToTrainIds(Dataset):
    def __init__(self, root: str, split: str = "train", img_size=(518, 518)):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size

        self.left_dir = self.root / "leftImg8bit" / split
        self.gt_dir   = self.root / "gtFine" / split

        assert self.left_dir.exists(), f"Non trovo {self.left_dir}"
        assert self.gt_dir.exists(),   f"Non trovo {self.gt_dir}"

        self.img_names = sorted(self.left_dir.rglob("*_leftImg8bit.png"))
        assert len(self.img_names) > 0, f"Nessuna immagine trovata in {self.left_dir}"

        self.mask_names = []
        for p in self.img_names:
            city = p.parent.name
            base = p.name.replace("_leftImg8bit.png", "")
            m = self.gt_dir / city / f"{base}_gtFine_labelIds.png"
            self.mask_names.append(m)

        missing = [str(m) for m in self.mask_names if not m.exists()]
        if len(missing) > 0:
            raise FileNotFoundError(
                f"Mancano {len(missing)} maschere gtFine. Esempi:\n" + "\n".join(missing[:10])
            )

        self.img_tf = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        self.mask_tf = transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img = Image.open(self.img_names[idx]).convert("RGB")
        img.load()
        img_t = self.img_tf(img)

        m = Image.open(self.mask_names[idx]).convert("L")
        m.load()
        m = self.mask_tf(m)
        mask_t = labelids_to_trainids(m)

        return img_t, mask_t, "city"
    

# ---------------------------
# Wrapper resize CNP
# ---------------------------
class CNPResizeWrapper(Dataset):
    def __init__(self, base_ds: Dataset, img_size=(518, 518)):
        self.base = base_ds
        self.img_size = img_size

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, mask, src = self.base[idx]  # mask: {0,1}
        img_r = F.interpolate(img.unsqueeze(0), size=self.img_size, mode="bilinear", align_corners=False).squeeze(0)
        mask_r = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=self.img_size, mode="nearest").squeeze(0).squeeze(0).long()
        return img_r, mask_r, src


# ---------------------------
# collate_fn: source = STR
# ---------------------------
def collate_city(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    masks = torch.stack([b[1] for b in batch], dim=0)
    return imgs, masks, "city"

def collate_cnp(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    masks = torch.stack([b[1] for b in batch], dim=0)
    return imgs, masks, "cnp"


# ---------------------------
# Mixed iterator (batch-level)
# ---------------------------
def cycle(dl):
    while True:
        for b in dl:
            yield b

class MixedFiniteIterable(IterableDataset):
    def __init__(self, dl_city, dl_cnp, city_ratio=3, cnp_ratio=1, steps_per_epoch=200):
        super().__init__()
        self.city = cycle(dl_city) if dl_city is not None else None
        self.cnp  = cycle(dl_cnp)
        self.pattern = (["city"] * int(city_ratio)) + (["cnp"] * int(cnp_ratio))
        self.steps_per_epoch = int(steps_per_epoch)

    def __iter__(self):
        n = 0
        while n < self.steps_per_epoch:
            for p in self.pattern:
                if n >= self.steps_per_epoch:
                    break
                if p == "city":
                    yield next(self.city) if self.city is not None else next(self.cnp)
                else:
                    yield next(self.cnp)
                n += 1


# ---------------------------
# Debug callback
# ---------------------------
class DebugPrintCallback(L.Callback):
    def __init__(self, every_n_steps=20, first_k_per_source=2):
        super().__init__()
        self.every_n_steps = every_n_steps
        self.first_k = first_k_per_source
        self.seen = {"city": 0, "cnp": 0}

    @staticmethod
    def _mask_stats(mask, source, ignore_index=255):
        with torch.no_grad():
            if source == "city":
                valid = (mask != ignore_index)
                uniq = torch.unique(mask[valid]) if valid.any() else torch.tensor([], device=mask.device)
                return f"city valid%={(valid.float().mean().item()*100):.1f} uniq={uniq[:20].tolist()}"
            else:
                uniq = torch.unique(mask)
                anom = (mask == 1).float().mean().item() * 100
                return f"cnp anom%={anom:.2f} uniq={uniq.tolist()}"

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        img, mask, source = batch
        step = int(trainer.global_step)

        if self.seen.get(source, 0) < self.first_k:
            self.seen[source] += 1
            print(f"\n[DBG first {source}] step={step} img={tuple(img.shape)} mask={tuple(mask.shape)} "
                  f"mask_stats=({self._mask_stats(mask, source, getattr(pl_module,'ignore_index',255))})")

            if hasattr(pl_module, "energy_map"):
                with torch.no_grad():
                    logits = pl_module(img)
                    E = pl_module.energy_map(logits)
                    print(f"[DBG energy] E mean={E.mean().item():.3f} min={E.min().item():.3f} max={E.max().item():.3f}")

        if step > 0 and (step % self.every_n_steps == 0):
            m = trainer.callback_metrics
            keys = list(m.keys())

            loss = m.get("train/loss_ce", m.get("train_loss", None))
            miou = m.get("train/mIoU", m.get("train_mIoU", None))
            val_miou = m.get("val/mIoU", m.get("val_mIoU", None))

            def fmt(x):
                if x is None:
                    return "NA"
                try:
                    return f"{float(x):.4f}"
                except Exception:
                    return str(x)

            msg = f"\n[DBG step={step}] keys={keys[:12]}..."
            msg += f" | train_loss={fmt(loss)} train_mIoU={fmt(miou)} val_mIoU={fmt(val_miou)}"

            if "train/energy_sep" in m:
                msg += f" | E_sep={fmt(m['train/energy_sep'])}"
            if "train/energy_in" in m:
                msg += f" | E_in={fmt(m['train/energy_in'])}"
            if "train/energy_out" in m:
                msg += f" | E_out={fmt(m['train/energy_out'])}"
            print(msg)

            if source == "cnp":
                uniq = torch.unique(mask).tolist()
                if uniq == [0] or uniq == [1]:
                    print(f"[WARN] CNP batch step={step} ha uniq={uniq} -> statistiche out/in possono diventare nan (non è crash).")

    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        if "train/mIoU" in m or "train_mIoU" in m:
            print(f"\n[DBG epoch_end] train_mIoU={m.get('train/mIoU', m.get('train_mIoU'))}")
        if "val/mIoU" in m or "val_mIoU" in m:
            print(f"[DBG epoch_end] val_mIoU={m.get('val/mIoU', m.get('val_mIoU'))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnp_zip", type=str, required=True)
    ap.add_argument("--city_root", type=str, default="")
    ap.add_argument("--img_h", type=int, default=518)
    ap.add_argument("--img_w", type=int, default=518)
    ap.add_argument("--batch_city", type=int, default=2)
    ap.add_argument("--batch_cnp", type=int, default=1)
    ap.add_argument("--mix_city", type=int, default=3)
    ap.add_argument("--mix_cnp", type=int, default=1)
    ap.add_argument("--steps_per_epoch", type=int, default=200)
    ap.add_argument("--max_epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--ckpt_dir", type=str, default="./checkpoints_clean")
    ap.add_argument("--log_dir", type=str, default="./logs_clean")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--print_every", type=int, default=20)
    args = ap.parse_args()

    L.seed_everything(args.seed, workers=True)
    img_size = (args.img_h, args.img_w)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # ---- datasets / loaders ----
    cnp_ds = CNPResizeWrapper(CNPZipDataset(args.cnp_zip), img_size=img_size)
    dl_cnp = DataLoader(
        cnp_ds,
        batch_size=args.batch_cnp,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        collate_fn=collate_cnp,
    )

    dl_city = None
    dl_val = None
    if args.city_root and Path(args.city_root).exists():
        city_ds = CityscapesZipLabelIdsToTrainIds(args.city_root, split="train", img_size=img_size)
        dl_city = DataLoader(
            city_ds,
            batch_size=args.batch_city,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            collate_fn=collate_city,
        )
        val_ds = CityscapesZipLabelIdsToTrainIds(args.city_root, split="val", img_size=img_size)
        dl_val = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_city,
        )

    mixed_iter = MixedFiniteIterable(
        dl_city=dl_city,
        dl_cnp=dl_cnp,
        city_ratio=args.mix_city,
        cnp_ratio=args.mix_cnp,
        steps_per_epoch=args.steps_per_epoch
    )
    dl_train = DataLoader(mixed_iter, batch_size=None, num_workers=0)

    # ---- model ----
    model = AnomalySegmenter(img_size=img_size, lr=args.lr)

    # ---- callbacks ----
    ckpt = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename="clean-{epoch:02d}-{step:06d}",
        save_last=True,
        save_top_k=0,               
        every_n_train_steps=200,   
    )

    dbg = DebugPrintCallback(every_n_steps=args.print_every, first_k_per_source=2)
    lrmon = LearningRateMonitor(logging_interval="epoch")

    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        max_epochs=args.max_epochs,
        callbacks=[ckpt, dbg, lrmon],
        default_root_dir=args.log_dir,
        log_every_n_steps=10,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="norm",
        check_val_every_n_epoch=1 if dl_val is not None else 999999,
        num_sanity_val_steps=2 if dl_val is not None else 0,
        enable_checkpointing=True,
    )

    trainer.fit(model, train_dataloaders=dl_train, val_dataloaders=dl_val)


if __name__ == "__main__":
    main()