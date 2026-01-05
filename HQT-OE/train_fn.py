# train_qheads_ft.py
import os
import argparse
import torch
from torch.utils.data import DataLoader, IterableDataset, Subset
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

# IMPORTANT: this is the "variant" segmenter file (keeps segmenter.py clean)
from models.segmenter_qheads_ft import AnomalySegmenter

from cnp_zip_dataset import CNPZipDataset
from train_sanity import CNPResizeWrapper, CityscapesFolderLabelIdsToTrainIds


def cycle(dl):
    while True:
        for b in dl:
            yield b


class MixedFiniteIterable(IterableDataset):
    """
    Mix City (ID) and CNP (OE/OOD) in a fixed pattern, for a finite number of steps/epoch.
    """
    def __init__(self, dl_city, dl_cnp, mix_city=3, mix_cnp=1, steps_per_epoch=200):
        super().__init__()
        self.city = cycle(dl_city) if dl_city is not None else None
        self.cnp = cycle(dl_cnp)
        self.pattern = ([0] * int(mix_city)) + ([1] * int(mix_cnp))  # 0=city, 1=cnp
        self.steps_per_epoch = int(steps_per_epoch)

    def __iter__(self):
        n = 0
        while n < self.steps_per_epoch:
            for p in self.pattern:
                if n >= self.steps_per_epoch:
                    break
                if p == 0:
                    yield next(self.city) if self.city is not None else next(self.cnp)
                else:
                    yield next(self.cnp)
                n += 1


class DebugCallback(L.Callback):
    def __init__(self, every_n_steps=20):
        self.every_n_steps = every_n_steps
        self.seen_city = 0
        self.seen_cnp = 0

    @staticmethod
    def _source_to_int(source):
        if isinstance(source, (list, tuple)):
            source = source[0]
        if torch.is_tensor(source):
            return int(source.flatten()[0].item())
        if isinstance(source, str):
            return 0 if source == "city" else 1
        return int(source)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        img, mask, source = batch
        source_i = self._source_to_int(source)

        if source_i == 0 and self.seen_city < 1:
            self.seen_city += 1
            valid = (mask != pl_module.ignore_index)
            vp = float(valid.float().mean().item() * 100.0)
            uq = torch.unique(mask[valid]).detach().cpu().tolist() if valid.any() else []
            print(f"[DBG first city] img={tuple(img.shape)} mask={tuple(mask.shape)} valid%={vp:.2f} uniq(sample)={uq[:15]}")

        if source_i == 1 and self.seen_cnp < 1:
            self.seen_cnp += 1
            uq = torch.unique(mask).detach().cpu().tolist()
            ap = float((mask > 0).float().mean().item() * 100.0)
            print(f"[DBG first oe]   img={tuple(img.shape)} mask={tuple(mask.shape)} anom%={ap:.2f} uniq={uq}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step and trainer.global_step % self.every_n_steps == 0:
            m = trainer.callback_metrics

            def fmt(x):
                if x is None:
                    return "NA"
                try:
                    return f"{float(x):.4f}"
                except Exception:
                    return "NA"

            print(
                f"[DBG step={trainer.global_step}] "
                f"loss_ce={fmt(m.get('train/loss_ce'))} "
                f"loss_en={fmt(m.get('train/loss_energy'))} "
                f"train_mIoU={fmt(m.get('train/mIoU'))} "
                f"val_mIoU={fmt(m.get('val_city/mIoU'))} "
                f"E_sep={fmt(m.get('train/energy_sep'))}"
            )


def _maybe_subset(ds, limit: int):
    if limit is None or limit <= 0:
        return ds
    try:
        n = len(ds)
        k = min(int(limit), int(n))
        return Subset(ds, list(range(k)))
    except Exception:
        return ds


def main():
    ap = argparse.ArgumentParser()

    # split CNP (best practice)
    ap.add_argument("--cnp_zip_train", type=str, required=True, help="Path to CNP train zip (80%)")
    ap.add_argument("--cnp_zip_val", type=str, default=None, help="Path to CNP val zip (20%) (optional)")

    ap.add_argument("--city_root", type=str, required=True)

    ap.add_argument("--img_h", type=int, default=1024)
    ap.add_argument("--img_w", type=int, default=1024)

    ap.add_argument("--batch_city", type=int, default=1)
    ap.add_argument("--batch_cnp", type=int, default=1)
    ap.add_argument("--mix_city", type=int, default=3)
    ap.add_argument("--mix_cnp", type=int, default=1)

    ap.add_argument("--steps_per_epoch", type=int, default=200)
    ap.add_argument("--max_epochs", type=int, default=5)

    # FT: usually lower LR
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--ckpt_dir", type=str, default="./checkpoints_ft")
    ap.add_argument("--log_dir", type=str, default="./logs_ft")
    ap.add_argument("--seed", type=int, default=0)

    # limit validation images (PER DATASET)
    ap.add_argument("--val_limit", type=int, default=500,
                    help="Limit number of validation images PER dataset (City val and CNP val)")

    # energy (still configurable)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--m_in", type=float, default=-12.0)
    ap.add_argument("--m_out", type=float, default=-2.0)
    ap.add_argument("--lambda_energy", type=float, default=0.15)

    # in FT typically 0 (you already have good weights). keep as arg anyway
    ap.add_argument("--warmup_epochs", type=int, default=0)

    # backbone config (must match the ckpt / eomt)
    ap.add_argument("--backbone_name", type=str, default="vit_base_patch14_reg4_dinov2")
    ap.add_argument("--num_blocks", type=int, default=3)
    ap.add_argument("--num_queries", type=int, default=100)

    # REQUIRED: fine-tune from this Lightning checkpoint
    ap.add_argument("--ft_ckpt", type=str, required=True, help="Path to the .ckpt to fine-tune from")

    # Optional: just validate the FT checkpoint
    ap.add_argument("--eval_only", action="store_true",
                    help="Run validation only on BOTH (City val + CNP val if provided) then exit")

    args = ap.parse_args()

    L.seed_everything(args.seed, workers=True)
    img_size = (args.img_h, args.img_w)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # ----------------------------
    # Datasets / loaders
    # ----------------------------
    cnp_train_ds = CNPResizeWrapper(CNPZipDataset(args.cnp_zip_train), img_size=img_size)
    dl_cnp_train = DataLoader(
        cnp_train_ds,
        batch_size=args.batch_cnp,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    dl_cnp_val = None
    if args.cnp_zip_val is not None:
        cnp_val_ds = CNPResizeWrapper(CNPZipDataset(args.cnp_zip_val), img_size=img_size)
        cnp_val_ds = _maybe_subset(cnp_val_ds, args.val_limit)
        dl_cnp_val = DataLoader(
            cnp_val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    city_train_ds = CityscapesFolderLabelIdsToTrainIds(args.city_root, split="train", img_size=img_size)
    dl_city = DataLoader(
        city_train_ds,
        batch_size=args.batch_city,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    city_val_ds = CityscapesFolderLabelIdsToTrainIds(args.city_root, split="val", img_size=img_size)
    city_val_ds = _maybe_subset(city_val_ds, args.val_limit)
    dl_city_val = DataLoader(
        city_val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    mixed_iter = MixedFiniteIterable(
        dl_city=dl_city,
        dl_cnp=dl_cnp_train,
        mix_city=args.mix_city,
        mix_cnp=args.mix_cnp,
        steps_per_epoch=args.steps_per_epoch,
    )
    dl_train = DataLoader(mixed_iter, batch_size=None, num_workers=0)

    # ----------------------------
    # Model (FT variant)
    # ----------------------------
    model = AnomalySegmenter(
        img_size=img_size,
        lr=args.lr,
        T=args.T,
        m_in=args.m_in,
        m_out=args.m_out,
        lambda_energy=args.lambda_energy,
        warmup_epochs=args.warmup_epochs,
        backbone_name=args.backbone_name,
        num_queries=args.num_queries,
        num_blocks=args.num_blocks,
        patch_size=16,
        train_epochs=args.max_epochs,

        # IMPORTANT: in fine-tuning we DO NOT re-init from .bin
        pretrained_eomt_bin=None,
        use_lora=True,
    )

    # ----------------------------
    # Callbacks / trainer
    # ----------------------------
    ckpt = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename="bestOOD-ft-{epoch:02d}-{step:06d}",
        save_last=True,
        save_top_k=2,
        monitor="val_ood/auprc_msp",
        mode="max",
        every_n_epochs=1,
    )

    lrmon = LearningRateMonitor(logging_interval="step")
    dbg = DebugCallback(every_n_steps=20)

    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        max_epochs=args.max_epochs,
        callbacks=[ckpt, lrmon, dbg],
        default_root_dir=args.log_dir,
        log_every_n_steps=10,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="norm",
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,
    )

    val_loaders = [dl_city_val]
    if dl_cnp_val is not None:
        val_loaders.append(dl_cnp_val)

    if args.eval_only:
        print("\n[INFO] Running validation ONLY on BOTH (City val + CNP val) from ft_ckpt...\n")
        results = trainer.validate(
            model,
            dataloaders=val_loaders,
            ckpt_path=args.ft_ckpt,
            verbose=True,
        )
        print("\n[INFO] Validate returned:")
        print(results)
        return

    # ----------------------------
    # Fine-tune from checkpoint
    # ----------------------------
    print(f"\n[INFO] Fine-tuning from ckpt: {args.ft_ckpt}\n")
    trainer.fit(
        model,
        train_dataloaders=dl_train,
        val_dataloaders=val_loaders,
        ckpt_path=args.ft_ckpt,  # <- THIS is the FT starting point
    )


if __name__ == "__main__":
    main()