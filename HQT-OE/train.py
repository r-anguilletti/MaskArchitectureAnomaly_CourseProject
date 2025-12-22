# train.py
import os
import argparse
import torch
from torch.utils.data import DataLoader, IterableDataset
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from models.segmenter import AnomalySegmenter

from cnp_zip_dataset import CNPZipDataset
from train_sanity import CityscapesZipLabelIdsToTrainIds, CNPResizeWrapper, CityscapesFolderLabelIdsToTrainIds


def cycle(dl):
    while True:
        for b in dl:
            yield b


class MixedFiniteIterable(IterableDataset):
    def __init__(self, dl_city, dl_cnp, mix_city=3, mix_cnp=1, steps_per_epoch=200):
        super().__init__()
        self.city = cycle(dl_city) if dl_city is not None else None
        self.cnp = cycle(dl_cnp)
        self.pattern = ([0] * int(mix_city)) + ([1] * int(mix_cnp))  # ✅ 0=city, 1=cnp
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
    def _stats_mask_city(mask, ignore_index=255):
        valid = (mask != ignore_index)
        valid_pct = float(valid.float().mean().item() * 100.0)
        uniq = torch.unique(mask[valid]).detach().cpu().tolist() if valid.any() else []
        return valid_pct, uniq[:25]

    @staticmethod
    def _stats_mask_cnp(mask01):
        uniq = torch.unique(mask01).detach().cpu().tolist()
        anom_pct = float((mask01 == 1).float().mean().item() * 100.0)
        return anom_pct, uniq

    @staticmethod
    def _source_to_int(source):
        # può arrivare come: 0/1, tensor([0,0]), ["city","city"], [0,0], "city"
        if isinstance(source, (list, tuple)):
            source = source[0]

        if torch.is_tensor(source):
            return int(source.flatten()[0].item())

        if isinstance(source, str):
            return 0 if source == "city" else 1

        return int(source)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        img, mask, source = batch

        # source può essere int o tensor batchato -> portiamolo a int
        source_i = self._source_to_int(source)

        if source_i == 0 and self.seen_city < 2:
            self.seen_city += 1
            vp, uq = self._stats_mask_city(mask, ignore_index=pl_module.ignore_index)
            print(
                f"[DBG first city] step={trainer.global_step} img={tuple(img.shape)} mask={tuple(mask.shape)} "
                f"valid%={vp:.1f} uniq={uq}"
            )

        if source_i == 1 and self.seen_cnp < 2:
            self.seen_cnp += 1
            ap, uq = self._stats_mask_cnp(mask)
            print(
                f"[DBG first cnp] step={trainer.global_step} img={tuple(img.shape)} mask={tuple(mask.shape)} "
                f"anom%={ap:.2f} uniq={uq}"
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step == 0:
            return

        if trainer.global_step % self.every_n_steps == 0:
            metrics = trainer.callback_metrics
            tl_ce = metrics.get("train/loss_ce")
            tl_en = metrics.get("train/loss_energy")
            miou = metrics.get("train/mIoU")
            vmiou = metrics.get("val/mIoU")
            e_sep = metrics.get("train/energy_sep")
            e_in = metrics.get("train/energy_in")
            e_out = metrics.get("train/energy_out")

            def fmt(x):
                if x is None:
                    return "NA"
                try:
                    return f"{float(x):.4f}"
                except Exception:
                    return "NA"

            print(
                f"[DBG step={trainer.global_step}] "
                f"loss_ce={fmt(tl_ce)} loss_energy={fmt(tl_en)} "
                f"train_mIoU={fmt(miou)} val_mIoU={fmt(vmiou)} "
                f"E_sep={fmt(e_sep)} E_in={fmt(e_in)} E_out={fmt(e_out)}"
            )

        img, mask, source = batch
        source_i = self._source_to_int(source)

        if source_i == 1:
            uniq = torch.unique(mask).detach().cpu().tolist()
            if uniq == [0]:
                print(f"[WARN] OE batch step={trainer.global_step} uniq=[0] (no anomalies). ...")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnp_zip", type=str, required=True)
    ap.add_argument("--city_root", type=str, required=True)
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

    ap.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    ap.add_argument("--log_dir", type=str, default="./logs")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--m_in", type=float, default=-12.0)
    ap.add_argument("--m_out", type=float, default=-6.0)
    ap.add_argument("--lambda_energy", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=3.0)
    ap.add_argument("--alpha", type=float, default=5.0)

    # ✅ RESUME OPTIONS
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last.ckpt in ckpt_dir (if exists).",
    )
    ap.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Optional explicit checkpoint path to resume from (overrides --resume).",
    )

    args = ap.parse_args()

    L.seed_everything(args.seed, workers=True)
    img_size = (args.img_h, args.img_w)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # ----------------------------
    # Datasets / loaders
    # ----------------------------
    # IMPORTANT: datasets should return:
    #   City: (img, trainIdsMask, 0)
    #   CNP : (img, oeMask01,     1)

    cnp_ds = CNPResizeWrapper(CNPZipDataset(args.cnp_zip), img_size=img_size)
    dl_cnp = DataLoader(
        cnp_ds,
        batch_size=args.batch_cnp,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    city_ds = CityscapesFolderLabelIdsToTrainIds(args.city_root, split="train", img_size=img_size)
    dl_city = DataLoader(
        city_ds,
        batch_size=args.batch_city,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_ds = CityscapesFolderLabelIdsToTrainIds(args.city_root, split="val", img_size=img_size)
    dl_val = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    mixed_iter = MixedFiniteIterable(
        dl_city=dl_city,
        dl_cnp=dl_cnp,
        mix_city=args.mix_city,
        mix_cnp=args.mix_cnp,
        steps_per_epoch=args.steps_per_epoch,
    )
    dl_train = DataLoader(mixed_iter, batch_size=None, num_workers=0)

    # ----------------------------
    # Model
    # ----------------------------
    model = AnomalySegmenter(
        img_size=img_size,
        lr=args.lr,
        T=args.T,
        m_in=args.m_in,
        m_out=args.m_out,
        lambda_energy=args.lambda_energy,
        gamma=args.gamma,
        alpha=args.alpha,
        use_balanced_energy=True,
    )

    # ----------------------------
    # Callbacks
    # ----------------------------
    ckpt = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename="seg-{epoch:02d}-{step:06d}",
        save_last=True,
        save_top_k=2,
        monitor="val/mIoU",
        mode="max",
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

    # ----------------------------
    # Resume logic
    # ----------------------------
    resume_ckpt = None
    if args.ckpt_path is not None:
        resume_ckpt = args.ckpt_path
        print(f"[RESUME] Using explicit ckpt: {resume_ckpt}")
    elif args.resume:
        last_path = os.path.join(args.ckpt_dir, "last.ckpt")
        if os.path.exists(last_path):
            resume_ckpt = last_path
            print(f"[RESUME] Resuming from: {resume_ckpt}")
        else:
            print(f"[RESUME] --resume set but no last.ckpt found in {args.ckpt_dir}. Starting fresh.")

    trainer.fit(model, train_dataloaders=dl_train, val_dataloaders=dl_val, ckpt_path=resume_ckpt)


if __name__ == "__main__":
    main()