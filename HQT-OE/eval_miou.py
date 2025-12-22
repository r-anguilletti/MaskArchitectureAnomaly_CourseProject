# eval_miou_cityscapes.py
import os
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np

from torchmetrics.classification import MulticlassJaccardIndex

from models.segmenter import AnomalySegmenter
from train_sanity import CityscapesFolderLabelIdsToTrainIds


def _to_logits(out):
    """
    Rende robusto l'output del modello:
    - se è un Tensor: ok
    - se è tuple/list: prende il primo tensor "logits-like"
    """
    if torch.is_tensor(out):
        return out
    if isinstance(out, (list, tuple)):
        for x in out:
            if torch.is_tensor(x):
                return x
    raise RuntimeError(f"Output modello non riconosciuto: {type(out)}")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path a last.ckpt o checkpoint .ckpt")
    ap.add_argument("--city_root", required=True, help="Root Cityscapes_Local (come training)")
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--img_h", type=int, default=518)
    ap.add_argument("--img_w", type=int, default=518)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="Se >0 valuta solo N batch")
    ap.add_argument("--debug_first", action="store_true", help="Stampa statistiche sul primo batch")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = (args.img_h, args.img_w)

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint non trovato: {args.ckpt}")

    print(f"\n--- Eval mIoU Cityscapes | device={device} | split={args.split} | img_size={img_size} ---")
    print(f"ckpt: {args.ckpt}\n")

    # Dataset identico al training
    ds = CityscapesFolderLabelIdsToTrainIds(
        args.city_root,
        split=args.split,
        img_size=img_size
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    # Modello identico al training
    model = AnomalySegmenter.load_from_checkpoint(args.ckpt)
    model.to(device)
    model.eval()

    num_classes = 19
    ignore_index = getattr(model, "ignore_index", 255)

    miou_metric = MulticlassJaccardIndex(
        num_classes=num_classes,
        ignore_index=ignore_index,
        average="macro"
    ).to(device)

    total_batches = 0
    total_pixels_valid = 0

    for bidx, batch in enumerate(dl):
        if args.limit > 0 and bidx >= args.limit:
            break

        # CityscapesFolderLabelIdsToTrainIds ritorna (img, mask, source)
        img, mask, _source = batch
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        out = model(img)                   # <-- STESSO forward del training
        logits = _to_logits(out)           # (B, 19, H, W) atteso
        preds = torch.argmax(logits, dim=1)

        # Update mIoU (torchmetrics gestisce ignore_index)
        miou_metric.update(preds, mask)

        valid = (mask != ignore_index)
        total_pixels_valid += int(valid.sum().item())

        if args.debug_first and bidx == 0:
            uniq_gt = torch.unique(mask[valid]).detach().cpu().tolist() if valid.any() else []
            uniq_pr = torch.unique(preds[valid]).detach().cpu().tolist() if valid.any() else []
            valid_pct = float(valid.float().mean().item() * 100.0)

            print("[DBG first batch]")
            print(" img:", tuple(img.shape))
            print(" mask:", tuple(mask.shape))
            print(f" valid%: {valid_pct:.2f}")
            print(" uniq_gt (<=25):", uniq_gt[:25])
            print(" uniq_pred(<=25):", uniq_pr[:25])
            print(" logits:", tuple(logits.shape))
            print()

        total_batches += 1

    miou = float(miou_metric.compute().item() * 100.0)

    print("---- RESULT ----")
    print(f"batches: {total_batches}")
    print(f"valid_pixels: {total_pixels_valid}")
    print(f"mIoU Cityscapes (19 classi): {miou:.2f}%\n")


if __name__ == "__main__":
    main()