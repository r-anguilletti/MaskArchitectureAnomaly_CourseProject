import os, shutil, zipfile, random, time
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from datasets.hybrid_anomaly import HybridAnomalyDataset

PATH_CITY = "/content/Cityscapes_Local"
PATH_COCO = "/content/COCO_Local"

IMG_SIZE = (518, 518)         
N_SAMPLES = 10000              
ANOMALY_CLASS_ID = 19          

TMP_OUT = Path("/content/tmp_cnp_from_hybrid")
OUT_IMG = TMP_OUT / "images"
OUT_MSK = TMP_OUT / "masks"

ZIP_LOCAL = Path("/content/cnp_dataset.zip")
ZIP_DRIVE = Path("/content/drive/MyDrive/Anomaly_Segmentation/cnp_dataset.zip")

ZIP_COMPRESSLEVEL = 3

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def denormalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    # x: [3,H,W], normalized ImageNet
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(3, 1, 1)
    y = x * std + mean
    return y.clamp(0, 1)


def reset_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def main():
    print("=== Pre-generate dataset from HybridAnomalyDataset ===")
    print("CITY:", PATH_CITY)
    print("COCO:", PATH_COCO)
    print("IMG_SIZE:", IMG_SIZE)
    print("N_SAMPLES:", N_SAMPLES)

    reset_dir(TMP_OUT)
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_MSK.mkdir(parents=True, exist_ok=True)

    # 1) init dataset
    ds = HybridAnomalyDataset(PATH_CITY, PATH_COCO, img_size=IMG_SIZE)
    print("Dataset initialized. len(ds) =", len(ds))

    t0 = time.time()
    anomalies = 0

    for i in range(N_SAMPLES):
        # random sample
        idx = random.randint(0, len(ds) - 1)

        img_t, mask_t = ds[idx] 

        # img_t: [3,H,W] float tensor (normalized)
        # mask_t: [H,W] long tensor con classi, anomalia = 19
        # salva immagine visibile (denorm -> uint8)
        img_vis = denormalize_imagenet(img_t).mul(255).byte().permute(1, 2, 0).cpu().numpy()
        Image.fromarray(img_vis).save(OUT_IMG / f"{i:07d}.png")

        m = (mask_t == ANOMALY_CLASS_ID).to(torch.uint8).mul(255).cpu().numpy()
        Image.fromarray(m).save(OUT_MSK / f"{i:07d}.png")

        if m.sum() > 0:
            anomalies += 1

        if (i + 1) % 200 == 0:
            elapsed = (time.time() - t0) / 60
            print(f"[{i+1}/{N_SAMPLES}] anomalies_in_samples={anomalies}  ({100*anomalies/(i+1):.1f}%)  elapsed={elapsed:.1f} min")

    # 2) zip
    if ZIP_LOCAL.exists():
        ZIP_LOCAL.unlink()

    print("📦 Zipping...")
    with zipfile.ZipFile(ZIP_LOCAL, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=ZIP_COMPRESSLEVEL) as z:
        for p in TMP_OUT.rglob("*"):
            if p.is_file():
                z.write(p, arcname=p.relative_to(TMP_OUT))

    ZIP_DRIVE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ZIP_LOCAL, ZIP_DRIVE)
    print("DONE")
    print("Saved zip to:", ZIP_DRIVE)


if __name__ == "__main__":
    main()

