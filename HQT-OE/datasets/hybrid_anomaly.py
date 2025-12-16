import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T
from torchvision.transforms import functional as F
from PIL import Image

from datasets.cityscapes_semantic import CityscapesSemantic
from datasets.coco_instance import COCOInstance


class HybridAnomalyDataset(Dataset):
    def __init__(
        self,
        cityscapes_root,
        coco_root,
        img_size=(518, 518),
        anomaly_class_idx=19
    ):
        self.img_size = img_size
        self.anomaly_class_idx = anomaly_class_idx

        # --------------------------------------------------
        # ❌ CLASSI COCO DA ESCLUDERE
        # --------------------------------------------------
        # 1 = person, 3 = car
        # (aggiungi altri ID se vuoi)
        self.EXCLUDED_COCO_IDS = {1, 3}

        # --------------------------------------------------
        # LUT Cityscapes
        # --------------------------------------------------
        self.id_map = torch.full((256,), 255, dtype=torch.long)
        mapping = {
            7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
            19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
            25: 12, 26: 13, 27: 14, 28: 15, 31: 16,
            32: 17, 33: 18
        }
        for k, v in mapping.items():
            self.id_map[k] = v

        # --------------------------------------------------
        # Cityscapes
        # --------------------------------------------------
        print(f"--> Loading Cityscapes from: {cityscapes_root}")
        self.cs_module = CityscapesSemantic(
            path=cityscapes_root,
            img_size=img_size,
            color_jitter_enabled=False
        )
        self.cs_module.setup()
        self.cityscapes_ds = self.cs_module.cityscapes_train_dataset
        self.cityscapes_ds.transforms = None

        # --------------------------------------------------
        # COCO
        # --------------------------------------------------
        print(f"--> Loading COCO from: {coco_root}")
        self.coco_module = COCOInstance(
            path=coco_root,
            img_size=img_size,
            color_jitter_enabled=False
        )
        self.coco_module.setup()
        self.coco_ds = self.coco_module.train_dataset
        self.coco_ds.transforms = None

        # --------------------------------------------------
        # Normalize
        # --------------------------------------------------
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.cityscapes_ds)

    # ==================================================
    # MASK HANDLING
    # ==================================================

    def _flatten_instance_masks(self, target_raw, height, width):
        semantic = torch.full((height, width), 255, dtype=torch.long)
        masks, labels = target_raw[0], target_raw[1]

        if isinstance(masks, torch.Tensor) and masks.dim() == 3:
            for i in range(masks.shape[0]):
                semantic[masks[i] > 0] = labels[i]
        return semantic

    def _extract_mask_universal(self, target_raw, height, width):
        if isinstance(target_raw, (list, tuple)) and len(target_raw) >= 2:
            return self._flatten_instance_masks(target_raw, height, width)

        if isinstance(target_raw, Image.Image):
            return torch.from_numpy(np.array(target_raw)).long()

        if isinstance(target_raw, torch.Tensor):
            return target_raw.long()

        return torch.full((height, width), 255, dtype=torch.long)

    def _smart_encode(self, target):
        valid = (target != 255) & (target != -1)
        if not valid.any():
            return target
        if target[valid].max() > 18:
            return self.id_map[target]
        return target

    # ==================================================
    # COCO CROP
    # ==================================================

    def _crop_object_from_coco(self, img_pil, mask):
        mask = mask.bool()
        ys, xs = torch.where(mask)

        if len(ys) == 0:
            return None, None

        y1, y2 = ys.min().item(), ys.max().item()
        x1, x2 = xs.min().item(), xs.max().item()

        img_t = F.pil_to_tensor(img_pil)
        crop_img = img_t[:, y1:y2 + 1, x1:x2 + 1]
        crop_mask = mask[y1:y2 + 1, x1:x2 + 1]

        return F.to_pil_image(crop_img), crop_mask

    # ==================================================
    # CUT & PASTE
    # ==================================================

    def cut_paste(self, base_img_pil, base_lbl, anom_crop_pil, anom_crop_mask):
        base_img = F.pil_to_tensor(base_img_pil).float()
        anom_img = F.pil_to_tensor(anom_crop_pil).float()
        anom_crop_mask = anom_crop_mask.float()

        scale = random.uniform(0.2, 0.6)
        H, W = base_img.shape[-2:]
        new_h, new_w = int(H * scale), int(W * scale)

        if new_h < 5 or new_w < 5:
            return base_img_pil, base_lbl

        anom_img = F.resize(anom_img, (new_h, new_w))
        mask = F.resize(
            anom_crop_mask.unsqueeze(0),
            (new_h, new_w),
            interpolation=T.InterpolationMode.NEAREST
        ).squeeze(0) > 0

        max_y, max_x = H - new_h, W - new_w
        if max_y <= 0 or max_x <= 0:
            return base_img_pil, base_lbl

        y = random.randint(0, max_y)
        x = random.randint(0, max_x)

        base_crop = base_img[:, y:y + new_h, x:x + new_w]
        mask_3d = mask.unsqueeze(0).expand_as(base_crop)

        composed = torch.where(mask_3d, anom_img, base_crop)
        base_img[:, y:y + new_h, x:x + new_w] = composed

        lbl_crop = base_lbl[y:y + new_h, x:x + new_w]
        lbl_crop[mask] = self.anomaly_class_idx
        base_lbl[y:y + new_h, x:x + new_w] = lbl_crop

        return F.to_pil_image(base_img.byte()), base_lbl

    # ==================================================
    # GETITEM
    # ==================================================

    def __getitem__(self, idx):
        img, target_raw = self.cityscapes_ds[idx]

        if not isinstance(img, Image.Image):
            img = F.to_pil_image(img)
        img = img.convert("RGB")

        semantic = self._extract_mask_universal(
            target_raw, img.height, img.width
        )
        semantic = self._smart_encode(semantic)

        # --------------------------------------------------
        # INIEZIONE ANOMALIA (COCO)
        # --------------------------------------------------
        if random.random() < 0.7:
            for _ in range(5):
                coco_idx = random.randint(0, len(self.coco_ds) - 1)
                anom_img, anom_target = self.coco_ds[coco_idx]

                if not isinstance(anom_img, Image.Image):
                    anom_img = F.to_pil_image(anom_img)
                anom_img = anom_img.convert("RGB")

                if not (
                    isinstance(anom_target, dict)
                    and "masks" in anom_target
                    and "labels" in anom_target
                ):
                    continue

                masks = anom_target["masks"]
                labels = anom_target["labels"]

                # -------- FILTRO CLASSI COCO --------
                valid_idxs = [
                    i for i, lbl in enumerate(labels.tolist())
                    if lbl not in self.EXCLUDED_COCO_IDS
                ]

                if len(valid_idxs) == 0:
                    continue

                chosen = random.choice(valid_idxs)
                mask = masks[chosen]

                # 🔥 FIX CRITICA: allinea mask all'immagine
                if mask.shape[-2:] != (anom_img.height, anom_img.width):
                    mask = F.resize(
                        mask.unsqueeze(0),
                        (anom_img.height, anom_img.width),
                        interpolation=T.InterpolationMode.NEAREST
                    ).squeeze(0)

                anom_crop, anom_mask_crop = self._crop_object_from_coco(
                    anom_img, mask
                )

                if anom_crop is not None:
                    img, semantic = self.cut_paste(
                        img, semantic, anom_crop, anom_mask_crop
                    )

                    if (semantic == self.anomaly_class_idx).any():
                        break

        # --------------------------------------------------
        # OUTPUT
        # --------------------------------------------------
        img = F.resize(img, self.img_size)
        img = F.pil_to_tensor(img).float() / 255.0
        img = self.normalize(img)

        semantic = F.resize(
            semantic.unsqueeze(0),
            self.img_size,
            interpolation=T.InterpolationMode.NEAREST
        ).squeeze(0)

        return img, semantic.long()