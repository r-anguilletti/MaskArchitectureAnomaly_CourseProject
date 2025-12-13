import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T
from torchvision.transforms import functional as F
from PIL import Image

# Importiamo le classi che gestiscono gli ZIP
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

        # 1. SETUP CITYSCAPES
        print(f"--> Caricamento Cityscapes Zips da: {cityscapes_root}")
        self.cs_module = CityscapesSemantic(
            path=cityscapes_root, 
            img_size=img_size, 
            color_jitter_enabled=False 
        )
        self.cs_module.setup() 
        self.cityscapes_ds = self.cs_module.cityscapes_train_dataset
        self.cityscapes_ds.transforms = None 

        # 2. SETUP COCO
        print(f"--> Caricamento COCO Zips da: {coco_root}")
        self.coco_module = COCOInstance(
            path=coco_root, 
            img_size=img_size,
            color_jitter_enabled=False
        )
        self.coco_module.setup()
        self.coco_ds = self.coco_module.val_dataset
        self.coco_ds.transforms = None 

        # 3. TRANSFORMS
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.cityscapes_ds)

    def cut_paste(self, base_img_pil, base_lbl, anom_img_pil, anom_mask):
        """
        Blending tra due immagini PIL.
        """
        # FIX: Usa pil_to_tensor invece di to_image per compatibilità
        base_img = F.pil_to_tensor(base_img_pil) # (3, H, W) uint8
        anom_img = F.pil_to_tensor(anom_img_pil) # (3, H, W) uint8
        
        if not isinstance(anom_mask, torch.Tensor):
            anom_mask = torch.as_tensor(anom_mask)

        # 1. Resize Anomalia
        scale = random.uniform(0.1, 0.5)
        h, w = base_img.shape[-2:]
        new_h, new_w = int(h * scale), int(w * scale)
        
        anom_resized = F.resize(anom_img, (new_h, new_w))
        mask_resized = F.resize(anom_mask.unsqueeze(0), (new_h, new_w), interpolation=T.InterpolationMode.NEAREST).squeeze(0)
        mask_bool = mask_resized > 0

        # 2. Posizione Random
        max_y, max_x = h - new_h, w - new_w
        if max_y <= 0 or max_x <= 0: 
            return base_img_pil, base_lbl
        
        start_y = random.randint(0, max_y)
        start_x = random.randint(0, max_x)

        # 3. Incolla
        base_crop = base_img[:, start_y:start_y+new_h, start_x:start_x+new_w]
        mask_expanded = mask_bool.expand_as(base_crop)
        base_crop[mask_expanded] = anom_resized[mask_expanded]
        base_img[:, start_y:start_y+new_h, start_x:start_x+new_w] = base_crop

        # 4. Aggiorna Label
        lbl_crop = base_lbl[start_y:start_y+new_h, start_x:start_x+new_w]
        lbl_crop[mask_bool] = self.anomaly_class_idx
        base_lbl[start_y:start_y+new_h, start_x:start_x+new_w] = lbl_crop

        return F.to_pil_image(base_img), base_lbl

    def __getitem__(self, idx):
        # A. Cityscapes
        img, target = self.cityscapes_ds[idx]
        if not isinstance(img, Image.Image):
            img = F.to_pil_image(img)
        img = img.convert('RGB')
        
        # Mappa Semantica
        semantic_map = torch.zeros((img.height, img.width), dtype=torch.long)
        if 'masks' in target and 'labels' in target:
             masks = target['masks']
             labels = target['labels']
             for i in range(len(labels)):
                 semantic_map[masks[i]] = labels[i]

        # B. COCO
        coco_idx = random.randint(0, len(self.coco_ds)-1)
        anom_img, anom_target = self.coco_ds[coco_idx]
        if not isinstance(anom_img, Image.Image):
            anom_img = F.to_pil_image(anom_img)
        anom_img = anom_img.convert('RGB')
        
        anom_mask = None
        if 'masks' in anom_target and len(anom_target['masks']) > 0:
            rand_obj = random.randint(0, len(anom_target['masks'])-1)
            anom_mask = anom_target['masks'][rand_obj]

        # C. Cut-Paste
        if random.random() < 0.5 and anom_mask is not None:
            img, semantic_map = self.cut_paste(img, semantic_map, anom_img, anom_mask)

        # D. Pipeline Finale
        img = F.resize(img, self.img_size, interpolation=T.InterpolationMode.BILINEAR)
        semantic_map = F.resize(semantic_map.unsqueeze(0), self.img_size, interpolation=T.InterpolationMode.NEAREST).squeeze(0)
        
        # FIX: Usa pil_to_tensor per compatibilità
        img_t = F.pil_to_tensor(img) 
        img_t = img_t.to(dtype=torch.float32) / 255.0 
        
        img_t = self.normalize(img_t)

        return img_t, semantic_map.long()