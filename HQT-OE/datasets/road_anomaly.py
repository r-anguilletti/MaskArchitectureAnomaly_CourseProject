import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from datasets.transforms import Transforms

class RoadAnomalyDataset(Dataset):
    def __init__(self, root_path, img_size=(518, 518), anomaly_class_idx=19):
        self.root = Path(root_path)
        self.img_size = img_size
        self.anomaly_class_idx = anomaly_class_idx
        
        # 1. Trova tutte le immagini .jpg
        # RoadAnomaly di solito le ha in una cartella 'frames' o 'images'
        self.img_files = sorted(list(self.root.rglob("*.jpg")))
        self.mask_files = []
        
        valid_imgs = []
        print(f"--> Indicizzazione RoadAnomaly in {root_path}...")
        
        # 2. Trova le maschere corrispondenti
        for img_p in self.img_files:
            # Le maschere possono avere estensioni diverse o suffissi
            # Esempio: image01.jpg -> image01.labels.png OPPURE image01.png
            candidates = [
                img_p.name.replace(".jpg", ".labels.png"),
                img_p.name.replace(".jpg", ".png")
            ]
            
            mask_p = None
            # Cerca il file maschera ricorsivamente in tutta la root
            for cand in candidates:
                found = list(self.root.rglob(cand))
                # Filtra per evitare di riprendere la stessa immagine jpg se ha lo stesso nome
                found = [f for f in found if ".jpg" not in f.name]
                if found:
                    mask_p = found[0]
                    break
            
            if mask_p:
                valid_imgs.append(img_p)
                self.mask_files.append(mask_p)
            
        self.img_files = valid_imgs
        print(f"✅ Trovate {len(self.img_files)} coppie immagine/maschera valide.")

        # 3. Trasformazioni (Solo Resize e Normalizzazione per la validazione)
        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=False, 
            scale_range=(1.0, 1.0)
        )

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        mask_path = self.mask_files[idx]
        
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        
        # Converti maschera in binaria (0=Sfondo, 1=Anomalia)
        mask_np = np.array(mask)
        # Qualsiasi pixel > 0 è considerato anomalia
        mask_bool = mask_np > 0
        
        # Costruisci Tensore Label
        label = torch.zeros_like(torch.tensor(mask_np), dtype=torch.long)
        label[mask_bool] = self.anomaly_class_idx
        
        target = {
            "masks": tv_tensors.Mask(label.unsqueeze(0)),
            "labels": torch.tensor([self.anomaly_class_idx]),
            "is_crowd": torch.tensor([False])
        }
        img = tv_tensors.Image(img)
        
        if self.transforms:
            img, target = self.transforms(img, target)
            
        return img, target['masks'].squeeze(0).long()