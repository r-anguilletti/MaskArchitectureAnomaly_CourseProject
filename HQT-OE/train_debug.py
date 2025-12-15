import os
import torch
import torchvision
import lightning as L
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from PIL import Image

# --- IMPORTA I TUOI MODULI ---
from datasets.hybrid_anomaly import HybridAnomalyDataset
from models.segmenter import AnomalySegmenter 

# ==========================================
# CONFIGURAZIONE CPU (LOW POWER)
# ==========================================
DEBUG_DIR = "debug_cpu_output"
# Modifica questi percorsi se necessario per puntare ai tuoi dati locali
PATH_CITY = "/content/Cityscapes_Local"  # Esempio: metti i dati nella stessa cartella
PATH_COCO = "/content/COCO_Local"

IMG_SIZE = (336, 336) 
BATCH_SIZE = 1         # <--- Tassativo a 1 per CPU
LR = 1e-3              

os.makedirs(DEBUG_DIR, exist_ok=True)

# ==========================================
# UTILS
# ==========================================
def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(tensor.device)
    return tensor * std + mean

def colorize_mask(mask):
    h, w = mask.shape
    colored = torch.zeros((3, h, w), device=mask.device)
    unique_vals = torch.unique(mask)
    
    colors = {
        0: [128, 64, 128],  # Road (Viola)
        19: [255, 0, 0],    # Anomaly (Rosso)
        255: [0, 0, 0]      # Ignore (Nero)
    }
    
    for val in unique_vals:
        v = val.item()
        if v in colors:
            c = torch.tensor(colors[v], device=mask.device).float() / 255.0
        else:
            torch.manual_seed(v)
            c = torch.rand(3, device=mask.device)
            
        mask_bool = (mask == val)
        colored[:, mask_bool] = c.view(3, 1)
        
    return colored

class DebugImageCallback(L.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # Salva a ogni epoca perché ne faremo poche
        batch = next(iter(trainer.train_dataloader))
        imgs, masks = batch
        imgs = imgs.to(pl_module.device)
        
        with torch.no_grad():
            logits = pl_module(imgs)
            preds = torch.argmax(logits, dim=1)
        
        imgs_vis = denormalize(imgs)
        masks_vis = torch.stack([colorize_mask(m) for m in masks])
        preds_vis = torch.stack([colorize_mask(p) for p in preds])
        
        # Grid più semplice per batch size 1
        grid = make_grid(torch.cat([imgs_vis, masks_vis, preds_vis], dim=0), nrow=1)
        
        save_path = os.path.join(DEBUG_DIR, f"cpu_test_epoch_{trainer.current_epoch}.png")
        save_image(grid, save_path)
        print(f"   📸 Snapshot salvato: {save_path}")

# ==========================================
# MAIN
# ==========================================
def main():
    print("--- 🐌 AVVIO DEBUG MODE (CPU) ---")
    print("Nota: Sarà lento. Serve solo per verificare che il codice non crashi.")

    # 1. VERIFICA DATI
    print("\n[1/3] Verifica Dati...")
    if not os.path.exists(PATH_CITY):
        # Fallback dummy se non hai i dati scaricati, giusto per testare il modello
        print(f"⚠️ ATTENZIONE: Percorso {PATH_CITY} non trovato.")
        print("   Assicurati di aver scaricato almeno 1 immagine o punta ai percorsi giusti.")
        # Se vuoi testare comunque, il codice si fermerà qui sotto.
    
    try:
        ds = HybridAnomalyDataset(PATH_CITY, PATH_COCO, img_size=IMG_SIZE)
        # num_workers=0 è cruciale su CPU locale per evitare blocchi
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) 
        
        imgs, masks = next(iter(loader))
        print(f"   ✅ Dati caricati. Shape: {imgs.shape}")
    except Exception as e:
        print(f"❌ Errore caricamento dati: {e}")
        return

    # 2. MODELLO
    print("\n[2/3] Caricamento Modello ViT (potrebbe richiedere download)...")
    model = AnomalySegmenter(
        img_size=IMG_SIZE,
        num_classes=20,
        lr=LR,
        backbone_name="vit_large_patch14_reg4_dinov2"
    )
    print("   ✅ Modello in memoria.")

    # 3. TRAINING SIMULATO
    print("\n[3/3] Test Loop (2 Epoche)...")
    
    trainer = L.Trainer(
        max_epochs=2,              # Solo 2 epoche
        overfit_batches=1,         # Usa sempre la stessa immagine
        accelerator="cpu",         # <--- CPU
        devices="auto",
        precision="32-true",       # <--- Float32 Standard
        enable_checkpointing=False,
        logger=False,
        callbacks=[DebugImageCallback()],
        log_every_n_steps=1,
        enable_progress_bar=True
    )

    trainer.fit(model, loader)

    print("\n--- ✅ TEST CPU COMPLETATO ---")
    print(f"Controlla '{DEBUG_DIR}' per vedere se le immagini sono state salvate.")

if __name__ == "__main__":
    # Niente ottimizzazioni matriciali su CPU
    main()