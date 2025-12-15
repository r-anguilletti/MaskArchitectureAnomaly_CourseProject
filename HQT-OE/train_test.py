import os
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

# Importa i tuoi moduli custom
from datasets.hybrid_anomaly import HybridAnomalyDataset
from datasets.road_anomaly import RoadAnomalyDataset
from models.segmenter import AnomalySegmenter 

# --- CONFIGURAZIONE PERCORSI (DATI LOCALI) ---
PATH_CITY = "/content/Cityscapes_Local"
PATH_COCO = "/content/COCO_Local"
PATH_ROAD_ANOMALY = "/content/drive/MyDrive/Anomaly_Segmentation/RoadAnomaly"
CHECKPOINT_DIR = "/content/drive/MyDrive/AnomalyProject/checkpoints"
LOG_DIR = "/content/drive/MyDrive/AnomalyProject/logs"

# --- IPERPARAMETRI OTTIMIZZATI ---
# Con i dati in locale e FP16, possiamo spingere di più:
BATCH_SIZE = 2        # Aumentato da 2 a 4 (Prova 4, se va OOM torna a 2)
NUM_WORKERS = 2       # Aumentato da 2 a 4 (Sfrutta tutta la CPU di Colab)
ACCUMULATE_GRAD = 1   # Ridotto a 1 perché il batch reale è già 4
IMG_SIZE = (336, 336)
MAX_EPOCHS = 6
LR = 1e-4             

# --- DEFINIZIONE CLASSI ---
NUM_CLASSES = 20      
ANOMALY_IDX = 19      
IGNORE_IDX = 255      

def main():
    print(f"--- Avvio Training (Fast Mode) ---")
    print(f"Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # --- A. DATASETS ---
    print("--> Caricamento Training Set (Hybrid)...")
    # Controllo di sicurezza: se hai dimenticato di lanciare la cella di setup
    if not os.path.exists(PATH_CITY) or not os.path.exists(PATH_COCO):
        raise FileNotFoundError(f"❌ ERRORE: Non trovo i dati in {PATH_CITY} o {PATH_COCO}. Hai lanciato la cella di 'SETUP CORRETTO (Zip Mode)'?")

    train_ds = HybridAnomalyDataset(
        cityscapes_root=PATH_CITY,
        coco_root=PATH_COCO,
        img_size=IMG_SIZE
    )
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, # <--- 4 Workers per massima velocità
        persistent_workers=True, # Mantiene i workers vivi tra le epoche
        pin_memory=True
    )

    print("--> Verifica Validation Set (RoadAnomaly)...")
    val_loader = None
    use_validation = False
    
    if os.path.exists(PATH_ROAD_ANOMALY):
        try:
            val_ds = RoadAnomalyDataset(
                root_path=PATH_ROAD_ANOMALY,
                img_size=IMG_SIZE
            )
            if len(val_ds) > 0:
                # Anche qui aumentiamo workers e batch size
                val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
                use_validation = True
                print(f"Validation Set caricato: {len(val_ds)} immagini.")
            else:
                print("Cartella RoadAnomaly vuota.")
        except Exception as e:
            print(f"Errore caricamento RoadAnomaly: {e}.")
    else:
        print(f"Dataset Validazione NON trovato in: {PATH_ROAD_ANOMALY}")

    # --- B. MODELLO ---
    print("--> Inizializzazione Modello (ViT-Large + LoRA)...")
    model = AnomalySegmenter(
        img_size=IMG_SIZE,
        num_classes=NUM_CLASSES,       
        anomaly_class_idx=ANOMALY_IDX, 
        ignore_index=IGNORE_IDX,       
        lr=LR,
        backbone_name="vit_large_patch14_reg4_dinov2"
    )

    # --- C. CALLBACKS ---
    monitor_metric = "val_loss" if use_validation else "train_loss"
    print(f"--> Monitoraggio checkpoint su: {monitor_metric}")

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="anomaly-fast-{epoch:02d}-{" + monitor_metric + ":.2f}",
        save_top_k=1,            
        monitor=monitor_metric,
        mode="min",
        save_last=True           
    )

    # --- D. TRAINER ---
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu",
        devices=1,
        
        # OTTIMIZZAZIONI VELOCITÀ
        precision="16-mixed",     # <--- Torniamo a FP16 (Veloce e leggero)
        accumulate_grad_batches=ACCUMULATE_GRAD,
        
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
        default_root_dir=LOG_DIR,
        check_val_every_n_epoch=1 if use_validation else 0, 
        limit_val_batches=1.0 if use_validation else 0,     

        # SICUREZZA
        gradient_clip_val=0.5,
        gradient_clip_algorithm="norm"
    )

    # --- E. START / RESUME ---
    ckpt_path = os.path.join(CHECKPOINT_DIR, "last.ckpt")
    
    if os.path.exists(ckpt_path):
        print(f"🔄 TROVATO CHECKPOINT! Riprendo da: {ckpt_path}")
        try:
            trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
        except Exception as e:
            print(f"⚠️ Checkpoint corrotto o incompatibile: {e}")
            print("Restarting from scratch...")
            trainer.fit(model, train_loader, val_loader)
    else:
        print("🆕 Training da zero.")
        trainer.fit(model, train_loader, val_loader)

    print("🎉 Training Concluso!")

if __name__ == "__main__":
    # Ottimizzazione extra per le operazioni di matrice su Tesla T4
    torch.set_float32_matmul_precision('medium')
    main()