import os
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

# Importa i moduli creati nelle settimane precedenti
from datasets.hybrid_anomaly import HybridAnomalyDataset
from datasets.road_anomaly import RoadAnomalyDataset
from models.segmenter import AnomalySegmenter

# --- CONFIGURAZIONE PER COLAB ---
# 1. Percorsi Dati (Assicurati che puntino alle cartelle DOVE HAI UNZIPPATO)
PATH_CITY = "/content/drive/MyDrive/Anomaly_Segmentation/Cityscapes"
PATH_COCO = "/content/drive/MyDrive/Anomaly_Segmentation/COCO"
# Percorso per la validazione (su Drive o locale se lo hai copiato)
PATH_ROAD_ANOMALY = "/content/drive/MyDrive/Anomaly_Segmentation/RoadAnomaly"

# 2. Iperparametri "Safe Mode" per GPU T4
BATCH_SIZE = 2        # Basso per non riempire la VRAM
ACCUMULATE_GRAD = 2   # Simula un batch size effettivo di 4 (2*2)
IMG_SIZE = (518, 518) # Risoluzione DINOv2
MAX_EPOCHS = 6        # Epoche totali (bastano poche grazie al pre-training)
LR = 1e-4             # Learning Rate

# 3. Percorsi Salvataggio (Su Drive per sicurezza)
CHECKPOINT_DIR = "/content/drive/MyDrive/AnomalyProject/checkpoints"
LOG_DIR = "/content/drive/MyDrive/AnomalyProject/logs"

def main():
    print(f"--- Avvio Training ---")
    print(f"Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # --- A. DATASETS ---
    print("--> Caricamento Training Set (Hybrid)...")
    train_ds = HybridAnomalyDataset(
        cityscapes_root=PATH_CITY,
        coco_root=PATH_COCO,
        img_size=IMG_SIZE
    )
    # pin_memory=True e num_workers=2 velocizzano il passaggio dati
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2,
        persistent_workers=True,
        pin_memory=True
    )

    print("--> Verifica Validation Set (RoadAnomaly)...")
    val_loader = None
    if os.path.exists(PATH_ROAD_ANOMALY):
        try:
            val_ds = RoadAnomalyDataset(
                root_path=PATH_ROAD_ANOMALY,
                img_size=IMG_SIZE
            )
            # Verifica che abbia trovato file
            if len(val_ds) > 0:
                val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=2)
                print(f"Validation Set caricato: {len(val_ds)} immagini.")
            else:
                print("Cartella trovata ma vuota. Validation disabilitata.")
        except Exception as e:
            print(f"Errore caricamento RoadAnomaly: {e}. Validation disabilitata.")
    else:
        print(f"Percorso RoadAnomaly non trovato: {PATH_ROAD_ANOMALY}. Validation disabilitata.")

    # --- B. MODELLO ---
    print("--> Inizializzazione Modello (ViT-Large + LoRA)...")
    # Usa 'vit_base_patch14_reg4_dinov2' se 'vit_large' crasha ancora
    model = AnomalySegmenter(
        img_size=IMG_SIZE,
        backbone_name="vit_large_patch14_reg4_dinov2", 
        lr=LR
    )

    # --- C. TRAINER ---
    # Salva il miglior modello basandosi sulla loss di training (o val_loss se presente)
    monitor_metric = "train_loss" # Cambia in 'val_loss' se la validazione funziona bene
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="eomt_lora-{epoch:02d}-{train_loss:.2f}",
        save_top_k=2,
        monitor=monitor_metric,
        mode="min",
        save_last=True # Salva sempre l'ultimo stato per poter riprendere
    )

    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",      # Risparmia ~40% di memoria
        accumulate_grad_batches=ACCUMULATE_GRAD,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,      # Log frequente
        default_root_dir=LOG_DIR,
        check_val_every_n_epoch=1  # Valida a fine epoca
    )

    # --- D. START + logica checkpoint ---
    ckpt_path = os.path.join(CHECKPOINT_DIR, "last.ckpt")
    
    if os.path.exists(ckpt_path):
        print(f"🔄 TROVATO CHECKPOINT! Riprendo il training da: {ckpt_path}")
        print("   (Saranno ripristinati: epoca, pesi e stato ottimizzatore)")
        trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
    else:
        print("🆕 Nessun checkpoint 'last.ckpt' trovato. Inizio training da zero.")
        trainer.fit(model, train_loader, val_loader)

    #print("--> Inizio Fit...")
    #trainer.fit(model, train_loader, val_loader)
    print("🎉 Training Concluso con Successo!")

if __name__ == "__main__":
    # Ottimizzazione TensorCores
    torch.set_float32_matmul_precision('medium')
    main()