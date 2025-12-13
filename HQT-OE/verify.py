"""""
import torch
import matplotlib.pyplot as plt
import os
import sys
from torchvision.transforms import v2 as T
from torchvision.utils import draw_segmentation_masks

# Aggiungi la directory corrente al path per trovare il modulo 'datasets'
sys.path.append(os.getcwd())

from datasets.hybrid_anomaly import HybridAnomalyDataset

# --- CONFIGURAZIONE PATH (Modifica con i tuoi percorsi di Colab LOCALI) ---
# Questi sono i percorsi dove avrai unzippato i dati in /content/
CITY_ROOT = "/content/drive/MyDrive/Anomaly_Segmentation/Cityscapes"
COCO_ROOT = "/content/drive/MyDrive/Anomaly_Segmentation/COCO" 
OUTPUT_DIR = "/content/drive/MyDrive/AnomalyProject/debug_output" # Cartella di output su Google Drive

def unnormalize_and_plot(dataset, num_samples=4):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    fig, axs = plt.subplots(num_samples, 2, figsize=(12, 4 * num_samples))
    plt.suptitle("Verifica Cut-Paste Augmentation", fontsize=16)

    print(f"Generazione di {num_samples} campioni di verifica...")

    for i in range(num_samples):
        # 1. Preleva un campione dal dataset
        # img: Tensor (C, H, W), target: Tensor (H, W)
        img, target_mask = dataset[i]

        # 2. Preparazione Immagine per visualizzazione
        # L'immagine è un tensor float o uint8. Convertiamo per visualizzarla.
        if img.dtype == torch.float:
            img_vis = img.permute(1, 2, 0).numpy()
            # Se hai normalizzato (mean/std), dovresti denormalizzare qui.
            # Assumiamo valori 0-1 o 0-255 per ora basandoci sui tuoi transforms.
            if img_vis.max() <= 1.0:
                img_vis = (img_vis * 255).astype('uint8')
            else:
                img_vis = img_vis.astype('uint8')
        elif img.dtype == torch.uint8:
            img_vis = img.permute(1, 2, 0).numpy()
        
        # 3. Preparazione Maschera
        # La maschera contiene l'ID classe (es. 19 per anomalia). 
        # Convertiamola in booleana per evidenziarla
        anomaly_mask = (target_mask == dataset.anomaly_class_idx)

        # Plot Immagine RGB
        axs[i, 0].imshow(img_vis)
        axs[i, 0].set_title(f"Sample {i}: Input Image")
        axs[i, 0].axis('off')

        # Plot Ground Truth (Anomalia evidenziata)
        axs[i, 1].imshow(img_vis) # Sfondo
        # Sovrapponi la maschera in rosso semi-trasparente
        axs[i, 1].imshow(anomaly_mask, cmap='Reds', alpha=0.6, interpolation='none')
        axs[i, 1].set_title(f"Label (Rosso = Anomalia ID {dataset.anomaly_class_idx})")
        axs[i, 1].axis('off')

    save_path = os.path.join(OUTPUT_DIR, "dataset_check.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Verifica completata! Immagine salvata in: {save_path}")
    plt.close()

if __name__ == "__main__":
    # Inizializza il dataset
    try:
        ds = HybridAnomalyDataset(
            cityscapes_root=CITY_ROOT,
            coco_root=COCO_ROOT,
            img_size=(518, 518), # Risoluzione DINOv2
            anomaly_class_idx=19 # O 255, o quello che decideremo
        )
        
        unnormalize_and_plot(ds)
        
    except FileNotFoundError as e:
        print(f"Errore Path: {e}")
        print("Assicurati di aver eseguito l'unzip dei dati nelle cartelle specificate in CITY_ROOT e COCO_ROOT.")
    except Exception as e:
        print(f"Errore Generico: {e}")
        import traceback
        traceback.print_exc()
"""""
"""""
import sys
import os
import torch

# Aggiungi la cartella corrente al path
sys.path.append(os.getcwd())

from models.segmenter import AnomalySegmenter

def check_model():
    print("--- Inizio Verifica Modello ---")
    
    # 1. Istanzia
    try:
        model_module = AnomalySegmenter(
            img_size=(518, 518),
            backbone_name="vit_large_patch14_reg4_dinov2", # O vit_base... se large esplode
            num_classes=20
        )
        print("Modello istanziato correttamente.")
    except Exception as e:
        print(f"Errore istanziazione: {e}")
        return

    # 2. Verifica LoRA
    # Controlliamo quanti parametri sono addestrabili
    total_params = sum(p.numel() for p in model_module.parameters())
    trainable_params = sum(p.numel() for p in model_module.parameters() if p.requires_grad)
    
    print(f"\n📊 Statistiche Parametri:")
    print(f"   Totali:       {total_params:,}")
    print(f"   Addestrabili: {trainable_params:,}")
    print(f"   Percentuale:  {100 * trainable_params / total_params:.2f}%")
    
    # Deve essere bassa (es. < 5-10%), ma non zero!
    if trainable_params == 0:
        print("ATTENZIONE: 0 parametri addestrabili. LoRA non sta funzionando!")
    else:
        print("LoRA attivo. Backbone congelato, adapter attivi.")

    # 3. Test Forward Pass (Dummy)
    print("\nTest Forward Pass (Dummy input)...")
    dummy_img = torch.randn(2, 3, 518, 518) # Batch size 2
    dummy_mask = torch.randint(0, 19, (2, 518, 518))
    
    try:
        loss = model_module.training_step((dummy_img, dummy_mask), 0)
        print(f"Forward Pass OK. Loss calcolata: {loss.item():.4f}")
    except Exception as e:
        print(f"Errore Forward Pass: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_model()
"""""

import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T

# Importa il tuo dataset
from datasets.hybrid_anomaly import HybridAnomalyDataset

# Configurazione
PATH_CITY = "/content/drive/MyDrive/Anomaly_Segmentation/Cityscapes"
PATH_COCO = "/content/drive/MyDrive/Anomaly_Segmentation/COCO"
IMG_SIZE = (518, 518)

def unnormalize(tensor):
    # Denormalizza per visualizzazione (ImageNet stats)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def main():
    print("--- CHECK DATALOADER ---")
    try:
        ds = HybridAnomalyDataset(
            cityscapes_root=PATH_CITY,
            coco_root=PATH_COCO,
            img_size=IMG_SIZE
        )
    except Exception as e:
        print(f"❌ Errore init dataset: {e}")
        return

    # Prendi un batch
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    images, masks = next(iter(loader))

    print(f"Batch Shape: {images.shape}")
    print(f"Mask Shape: {masks.shape}")
    print(f"Labels uniche nel batch: {torch.unique(masks)}")

    # Verifica se c'è la classe 19 (Anomalia)
    if 19 in masks:
        print("✅ Classe 19 (Anomalia) trovata nel batch!")
    else:
        print("⚠️ ATTENZIONE: Nessuna anomalia (classe 19) in questo batch random.")

    # Plot
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    for i in range(4):
        # Immagine
        img_vis = unnormalize(images[i]).permute(1, 2, 0).numpy()
        img_vis = np.clip(img_vis, 0, 1)
        axs[0, i].imshow(img_vis)
        axs[0, i].set_title(f"Input {i}")
        axs[0, i].axis("off")

        # Maschera
        mask_vis = masks[i].numpy()
        # Evidenzia anomalia in rosso
        axs[1, i].imshow(mask_vis, cmap="gray")
        # Sovrapponi rosso dove c'è 19
        anom_mask = (mask_vis == 19)
        if anom_mask.sum() > 0:
            axs[1, i].imshow(anom_mask, cmap="Reds", alpha=0.5)
            axs[1, i].set_title(f"Mask {i} (Red=Anomaly)")
        else:
            axs[1, i].set_title(f"Mask {i} (No Anomaly)")
        axs[1, i].axis("off")

    plt.tight_layout()
    plt.savefig("dataloader_check.png")
    print("📸 Salvato 'dataloader_check.png'. Controlla se vedi oggetti incollati con la maschera rossa corretta.")

if __name__ == "__main__":
    main()