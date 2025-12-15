import torch
import numpy as np
import cv2
import os
from torchvision.transforms import functional as F
from datasets.hybrid_anomaly import HybridAnomalyDataset

# --- CONFIGURA I PERCORSI ---
PATH_CITY = "/content/Cityscapes_Local" 
PATH_COCO = "/content/COCO_Local"
OUTPUT_DIR = "debug_data_proof"
IMG_SIZE = (518, 518)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def denormalize(tensor):
    """Converte il tensore normalizzato (ImageNet) in immagine visualizzabile."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    return tensor.clamp(0, 1)

def main():
    print(f"--- 🕵️‍♂️ DATASET INSPECTION START ---")
    
    # 1. Carichiamo il Dataset
    try:
        ds = HybridAnomalyDataset(PATH_CITY, PATH_COCO, img_size=IMG_SIZE)
        print("✅ Dataset inizializzato correttamente.")
    except Exception as e:
        print(f"❌ Errore init dataset: {e}")
        return

    # 2. Iteriamo su 10 campioni casuali
    print(f"--- Generazione di 10 campioni di prova in '{OUTPUT_DIR}' ---")
    
    found_anomalies = 0
    
    for i in range(10):
        # Prendiamo un indice a caso per variare l'immagine di base
        idx = np.random.randint(0, len(ds))
        
        # Chiamata a __getitem__ (qui avviene il cut-paste)
        img_tensor, mask_tensor = ds[idx]
        
        # Analisi Maschera
        # La classe anomalia è 19 (definito nel tuo hybrid_anomaly.py)
        anomaly_pixels = (mask_tensor == 19).sum().item()
        has_anomaly = anomaly_pixels > 0
        
        status = "🔴 ANOMALIA PRESENTE" if has_anomaly else "⚪ Pulita"
        if has_anomaly: found_anomalies += 1
        print(f"Sample {i}: {status} (Pixel anomali: {anomaly_pixels})")

        # --- VISUALIZZAZIONE ---
        # 1. Immagine Originale
        img_vis = denormalize(img_tensor).permute(1, 2, 0).numpy()
        img_vis = (img_vis * 255).astype(np.uint8)
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)

        # 2. Creiamo un overlay rosso per l'anomalia
        if has_anomaly:
            overlay = img_vis.copy()
            # Maschera booleana dove c'è l'anomalia
            mask_bool = (mask_tensor == 19).numpy()
            
            # Coloriamo di rosso (BGR: 0, 0, 255) quei pixel
            overlay[mask_bool] = [0, 0, 255]
            
            # Fondiamo l'immagine originale con l'overlay (trasparenza)
            cv2.addWeighted(overlay, 0.5, img_vis, 0.5, 0, img_vis)
            
            # Disegniamo un contorno verde intorno all'anomalia per vederla meglio
            contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_vis, contours, -1, (0, 255, 0), 2)

        # Salva
        filename = os.path.join(OUTPUT_DIR, f"proof_{i}_{'ANOMALY' if has_anomaly else 'CLEAN'}.png")
        cv2.imwrite(filename, img_vis)

    print("-" * 30)
    print(f"Totale immagini generate: 10")
    print(f"Immagini con anomalie: {found_anomalies}")
    if found_anomalies > 0:
        print("✅ IL SISTEMA FUNZIONA: Le anomalie vengono iniettate.")
    else:
        print("⚠️ NESSUNA ANOMALIA: Controlla la probabilità in hybrid_anomaly.py o i percorsi COCO.")

if __name__ == "__main__":
    main()