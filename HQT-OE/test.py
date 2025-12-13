import os
import glob
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from argparse import ArgumentParser
from torchvision.transforms import v2 as T
import matplotlib.pyplot as plt

from models.segmenter import AnomalySegmenter

# --- CONFIGURAZIONE ---
IMG_SIZE = (518, 518)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Trasformazioni
input_transform = T.Compose([
    T.Resize(IMG_SIZE, interpolation=T.InterpolationMode.BILINEAR),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=MEAN, std=STD)
])

target_transform = T.Compose([
    T.Resize(IMG_SIZE, interpolation=T.InterpolationMode.NEAREST),
])

def unnormalize(tensor):
    """Per visualizzare l'immagine originale"""
    tensor = tensor.clone().detach().cpu()
    for t, m, s in zip(tensor, MEAN, STD):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1).permute(1, 2, 0).numpy()

def main():
    parser = ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, help="Pattern glob (es. path/*.webp)")
    parser.add_argument("--ckpt", default="/content/drive/MyDrive/AnomalyProject/checkpoints/last.ckpt")
    parser.add_argument("--method", default="energy")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Carica Modello
    print(f"--> Caricamento Checkpoint: {args.ckpt}")
    if not os.path.exists(args.ckpt):
        print("❌ Errore: Checkpoint non trovato!")
        return

    try:
        model = AnomalySegmenter.load_from_checkpoint(args.ckpt)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"❌ Errore caricamento modello: {e}")
        return

    # 2. Prendi la PRIMA immagine trovata
    file_list = []
    for pattern in args.input:
        file_list.extend(glob.glob(os.path.expanduser(pattern), recursive=True))
    
    if not file_list:
        print("❌ Nessuna immagine trovata!")
        return

    path = sorted(file_list)[0] 
    print(f"\n🔬 ANALISI DEBUG SU: {os.path.basename(path)}")
    print(f"   Path completo: {path}")

    # A. Inferenza
    try:
        img_pil = Image.open(path).convert("RGB")
    except:
        print("❌ Impossibile aprire immagine.")
        return

    img_tensor = input_transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        mask_logits, class_logits = model(img_tensor)
        final_mask = mask_logits[-1][0] # [Q, H_patch, W_patch]
        final_class = class_logits[-1][0] # [Q, C+1]
        
        # --- FIX: Upsample corretto ---
        # Input deve essere [1, Q, H, W] -> Output [1, Q, 518, 518]
        final_mask_up = F.interpolate(
            final_mask.unsqueeze(0), # Aggiungi batch dim -> [1, Q, H, W]
            size=IMG_SIZE, 
            mode="bilinear",
            align_corners=False
        ).squeeze(0) # Rimuovi batch dim -> [Q, 518, 518]
        
        # Logits -> Pixel
        prob_cls = F.softmax(final_class, dim=-1)[..., :-1] 
        prob_msk = torch.sigmoid(final_mask_up)
        
        # Pixel Logits: (C, Q) @ (Q, HW) -> (C, HW)
        pixel_logits = torch.mm(prob_cls.T, prob_msk.flatten(1))
        pixel_logits = pixel_logits.view(-1, IMG_SIZE[0], IMG_SIZE[1]) # (C, 518, 518)
        
        # CALCOLO SCORE
        # Energy = -LogSumExp
        # Score = -1 * Energy = LogSumExp (se vogliamo confidenza)
        # Score Anomalia = -LogSumExp (se vogliamo anomalia alta)
        # Proviamo il calcolo corretto per visualizzazione anomalia
        logsumexp = torch.logsumexp(pixel_logits, dim=0)
        anomaly_map = -1.0 * logsumexp
        
        # Statistiche
        print(f"\n📊 STATISTICHE SCORE MODELLO:")
        print(f"   LogSumExp (Confidenza) -> Min: {logsumexp.min():.2f}, Max: {logsumexp.max():.2f}")
        print(f"   Anomaly Score (-LSE)   -> Min: {anomaly_map.min():.2f}, Max: {anomaly_map.max():.2f}")

    # B. Caricamento GT
    pathGT = None
    if "images" in path: # SMIYC
        candidate = path.replace("images", "labels_masks")
        candidate = os.path.splitext(candidate)[0] + ".png"
        if os.path.exists(candidate): pathGT = candidate
        elif os.path.exists(candidate.replace(".png", "_labels_semantic.png")):
            pathGT = candidate.replace(".png", "_labels_semantic.png")
    
    # RoadAnomaly Originale
    if not pathGT and "RoadAnomaly" in path and "21" not in path:
        candidate = path.replace(".jpg", ".labels.png")
        if os.path.exists(candidate): pathGT = candidate

    final_gt = None
    if pathGT:
        gt_img = Image.open(pathGT)
        gt_img = target_transform(gt_img)
        gt_arr = np.array(gt_img)
        
        print(f"\n🏷️ ANALISI GROUND TRUTH:")
        print(f"   Path GT: {pathGT}")
        vals = np.unique(gt_arr)
        print(f"   Valori unici nella maschera: {vals}")
        
        # Mappatura Visuale
        final_gt = np.zeros_like(gt_arr)
        if "RoadAnomaly" in path and "21" not in path:
            final_gt[gt_arr == 2] = 1 # Anomalia=2
        elif "FS" in path or "LostFound" in path:
            final_gt[gt_arr > 1] = 1
        else: # SMIYC
            final_gt[gt_arr == 1] = 1
    else:
        print("⚠️ Maschera GT non trovata.")

    # C. Visualizzazione
    print("\n💾 Salvataggio debug_vis.png ...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Immagine Originale
    img_vis = unnormalize(img_tensor.squeeze())
    axes[0].imshow(img_vis)
    axes[0].set_title("Input Image")
    axes[0].axis('off')
    
    # 2. Anomaly Score (Heatmap)
    score_vis = anomaly_map.cpu().numpy()
    # Normalizza per visualizzazione
    score_vis = (score_vis - score_vis.min()) / (score_vis.max() - score_vis.min() + 1e-8)
    
    im = axes[1].imshow(score_vis, cmap='jet')
    axes[1].set_title("Prediction (-Energy)\nRed=Anomaly, Blue=Normal")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 3. GT
    if final_gt is not None:
        axes[2].imshow(final_gt, cmap='gray')
        axes[2].set_title("Ground Truth\nWhite=Anomaly")
    else:
        axes[2].text(0.5, 0.5, "No GT Found", ha='center')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig("debug_vis.png")
    print("✅ Fatto! Apri 'debug_vis.png'.")

if __name__ == "__main__":
    main()