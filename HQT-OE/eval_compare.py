import os
import glob
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from argparse import ArgumentParser
from sklearn.metrics import average_precision_score, roc_curve
from torchvision.transforms import v2 as T
from tqdm import tqdm

from models.segmenter import AnomalySegmenter

# --- CONFIGURAZIONE ---
IMG_SIZE = (518, 518)
NUM_CLASSES = 19

input_transform = T.Compose([
    T.Resize(IMG_SIZE, interpolation=T.InterpolationMode.BILINEAR),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

target_transform = T.Compose([
    T.Resize(IMG_SIZE, interpolation=T.InterpolationMode.NEAREST),
])

def fpr_at_95_tpr(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    if len(tpr) == 0: return 0.0
    idxs = np.where(tpr >= 0.95)[0]
    if len(idxs) == 0: return 1.0
    return float(fpr[idxs[0]])

def main():
    parser = ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, help="Path immagini (wildcard supportata)")
    parser.add_argument("--ckpt", default="/content/drive/MyDrive/AnomalyProject/checkpoints/last.ckpt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- COMPARAZIONE METRICHE ---")

    if not os.path.exists(args.ckpt):
        print("❌ Checkpoint non trovato.")
        return

    try:
        model = AnomalySegmenter.load_from_checkpoint(args.ckpt)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"❌ Errore modello: {e}")
        return

    file_list = []
    for pattern in args.input:
        file_list.extend(glob.glob(os.path.expanduser(pattern), recursive=True))
    
    file_list = sorted(list(set(file_list)))
    print(f"--> Analisi su {len(file_list)} immagini...")
    
    if len(file_list) == 0:
        print("⚠️ Nessuna immagine trovata. Controlla il path.")
        return

    # Accumulatori per ogni metrica
    scores_msp = []
    scores_maxlogit = []
    scores_energy = []
    scores_entropy = []
    all_labels = []

    for path in tqdm(file_list):
        try:
            img_pil = Image.open(path).convert("RGB")
        except: continue
        
        img_tensor = input_transform(img_pil).unsqueeze(0).to(device)

        # 1. INFERENZA
        with torch.no_grad():
            mask_logits, class_logits = model(img_tensor)
            final_mask = mask_logits[-1][0] # [Q, H, W]
            final_class = class_logits[-1][0] # [Q, C+1]

            # --- FIX DIMENSIONI ---
            # interpolate vuole [N, C, H, W]. 
            # Se final_mask è [Q, H, W] (3 dim), aggiungiamo dim 0 -> [1, Q, H, W]
            if final_mask.ndim == 3:
                final_mask_in = final_mask.unsqueeze(0)
            else:
                final_mask_in = final_mask

            # Upsample
            final_mask_up = F.interpolate(
                final_mask_in, 
                size=IMG_SIZE, mode="bilinear", align_corners=False
            ).squeeze(0)

            # Ricostruzione Pixel Logits
            prob_cls = F.softmax(final_class, dim=-1)[..., :-1] 
            prob_msk = torch.sigmoid(final_mask_up)
            
            # (C, Q) @ (Q, HW)
            pixel_logits = torch.mm(prob_cls.T, prob_msk.flatten(1)) # (C, HW)
            
            # Reshape a (C, H, W)
            pixel_logits = pixel_logits.view(-1, IMG_SIZE[0], IMG_SIZE[1])
            
            # --- CALCOLO DI TUTTE LE METRICHE ---
            
            # A. MSP (1 - MaxProb)
            probs = F.softmax(pixel_logits, dim=0)
            msp_map = 1.0 - probs.max(dim=0).values
            
            # B. MaxLogit (-MaxLogit)
            maxlogit_map = -1.0 * pixel_logits.max(dim=0).values
            
            # C. Energy (-LogSumExp)
            # Energy = -LogSumExp. Anomaly Score = Energy (Alto per OOD)
            energy_map = -1.0 * torch.logsumexp(pixel_logits, dim=0)
            
            # D. Entropy
            eps = 1e-8
            entropy_map = -(probs * (probs + eps).log()).sum(dim=0)

            # Move to CPU
            s_msp = msp_map.cpu().numpy().flatten()
            s_ml = maxlogit_map.cpu().numpy().flatten()
            s_en = energy_map.cpu().numpy().flatten()
            s_ent = entropy_map.cpu().numpy().flatten()

        # 2. GT LOADING (SMIYC Logic & RoadAnomaly)
        pathGT = None
        
        # Logica SMIYC
        if "images" in path:
            cand = path.replace("images", "labels_masks")
            cand = os.path.splitext(cand)[0] + ".png"
            if os.path.exists(cand): pathGT = cand
            elif os.path.exists(cand.replace(".png", "_labels_semantic.png")):
                pathGT = cand.replace(".png", "_labels_semantic.png")
        
        # Logica RoadAnomaly
        if not pathGT and "RoadAnomaly" in path and "21" not in path:
            cand = path.replace(".jpg", ".labels.png")
            if os.path.exists(cand): pathGT = cand
            
        # Logica FS
        if not pathGT and "leftImg8bit" in path:
            cand = path.replace("leftImg8bit", "gtCoarse").replace("_leftImg8bit", "_gtCoarse_labelIds")
            base = os.path.splitext(cand)[0]
            if os.path.exists(base + ".png"): pathGT = base + ".png"

        if not pathGT or not os.path.exists(pathGT): continue

        gt_img = Image.open(pathGT)
        gt_img = target_transform(gt_img)
        gt_arr = np.array(gt_img).flatten()

        # Normalizza Label
        # RoadAnomaly21: 1=Anomaly, 0=Normal.
        # FS: 0=Void, 1=Road, >1=Obstacle.
        
        valid_mask = np.ones_like(gt_arr, dtype=bool)
        labels = np.zeros_like(gt_arr)
        
        if "RoadAnomaly" in path and "21" not in path: # Original
            # 0=Void(Ignore), 1=Road(Normal), 2=Anomaly
            valid_mask = (gt_arr != 0) 
            labels[gt_arr == 2] = 1
        elif "FS" in path or "LostFound" in path:
            # 0=Void, 1=Road, >1=Obstacle
            valid_mask = (gt_arr != 0)
            labels[gt_arr > 1] = 1
        else: # SMIYC / RoadAnomaly21
            # 0=Normal, 1=Anomaly, 255=Ignore
            valid_mask = (gt_arr != 255)
            labels[gt_arr == 1] = 1

        # Applica Filtro
        if valid_mask.sum() == 0: continue

        all_labels.append(labels[valid_mask])
        scores_msp.append(s_msp[valid_mask])
        scores_maxlogit.append(s_ml[valid_mask])
        scores_energy.append(s_en[valid_mask])
        scores_entropy.append(s_ent[valid_mask])

    # 3. REPORT FINALE
    if not all_labels:
        print("❌ Nessun dato valido.")
        return

    print("\nCalcolo statistiche globali (concatenazione)...")
    cat_labels = np.concatenate(all_labels)
    
    methods = {
        "MSP": np.concatenate(scores_msp),
        "MaxLogit": np.concatenate(scores_maxlogit),
        "Energy": np.concatenate(scores_energy),
        "MaxEntropy": np.concatenate(scores_entropy)
    }

    print("\n" + "="*60)
    print(f"{'METRIC':<15} | {'AuPRC (%)':<15} | {'FPR95 (%)':<15}")
    print("-" * 60)
    
    for name, scores in methods.items():
        try:
            auprc = average_precision_score(cat_labels, scores) * 100
            fpr95 = fpr_at_95_tpr(scores, cat_labels) * 100
            print(f"{name:<15} | {auprc:<15.2f} | {fpr95:<15.2f}")
        except Exception as e:
            print(f"{name:<15} | Errore calcolo ({e})")
            
    print("="*60 + "\n")

if __name__ == "__main__":
    main()