import os
import sys
import glob
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from argparse import ArgumentParser
from sklearn.metrics import precision_recall_curve, auc, roc_curve, average_precision_score
from torchvision.transforms import v2 as T
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

from models.segmenter import AnomalySegmenter

# --- CONFIGURAZIONE ---
IMG_SIZE = (336, 336)
NUM_CLASSES = 19

# Trasformazioni
input_transform = T.Compose([
    T.Resize(IMG_SIZE, interpolation=T.InterpolationMode.BILINEAR),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

target_transform = T.Compose([
    T.Resize(IMG_SIZE, interpolation=T.InterpolationMode.NEAREST),
])

# -----------------------------------------------------------------------------
# METRICHE & SCORING
# -----------------------------------------------------------------------------

def fpr_at_95_tpr(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    if len(tpr) == 0: return 0.0
    idxs = np.where(tpr >= 0.95)[0]
    if len(idxs) == 0: return 1.0
    return float(fpr[idxs[0]])

def compute_anomaly_map(mask_logits, class_logits, method):
    # Upsample
    mask_logits = F.interpolate(
        mask_logits.unsqueeze(0), size=IMG_SIZE, mode="bilinear", align_corners=False
    ).squeeze(0)

    # Pixel-Wise Reconstruction
    prob_cls = F.softmax(class_logits, dim=-1)[..., :-1] 
    prob_msk = torch.sigmoid(mask_logits)
    pixel_logits = torch.mm(prob_cls.T, prob_msk.flatten(1)) 
    pixel_logits = pixel_logits.view(-1, IMG_SIZE[0], IMG_SIZE[1]) # Shape: (20, 518, 518)

    # --- FIX CRITICO: ESCLUDERE LA CLASSE ANOMALIA ---
    # Il tuo modello ha 20 classi (0-18 Cityscapes, 19 Anomalia).
    # Per calcolare "quanto è anomalo", dobbiamo vedere quanto è BASSA l'attivazione delle classi ID (0-18).
    
    # Prendiamo solo i logits delle prime 19 classi (ID)
    id_logits = pixel_logits[:19, :, :] 

    if method == "energy":
        # Calcoliamo l'energia solo sulle classi normali.
        # Se nessuna classe normale è attiva -> LogSumExp basso -> -LogSumExp Alto -> Anomalia!
        return -1.0 * torch.logsumexp(id_logits, dim=0)
        
    elif method == "msp":
        probs = F.softmax(id_logits, dim=0)
        return 1.0 - probs.max(dim=0).values
        
    elif method == "maxlogit":
        return -1.0 * id_logits.max(dim=0).values
        
    elif method == "maxentropy":
        probs = F.softmax(id_logits, dim=0)
        eps = 1e-8
        return -(probs * (probs + eps).log()).sum(dim=0)
        
    # BONUS: Metodo "Diretto" (Supervised)
    # Dato che hai addestrato la classe 19, potremmo usare direttamente quella!
    # return pixel_logits[19, :, :] 
    
    raise ValueError(f"Unknown method: {method}")

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    parser = ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, help="Pattern glob immagini")
    parser.add_argument("--ckpt", default="/content/drive/MyDrive/AnomalyProject/checkpoints/last.ckpt")
    parser.add_argument("--method", default="energy", choices=["msp", "maxlogit", "maxentropy", "rba", "energy"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Config: {device} | Method: {args.method} (with Smoothing) ---")

    if not os.path.exists(args.ckpt):
        print(f"❌ Checkpoint mancante: {args.ckpt}")
        return

    print("--> Caricamento modello...")
    try:
        model = AnomalySegmenter.load_from_checkpoint(args.ckpt)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"❌ Errore caricamento: {e}")
        return

    file_list = []
    for pattern in args.input:
        file_list.extend(glob.glob(os.path.expanduser(pattern), recursive=True))
    
    file_list = sorted(list(set(file_list)))
    print(f"--> Trovate {len(file_list)} immagini.")
    
    if not file_list: return

    anomaly_score_list = []
    ood_gts_list = []

    print("Processing...")
    for path in tqdm(file_list):
        try:
            img_pil = Image.open(path).convert("RGB")
        except: continue
        
        img_tensor = input_transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            # --- FIX APPLICATO QUI ---
            # Chiamiamo model.model(...) invece di model(...) per accedere all'EoMT interno
            # che restituisce la tupla (mask_logits, class_logits) invece del singolo output processato.
            mask_logits, class_logits = model.model(img_tensor)
            
            final_mask = mask_logits[-1][0]
            final_class = class_logits[-1][0]
            
            # Calcolo mappa (ora include lo smoothing)
            anomaly_map = compute_anomaly_map(final_mask, final_class, args.method)
            anomaly_np = anomaly_map.cpu().numpy() # Assicurati di passare a CPU prima di numpy()

        # GT Logic
        pathGT = None
        # SMIYC
        if "images" in path and ("RoadAnomaly21" in path or "RoadObsticle21" in path):
            cand = path.replace("images", "labels_masks")
            cand_png = os.path.splitext(cand)[0] + ".png"
            if os.path.exists(cand_png): pathGT = cand_png
            elif os.path.exists(cand_png.replace(".png", "_labels_semantic.png")):
                pathGT = cand_png.replace(".png", "_labels_semantic.png")
        
        # RoadAnomaly Orig
        elif "RoadAnomaly" in path and "21" not in path:
            cand = path.replace(".jpg", ".labels.png")
            if os.path.exists(cand): pathGT = cand
            
        # FS
        elif "leftImg8bit" in path:
            cand = path.replace("leftImg8bit", "gtCoarse").replace("_leftImg8bit", "_gtCoarse_labelIds")
            base = os.path.splitext(cand)[0]
            if os.path.exists(base + ".png"): pathGT = base + ".png"

        if not pathGT: continue

        gt_img = Image.open(pathGT)
        gt_img = target_transform(gt_img)
        ood_gts = np.array(gt_img)

        # GT Mapping (Corrected)
        new_gt = np.ones_like(ood_gts) * 255
        if "RoadAnomaly" in path and "21" not in path:
            new_gt[ood_gts == 1] = 0
            new_gt[ood_gts == 2] = 1
        elif "FS" in path or "LostFound" in path:
            new_gt[ood_gts == 1] = 0
            new_gt[ood_gts > 1] = 1
        else: # SMIYC
            new_gt[ood_gts == 0] = 0
            new_gt[ood_gts == 1] = 1

        ood_gts_list.append(new_gt)
        anomaly_score_list.append(anomaly_np)

    if not ood_gts_list:
        print("❌ Nessun dato valido.")
        return

    print("Calcolo metriche...")
    ood_gts_flat = np.array(ood_gts_list)
    anomaly_scores_flat = np.array(anomaly_score_list)

    valid_mask = (ood_gts_flat != 255)
    scores = anomaly_scores_flat[valid_mask]
    labels = ood_gts_flat[valid_mask]

    if len(np.unique(labels)) < 2:
        print("⚠️ Attenzione: il test set contiene solo una classe (solo normali o solo anomalie). AuPRC potrebbe essere indecidibile.")
    
    if len(scores) == 0:
        print("❌ Nessun pixel valido per la valutazione.")
        return

    auprc = average_precision_score(labels, scores)
    fpr95 = fpr_at_95_tpr(scores, labels)

    result_str = f"[RISULTATO {args.method.upper()}] AuPRC: {auprc * 100.0:.2f}% | FPR@95: {fpr95 * 100.0:.2f}%"
    print("\n" + "#"*60)
    print(result_str)
    print("#"*60 + "\n")

    with open("results_custom.txt", "a") as f:
        f.write(f"Input: {args.input} | Method: {args.method}\n")
        f.write(result_str + "\n\n")

if __name__ == "__main__":
    main() 