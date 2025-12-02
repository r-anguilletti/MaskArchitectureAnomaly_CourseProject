# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
import glob
import torch
import torch.nn.functional as F
import random
from PIL import Image
import numpy as np
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr
from sklearn.metrics import average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor

# -------------------------------------------------------------------
# CONFIGURAZIONE PATH E IMPORT
# -------------------------------------------------------------------
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CUR_DIR)
EOMT_ROOT = os.path.join(PROJECT_ROOT, "eomt")
if EOMT_ROOT not in sys.path:
    sys.path.append(EOMT_ROOT)

from models.vit import ViT
from models.eomt import EoMT

# -------------------------------------------------------------------
# SETUP
# -------------------------------------------------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

NUM_CLASSES = 20  # Assumiamo 19 classi + background/void
NUM_Q = 100
NUM_BLOCKS = 3
BACKBONE_NAME = "vit_base_patch14_reg4_dinov2"

input_transform = Compose([
    Resize((512, 1024), Image.BILINEAR),
    ToTensor(),
])

target_transform = Compose([
    Resize((512, 1024), Image.NEAREST),
])

def get_msp_score(logits, temperature=1.0):
    """
    Calcola l'anomaly score (1 - Max Softmax Prob) applicando la temperatura.
    logits: Tensor [C, H, W]
    temperature: float
    """
    # Scaliamo i logits
    scaled_logits = logits / temperature
    
    # Softmax sui canali (dim=0 perché l'input è C,H,W)
    probs = F.softmax(scaled_logits, dim=0)
    
    # MSP = 1 - max probability
    max_prob, _ = torch.max(probs, dim=0)
    anomaly_score = 1.0 - max_prob
    
    return anomaly_score.cpu().numpy()

def main():
    parser = ArgumentParser()
    parser.add_argument("--input", default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp", nargs="+")
    parser.add_argument("--loadDir", default="../trained_models/")
    parser.add_argument("--loadWeights", default="eomt_cityscapes_semantic.pth")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    # Define temperatures to test
    # Target values required by table: 0.5, 0.75, 1.1
    # Search values for 'Best T': generic range around 1.0
    target_temps = [0.5, 0.75, 1.0, 1.1] 
    search_temps = [0.1, 0.2, 0.3, 0.9, 1.2, 1.3, 1.5, 2.0, 2.3, 2.5, 3.0, 4.0, 5.0]
    all_temps = sorted(list(set(target_temps + search_temps)))

    # Dizionario per accumulare gli score: { 0.5: [], 0.75: [], ... }
    anomaly_scores_dict = {t: [] for t in all_temps}
    ood_gts_list = []

    # -------------------------------
    # CARICAMENTO MODELLO
    # -------------------------------
    weightspath = os.path.join(args.loadDir, args.loadWeights)
    print(f"Loading weights: {weightspath}")

    encoder = ViT(img_size=(512, 1024), backbone_name=BACKBONE_NAME)
    model = EoMT(
        encoder=encoder,
        num_classes=NUM_CLASSES,
        num_q=NUM_Q,
        num_blocks=NUM_BLOCKS,
        masked_attn_enabled=False
    )

    if not args.cpu:
        model = model.cuda()

    checkpoint = torch.load(weightspath, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            if name.startswith("module.") and name.split("module.")[-1] in own_state:
                own_state[name.split("module.")[-1]].copy_(param)
        else:
            own_state[name].copy_(param)
    
    print("Model loaded. Starting Temperature Scaling evaluation...")
    model.eval()

    # -------------------------------
    # INFERENZA
    # -------------------------------
    image_paths = []
    for pattern in args.input:
        image_paths.extend(glob.glob(os.path.expanduser(pattern)))

    print(f"Processing {len(image_paths)} images with temperatures: {all_temps}")

    for path in image_paths:
        # 1. Prepare Input
        try:
            img_pil = Image.open(path).convert("RGB")
        except:
            continue
            
        images = input_transform(img_pil).unsqueeze(0).float()
        if not args.cpu:
            images = images.cuda()

        with torch.no_grad():
            # 2. Forward Pass EoMT
            mask_logits_per_layer, class_logits_per_layer = model(images)
            
            mask_logits = mask_logits_per_layer[-1]    # [B, Q, H', W']
            class_logits = class_logits_per_layer[-1]  # [B, Q, C+1]

            # Rimuoviamo la classe "void" (ultima) per ottenere i logit delle classi note
            class_logits = class_logits[..., :NUM_CLASSES]   # [B, Q, C]

            # 3. Calcolo dei SEMSEG LOGITS (i logit per pixel)
            # Combinazione lineare: logit semantici = somma(class_logits * mask_logits)
            # Questo ci dà una mappa [B, C, H', W'] che rappresenta i "logit" grezzi per pixel
            semseg_logits = torch.einsum("bqc,bqhw->bchw", class_logits, mask_logits)
            
            # Upsampling alla dimensione originale (512x1024)
            semseg_logits = F.interpolate(
                semseg_logits, size=(512, 1024), mode="bilinear", align_corners=False
            )
            
            # Rimuoviamo la dimensione batch -> [C, H, W]
            pixel_logits = semseg_logits.squeeze(0)

            # 4. Calcolo MSP per ogni temperatura
            for t in all_temps:
                score = get_msp_score(pixel_logits, temperature=t)
                anomaly_scores_dict[t].append(score)

        # -------------------------------
        # CARICAMENTO GROUND TRUTH
        # -------------------------------
        pathGT = path.replace("images", "labels_masks")
        if "RoadObsticle21" in pathGT: pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT: pathGT = pathGT.replace("jpg", "png")
        if "RoadAnomaly" in pathGT: pathGT = pathGT.replace("jpg", "png")
        if not os.path.exists(pathGT): pathGT = pathGT.replace(".png", ".jpg")

        try:
            mask_gt = Image.open(pathGT)
            mask_gt = target_transform(mask_gt)
            ood_gts = np.array(mask_gt)

            if "RoadAnomaly" in pathGT:
                ood_gts = np.where((ood_gts == 2), 1, ood_gts)
            if "LostAndFound" in pathGT:
                ood_gts = np.where((ood_gts == 0), 255, ood_gts)
                ood_gts = np.where((ood_gts == 1), 0, ood_gts)
                ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)
            if "Streethazard" in pathGT:
                ood_gts = np.where((ood_gts == 14), 255, ood_gts)
                ood_gts = np.where((ood_gts < 20), 0, ood_gts)
                ood_gts = np.where((ood_gts == 255), 1, ood_gts)

            if 1 not in np.unique(ood_gts):
                # Se l'immagine viene saltata, dobbiamo rimuovere gli score appena aggiunti
                for t in all_temps:
                    anomaly_scores_dict[t].pop()
                continue
            
            ood_gts_list.append(ood_gts)

        except Exception as e:
            print(f"GT Error {path}: {e}")
            # Cleanup se fallisce il caricamento GT
            for t in all_temps:
                if len(anomaly_scores_dict[t]) > len(ood_gts_list):
                    anomaly_scores_dict[t].pop()
            continue
        
        # Memory cleanup
        del semseg_logits, pixel_logits
        torch.cuda.empty_cache()

    # -------------------------------
    # CALCOLO METRICHE E STAMPA
    # -------------------------------
    print("\n" + "="*50)
    print(f"RESULTS FOR: {args.input[0]}")
    print("="*50)

    ood_gts = np.array(ood_gts_list)
    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)
    
    # Template label per le metriche (0=In-Dist, 1=Anomaly)
    val_label = np.concatenate((np.zeros(ind_mask.sum()), np.ones(ood_mask.sum())))

    best_auprc = 0
    best_t = 1.0

    # Risultati per la tabella
    for t in all_temps:
        scores = np.array(anomaly_scores_dict[t])
        
        ood_out = scores[ood_mask]
        ind_out = scores[ind_mask]
        val_out = np.concatenate((ind_out, ood_out))

        prc_auc = average_precision_score(val_label, val_out)
        fpr = fpr_at_95_tpr(val_out, val_label)
        
        # Formattazione output
        prefix = " "
        if t in [0.5, 0.75, 1.1]: prefix = "*" # Evidenzia quelli richiesti specificamente
        
        print(f"{prefix} T = {t:<4} | AuPRC: {prc_auc*100.0} | FPR@95: {fpr*100.0}")

        # Check Best T
        if prc_auc > best_auprc:
            best_auprc = prc_auc
            best_t = t

    print("-" * 50)
    print(f"BEST T FOUND: {best_t} (AuPRC: {best_auprc*100.0})")
    print("-" * 50)

if __name__ == "__main__":
    main()