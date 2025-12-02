# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
import glob
import torch
import torch.nn.functional as F
import random
from PIL import Image
import numpy as np
import os.path as osp
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
# SETUP & SEED
# -------------------------------------------------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

NUM_CLASSES = 20  # Cityscapes classes (solitamente 19 + background o void, verifica la tua config)
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

def main():
    parser = ArgumentParser()
    parser.add_argument("--input", default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp", nargs="+")
    parser.add_argument("--loadDir", default="../trained_models/")
    parser.add_argument("--loadWeights", default="eomt_cityscapes_semantic.pth")
    parser.add_argument("--subset", default="val")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    # Liste risultati
    anomaly_score_list = []
    ood_gts_list = []

    if not os.path.exists("results_rba.txt"):
        open("results_rba.txt", "w").close()
    file = open("results_rba.txt", "a")

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

    # Caricamento pesi
    checkpoint = torch.load(weightspath, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Pulizia chiavi state_dict
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            # Gestione prefisso "module."
            if name.startswith("module.") and name.split("module.")[-1] in own_state:
                own_state[name.split("module.")[-1]].copy_(param)
        else:
            own_state[name].copy_(param)
    
    print("Model loaded successfully for RbA evaluation.")
    model.eval()

    # -------------------------------
    # INFERENZA
    # -------------------------------
    # Espansione glob pattern
    image_paths = []
    for pattern in args.input:
        image_paths.extend(glob.glob(os.path.expanduser(pattern)))

    print(f"Processing {len(image_paths)} images...")

    for path in image_paths:
        # 1. Prepare Input
        img_pil = Image.open(path).convert("RGB")
        images = input_transform(img_pil).unsqueeze(0).float()
        if not args.cpu:
            images = images.cuda()

        with torch.no_grad():
            # 2. Forward Pass
            # Otteniamo i logits grezzi dal modello (Mask Transformer)
            mask_logits_per_layer, class_logits_per_layer = model(images)
            
            # Usiamo l'output dell'ultimo layer
            mask_logits = mask_logits_per_layer[-1]    # [B, Q, H_feat, W_feat]
            class_logits = class_logits_per_layer[-1]  # [B, Q, K+1] (K classi + void/no-object)

            # ---------------------------------------------------------
            # IMPLEMENTAZIONE RbA (Rejected by All) SCORING
            # ---------------------------------------------------------
            # RbA si basa sulle probabilità reali, non sui logits grezzi combinati.
            
            # A. Probabilità di Classe per ogni Query: Softmax su K+1 classi
            #    Shape: [B, Q, K+1]
            class_probs = F.softmax(class_logits, dim=-1)
            
            # B. Probabilità della Maschera per ogni Query: Sigmoid
            #    Shape: [B, Q, H_feat, W_feat]
            mask_probs = torch.sigmoid(mask_logits)
            
            # C. Calcolo della Mappa di Probabilità Semantica per le Classi NOTE
            #    P(class=c | pixel) = Sum_over_Queries ( P(class=c|query) * P(mask|query) )
            #    Consideriamo solo le prime NUM_CLASSES (escludiamo l'ultima classe "void" dai logit di classe)
            #    Shape risultante: [B, NUM_CLASSES, H_feat, W_feat]
            
            # Selezioniamo solo le probabilità delle classi in-distribution (0...K-1)
            class_probs_known = class_probs[..., :NUM_CLASSES] 
            
            # Moltiplicazione matriciale (einsum) per aggregare le query
            prob_map_known = torch.einsum("bqc,bqhw->bchw", class_probs_known, mask_probs)
            
            # D. Calcolo Score RbA
            #    RbA definisce l'anomalia come "essere rifiutato da tutte le classi note".
            #    Score = 1.0 - Somma(Probabilità di tutte le classi note)
            #    Questo è matematicamente equivalente alla probabilità assegnata alla classe "void" 
            #    o "residua" dal modello.
            
            # Somma su tutte le classi note -> [B, H_feat, W_feat]
            total_known_prob = torch.sum(prob_map_known, dim=1)
            
            # Score finale (più alto = più anomalo)
            rba_score_map = 1.0 - total_known_prob

            # Interpolazione alla risoluzione originale (512, 1024)
            rba_score_map = F.interpolate(
                rba_score_map.unsqueeze(1), # Aggiungi dim canali per interpolate
                size=(512, 1024), 
                mode="bilinear", 
                align_corners=False
            ).squeeze(1) # Rimuovi dim canali
            
            # Converti in numpy per metriche
            anomaly_score = rba_score_map.squeeze(0).cpu().numpy()

        # -------------------------------
        # CARICAMENTO GROUND TRUTH (GT)
        # -------------------------------
        pathGT = path.replace("images", "labels_masks")
        if "RoadObsticle21" in pathGT: pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT: pathGT = pathGT.replace("jpg", "png")
        if "RoadAnomaly" in pathGT: pathGT = pathGT.replace("jpg", "png")
        
        # Fallback estensione
        if not os.path.exists(pathGT):
            pathGT = pathGT.replace(".png", ".jpg")

        try:
            mask_gt = Image.open(pathGT)
            mask_gt = target_transform(mask_gt)
            ood_gts = np.array(mask_gt)

            # Mappatura etichette OOD specifica per dataset
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
                continue
            
            ood_gts_list.append(ood_gts)
            anomaly_score_list.append(anomaly_score)

        except Exception as e:
            print(f"Error loading GT for {path}: {e}")
            continue

        # Memory Cleanup
        del mask_logits, class_logits, prob_map_known, rba_score_map, total_known_prob
        torch.cuda.empty_cache()

    # -------------------------------
    # CALCOLO METRICHE FINALI
    # -------------------------------
    print("Calculating Metrics...")
    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    # Flattening
    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((np.zeros(len(ind_out)), np.ones(len(ood_out))))

    # Metriche
    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f"RbA Results for {args.input}:")
    print(f"  AUPRC score: {prc_auc * 100.0}")
    print(f"  FPR@TPR95:   {fpr * 100.0}")

    file.write(f"\nResults for {args.input}:\n")
    file.write(f"  AUPRC score RbA: {prc_auc * 100.0}\n")
    file.write(f"  FPR@TPR95 RbA:   {fpr * 100.0}\n")
    file.close()

if __name__ == "__main__":
    main()