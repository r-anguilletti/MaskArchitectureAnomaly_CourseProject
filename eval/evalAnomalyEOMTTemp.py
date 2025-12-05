# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
import glob
import random
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from argparse import ArgumentParser
from sklearn.metrics import average_precision_score, roc_curve
from torchvision.transforms import Compose, Resize, ToTensor

# -----------------------------------------------------------------------------
# SETUP PATH & IMPORTS
# -----------------------------------------------------------------------------
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(CUR_DIR, "..")
EOMT_ROOT = os.path.join(PROJECT_ROOT, "eomt")

if EOMT_ROOT not in sys.path:
    sys.path.insert(0, EOMT_ROOT)

from models.vit import ViT
from models.eomt import EoMT
from training.lightning_module import LightningModule

# -----------------------------------------------------------------------------
# CONFIGURAZIONE & PARAMETRI
# -----------------------------------------------------------------------------
SEED = 42
NUM_CLASSES = 19              # Classi Cityscapes (senza void)
IMG_SIZE = (1024, 1024)       # Coerente con lo script evalAnomalyEOMT.py
NUM_QUERIES = 100
NUM_BLOCKS = 3
BACKBONE_NAME = "vit_base_patch14_reg4_dinov2"

# Setup riproducibilità
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Trasformazioni Input/Target
input_transform = Compose([
    Resize(IMG_SIZE, Image.BILINEAR),
    ToTensor(),
])

target_transform = Compose([
    Resize(IMG_SIZE, Image.NEAREST),
])

# -----------------------------------------------------------------------------
# FUNZIONI DI UTILITÀ
# -----------------------------------------------------------------------------
def fpr_at_95_tpr(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Calcola il False Positive Rate (FPR) quando il True Positive Rate (TPR) è al 95%.

    Args:
        scores: array dei punteggi di anomalia.
        labels: array binario delle etichette (1 = anomalia, 0 = in-distribution).
    """
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)

    idxs = np.where(tpr >= 0.95)[0]
    if len(idxs) == 0:
        return 1.0  # fallback pessimistico

    return float(fpr[idxs[0]])


def load_eomt_model(ckpt_path: str, device: torch.device) -> EoMT:
    """
    Inizializza l'architettura EoMT (Backbone ViT + Decoder) e carica i pesi
    usando la logica del LightningModule originale (come in evalAnomalyEOMT.py).
    """
    print(f"--> Inizializzazione Backbone: {BACKBONE_NAME}")
    encoder = ViT(img_size=IMG_SIZE, backbone_name=BACKBONE_NAME)

    print(f"--> Inizializzazione EoMT Network (Classes: {NUM_CLASSES}, Queries: {NUM_QUERIES})")
    network = EoMT(
        encoder=encoder,
        num_classes=NUM_CLASSES,
        num_q=NUM_QUERIES,
        num_blocks=NUM_BLOCKS,
        masked_attn_enabled=True,
    )

    # LightningModule si occupa del caricamento pesi
    lm = LightningModule(
        network=network,
        img_size=IMG_SIZE,
        num_classes=NUM_CLASSES,
        attn_mask_annealing_enabled=False,
        attn_mask_annealing_start_steps=None,
        attn_mask_annealing_end_steps=None,
        lr=1e-4,
        llrd=0.8,
        llrd_l2_enabled=True,
        lr_mult=1.0,
        weight_decay=0.05,
        poly_power=0.9,
        warmup_steps=(500, 1000),
        ckpt_path=ckpt_path,
        delta_weights=False,
        load_ckpt_class_head=True,
    )

    model = lm.network
    model.to(device)
    model.eval()
    return model


def get_msp_score(pixel_logits: torch.Tensor, temperature: float = 1.0) -> np.ndarray:
    """
    Calcola l'anomaly score MSP (1 - Max Softmax Prob) applicando la temperatura.

    Args:
        pixel_logits: Tensor [C, H, W] con i logits semantici per pixel.
        temperature: valore di temperatura per il temperature scaling.
    """
    # Scaling dei logits
    scaled_logits = pixel_logits / temperature

    # Softmax sui canali (classe) -> probabilità per pixel
    probs = F.softmax(scaled_logits, dim=0)

    # MSP = 1 - max(P(y|x))
    max_prob, _ = torch.max(probs, dim=0)
    anomaly_score = 1.0 - max_prob

    return anomaly_score.detach().cpu().numpy()


# -----------------------------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------------------------
def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        nargs="+",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        help="Path o glob pattern immagini input"
    )
    parser.add_argument(
        "--loadDir",
        default="../trained_models/",
        help="Cartella dei modelli salvati"
    )
    parser.add_argument(
        "--loadWeights",
        default="eomt_cityscapes.bin",
        help="Nome file checkpoint (come in evalAnomalyEOMT.py)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Forza esecuzione su CPU"
    )
    args = parser.parse_args()

    # Temperature da testare
    # Target per tabella: 0.5, 0.75, 1.1
    # Search values per 'best T'
    target_temps = [0.5, 0.75, 1.0, 1.1]
    search_temps = [0.3, 0.9, 1.2, 1.5, 2.0, 2.5, 3.0, 5.0, 7.0, 10.0, 11.0]
    all_temps = sorted(list(set(target_temps + search_temps)))

    # Dizionario: {T: [anomaly_map_img0, anomaly_map_img1, ...]}
    anomaly_scores_dict = {t: [] for t in all_temps}
    ood_gts_list = []

    # Configurazione device
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device in uso: {device}")

    ckpt_full_path = os.path.join(args.loadDir, args.loadWeights)
    print(f"Caricamento pesi da: {ckpt_full_path}")

    # Caricamento modello EoMT
    try:
        model = load_eomt_model(ckpt_full_path, device)
        print("Modello EoMT caricato con successo.")
    except Exception as e:
        print(f"Errore critico nel caricamento del modello: {e}")
        return

    # Costruzione file list
    image_paths = []
    for pattern in args.input:
        image_paths.extend(glob.glob(os.path.expanduser(pattern)))

    print(f"Processo {len(image_paths)} immagini con temperature: {all_temps}")

    # -------------------------------------------------------------------------
    # INFERENZA
    # -------------------------------------------------------------------------
    for path in image_paths:
        print(f"Processing: {path}")

        # 1. Input image
        try:
            img_pil = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Errore nel caricamento immagine {path}: {e}")
            continue

        img_tensor = input_transform(img_pil).unsqueeze(0).float().to(device)

        with torch.no_grad():
            # Forward EoMT
            mask_logits_layers, class_logits_layers = model(img_tensor)

            final_mask_logits = mask_logits_layers[-1]    # [B, Q, h, w]
            final_class_logits = class_logits_layers[-1]  # [B, Q, C+1]

            # Upsample dei mask logits alla dimensione IMG_SIZE
            final_mask_logits = F.interpolate(
                final_mask_logits, size=IMG_SIZE, mode="bilinear", align_corners=False
            )

            # Conversione in semantic logits per-pixel [B, C, H, W]
            per_pixel_logits = LightningModule.to_per_pixel_logits_semantic(
                final_mask_logits, final_class_logits
            )
            pixel_logits = per_pixel_logits[0]  # [C, H, W]

            # 2. MSP per ogni temperatura
            for t in all_temps:
                score_map = get_msp_score(pixel_logits, temperature=t)
                anomaly_scores_dict[t].append(score_map)

        # ---------------------------------------------------------------------
        # GROUND TRUTH
        # ---------------------------------------------------------------------
        pathGT = path.replace("images", "labels_masks")
        if "RoadObsticle21" in pathGT:
            pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
            pathGT = pathGT.replace("jpg", "png")
        if "RoadAnomaly" in pathGT:
            pathGT = pathGT.replace("jpg", "png")
        if not os.path.exists(pathGT):
            pathGT = pathGT.replace(".png", ".jpg")

        try:
            gt_img = Image.open(pathGT)
            gt_img = target_transform(gt_img)
            ood_gts = np.array(gt_img)

            # Mapping dataset-specific -> (0 in-dist, 1 OOD, 255 ignore)
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

            # Se non ci sono pixel OOD validi in questa immagine,
            # rimuoviamo gli anomaly score appena aggiunti
            if 1 not in np.unique(ood_gts):
                for t in all_temps:
                    if len(anomaly_scores_dict[t]) > len(ood_gts_list):
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

        # Pulizia memoria
        del img_tensor, pixel_logits
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # CALCOLO METRICHE
    # -------------------------------------------------------------------------
    if not ood_gts_list:
        print("Nessun dato valido raccolto per la valutazione.")
        return

    print("\n" + "=" * 50)
    print(f"RESULTS FOR: {args.input[0]}")
    print("=" * 50)

    ood_gts_all = np.array(ood_gts_list)
    ood_mask = (ood_gts_all == 1)
    ind_mask = (ood_gts_all == 0)

    # Template label per le metriche (0=In-Dist, 1=Anomaly)
    val_label = np.concatenate((np.zeros(ind_mask.sum()), np.ones(ood_mask.sum())))

    best_auprc = 0.0
    best_t = 1.0

    # Stampa tipo tabella:
    #  T | AuPRC | FPR@95
    for t in all_temps:
        scores_all = np.array(anomaly_scores_dict[t])

        # Estraggo i valori solo sui pixel validi
        ood_out = scores_all[ood_mask]
        ind_out = scores_all[ind_mask]
        val_out = np.concatenate((ind_out, ood_out))

        prc_auc = average_precision_score(val_label, val_out)
        fpr = fpr_at_95_tpr(val_out, val_label)

        # Evidenzia i T richiesti dalla tabella
        prefix = " "
        if t in [0.5, 0.75, 1.1]:
            prefix = "*"

        print(f"{prefix} T = {t:<4} | AuPRC: {prc_auc * 100.0:.6f} | FPR@95: {fpr * 100.0:.6f}")

        if prc_auc > best_auprc:
            best_auprc = prc_auc
            best_t = t

    print("-" * 50)
    print(f"BEST T FOUND: {best_t} (AuPRC: {best_auprc * 100.0:.2f})")
    print("-" * 50)


if __name__ == "__main__":
    main()