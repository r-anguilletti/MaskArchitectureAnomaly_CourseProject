# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
import glob
import torch
import torch.nn.functional as F
import random
from PIL import Image
import numpy as np

import cv2  # usato solo se ti serve salvare visualizzazioni, altrimenti puoi rimuoverlo

import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr, plot_barcode
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# -------------------------------------------------------------------
# AGGIUNTA DELLA CARTELLA eomt/ AL PYTHONPATH
# -------------------------------------------------------------------
# Struttura:
#   MaskArchitectureAnomaly_CourseProject/
#       eomt/
#           models/vit.py, eomt.py, ...
#       eval/
#           evalAnomaly_eomt.py  <-- questo file
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CUR_DIR)          # MaskArchitectureAnomaly_CourseProject/
EOMT_ROOT = os.path.join(PROJECT_ROOT, "eomt")   # .../MaskArchitectureAnomaly_CourseProject/eomt
if EOMT_ROOT not in sys.path:
    sys.path.append(EOMT_ROOT)

from models.vit import ViT
from models.eomt import EoMT

# -------------------------------------------------------------------
# SEED & PARAMETRI
# -------------------------------------------------------------------
seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CLASSES = 20
# Hyperparam EoMT presi dalla config che mi hai dato
NUM_Q = 100
NUM_BLOCKS = 3
BACKBONE_NAME = "vit_base_patch14_reg4_dinov2"

# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Per EoMT: la normalizzazione viene fatta dentro l'encoder (pixel_mean/std),
# quindi qui facciamo solo resize + ToTensor.
input_transform = Compose(
    [
        Resize((512, 1024), Image.BILINEAR),
        ToTensor(),
    ]
)

target_transform = Compose(
    [
        Resize((512, 1024), Image.NEAREST),
    ]
)


def compute_anomaly_from_logits(logits: torch.Tensor, method: str = "msp") -> np.ndarray:
    """
    Calcola la mappa di anomaly score a partire da logits per classe/pixel.

    logits: torch.Tensor [1, C, H, W] oppure [C, H, W]
    method: "msp", "maxlogit", "entropy"
    """
    # Assicuriamoci di avere [C, H, W]
    if logits.dim() == 4:
        # [1, C, H, W] -> [C, H, W]
        logits = logits.squeeze(0)

    # softmax sui canali di classe
    probs = torch.softmax(logits, dim=0)  # [C, H, W]

    if method == "msp":
        # MSP: 1 - max softmax probability
        max_prob, _ = probs.max(dim=0)  # [H, W]
        score = 1.0 - max_prob

    elif method == "maxlogit":
        # MaxLogit: - max logit (più basso il logit -> più alto lo score)
        max_logit, _ = logits.max(dim=0)  # [H, W]
        score = -max_logit

    elif method == "entropy":
        # Entropy: -sum p*log(p)
        log_probs = torch.log(probs.clamp(min=1e-12))
        entropy = -torch.sum(probs * log_probs, dim=0)  # [H, W]
        score = entropy

    else:
        raise ValueError(f"Unknown method: {method}")

    return score.cpu().numpy()  # [H, W] in numpy


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument("--loadDir", default="../trained_models/")
    parser.add_argument("--loadWeights", default="eomt_cityscapes_semantic.pth")
    parser.add_argument("--subset", default="val", help="can be val or train (must have labels)")
    parser.add_argument(
        "--datadir",
        default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()

    # Liste per le tre metriche
    anomaly_msp_list = []
    anomaly_maxlogit_list = []
    anomaly_entropy_list = []
    ood_gts_list = []

    if not os.path.exists("results.txt"):
        open("results.txt", "w").close()
    file = open("results.txt", "a")

    modelpath = os.path.join(args.loadDir, "eomt.py")  # solo per stampa
    weightspath = os.path.join(args.loadDir, args.loadWeights)

    print("Loading model: " + modelpath)
    print("Loading weights: " + weightspath)

    # -------------------------------
    # ISTANZIA EoMT (encoder + network)
    # -------------------------------
    # Encoder ViT (firma come in inference.ipynb: ViT(img_size=..., backbone_name=...))
    encoder = ViT(img_size=(512, 1024), backbone_name=BACKBONE_NAME)

    # Modello EoMT
    model = EoMT(
        encoder=encoder,
        num_classes=NUM_CLASSES,
        num_q=NUM_Q,
        num_blocks=NUM_BLOCKS,
        masked_attn_enabled=False,  # per eval non usiamo masked attention
    )

    if not args.cpu:
        model = model.cuda()

    def load_my_state_dict(model, state_dict):
        """Carica lo state_dict gestendo eventuale 'module.' davanti ai nomi."""
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                # prova a togliere "module." se viene da DataParallel
                if name.startswith("module.") and name.split("module.")[-1] in own_state:
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    # Carica checkpoint (sia formato puro che Lightning-style con "state_dict")
    ckpt = torch.load(weightspath, map_location=lambda storage, loc: storage)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model = load_my_state_dict(model, ckpt)

    print("Model and weights LOADED successfully")
    model.eval()

    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        print(path)
        images = (
            input_transform((Image.open(path).convert("RGB")))
            .unsqueeze(0)
            .float()
        )

        if not args.cpu:
            images = images.cuda()

        with torch.no_grad():
            # ==========================
            # FORWARD PASS EoMT
            # ==========================
            # EoMT.forward ritorna:
            #   mask_logits_per_layer: lista di [B, Q, H', W']
            #   class_logits_per_layer: lista di [B, Q, C+1]
            mask_logits_per_layer, class_logits_per_layer = model(images)

        # Prendiamo l'ultimo layer (come in inference.ipynb)
        mask_logits = mask_logits_per_layer[-1]          # [B, Q, H', W']
        class_logits = class_logits_per_layer[-1]        # [B, Q, C+1]

        # Rimuoviamo il canale "no-object" (ultimo)
        class_logits = class_logits[..., :NUM_CLASSES]   # [B, Q, C]

        # Combiniamo query class + query mask in logits semantici per pixel:
        # semseg_logits[b, c, h, w] = sum_q class_logits[b, q, c] * mask_logits[b, q, h, w]
        semseg_logits = torch.einsum("bqc,bqhw->bchw", class_logits, mask_logits)  # [B, C, H', W']

        # Riportiamo alla risoluzione di 512x1024 (come GT)
        semseg_logits = F.interpolate(
            semseg_logits, size=(512, 1024), mode="bilinear", align_corners=False
        )

        # ==========================
        # ANOMALY SCORES (tre metodi)
        # ==========================
        anomaly_msp = compute_anomaly_from_logits(semseg_logits, method="msp")
        anomaly_maxlogit = compute_anomaly_from_logits(semseg_logits, method="maxlogit")
        anomaly_entropy = compute_anomaly_from_logits(semseg_logits, method="entropy")

        # ==========================
        # GROUND TRUTH LOADING
        # ==========================
        pathGT = path.replace("images", "labels_masks")
        if "RoadObsticle21" in pathGT:
            pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
            pathGT = pathGT.replace("jpg", "png")
        if "RoadAnomaly" in pathGT:
            pathGT = pathGT.replace("jpg", "png")

        mask = Image.open(pathGT)
        mask = target_transform(mask)
        ood_gts = np.array(mask)

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
        else:
            ood_gts_list.append(ood_gts)
            anomaly_msp_list.append(anomaly_msp)
            anomaly_maxlogit_list.append(anomaly_maxlogit)
            anomaly_entropy_list.append(anomaly_entropy)

        del (
            mask_logits_per_layer,
            class_logits_per_layer,
            mask_logits,
            class_logits,
            semseg_logits,
            anomaly_msp,
            anomaly_maxlogit,
            anomaly_entropy,
            ood_gts,
            mask,
        )
        torch.cuda.empty_cache()

    file.write("\n")

    # ==========================
    # CALCOLO METRICHE
    # ==========================
    ood_gts = np.array(ood_gts_list)
    scores_msp = np.array(anomaly_msp_list)
    scores_maxlogit = np.array(anomaly_maxlogit_list)
    scores_entropy = np.array(anomaly_entropy_list)

    ood_mask = ood_gts == 1
    ind_mask = ood_gts == 0

    # Label OOD / in-distribution
    ood_label = np.ones(np.count_nonzero(ood_mask))
    ind_label = np.zeros(np.count_nonzero(ind_mask))
    val_label = np.concatenate((ind_label, ood_label))

    # ---- MSP ----
    ood_out_msp = scores_msp[ood_mask]
    ind_out_msp = scores_msp[ind_mask]
    val_out_msp = np.concatenate((ind_out_msp, ood_out_msp))

    prc_auc_msp = average_precision_score(val_label, val_out_msp)
    fpr_msp = fpr_at_95_tpr(val_out_msp, val_label)

    print(f"AUPRC score MSP: {prc_auc_msp * 100.0}")
    print(f"FPR@TPR95 MSP: {fpr_msp * 100.0}")

    # ---- MaxLogit ----
    ood_out_maxlogit = scores_maxlogit[ood_mask]
    ind_out_maxlogit = scores_maxlogit[ind_mask]
    val_out_maxlogit = np.concatenate((ind_out_maxlogit, ood_out_maxlogit))

    prc_auc_maxlogit = average_precision_score(val_label, val_out_maxlogit)
    fpr_maxlogit = fpr_at_95_tpr(val_out_maxlogit, val_label)

    print(f"AUPRC score MaxLogit: {prc_auc_maxlogit * 100.0}")
    print(f"FPR@TPR95 MaxLogit: {fpr_maxlogit * 100.0}")

    # ---- Entropy ----
    ood_out_entropy = scores_entropy[ood_mask]
    ind_out_entropy = scores_entropy[ind_mask]
    val_out_entropy = np.concatenate((ind_out_entropy, ood_out_entropy))

    prc_auc_entropy = average_precision_score(val_label, val_out_entropy)
    fpr_entropy = fpr_at_95_tpr(val_out_entropy, val_label)

    print(f"AUPRC score Entropy: {prc_auc_entropy * 100.0}")
    print(f"FPR@TPR95 Entropy: {fpr_entropy * 100.0}")

    file.write(
        "    AUPRC score MSP:"
        + str(prc_auc_msp * 100.0)
        + "   FPR@TPR95 MSP:"
        + str(fpr_msp * 100.0)
        + "\n"
        + "    AUPRC score MaxLogit:"
        + str(prc_auc_maxlogit * 100.0)
        + "   FPR@TPR95 MaxLogit:"
        + str(fpr_maxlogit * 100.0)
        + "\n"
        + "    AUPRC score Entropy:"
        + str(prc_auc_entropy * 100.0)
        + "   FPR@TPR95 Entropy:"
        + str(fpr_entropy * 100.0)
        + "\n"
    )
    file.close()


if __name__ == "__main__":
    main()