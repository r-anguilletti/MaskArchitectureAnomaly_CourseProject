# Code to calculate IoU (mean and per-class) in a dataset
# Adapted for EoMT from the original ERFNet evaluation code
#######################

import numpy as np
import torch
import torch.nn.functional as F
import os
import sys
import time
from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage

# -------------------------------------------------------------------
# GESTIONE PATH per importare i moduli custom (eomt, vit, dataset)
# -------------------------------------------------------------------
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CUR_DIR)          
EOMT_ROOT = os.path.join(PROJECT_ROOT, "eomt")   

# Aggiunge il path solo se non esiste già
if EOMT_ROOT not in sys.path:
    sys.path.append(EOMT_ROOT)

# Imports locali
from dataset import cityscapes
from transform import Relabel, ToLabel, Colorize
from iouEval import iouEval, getColorEntry

# Imports Modello
from models.vit import ViT
from models.eomt import EoMT

# -------------------------------------------------------------------
# CONFIGURAZIONE CLASSI
# -------------------------------------------------------------------
# Il modello EOMT è allenato per predire 19 classi (+1 classe void interna ai logit).
NUM_CLASSES_MODEL = 19 

# Per la valutazione, usiamo 20 classi.
# MOTIVO: Le label originali hanno 255 come "ignore". Noi mappiamo 255 -> 19.
# Quindi l'one-hot encoding deve poter ospitare l'indice 19 (quindi size 20).
# L'evaluator poi ignorerà l'indice 19 nel calcolo finale.
NUM_CLASSES_EVAL = 20 

# Parametri EOMT (Risoluzione Cityscapes)
IMG_HEIGHT = 1024
IMG_WIDTH = 2048
NUM_QUERIES = 100 

# Trasformazioni Input
input_transform_cityscapes = Compose([
    Resize((IMG_HEIGHT, IMG_WIDTH), Image.BILINEAR),
    ToTensor(),
])

# Trasformazioni Label (Ground Truth)
target_transform_cityscapes = Compose([
    Resize((IMG_HEIGHT, IMG_WIDTH), Image.NEAREST),
    ToLabel(),
    Relabel(255, 19),   # Mappa 255 (ignore) a 19 (che è la 20esima classe)
])

def main(args):

    # Costruzione percorso pesi
    # Se loadDir è specificato e loadWeights non è assoluto, li unisce.
    if args.loadDir and not os.path.isabs(args.loadWeights):
        weightspath = os.path.join(args.loadDir, args.loadWeights)
    else:
        weightspath = args.loadWeights

    print ("Loading weights: " + weightspath)

    # -------------------------------------------------
    # 1. INIZIALIZZAZIONE MODELLO (EoMT + ViT)
    # -------------------------------------------------
    encoder = ViT(
        img_size=(IMG_HEIGHT, IMG_WIDTH), 
        patch_size=16, 
        backbone_name="vit_large_patch14_reg4_dinov2", 
        ckpt_path=None
    )

    model = EoMT(
        encoder=encoder,
        num_classes=NUM_CLASSES_MODEL, # 19 classi
        num_q=NUM_QUERIES,
        num_blocks=4,
        masked_attn_enabled=True
    )

    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    # -------------------------------------------------
    # 2. CARICAMENTO PESI ROBUSTO
    # -------------------------------------------------
    if not os.path.exists(weightspath):
        print(f"ERROR: Weights file not found at {weightspath}")
        return

    print(f"Loading checkpoint from {weightspath}...")
    checkpoint = torch.load(weightspath, map_location='cpu')
    
    # Gestione annidamento dizionari
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Pulizia prefisso 'module.' (necessario se allenato in DataParallel ma caricato qui diversamente)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    # Caricamento strict=False (ViT rimuove token, etc.)
    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)
    
    print ("Model and weights LOADED successfully")

    model.eval()

    if(not os.path.exists(args.datadir)):
        print (f"Error: datadir '{args.datadir}' does not exist")

    # -------------------------------------------------
    # 3. DATALOADER
    # -------------------------------------------------
    loader = DataLoader(
        cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset), 
        num_workers=args.num_workers, 
        batch_size=args.batch_size, 
        shuffle=False
    )

    # Inizializza iouEval con 20 classi totali, dicendo di ignorare l'indice 19
    iouEvalVal = iouEval(NUM_CLASSES_EVAL, ignoreIndex=19)

    start = time.time()

    print("Starting evaluation loop...")
    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        if (not args.cpu):
            images = images.cuda()
            labels = labels.cuda()

        inputs = Variable(images)
        with torch.no_grad():
            # -------------------------------------------------
            # 4. FORWARD & DECODING
            # -------------------------------------------------
            mask_logits_list, class_logits_list = model(inputs)
            
            # Prendi output dell'ultimo blocco
            mask_logits = mask_logits_list[-1]   # [B, Q, H_feat, W_feat]
            class_logits = class_logits_list[-1] # [B, Q, C+1]

            # Probabilità Classi (Softmax) e Maschere (Sigmoid)
            out_prob = F.softmax(class_logits, dim=-1)
            out_mask = F.sigmoid(mask_logits)

            # Rimuovi l'ultima classe dai logit (assume sia 'no object'/'void')
            # Rimangono le 19 classi semantiche
            out_prob = out_prob[:, :, :-1] 
            
            # Matrix Multiplication: combina classi e maschere
            # [B, Q, 19] x [B, Q, H, W] -> [B, 19, H, W]
            sem_seg = torch.einsum("bqc, bqhw -> bchw", out_prob, out_mask)

            # Upsample bilineare alla risoluzione originale (1024x2048)
            sem_seg = F.interpolate(sem_seg, size=(IMG_HEIGHT, IMG_WIDTH), mode='bilinear', align_corners=False)

            # Argmax per ottenere la classe vincente per ogni pixel (Indici 0-18)
            pred_labels = sem_seg.max(1)[1].unsqueeze(1).data

        # -------------------------------------------------
        # 5. AGGIORNAMENTO METRICHE
        # -------------------------------------------------
        # pred_labels: [0-18]
        # labels: [0-18] e [19] (dove era 255)
        iouEvalVal.addBatch(pred_labels, labels)

        filenameSave = filename[0].split("leftImg8bit/")[1] 
        # Feedback visivo ogni 10 step
        if step % 10 == 0:
            print (f"Step {step}: {filenameSave}")

    # -------------------------------------------------
    # 6. CALCOLO E STAMPA RISULTATI
    # -------------------------------------------------
    iouVal, iou_classes = iouEvalVal.getIoU()

    iou_classes_str = []
    for i in range(iou_classes.size(0)):
        # Formatta con colore in base al punteggio
        iouStr = getColorEntry(iou_classes[i])+'{:0.2f}'.format(iou_classes[i]*100) + '\033[0m'
        iou_classes_str.append(iouStr)

    print("---------------------------------------")
    print("Took ", time.time()-start, "seconds")
    print("=======================================")
    print("Per-Class IoU:")
    
    classes_names = [
        "Road", "Sidewalk", "Building", "Wall", "Fence", "Pole", "Traffic Light", 
        "Traffic Sign", "Vegetation", "Terrain", "Sky", "Person", "Rider", "Car", 
        "Truck", "Bus", "Train", "Motorcycle", "Bicycle"
    ]
    
    # Stampa solo le prime 19 classi (ignorando l'eventuale 20esima 'ignore')
    for i in range(min(len(iou_classes_str), len(classes_names))):
        print(f"{iou_classes_str[i]} {classes_names[i]}")
        
    print("=======================================")
    iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
    print ("MEAN IoU: ", iouStr, "%")

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')

    parser.add_argument('--loadDir', default="") 
    parser.add_argument('--loadWeights', required=True)
    parser.add_argument('--loadModel', default="eomt.py")
    parser.add_argument('--subset', default="val")
    parser.add_argument('--datadir', required=True)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')

    main(parser.parse_args())