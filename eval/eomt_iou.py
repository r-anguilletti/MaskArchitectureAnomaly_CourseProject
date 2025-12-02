# Code to calculate IoU (mean and per-class) in a dataset
# Nov 2017
# Eduardo Romera
#######################

import numpy as np
import torch
import torch.nn.functional as F
import os
import importlib
import time

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage

from dataset import cityscapes
from eomt.models.vit import ViT
from eomt.models.eomt import EoMT
from transform import Relabel, ToLabel, Colorize
from iouEval import iouEval, getColorEntry

NUM_CHANNELS = 3
NUM_CLASSES = 19 
# Cityscapes ha 19 classi di valutazione (ignora la 20esima background/void se gestita diversamente)

# Parametri specifici per EoMT
IMG_HEIGHT = 1024
IMG_WIDTH = 2048
NUM_QUERIES = 100 # <--- IMPORTANTE: Verifica questo numero nel config del tuo training (spesso è 100)


image_transform = ToPILImage()
input_transform_cityscapes = Compose([
    # Se le immagini sono già 1024x2048 su disco, Resize potrebbe non servire, 
    # ma lo lasciamo per sicurezza o se usi crop.
    Resize((IMG_HEIGHT, IMG_WIDTH), Image.BILINEAR), 
    ToTensor(),
    # NOTA: Non aggiungiamo Normalize qui, perché EoMT lo fa internamente nel forward.
])

# Assicurati che anche le label siano ridimensionate correttamente
target_transform_cityscapes = Compose([
    Resize((IMG_HEIGHT, IMG_WIDTH), Image.NEAREST),
    ToLabel(),
    Relabel(255, 19), 
])

def main(args):

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    # 1. Inizializza il Backbone ViT
    # Nota: patch_size=14 o 16 dipende da come hai allenato. Controlla il tuo config.
    encoder = ViT(
        img_size=(IMG_HEIGHT, IMG_WIDTH), 
        patch_size=16, 
        backbone_name="vit_large_patch14_reg4_dinov2", # O il nome usato in training
        ckpt_path=None # Non carichiamo pesi pretreinati di imagenet ora, caricheremo il bin completo
    )

    # 2. Inizializza EoMT
    model = EoMT(
        encoder=encoder,
        num_classes=NUM_CLASSES,
        num_q=NUM_QUERIES,
        num_blocks=4, # Verifica se in training era 4
        masked_attn_enabled=True
    )

    print(f"Loading weights from: {weightspath}")
    checkpoint = torch.load(weightspath, map_location='cpu')
    
    # 1. Gestione caso in cui i pesi siano dentro una sotto-chiave
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']

    # 2. Rimuovi il prefisso "module." (fix veloce con dict comprehension)
    clean_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}

    # 3. Carica con strict=False (ignora le chiavi cancellate da vit.py)
    model.load_state_dict(clean_state_dict, strict=False)
    
    print("Weights loaded!")

    if (not args.cpu):
         model = torch.nn.DataParallel(model).cuda()

    model.eval()

    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")


    loader = DataLoader(cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset), num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)


    iouEvalVal = iouEval(NUM_CLASSES)

    start = time.time()

    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        if (not args.cpu):
            images = images.cuda()
            labels = labels.cuda()

        inputs = Variable(images)
        with torch.no_grad():
            # EoMT restituisce liste di output per ogni blocco. Prendiamo l'ultimo (-1).
            mask_logits_list, class_logits_list = model(inputs)
            
            # [Batch, Queries, H_feat, W_feat]
            mask_logits = mask_logits_list[-1] 
            # [Batch, Queries, NumClasses + 1]
            class_logits = class_logits_list[-1]

            # 1. Calcola le probabilità
            # Softmax sulle classi (lungo l'ultima dim)
            out_prob = F.softmax(class_logits, dim=-1)
            # Sigmoide sulle maschere
            out_mask = F.sigmoid(mask_logits)

            # 2. Moltiplicazione matriciale per ottenere la mappa semantica [B, NumClasses, H_feat, W_feat]
            # Escludiamo l'ultima classe che solitamente è "No Object" o "Void" in questi modelli (index -1)
            # Verifica se il tuo modello usa index 0 o index -1 per background. 
            # Assumiamo standard Mask2Former: le prime K sono classi, l'ultima è void.
            out_prob = out_prob[:, :, :-1] # Rimuovi classe void dai logit
            
            # Einsum: Somma pesata delle maschere in base alla probabilità della classe
            # b=batch, q=query, c=class, h=height, w=width
            sem_seg = torch.einsum("bqc, bqhw -> bchw", out_prob, out_mask)

            # 3. Upsample alla dimensione originale (1024x2048)
            sem_seg = F.interpolate(sem_seg, size=(IMG_HEIGHT, IMG_WIDTH), mode='bilinear', align_corners=False)

            # 4. Argmax per ottenere la classe vincente per ogni pixel
            pred_labels = sem_seg.max(1)[1].unsqueeze(1).data

        # Passa il risultato all'evaluator
        iouEvalVal.addBatch(pred_labels, labels)

        filenameSave = filename[0].split("leftImg8bit/")[1] 

        print (step, filenameSave)


    iouVal, iou_classes = iouEvalVal.getIoU()

    iou_classes_str = []
    for i in range(iou_classes.size(0)):
        iouStr = getColorEntry(iou_classes[i])+'{:0.2f}'.format(iou_classes[i]*100) + '\033[0m'
        iou_classes_str.append(iouStr)

    print("---------------------------------------")
    print("Took ", time.time()-start, "seconds")
    print("=======================================")
    #print("TOTAL IOU: ", iou * 100, "%")
    print("Per-Class IoU:")
    print(iou_classes_str[0], "Road")
    print(iou_classes_str[1], "sidewalk")
    print(iou_classes_str[2], "building")
    print(iou_classes_str[3], "wall")
    print(iou_classes_str[4], "fence")
    print(iou_classes_str[5], "pole")
    print(iou_classes_str[6], "traffic light")
    print(iou_classes_str[7], "traffic sign")
    print(iou_classes_str[8], "vegetation")
    print(iou_classes_str[9], "terrain")
    print(iou_classes_str[10], "sky")
    print(iou_classes_str[11], "person")
    print(iou_classes_str[12], "rider")
    print(iou_classes_str[13], "car")
    print(iou_classes_str[14], "truck")
    print(iou_classes_str[15], "bus")
    print(iou_classes_str[16], "train")
    print(iou_classes_str[17], "motorcycle")
    print(iou_classes_str[18], "bicycle")
    print("=======================================")
    iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
    print ("MEAN IoU: ", iouStr, "%")

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')

    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="../eomt/trained_models/pytorch_model.bin")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')

    main(parser.parse_args())
