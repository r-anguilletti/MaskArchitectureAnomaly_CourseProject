import os
import sys
import time
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

# -----------------------------------------------------------------------------
# FIX IMPORT: Aggiunta dinamica dei path
# -----------------------------------------------------------------------------
# 1. Troviamo la cartella dove si trova questo script (eval)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Troviamo la cartella genitore (MaskArchitectureAnomaly_CourseProject)
parent_dir = os.path.dirname(current_dir)
# 3. Costruiamo il path per la cartella 'eomt'
eomt_dir = os.path.join(parent_dir, 'eomt')

# Aggiungiamo 'eomt' al system path così Python trova la cartella 'models'
if eomt_dir not in sys.path:
    sys.path.append(eomt_dir)
    print(f"--> [System] Aggiunto al path: {eomt_dir}")

# Ora possiamo importare 'models' direttamente perché siamo dentro 'eomt' col path
try:
    from models.vit import ViT
    from models.eomt import EoMT
except ImportError as e:
    # Fallback: se siamo già nella root corretta
    from eomt.models.vit import ViT
    from eomt.models.eomt import EoMT

from iouEval import iouEval, getColorEntry
from dataset import cityscapes 

# -----------------------------------------------------------------------------
# TRASFORMAZIONI TARGET
# -----------------------------------------------------------------------------
class ToTargetTensor:
    """Converte la PIL Image della label in un Tensore (Long)."""
    def __call__(self, pic):
        return torch.from_numpy(np.array(pic)).long()

# -----------------------------------------------------------------------------
# UTILS
# -----------------------------------------------------------------------------
NUM_CLASSES = 19
# Usiamo 19 come indice interno per "ignore". 
# iouEval avrà bisogno di 20 classi (0-18 valide, 19 ignore)
IGNORE_INDEX = 19 

def get_semantic_segmentation(mask_logits, class_logits, target_size):
    # mask_logits: [Q, H_enc, W_enc]
    # class_logits: [Q, K+1]
    
    class_probs = F.softmax(class_logits, dim=-1) # [Q, K+1]
    class_probs = class_probs[:, :-1] # Rimuovi background [Q, K]
    
    mask_probs = torch.sigmoid(mask_logits) # [Q, H_enc, W_enc]
    
    # Matmul: (K, Q) @ (Q, HW) -> (K, HW)
    num_queries = mask_probs.shape[0]
    h_enc, w_enc = mask_probs.shape[1], mask_probs.shape[2]
    
    mask_probs_flat = mask_probs.view(num_queries, -1)
    class_probs_t = class_probs.transpose(0, 1)
    
    semantic_logits = torch.matmul(class_probs_t, mask_probs_flat)
    semantic_logits = semantic_logits.view(NUM_CLASSES, h_enc, w_enc)
    
    # Upsample alla risoluzione originale
    semantic_logits = F.interpolate(
        semantic_logits.unsqueeze(0), 
        size=target_size, 
        mode='bilinear', 
        align_corners=False
    ).squeeze(0)
    
    pred_map = torch.argmax(semantic_logits, dim=0)
    return pred_map

def load_eomt_checkpoint(model, weight_path, device):
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Non trovo i pesi in: {weight_path}")
        
    print(f"--> Caricamento pesi da: {weight_path}")
    checkpoint = torch.load(weight_path, map_location='cpu')
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        if name.startswith("network."): name = name[8:]
        elif name.startswith("module."): name = name[7:]
        new_state_dict[name] = v
        
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"    [Info] Chiavi mancanti: {len(missing)} (normale se mancano loss/aux heads)")
    
    model.to(device)
    model.eval()
    return model

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main(args):
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"--- Configurazione: {device} | Risoluzione Input: {args.img_size} ---")

    # 1. Trasformazioni
    input_transform = Compose([
        Resize((args.img_size, args.img_size), Image.BILINEAR),
        ToTensor(),
    ])
    
    target_transform = Compose([
        ToTargetTensor(), 
    ])
    
    # 2. Dataset
    dataset = cityscapes(
        args.datadir, 
        input_transform, 
        target_transform, 
        subset=args.subset
    )
    
    loader = DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False
    )

    # 3. Modello
    print(f"--> Init Backbone: {args.backbone}")
    encoder = ViT(img_size=(args.img_size, args.img_size), backbone_name=args.backbone, patch_size=16)

    print(f"--> Init EoMT (K={NUM_CLASSES}, Q={args.num_queries})")
    model = EoMT(
        encoder=encoder,
        num_classes=NUM_CLASSES,
        num_q=args.num_queries,
        num_blocks=args.num_blocks, 
        masked_attn_enabled=True,
    )

    model = load_eomt_checkpoint(model, args.loadWeights, device)

    # 4. Valutazione [FIX CRASH CUDA + IGNORE INDEX]
    iouEvalVal = iouEval(NUM_CLASSES + 1, ignoreIndex=IGNORE_INDEX)
    
    start = time.time()
    print("--> Inizio Valutazione...")

    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        if (not args.cpu):
            images = images.to(device)
            labels = labels.to(device)

        # [CRUCIAL FIX] Mappiamo i label void (255) a 19 per evitare crash CUDA
        labels[labels == 255] = IGNORE_INDEX
        labels[labels > IGNORE_INDEX] = IGNORE_INDEX

        h_orig, w_orig = labels.shape[-2], labels.shape[-1]

        with torch.no_grad():
            mask_logits_layers, class_logits_layers = model(images)
            final_mask_logits = mask_logits_layers[-1]
            final_class_logits = class_logits_layers[-1]

            batch_preds = []
            for b in range(images.shape[0]):
                pred_mask = get_semantic_segmentation(
                    final_mask_logits[b], 
                    final_class_logits[b], 
                    target_size=(h_orig, w_orig)
                )
                batch_preds.append(pred_mask)
            
            outputs = torch.stack(batch_preds)

        if labels.dim() == 3:
            labels = labels.unsqueeze(1)
        
        iouEvalVal.addBatch(outputs.unsqueeze(1), labels)

        if step % 50 == 0:
            print(f"Step {step}/{len(loader)} - {filename[0]}")

    # 5. Risultati
    iouVal, iou_classes = iouEvalVal.getIoU()

    print("\n" + "="*40)
    print(f"Took {time.time()-start:.1f} seconds")
    print("="*40)
    
    class_names = [
        "Road", "Sidewalk", "Building", "Wall", "Fence", "Pole", "Traffic Light",
        "Traffic Sign", "Vegetation", "Terrain", "Sky", "Person", "Rider", 
        "Car", "Truck", "Bus", "Train", "Motorcycle", "Bicycle"
    ]
    
    print("Per-Class IoU:")
    for i in range(len(class_names)):
        if i < len(iou_classes):
            c_name = class_names[i]
            val = iou_classes[i] * 100
            print(f"{c_name:15s}: {getColorEntry(val/100)}{val:.2f}%{colors.ENDC}")

    print("="*40)
    final_miou = iouVal * 100
    print(f"MEAN IoU: {getColorEntry(iouVal)}{final_miou:.2f}%{colors.ENDC}")
    print("="*40)

class colors:
    RED       = '\033[31;1m'
    GREEN     = '\033[32;1m'
    YELLOW    = '\033[33;1m'
    BLUE      = '\033[34;1m'
    CYAN      = '\033[36;1m'
    BOLD      = '\033[1m'
    ENDC      = '\033[0m'

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--datadir', required=True)
    parser.add_argument('--loadWeights', required=True)
    parser.add_argument('--img_size', type=int, default=1024)
    parser.add_argument('--backbone', default="vit_base_patch14_reg4_dinov2")
    parser.add_argument('--num_queries', type=int, default=100)
    parser.add_argument('--num_blocks', type=int, default=3)
    parser.add_argument('--subset', default="val")
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()
    main(args)