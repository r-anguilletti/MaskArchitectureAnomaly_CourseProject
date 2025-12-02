import os
import glob
import torch
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr, plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

input_transform = Compose([
    Resize((512, 1024), Image.BILINEAR),
    ToTensor(),
])

target_transform = Compose([
    Resize((512, 1024), Image.NEAREST),
])

def get_rba_masks(H, W, device):
    """Genera le maschere per l'ensemble RbA."""
    masks = []
    # 1. Originale (tutto visibile)
    masks.append(torch.ones((1, 1, H, W), device=device))
    
    # 2. Scacchiera
    check_size = 64
    mask_check = torch.ones((H, W), device=device)
    for i in range(0, H, check_size):
        for j in range(0, W, check_size):
            if ((i // check_size) + (j // check_size)) % 2 == 0:
                mask_check[i:i+check_size, j:j+check_size] = 0
    masks.append(mask_check.unsqueeze(0).unsqueeze(0))
    
    # 3. Scacchiera Inversa
    masks.append(1.0 - mask_check.unsqueeze(0).unsqueeze(0))
    
    # 4. Strisce Verticali
    mask_v = torch.ones((H, W), device=device)
    mask_v[:, ::128] = 0
    masks.append(mask_v.unsqueeze(0).unsqueeze(0))
    
    return masks

def main():
    parser = ArgumentParser()
    parser.add_argument("--input", default="/path/to/dataset/*.png", nargs="+")  
    parser.add_argument('--loadDir', default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    # Liste per salvare i risultati
    anomaly_score_list = []
    ood_gts_list = []

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print(f"Loading model: {modelpath}")
    print(f"Loading weights: {weightspath}")

    model = ERFNet(NUM_CLASSES)
    if not args.cpu:
        model = torch.nn.DataParallel(model).cuda()

    # Caricamento pesi custom
    def load_my_state_dict(model, state_dict):
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    model.eval()
    
    # --- CICLO DI INFERENZA ---
    print(f"Processing {len(glob.glob(os.path.expanduser(str(args.input[0]))))} images...")
    
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        
        # Carica e trasforma immagine
        img_pil = Image.open(path).convert('RGB')
        images = input_transform(img_pil).unsqueeze(0).float().cuda()
        
        B, C, H, W = images.shape
        masks = get_rba_masks(H, W, images.device)
        
        accumulated_probs = torch.zeros((B, NUM_CLASSES, H, W)).cuda()
        count_transforms = 0

        with torch.no_grad():
            for mask in masks:
                # 1. Immagine Mascherata
                masked_input = images * mask
                logits = model(masked_input)
                probs = torch.nn.functional.softmax(logits, dim=1)
                accumulated_probs += probs
                count_transforms += 1
                
                # 2. Immagine Mascherata + FLIP
                masked_input_flip = torch.flip(masked_input, dims=[3])
                logits_flip = model(masked_input_flip)
                logits_flip = torch.flip(logits_flip, dims=[3]) # Flip back
                probs_flip = torch.nn.functional.softmax(logits_flip, dim=1)
                accumulated_probs += probs_flip
                count_transforms += 1
        
        # --- RbA Logic: Calcolo Entropia sulla Media ---
        mean_probs = accumulated_probs / count_transforms
        
        # Calcolo Entropia
        log_prob_mean = torch.log(mean_probs + 1e-10)
        entropy_tensor = -torch.sum(mean_probs * log_prob_mean, dim=1)
        entropy_score = entropy_tensor.squeeze(0).data.cpu().numpy()

        # --- Gestione Ground Truth (Label) ---
        pathGT = path.replace("images", "labels_masks")                
        if "RoadObsticle21" in pathGT: pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT: pathGT = pathGT.replace("jpg", "png")                
        if "RoadAnomaly" in pathGT: pathGT = pathGT.replace("jpg", "png")  

        if not os.path.exists(pathGT):
            # Fallback per estensioni diverse se necessario
            pathGT = pathGT.replace(".png", ".jpg") 
        
        try:
            mask_gt = Image.open(pathGT)
            mask_gt = target_transform(mask_gt)
            ood_gts = np.array(mask_gt)

            # Mappatura etichette per i vari dataset
            if "RoadAnomaly" in pathGT:
                ood_gts = np.where((ood_gts==2), 1, ood_gts)
            if "LostAndFound" in pathGT:
                ood_gts = np.where((ood_gts==0), 255, ood_gts)
                ood_gts = np.where((ood_gts==1), 0, ood_gts)
                ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)
            if "Streethazard" in pathGT:
                ood_gts = np.where((ood_gts==14), 255, ood_gts)
                ood_gts = np.where((ood_gts<20), 0, ood_gts)
                ood_gts = np.where((ood_gts==255), 1, ood_gts)

            if 1 not in np.unique(ood_gts):
                continue
            
            ood_gts_list.append(ood_gts)
            anomaly_score_list.append(entropy_score) # RbA usa score basato su Entropia

        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue

        # Pulizia memoria
        del accumulated_probs, mean_probs, entropy_tensor, images
        torch.cuda.empty_cache()

    # --- Calcolo Metriche Finali ---
    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((np.zeros(len(ind_out)), np.ones(len(ood_out))))

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    print("\n" + "="*30)
    print(f"RESULTS FOR RbA")
    print(f"AuPRC: {prc_auc*100.0:.2f}")
    print(f"FPR@95: {fpr*100.0:.2f}")
    print("="*30 + "\n")

if __name__ == '__main__':
    main()