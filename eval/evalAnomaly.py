# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20
# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

input_transform = Compose(
    [
        Resize((512, 1024), Image.BILINEAR),
        ToTensor(),
        # Normalize([.485, .456, .406], [.229, .224, .225]),
    ]
)

target_transform = Compose(
    [
        Resize((512, 1024), Image.NEAREST),
    ]
)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    anomaly_score_list = []
    ood_gts_list = []
    max_logit_list = []
    entropy_list = []

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    model = ERFNet(NUM_CLASSES)

    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print ("Model and weights LOADED successfully")
    model.eval()
    
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        print(path)
        images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().cuda()
        #images = images.permute(0,3,1,2) #Action already done by ToTensor()
        with torch.no_grad():
            result = model(images)
        #MSP
        anomaly_result = 1.0 - np.max(result.squeeze(0).data.cpu().numpy(), axis=0)
        #MAX LOGIT low logit means anomaly ->  we use the -max logit as anomaly score 
        max_logit_tensor = torch.max(result, dim=1)[0]
        max_logit_score = -max_logit_tensor.squeeze(0).data.cpu().numpy()
        #ENTROPY -> - sum(p*log(p))
        # Compute log-softmax probabilities
        softmax = torch.nn.Softmax(dim=1)  
        prob = softmax(result)
        log_prob = torch.log(prob + 1e-10)
        # Compute entropy (adding a small constant to avoid log(0))
        entropy_tensor = -torch.sum(prob * log_prob, dim=1)
        entropy_score = entropy_tensor.squeeze(0).data.cpu().numpy()        
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
        else:
             ood_gts_list.append(ood_gts)
             anomaly_score_list.append(anomaly_result)
             #Max Logit
             max_logit_list.append(max_logit_score)
             #Entropy
             entropy_list.append(entropy_score)
        del result, anomaly_result, ood_gts, mask, max_logit_tensor, entropy_tensor, prob, log_prob #free GPU memory
        torch.cuda.empty_cache()

    file.write( "\n")

    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))
    
    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f'AUPRC score MSP: {prc_auc*100.0}')
    print(f'FPR@TPR95 MSP: {fpr*100.0}')

    # Max Logit
    max_logit_scores = np.array(max_logit_list)
    ood_out_maxlogit = max_logit_scores[ood_mask]
    ind_out_maxlogit = max_logit_scores[ind_mask]
    val_out_maxlogit = np.concatenate((ind_out_maxlogit, ood_out_maxlogit))

    prc_auc_maxlogit = average_precision_score(val_label, val_out_maxlogit)
    fpr_maxlogit = fpr_at_95_tpr(val_out_maxlogit, val_label)

    print(f'AUPRC score MaxLogit: {prc_auc_maxlogit*100.0}')
    print(f'FPR@TPR95 MaxLogit: {fpr_maxlogit*100.0}')

    # Entropy
    entropy_scores = np.array(entropy_list)
    ood_out_entropy = entropy_scores[ood_mask]
    ind_out_entropy = entropy_scores[ind_mask]
    val_out_entropy = np.concatenate((ind_out_entropy, ood_out_entropy))

    prc_auc_entropy = average_precision_score(val_label, val_out_entropy)
    fpr_entropy = fpr_at_95_tpr(val_out_entropy, val_label)

    print(f'AUPRC score Entropy: {prc_auc_entropy*100.0}')
    print(f'FPR@TPR95 Entropy: {fpr_entropy*100.0}')

    file.write(('    AUPRC score MSP:' + str(prc_auc*100.0) + '   FPR@TPR95 MSP:' + str(fpr*100.0) + '\n'
                '    AUPRC score MaxLogit:' + str(prc_auc_maxlogit*100.0) + '   FPR@TPR95 MaxLogit:' + str(fpr_maxlogit*100.0) + '\n'
                '    AUPRC score Entropy:' + str(prc_auc_entropy*100.0) + '   FPR@TPR95 Entropy:' + str(fpr_entropy*100.0) + '\n'))
    file.close()

if __name__ == '__main__':
    main()
