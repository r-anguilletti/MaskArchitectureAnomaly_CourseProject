
Contenuti in evidenza della cartella
Codice Python per AnomalySegmenter che implementa ViT ed EoMT con logica LoRA per la segmentazione.

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from peft import LoraConfig, get_peft_model
from torchmetrics.classification import MulticlassJaccardIndex

from models.vit import ViT
from models.eomt import EoMT


class AnomalySegmenter(L.LightningModule):
    def __init__(
        self,
        img_size=(518, 518),
        num_classes=20,                 # include anomaly slot (0-18 + 19)
        anomaly_class_idx=19,
        ignore_index=255,
        lr=5e-5,
        weight_decay=1e-2,
        backbone_name="vit_large_patch14_reg4_dinov2",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.anomaly_class_idx = anomaly_class_idx
        self.ignore_index = ignore_index

        # --------------------------------------------------
        # ARCHITETTURA
        # --------------------------------------------------
        self.encoder = ViT(
            img_size=img_size,
            backbone_name=backbone_name,
            patch_size=14
        )
        self.base_model = EoMT(
            encoder=self.encoder,
            num_classes=num_classes,
            num_q=100,
            num_blocks=4
        )
        self._setup_lora()

        # --------------------------------------------------
        # LOSS
        # --------------------------------------------------
        # ignore_index gestito manualmente nel training_step per evitare NaN
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        # --------------------------------------------------
        # METRICHE (NO ANOMALY)
        # --------------------------------------------------
        # Calcoliamo IoU solo sulle classi 'normali' (0-18)
        self.train_iou = MulticlassJaccardIndex(
            num_classes=num_classes - 1,   
            ignore_index=self.ignore_index,
            average="macro"
        )
        self.val_iou = MulticlassJaccardIndex(
            num_classes=num_classes - 1,
            ignore_index=self.ignore_index,
            average="macro"
        )

    # ==================================================
    # LoRA
    # ==================================================
    def _setup_lora(self):
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["qkv"],
            bias="none",
            modules_to_save=[
                "class_head", "mask_head",
                "upscale", "conv1", "conv2"
            ],
        )
        self.model = get_peft_model(self.base_model, peft_config)

    # ==================================================
    # FORWARD
    # ==================================================
    def forward(self, x):
        mask_logits_list, class_logits_list = self.model(x)

        mask_logits = mask_logits_list[-1]
        class_logits = class_logits_list[-1]

        seg_logits = self._build_segmentation_logits(
            mask_logits,
            class_logits,
            target_hw=(x.shape[-2], x.shape[-1])
        )

        # Sicurezza numerica: clampiamo i logit finali per evitare esplosioni
        return seg_logits.clamp(-10.0, 10.0)

    def _build_segmentation_logits(self, mask_logits, class_logits, target_hw):
        B, Q, Hp, Wp = mask_logits.shape
        C = class_logits.shape[-1] - 1   # esclude no-object

        class_logits = class_logits[..., :C].permute(0, 2, 1)
        mask_logits = mask_logits

        # Clamp intermedio cruciale per stabilità FP16
        seg_flat = torch.bmm(
            class_logits.clamp(-10, 10),
            mask_logits.view(B, Q, -1).clamp(-10, 10)
        )
        seg = seg_flat.view(B, C, Hp, Wp)

        if (Hp, Wp) != target_hw:
            seg = F.interpolate(
                seg,
                size=target_hw,
                mode="bilinear",
                align_corners=False
            )
        return seg

    # ==================================================
    # TRAINING STEP (PATCHATO)
    # ==================================================
    def training_step(self, batch, batch_idx):
        img, mask = batch
        seg_logits = self(img).float()

        # -------------------------
        # 1. CE LOSS (ANTI-NAN FIX)
        # -------------------------
        mask_ce = mask.clone()
        # Nascondiamo l'anomalia alla CE (deve imparare solo le classi normali)
        mask_ce[mask_ce == self.anomaly_class_idx] = self.ignore_index
        
        # Conta quanti pixel validi ci sono (che non siano 255)
        valid_pixels = (mask_ce != self.ignore_index).sum()
        
        if valid_pixels > 0:
            loss_ce = self.ce_loss(seg_logits, mask_ce)
        else:
            # Se il batch contiene SOLO anomalie e ignore, la CE loss esplode (div/0).
            # Restituiamo 0.0 con requires_grad=True per non rompere il grafo.
            loss_ce = torch.tensor(0.0, device=self.device, requires_grad=True)

        # -------------------------
        # 2. ENERGY LOSS (OOD)
        # -------------------------
        loss_energy = self._energy_loss(seg_logits, mask)

        total_loss = loss_ce + 0.1 * loss_energy

        # -------------------------
        # 3. METRICA SICURA (ANTI-CRASH FIX)
        # -------------------------
        preds = torch.argmax(seg_logits, dim=1)
        preds_id = preds.clone()
        preds_id[preds_id == self.anomaly_class_idx] = self.ignore_index

        mask_id = mask.clone()
        mask_id[mask_id == self.anomaly_class_idx] = self.ignore_index

        # FILTRO MANUALE: Passiamo alla metrica solo i pixel validi (non 255)
        # Altrimenti torchmetrics cerca di creare una matrice enorme e crasha.
        valid_mask = (mask_id != self.ignore_index)
        if valid_mask.any():
            self.train_iou(preds_id[valid_mask], mask_id[valid_mask])

        self.log_dict(
            {
                "train_loss": total_loss,
                "train_ce": loss_ce,
                "train_energy": loss_energy,
                "train_mIoU": self.train_iou,
            },
            prog_bar=True,
            on_epoch=True,
        )

        return total_loss

    # ==================================================
    # ENERGY LOSS
    # ==================================================
    def _energy_loss(self, seg_logits, mask):
        T = 1.0
        # LogSumExp sulle classi in-distribution
        # Nota: seg_logits ha 20 canali, usiamo tutti per il calcolo dell'energia
        # ma l'obiettivo è spingere giù l'energia delle anomalie.
        
        # Prendiamo solo i logit delle classi ID (0-18)
        id_logits = seg_logits[:, :self.anomaly_class_idx, :, :]
        energy = -torch.logsumexp(id_logits / T, dim=1)

        in_mask = (mask != self.anomaly_class_idx) & (mask != self.ignore_index)
        out_mask = (mask == self.anomaly_class_idx)

        m_in, m_out = -7.0, -5.0
        loss = torch.tensor(0.0, device=self.device)

        if in_mask.any():
            loss += torch.mean(F.relu(energy[in_mask] - m_in) ** 2)
        if out_mask.any():
            loss += torch.mean(F.relu(m_out - energy[out_mask]) ** 2)

        return loss

    # ==================================================
    # VALIDATION STEP (PATCHATO)
    # ==================================================
    def validation_step(self, batch, batch_idx):
        img, mask = batch
        seg_logits = self(img).float()

        # Fix Anti-Nan anche in validazione
        mask_ce = mask.clone()
        mask_ce[mask_ce == self.anomaly_class_idx] = self.ignore_index
        
        valid_pixels = (mask_ce != self.ignore_index).sum()
        if valid_pixels > 0:
            val_loss = self.ce_loss(seg_logits, mask_ce)
        else:
            val_loss = torch.tensor(0.0, device=self.device)

        # Fix Metrica
        preds = torch.argmax(seg_logits, dim=1)
        preds[preds == self.anomaly_class_idx] = self.ignore_index
        mask[mask == self.anomaly_class_idx] = self.ignore_index

        valid_mask = (mask != self.ignore_index)
        if valid_mask.any():
            self.val_iou(preds[valid_mask], mask[valid_mask])

        self.log_dict(
            {
                "val_loss": val_loss,
                "val_mIoU": self.val_iou,
            },
            prog_bar=True,
            on_epoch=True,
        )

        return val_loss

    # ==================================================
    # OPTIMIZER
    # ==================================================
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=100, # Adatta alle tue epoche totali stimate
            eta_min=1e-6
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "interval": "epoch",
            },
        }