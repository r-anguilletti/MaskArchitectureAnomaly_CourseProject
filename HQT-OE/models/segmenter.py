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
        num_classes=20,                 # include anomaly slot
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
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        # --------------------------------------------------
        # METRICHE (NO ANOMALY)
        # --------------------------------------------------
        self.train_iou = MulticlassJaccardIndex(
            num_classes=num_classes - 1,   # SOLO classi 0–18
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

        # sicurezza numerica
        return seg_logits.clamp(-10.0, 10.0)

    def _build_segmentation_logits(self, mask_logits, class_logits, target_hw):
        B, Q, Hp, Wp = mask_logits.shape
        C = class_logits.shape[-1] - 1   # esclude no-object

        class_logits = class_logits[..., :C].permute(0, 2, 1)
        mask_logits = mask_logits

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
    # TRAINING
    # ==================================================
    def training_step(self, batch, batch_idx):
        img, mask = batch
        seg_logits = self(img).float()

        # -------------------------
        # CE SOLO IN-DISTRIBUTION
        # -------------------------
        mask_ce = mask.clone()
        mask_ce[mask_ce == self.anomaly_class_idx] = self.ignore_index

        loss_ce = self.ce_loss(seg_logits, mask_ce)

        # -------------------------
        # ENERGY LOSS (OOD)
        # -------------------------
        loss_energy = self._energy_loss(seg_logits, mask)

        total_loss = loss_ce + 0.1 * loss_energy

        # -------------------------
        # METRICA (NO ANOMALY)
        # -------------------------
        preds = torch.argmax(seg_logits, dim=1)
        preds_id = preds.clone()
        preds_id[preds_id == self.anomaly_class_idx] = self.ignore_index

        mask_id = mask.clone()
        mask_id[mask_id == self.anomaly_class_idx] = self.ignore_index

        self.train_iou(preds_id, mask_id)

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
        energy = -torch.logsumexp(seg_logits / T, dim=1)

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
    # VALIDATION
    # ==================================================
    def validation_step(self, batch, batch_idx):
        img, mask = batch
        seg_logits = self(img).float()

        mask_ce = mask.clone()
        mask_ce[mask_ce == self.anomaly_class_idx] = self.ignore_index
        val_loss = self.ce_loss(seg_logits, mask_ce)

        preds = torch.argmax(seg_logits, dim=1)
        preds[preds == self.anomaly_class_idx] = self.ignore_index
        mask[mask == self.anomaly_class_idx] = self.ignore_index

        self.val_iou(preds, mask)

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
            T_max=100,
            eta_min=1e-6
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "interval": "epoch",
            },
        }