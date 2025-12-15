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
        num_classes=20,
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

        # 1. Architettura
        self.encoder = ViT(img_size=img_size, backbone_name=backbone_name, patch_size=14)
        self.base_model = EoMT(encoder=self.encoder, num_classes=num_classes, num_q=100, num_blocks=4)
        self._setup_lora()

        # 2. Loss & Metriche
        self.criterion_ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.train_iou = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=ignore_index, average='macro')
        self.val_iou = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=ignore_index, average='macro')

    def _setup_lora(self):
        peft_config = LoraConfig(
            r=8, lora_alpha=16, target_modules=["qkv"], lora_dropout=0.05, bias="none",
            modules_to_save=["class_head", "mask_head", "upscale", "conv1", "conv2"],
        )
        self.model = get_peft_model(self.base_model, peft_config)

    def forward(self, x):
        mask_logits_list, class_logits_list = self.model(x)
        mask_logits = mask_logits_list[-1]
        class_logits = class_logits_list[-1]

        target_hw = (x.shape[-2], x.shape[-1])
        seg_logits = self._build_segmentation_logits(mask_logits, class_logits, target_hw)
        
        # Clamp a 10.0 è sicuro per exp() anche in FP16 (e^10 ~= 22000 < 65000)
        seg_logits = seg_logits.clamp(-10.0, 10.0)
        return seg_logits

    def _build_segmentation_logits(self, mask_logits, class_logits, target_hw):
        B, Q, Hp, Wp = mask_logits.shape
        _, _, Cplus1 = class_logits.shape
        C = Cplus1 - 1 

        class_logits = class_logits[..., :C].permute(0, 2, 1).clamp(-10.0, 10.0)
        mask_logits = mask_logits.clamp(-10.0, 10.0)
        flat_masks = mask_logits.view(B, Q, -1)

        seg_logits_flat = torch.bmm(class_logits, flat_masks)
        seg_logits = seg_logits_flat.view(B, C, Hp, Wp)

        if (Hp, Wp) != target_hw:
            seg_logits = F.interpolate(seg_logits, size=target_hw, mode="bilinear", align_corners=False)
        return seg_logits

    def training_step(self, batch, batch_idx):
        img, mask = batch
        
        # Forward pass
        seg_logits = self(img)

        # --- SICUREZZA FP32 ---
        # Convertiamo i logits in float32 per evitare NaN nella Loss e Energy calculation
        seg_logits_fp32 = seg_logits.float()
        
        # Calcolo Loss
        loss_ce = self.criterion_ce(seg_logits_fp32, mask)
        loss_energy = self._compute_energy_loss(seg_logits_fp32, mask)

        # Somma pesata
        total_loss = loss_ce + 0.1 * loss_energy

        # Logging
        self.log("train_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_ce", loss_ce, on_step=False, on_epoch=True)
        self.log("train_en", loss_energy, on_step=False, on_epoch=True)
        
        # Metrica IoU
        preds = torch.argmax(seg_logits, dim=1)
        self.train_iou(preds, mask)
        self.log("train_mIoU", self.train_iou, on_step=False, on_epoch=True, prog_bar=True)

        return total_loss

    def _compute_energy_loss(self, seg_logits, mask):
        # Qui seg_logits è già float32 grazie al training_step
        T = 1.0
        energy_map = -torch.logsumexp(seg_logits / T, dim=1)

        mask_in = (mask != self.anomaly_class_idx) & (mask != self.ignore_index)
        mask_out = (mask == self.anomaly_class_idx)
        m_in, m_out = -7.0, -5.0

        # FIX: Inizializzazione sicura sul device corretto
        loss_energy = torch.tensor(0.0, device=self.device, dtype=seg_logits.dtype)
        
        if mask_in.sum() > 0:
            loss_energy += torch.mean(F.relu(energy_map[mask_in] - m_in) ** 2)
        if mask_out.sum() > 0:
            loss_energy += torch.mean(F.relu(m_out - energy_map[mask_out]) ** 2)
            
        return loss_energy

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        seg_logits = self(img)
        # Assicuriamoci che anche in validation sia float32
        val_loss = self.criterion_ce(seg_logits.float(), mask) 
        
        preds = torch.argmax(seg_logits, dim=1)
        self.val_iou(preds, mask)

        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True)
        self.log("val_mIoU", self.val_iou, prog_bar=True, on_epoch=True)
        return val_loss

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-6)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}