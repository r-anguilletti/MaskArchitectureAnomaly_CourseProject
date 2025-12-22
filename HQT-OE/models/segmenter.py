# models/segmenter.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics.classification import MulticlassJaccardIndex
from peft import LoraConfig, get_peft_model

from models.vit import ViT
from models.eomt import EoMT


class AnomalySegmenter(L.LightningModule):
    """
    Stable segmenter:
      - Cityscapes: CE on ID classes (0..C-1)
      - OE cut&paste: Energy-based separation
      - Balanced Energy Regularization (CVPR'23 style): focuses harder OOD pixels

    Convention:
      source == 0  -> Cityscapes (ID)
      source == 1  -> CNP/OE (OOD mask 0/1)
    """

    def __init__(
        self,
        img_size=(518, 518),
        num_id_classes=19,
        ignore_index=255,
        lr=5e-5,
        weight_decay=1e-2,
        backbone_name="vit_large_patch14_reg4_dinov2",
        # Energy setup
        T=1.0,
        m_in=-12.0,
        m_out=-6.0,
        lambda_energy=0.1,
        # Balanced energy (paper-inspired)
        use_balanced_energy=True,
        gamma=3.0,
        alpha=5.0,
        prior_momentum=0.99,
        # LoRA
        lora_r=8, #16
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules=("qkv",), #target_modules = ("qkv", "proj")
        modules_to_save=("class_head", "mask_head"),
        # Stability
        clamp_logits=10.0,
        grad_clip_hint=0.5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_id_classes = num_id_classes
        self.ignore_index = ignore_index

        self.lr = lr
        self.weight_decay = weight_decay

        self.T = float(T)
        self.m_in = float(m_in)
        self.m_out = float(m_out)
        self.lambda_energy = float(lambda_energy)

        self.use_balanced_energy = bool(use_balanced_energy)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.prior_momentum = float(prior_momentum)

        self.clamp_logits = float(clamp_logits)

        # ----------------------------
        # Architecture (ID-only)
        # ----------------------------
        self.encoder = ViT(img_size=img_size, backbone_name=backbone_name, patch_size=14)
        self.base_model = EoMT(encoder=self.encoder, num_classes=num_id_classes, num_q=100, num_blocks=4)

        self._setup_lora(
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=list(lora_target_modules),
            modules_to_save=list(modules_to_save),
        )

        # ----------------------------
        # Loss / Metrics
        # ----------------------------
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        self.train_miou = MulticlassJaccardIndex(
            num_classes=num_id_classes,
            ignore_index=self.ignore_index,
            average="macro"
        )
        self.val_miou = MulticlassJaccardIndex(
            num_classes=num_id_classes,
            ignore_index=self.ignore_index,
            average="macro"
        )

        # Running estimate of p(y|o)
        prior = torch.ones(num_id_classes, dtype=torch.float32) / float(num_id_classes)
        self.register_buffer("ood_prior", prior)

    # ----------------------------
    # Utility: normalize source
    # ----------------------------
    def _source_to_int(self, source):
        # source può arrivare come: "city", "cnp", 0, 1, tensor([0]), oppure ["city", ...]
        if isinstance(source, (list, tuple)):
            source = source[0]

        if torch.is_tensor(source):
            source = int(source.flatten()[0].item())

        if isinstance(source, str):
            return 0 if source == "city" else 1

        return int(source)

    # ----------------------------
    # LoRA
    # ----------------------------
    def _setup_lora(self, r, alpha, dropout, target_modules, modules_to_save):
        peft_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
            modules_to_save=modules_to_save,
        )
        self.model = get_peft_model(self.base_model, peft_config)

    # ----------------------------
    # Forward: ID logits
    # ----------------------------
    def forward(self, x):
        mask_logits_list, class_logits_list = self.model(x)
        mask_logits = mask_logits_list[-1]     # (B,Q,Hp,Wp)
        class_logits = class_logits_list[-1]   # (B,Q,C) or (B,Q,C+1)

        seg_logits = self._build_segmentation_logits(mask_logits, class_logits, target_hw=(x.shape[-2], x.shape[-1]))
        if self.clamp_logits > 0:
            seg_logits = seg_logits.clamp(-self.clamp_logits, self.clamp_logits)
        return seg_logits

    def _build_segmentation_logits(self, mask_logits, class_logits, target_hw):
        B, Q, Hp, Wp = mask_logits.shape
        C_eff = class_logits.shape[-1]

        # Drop no-object if present
        if C_eff == self.num_id_classes + 1:
            class_logits = class_logits[..., : self.num_id_classes]
        else:
            class_logits = class_logits[..., : self.num_id_classes]

        # (B,Q,C)->(B,C,Q)
        class_logits = class_logits.permute(0, 2, 1)

        seg_flat = torch.bmm(
            class_logits.clamp(-self.clamp_logits, self.clamp_logits),
            mask_logits.view(B, Q, -1).clamp(-self.clamp_logits, self.clamp_logits),
        )
        seg = seg_flat.view(B, self.num_id_classes, Hp, Wp)

        if (Hp, Wp) != target_hw:
            seg = F.interpolate(seg, size=target_hw, mode="bilinear", align_corners=False)
        return seg

    # ----------------------------
    # Energy
    # ----------------------------
    def energy_map(self, logits):
        return -self.T * torch.logsumexp(logits / self.T, dim=1)

    @torch.no_grad()
    def _update_ood_prior(self, logits, oe_mask01):
        out_mask = (oe_mask01 == 1)
        if not out_mask.any():
            return

        probs = torch.softmax(logits, dim=1)  # (B,C,H,W)
        p = probs.permute(0, 2, 3, 1)[out_mask].mean(dim=0)  # (C,)
        if torch.isnan(p).any():
            return

        m = self.prior_momentum
        self.ood_prior.mul_(m).add_((1.0 - m) * p)

        s = self.ood_prior.sum().clamp_min(1e-6)
        self.ood_prior.div_(s)

    def _balanced_weights(self, logits, oe_mask01):
        out_mask = (oe_mask01 == 1)
        B, C, H, W = logits.shape
        w = torch.zeros((B, H, W), device=logits.device, dtype=logits.dtype)
        if not out_mask.any():
            return w

        probs = torch.softmax(logits, dim=1)
        prior = self.ood_prior.view(1, C, 1, 1).to(probs.dtype)

        Z = (probs * prior).sum(dim=1).clamp_min(1e-8)
        Zg = Z ** self.gamma

        denom = Zg[out_mask].sum().clamp_min(1e-6)
        Zg_norm = Zg / denom * float(out_mask.sum())  # avg ~1 on OOD pixels
        w[out_mask] = Zg_norm[out_mask]
        return w

    def energy_loss(self, logits, oe_mask01):
        E = self.energy_map(logits)  # (B,H,W)
        in_mask = (oe_mask01 == 0)
        out_mask = (oe_mask01 == 1)

        loss_in = torch.tensor(0.0, device=self.device)
        loss_out = torch.tensor(0.0, device=self.device)

        if in_mask.any():
            loss_in = torch.mean(F.relu(E[in_mask] - self.m_in) ** 2)

        if out_mask.any():
            if self.use_balanced_energy:
                w_out = self._balanced_weights(logits, oe_mask01)
                m_out_adapt = self.m_out + self.alpha * w_out
                hinge = F.relu(m_out_adapt - E) ** 2
                loss_out = (hinge[out_mask] * w_out[out_mask]).mean()
            else:
                loss_out = torch.mean(F.relu(self.m_out - E[out_mask]) ** 2) #loss_out = torch.mean((F.relu(self.m_out - E[out_mask]) ** 2) * 5.0)

        loss = loss_in + loss_out

        with torch.no_grad():
            mean_in = E[in_mask].mean() if in_mask.any() else torch.tensor(float("nan"), device=self.device)
            mean_out = E[out_mask].mean() if out_mask.any() else torch.tensor(float("nan"), device=self.device)
            sep = mean_out - mean_in

        return loss, loss_in, loss_out, mean_in, mean_out, sep

    # ----------------------------
    # Lightning steps
    # ----------------------------
    def training_step(self, batch, batch_idx):
        img, mask, source = batch
        source = self._source_to_int(source)

        logits = self(img).float()

        if source == 0:  # City
            loss_ce = self.ce_loss(logits, mask)

            preds = torch.argmax(logits, dim=1)
            valid = (mask != self.ignore_index)
            if valid.any():
                self.train_miou(preds[valid], mask[valid])

            bs = img.shape[0]
            self.log("train/loss_ce", loss_ce, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
            self.log("train/mIoU", self.train_miou, prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
            return loss_ce

        else:  # CNP/OE (mask is 0/1)
            self._update_ood_prior(logits.detach(), mask)

            loss_e, loss_in, loss_out, mean_in, mean_out, sep = self.energy_loss(logits, mask)
            loss = self.lambda_energy * loss_e

            bs = img.shape[0]
            self.log("train/loss_energy", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
            self.log("train/energy_in", mean_in, prog_bar=False, on_step=True, on_epoch=True, batch_size=bs)
            self.log("train/energy_out", mean_out, prog_bar=False, on_step=True, on_epoch=True, batch_size=bs)
            self.log("train/energy_sep", sep, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
            self.log("train/hinge_in", loss_in, prog_bar=False, on_step=True, on_epoch=True, batch_size=bs)
            self.log("train/hinge_out", loss_out, prog_bar=False, on_step=True, on_epoch=True, batch_size=bs)

            with torch.no_grad():
                p = self.ood_prior.clamp_min(1e-8)
                ent = -(p * p.log()).sum()
            self.log("train/ood_prior_entropy", ent, prog_bar=False, on_step=True, on_epoch=True, batch_size=bs)

            return loss

    def validation_step(self, batch, batch_idx):
        img, mask, source = batch
        source = self._source_to_int(source)

        if source != 0:
            return None

        logits = self(img).float()
        loss = self.ce_loss(logits, mask)

        preds = torch.argmax(logits, dim=1)
        valid = (mask != self.ignore_index)
        if valid.any():
            self.val_miou(preds[valid], mask[valid])

        bs = img.shape[0]
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
        self.log("val/mIoU", self.val_miou, prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-6)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}

    @torch.no_grad()
    def anomaly_score(self, x):
        logits = self(x).float()
        E = self.energy_map(logits)
        return logits, E