# models/segmenter.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics.classification import MulticlassJaccardIndex, BinaryAveragePrecision
from peft import LoraConfig, get_peft_model

from models.vit import ViT
from models.eomt import EoMT


class AnomalySegmenter(L.LightningModule):
    def __init__(
        self,
        img_size=(1024, 1024),
        num_id_classes=19,
        ignore_index=255,
        lr=1e-5,
        weight_decay=1e-3,
        train_epochs=10,
        backbone_name="vit_base_patch14_reg4_dinov2",
        patch_size=16,
        num_queries=100,
        num_blocks=3,
        masked_attn_enabled=True,
        pretrained_eomt_bin=None,

        # Energy (BASE VALUES kept for backward-compat)
        T=1.0,
        m_in=-12.0,
        m_out=-2.0,          # (base value; during training we will schedule toward -6)
        lambda_energy=0.15,     # (base value; during training we will schedule toward 0.35)
        warmup_epochs=2,

        # LoRA
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules=("qkv",),
        modules_to_save=("class_head", "mask_head"),
        clamp_logits=50.0,

        # OOD validation
        ood_val_sample_pixels=20000,

    ):
        super().__init__()
        self.save_hyperparameters()

        # basics
        self.num_id_classes = int(num_id_classes)
        self.ignore_index = int(ignore_index)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.train_epochs = int(train_epochs)
        self.warmup_epochs = int(warmup_epochs)

        # energy params (store base)
        self.T = float(T)
        self.m_in = float(m_in)
        self.m_out = float(m_out)                  # base
        self.lambda_energy = float(lambda_energy)  # base
        self.clamp_logits = float(clamp_logits)

        self.ood_val_sample_pixels = int(ood_val_sample_pixels)

        self.energy_lambda_final = 0.35
        self.energy_lambda_start_epoch = 20
        self.energy_lambda_ramp_epochs = 5  # 15..20

        # m_out schedule: start softer, end at -6
        self.m_out_start = -2.0
        self.m_out_final = -6.0
        self.m_out_start_epoch = 20
        self.m_out_ramp_epochs = 5

        # model
        self.encoder = ViT(
            img_size=img_size,
            backbone_name=backbone_name,
            patch_size=int(patch_size),
            ckpt_path="disable_timm_pretrained",
        )

        self.base_model = EoMT(
            encoder=self.encoder,
            num_classes=self.num_id_classes,
            num_q=int(num_queries),
            num_blocks=int(num_blocks),
            masked_attn_enabled=bool(masked_attn_enabled),
        )

        if pretrained_eomt_bin is not None:
            self._load_eomt_bin(pretrained_eomt_bin)

        if use_lora:
            self._setup_lora(
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
                target_modules=list(lora_target_modules),
                modules_to_save=list(modules_to_save),
            )
        else:
            self.model = self.base_model

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        self.train_miou = MulticlassJaccardIndex(
            num_classes=self.num_id_classes,
            ignore_index=self.ignore_index,
            average="macro",
        )
        self.val_miou = MulticlassJaccardIndex(
            num_classes=self.num_id_classes,
            ignore_index=self.ignore_index,
            average="macro",
        )

        self.val_ood_auprc_msp = BinaryAveragePrecision()
        self._ood_metric_updated = False

    # ----------------------------
    # Loading / LoRA
    # ----------------------------
    def _load_eomt_bin(self, bin_path: str):
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"[LOAD_EOMT_BIN] not found: {bin_path}")

        sd = torch.load(bin_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        if not isinstance(sd, dict):
            raise ValueError("[LOAD_EOMT_BIN] Expected a state_dict-like dict.")

        sd = {k: v for k, v in sd.items() if k.startswith("network.")}
        sd = {k.replace("network.", "", 1): v for k, v in sd.items()}

        missing, unexpected = self.base_model.load_state_dict(sd, strict=False)
        print(f"[LOAD_EOMT_BIN] loaded from: {bin_path}")
        print(f"[LOAD_EOMT_BIN] missing={len(missing)} unexpected={len(unexpected)}")

    def _setup_lora(self, r, alpha, dropout, target_modules, modules_to_save):
        peft_config = LoraConfig(
            r=int(r),
            lora_alpha=int(alpha),
            lora_dropout=float(dropout),
            target_modules=target_modules,
            bias="none",
            modules_to_save=modules_to_save,
        )
        self.model = get_peft_model(self.base_model, peft_config)

    def _source_to_int(self, source):
        if isinstance(source, (list, tuple)):
            source = source[0]
        if torch.is_tensor(source):
            return int(source.flatten()[0].item())
        if isinstance(source, str):
            return 0 if source == "city" else 1
        return int(source)

    def _lambda_energy_now(self) -> float:
          """
          Requested schedule:
          - epochs < 20: 0.15
          - 20..25: linear ramp to 0.35
          - >=25: 0.35
          """
          e = int(self.current_epoch)
          start = int(self.energy_lambda_start_epoch)
          ramp = max(1, int(self.energy_lambda_ramp_epochs))
          final = float(self.energy_lambda_final)

          if e < start:
              return 0.15
          if e >= start + ramp:
              return final

          t = (e - start) / float(ramp)
          t = max(0.0, min(1.0, t))
          return final * t

    # ----------------------------
    # m_out schedule
    # ----------------------------
    def _m_out_now(self) -> float:
        """
        Linear schedule for m_out:
        - epochs < 15: m_out_start (-2.0)
        - 15..20: ramp to m_out_final (-6.0)
        - >=20: m_out_final
        """
        e = int(self.current_epoch)
        start = int(self.m_out_start_epoch)
        ramp = max(1, int(self.m_out_ramp_epochs))
        m0 = float(self.m_out_start)
        mf = float(self.m_out_final)

        if e < start:
            return m0
        if e >= start + ramp:
            return mf

        t = (e - start) / float(ramp)
        t = max(0.0, min(1.0, t))
        return m0 + t * (mf - m0)

    # ----------------------------
    # Forward -> per-pixel logits (B,C,H,W)
    # ----------------------------
    def forward(self, x):
        mask_logits_layers, class_logits_layers = self.model(x)
        mask_logits = mask_logits_layers[-1]     # (B,Q,h,w)
        class_logits = class_logits_layers[-1]   # (B,Q,C+1) or (B,Q,C)

        if mask_logits.shape[-2:] != x.shape[-2:]:
            mask_logits = F.interpolate(mask_logits, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if class_logits.shape[-1] == self.num_id_classes + 1:
            class_logits = class_logits[..., : self.num_id_classes]
        else:
            class_logits = class_logits[..., : self.num_id_classes]

        mask_probs = mask_logits.sigmoid()                 # (B,Q,H,W)
        class_probs = torch.softmax(class_logits, dim=-1)  # (B,Q,C)

        seg_probs = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)  # (B,C,H,W)

        eps = 1e-6
        denom = seg_probs.sum(dim=1, keepdim=True).clamp_min(eps)
        seg_probs = (seg_probs / denom).clamp_min(eps)

        seg_logits = torch.log(seg_probs)
        seg_logits = seg_logits - seg_logits.mean(dim=1, keepdim=True)

        if self.clamp_logits > 0:
            seg_logits = seg_logits.clamp(-self.clamp_logits, self.clamp_logits)

        return seg_logits

    # ----------------------------
    # seg_probs (for MSP)
    # ----------------------------
    @torch.no_grad()
    def seg_probs(self, x: torch.Tensor) -> torch.Tensor:
        mask_logits_layers, class_logits_layers = self.model(x)
        mask_logits = mask_logits_layers[-1]
        class_logits = class_logits_layers[-1]

        if mask_logits.shape[-2:] != x.shape[-2:]:
            mask_logits = F.interpolate(mask_logits, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if class_logits.shape[-1] == self.num_id_classes + 1:
            class_logits = class_logits[..., : self.num_id_classes]
        else:
            class_logits = class_logits[..., : self.num_id_classes]

        mask_probs = mask_logits.sigmoid()
        class_probs = torch.softmax(class_logits, dim=-1)
        seg_probs = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)

        eps = 1e-6
        denom = seg_probs.sum(dim=1, keepdim=True).clamp_min(eps)
        seg_probs = (seg_probs / denom).clamp_min(eps)
        return seg_probs

    # ----------------------------
    # OOD scores
    # ----------------------------
    def energy_map(self, logits):
        return -self.T * torch.logsumexp(logits / self.T, dim=1)  # (B,H,W)

    def _sample_pixels(self, scores, targets01, max_pixels):
        s = scores.reshape(-1)
        t = targets01.reshape(-1).to(torch.int64)
        n = s.numel()
        if n <= max_pixels:
            return s, t
        idx = torch.randperm(n, device=s.device)[:max_pixels]
        return s[idx], t[idx]

    # ----------------------------
    # Energy loss (train)
    # ----------------------------
    def energy_loss(self, logits, oe_mask01):
        """
        Uses scheduled m_out (via _m_out_now()).
        (Energy is disabled anyway because _lambda_energy_now() returns 0.)
        """
        E = self.energy_map(logits)
        in_mask = (oe_mask01 == 0)
        out_mask = (oe_mask01 == 1)

        loss_in = torch.tensor(0.0, device=logits.device)
        loss_out = torch.tensor(0.0, device=logits.device)

        m_out_used = float(self._m_out_now())

        if in_mask.any():
            loss_in = torch.mean(F.relu(E[in_mask] - self.m_in) ** 2)
        if out_mask.any():
            loss_out = torch.mean(F.relu(m_out_used - E[out_mask]) ** 2)

        loss = loss_in + loss_out

        with torch.no_grad():
            mean_in = E[in_mask].mean() if in_mask.any() else torch.tensor(float("nan"), device=logits.device)
            mean_out = E[out_mask].mean() if out_mask.any() else torch.tensor(float("nan"), device=logits.device)
            sep = mean_out - mean_in

        return loss, loss_in, loss_out, mean_in, mean_out, sep

    # ----------------------------
    # Lightning steps
    # ----------------------------
    def training_step(self, batch, batch_idx):
        img, mask, source = batch
        source = self._source_to_int(source)

        logits = self(img).float()
        bs = img.shape[0]

        is_warm = float(
            (source == 1) and getattr(self, "warmup_epochs", 0) and (self.current_epoch < self.warmup_epochs)
        )
        self.log("train/is_warmup", is_warm, prog_bar=False, on_step=False, on_epoch=True, batch_size=bs)

        # City (ID)
        if source == 0:
            loss_ce = self.ce_loss(logits, mask)

            loss = loss_ce

            preds = torch.argmax(logits, dim=1)
            valid = (mask != self.ignore_index)
            if valid.any():
                self.train_miou(preds[valid], mask[valid])

            self.log("train/loss_ce", loss_ce, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
            self.log("train/mIoU", self.train_miou, prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
            return loss

        # CNP / OE (OOD) + WARM-UP
        if getattr(self, "warmup_epochs", 0) and self.current_epoch < self.warmup_epochs:
            zero = logits.sum() * 0.0
            self.log("train/loss_energy", zero, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
            self.log("train/energy_sep", zero, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
            return zero

        oe_mask01 = (mask > 0).to(torch.int64)
        loss_e, *_ , sep = self.energy_loss(logits, oe_mask01)

        lam_e = self._lambda_energy_now()
        loss = lam_e * loss_e

        self.log("train/loss_energy", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
        self.log("train/lambda_energy", lam_e, prog_bar=False, on_step=False, on_epoch=True, batch_size=bs)
        self.log("train/m_out_used", float(self._m_out_now()), prog_bar=False, on_step=False, on_epoch=True, batch_size=bs)
        self.log("train/energy_sep", sep, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        img, mask, source = batch

        # Cityscapes val
        if dataloader_idx == 0:
            logits = self(img).float()
            loss = self.ce_loss(logits, mask)

            preds = torch.argmax(logits, dim=1)
            valid = (mask != self.ignore_index)
            if valid.any():
                self.val_miou(preds[valid], mask[valid])

            bs = img.shape[0]
            self.log("val_city/loss", loss, prog_bar=True, on_step=False, on_epoch=True,
                     batch_size=bs, add_dataloader_idx=False)
            self.log("val_city/mIoU", self.val_miou, prog_bar=True, on_step=False, on_epoch=True,
                     batch_size=bs, add_dataloader_idx=False)
            return loss

        # CNP val (OOD)
        with torch.no_grad():
            seg_probs = self.seg_probs(img)  # (B,C,H,W)

            msp = seg_probs.max(dim=1).values  # (B,H,W)
            msp_score = 1.0 - msp              # (B,H,W)

            logits = self(img).float()
            E = self.energy_map(logits)

            targets01 = (mask > 0).to(torch.int64)

            in_mask = (targets01 == 0)
            out_mask = (targets01 == 1)

            bs = img.shape[0]

            if in_mask.any() and out_mask.any():
                msp_in = msp_score[in_mask].mean()
                msp_out = msp_score[out_mask].mean()
                msp_sep = msp_out - msp_in

                e_in = E[in_mask].mean()
                e_out = E[out_mask].mean()
                e_sep = e_out - e_in

                self.log("val_ood/msp_sep", msp_sep, prog_bar=True, on_step=False, on_epoch=True,
                         batch_size=bs, add_dataloader_idx=False)
                self.log("val_ood/energy_sep", e_sep, prog_bar=True, on_step=False, on_epoch=True,
                         batch_size=bs, add_dataloader_idx=False)

                s, t = self._sample_pixels(msp_score, targets01, self.ood_val_sample_pixels)
                self.val_ood_auprc_msp.update(s, t)
                self._ood_metric_updated = True

    def on_validation_epoch_end(self):
        if self._ood_metric_updated:
            auprc = self.val_ood_auprc_msp.compute()
            self.log("val_ood/auprc_msp", auprc, prog_bar=True, on_step=False, on_epoch=True,
                     add_dataloader_idx=False)

        self.val_ood_auprc_msp.reset()
        self._ood_metric_updated = False

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.train_epochs, eta_min=1e-6)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}

    @torch.no_grad()
    def anomaly_score(self, x):
        logits = self(x).float()
        E = self.energy_map(logits)
        return logits, E