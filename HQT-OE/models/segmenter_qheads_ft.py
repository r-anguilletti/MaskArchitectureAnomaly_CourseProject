# models/segmenter_qheads_ft.py
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
    """
    Variant: Query + Heads fine-tuning only (paper-friendly)
    - Keep segmenter.py as baseline
    - Train only: base_model.q, class_head, mask_head
    - Use PEFT to manage trainable modules (modules_to_save) instead of manual freezing.
    """

    def __init__(
        self,
        img_size=(1024, 1024),
        num_id_classes=19,
        ignore_index=255,
        lr=1e-6,
        weight_decay=1e-3,
        train_epochs=10,

        backbone_name="vit_base_patch14_reg4_dinov2",
        patch_size=16,

        num_queries=100,
        num_blocks=3,
        masked_attn_enabled=True,

        pretrained_eomt_bin=None,   # <- ora accetta .ckpt o .bin

        # Energy
        T=1.0,
        m_in=-12.0,
        m_out=-6.0,
        lambda_energy=0.1,

        warmup_epochs=0,

        # PEFT/LoRA wrapper (used mainly for modules_to_save)
        lora_r=4,
        lora_alpha=16,
        lora_dropout=0.0,

        clamp_logits=50.0,

        # OOD validation
        ood_val_sample_pixels=20000,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_id_classes = int(num_id_classes)
        self.ignore_index = int(ignore_index)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.train_epochs = int(train_epochs)
        self.warmup_epochs = int(warmup_epochs)

        self.T = float(T)
        self.m_in = float(m_in)
        self.m_out = float(m_out)
        self.lambda_energy = float(lambda_energy)

        self.clamp_logits = float(clamp_logits)
        self.ood_val_sample_pixels = int(ood_val_sample_pixels)

        # ----------------------------
        # Build base model (same as baseline)
        # ----------------------------
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
            self._load_eomt_ckpt_or_bin(pretrained_eomt_bin)

        # ----------------------------
        # PEFT wrapper:
        # keep trainable only: q, class_head, mask_head
        # ----------------------------
        self.model = self._wrap_with_peft_qheads_only(
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

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

        # Print trainables once (helps debugging / paper)
        self._printed_trainables = False

    # ----------------------------
    # Loading (.ckpt Lightning OR .bin weights)
    # ----------------------------
    def _load_eomt_ckpt_or_bin(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"[LOAD_EOMT] not found: {path}")

        ckpt = torch.load(path, map_location="cpu")

        # Extract state_dict
        if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            sd = ckpt["state_dict"]
        elif isinstance(ckpt, dict):
            sd = ckpt
        else:
            raise ValueError("[LOAD_EOMT] Expected a dict-like checkpoint")

        cleaned = {}
        for k, v in sd.items():
            # Strip common wrappers
            if k.startswith("model."):
                k = k[len("model."):]
            if k.startswith("network."):
                k = k[len("network."):]
            if k.startswith("base_model."):
                k = k[len("base_model."):]
            cleaned[k] = v

        missing, unexpected = self.base_model.load_state_dict(cleaned, strict=False)
        print(f"[LOAD_EOMT] loaded from: {path}")
        print(f"[LOAD_EOMT] missing={len(missing)} unexpected={len(unexpected)}")

    # ----------------------------
    # PEFT wrapper (robust)
    # ----------------------------
    def _wrap_with_peft_qheads_only(self, lora_r, lora_alpha, lora_dropout):
        """
        We want ONLY q + heads trainable.
        PEFT sometimes refuses target_modules=[] depending on version.
        So:
          - try clean config with target_modules=[]
          - if it errors, fallback to target_modules=["qkv"] but freeze LoRA params immediately.
        """
        modules_to_save = ["q", "class_head", "mask_head"]

        # Try clean: no LoRA injection, only modules_to_save.
        try:
            peft_config = LoraConfig(
                r=int(lora_r),
                lora_alpha=int(lora_alpha),
                lora_dropout=float(lora_dropout),
                target_modules=[],   # no LoRA injection
                bias="none",
                modules_to_save=modules_to_save,
            )
            model = get_peft_model(self.base_model, peft_config)
            # If some versions still create adapters unexpectedly, freeze them
            self._freeze_lora_adapters_only(model)
            return model

        except Exception as e:
            print(f"[PEFT] target_modules=[] failed ({type(e).__name__}: {e})")
            print("[PEFT] Fallback: inject on ['qkv'] but freeze LoRA params (still trains only q + heads).")

            peft_config = LoraConfig(
                r=int(lora_r),
                lora_alpha=int(lora_alpha),
                lora_dropout=float(lora_dropout),
                target_modules=["qkv"],  # fallback that PEFT likes
                bias="none",
                modules_to_save=modules_to_save,
            )
            model = get_peft_model(self.base_model, peft_config)
            self._freeze_lora_adapters_only(model)
            return model

    # ----------------------------
    # Freeze only LoRA adapter weights
    # ----------------------------
    def _freeze_lora_adapters_only(self, peft_model):
        for n, p in peft_model.named_parameters():
            if "lora_" in n:
                p.requires_grad = False

    # ----------------------------
    # (Optional) print trainable parameters once
    # ----------------------------
    def on_fit_start(self):
        if not self._printed_trainables:
            self._printed_trainables = True
            self._print_trainables()

    def _print_trainables(self):
        total = 0
        trainable = 0
        print("\n========== TRAINABLE PARAMETERS (q+heads only) ==========")
        for n, p in self.named_parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
                print(f"[TRAIN] {n}")
        print("--------------------------------------------------------")
        print(f"Trainable params: {trainable/1e6:.3f} M")
        print(f"Total params:     {total/1e6:.3f} M")
        print(f"Ratio:            {100.0*trainable/max(1,total):.4f}%")
        print("========================================================\n")

    def _source_to_int(self, source):
        if isinstance(source, (list, tuple)):
            source = source[0]
        if torch.is_tensor(source):
            return int(source.flatten()[0].item())
        if isinstance(source, str):
            return 0 if source == "city" else 1
        return int(source)

    # ----------------------------
    # Query->pixel: produce per-pixel logits (B,C,H,W)
    # ----------------------------
    def forward(self, x):
        mask_logits_layers, class_logits_layers = self.model(x)
        mask_logits = mask_logits_layers[-1]     # (B,Q,h,w)
        class_logits = class_logits_layers[-1]   # (B,Q,C+1) or (B,Q,C)

        # upsample masks to image resolution
        if mask_logits.shape[-2:] != x.shape[-2:]:
            mask_logits = F.interpolate(
                mask_logits, size=x.shape[-2:], mode="bilinear", align_corners=False
            )

        # drop "no-object" if present
        if class_logits.shape[-1] == self.num_id_classes + 1:
            class_logits = class_logits[..., : self.num_id_classes]
        else:
            class_logits = class_logits[..., : self.num_id_classes]

        mask_probs = mask_logits.sigmoid()                 # (B,Q,H,W)
        class_probs = torch.softmax(class_logits, dim=-1)  # (B,Q,C)

        seg_probs = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)  # (B,C,H,W)

        # normalize to valid distribution
        eps = 1e-6
        denom = seg_probs.sum(dim=1, keepdim=True).clamp_min(eps)
        seg_probs = (seg_probs / denom).clamp_min(eps)

        # probs -> logits for CE (center for stability)
        seg_logits = torch.log(seg_probs)
        seg_logits = seg_logits - seg_logits.mean(dim=1, keepdim=True)

        # clamp [-50, 50]
        if self.clamp_logits > 0:
            seg_logits = seg_logits.clamp(-self.clamp_logits, self.clamp_logits)

        return seg_logits

    @torch.no_grad()
    def seg_probs_eomt_style(self, x: torch.Tensor) -> torch.Tensor:
        mask_logits_layers, class_logits_layers = self.model(x)
        mask_logits = mask_logits_layers[-1]
        class_logits = class_logits_layers[-1]

        if mask_logits.shape[-2:] != x.shape[-2:]:
            mask_logits = F.interpolate(
                mask_logits, size=x.shape[-2:], mode="bilinear", align_corners=False
            )

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
        E = self.energy_map(logits)
        in_mask = (oe_mask01 == 0)
        out_mask = (oe_mask01 == 1)

        loss_in = torch.tensor(0.0, device=logits.device)
        loss_out = torch.tensor(0.0, device=logits.device)

        if in_mask.any():
            loss_in = torch.mean(F.relu(E[in_mask] - self.m_in) ** 2)
        if out_mask.any():
            loss_out = torch.mean(F.relu(self.m_out - E[out_mask]) ** 2)

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
            preds = torch.argmax(logits, dim=1)
            valid = (mask != self.ignore_index)
            if valid.any():
                self.train_miou(preds[valid], mask[valid])

            self.log("train/loss_ce", loss_ce, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
            self.log("train/mIoU", self.train_miou, prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
            return loss_ce

        # CNP / OE warmup
        if getattr(self, "warmup_epochs", 0) and self.current_epoch < self.warmup_epochs:
            zero = logits.sum() * 0.0
            self.log("train/loss_energy", zero, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
            self.log("train/energy_sep", zero, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
            return zero

        # after warmup: energy loss
        oe_mask01 = (mask > 0).to(torch.int64)
        loss_e, *_ , sep = self.energy_loss(logits, oe_mask01)
        loss = self.lambda_energy * loss_e

        self.log("train/loss_energy", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
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
            seg_probs = self.seg_probs_eomt_style(img)   # (B,C,H,W)
            msp = seg_probs.max(dim=1).values            # (B,H,W)
            msp_score = 1.0 - msp                        # (B,H,W)

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