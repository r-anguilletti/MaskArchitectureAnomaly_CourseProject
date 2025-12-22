# models/segmenter.py
import os
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
    Fine-tuning EoMT baseline + OE:
      - Cityscapes: CE su 19 classi
      - OE cut&paste: energy separation (OOD mask 0/1)
    Convention:
      source == 0 -> City (ID)
      source == 1 -> OE  (mask 0/1)
    """

    def __init__(
        self,
        img_size=(1024, 1024),
        num_id_classes=19,
        ignore_index=255,
        lr=1e-5,
        weight_decay=1e-2,

        # ✅ QUESTO DEVE MATCHARE IL .bin:
        # patch16 + embed_dim 768 + reg tokens (reg4)
        backbone_name="vit_base_patch14_reg4_dinov2",
        patch_size=16,

        num_queries=100,
        num_blocks=3,
        masked_attn_enabled=True,

        pretrained_eomt_bin=None,

        # Energy
        T=1.0,
        m_in=-12.0,
        m_out=-6.0,
        lambda_energy=0.1,

        # LoRA
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules=("qkv",),
        modules_to_save=("class_head", "mask_head"),

        clamp_logits=10.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_id_classes = int(num_id_classes)
        self.ignore_index = int(ignore_index)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

        self.T = float(T)
        self.m_in = float(m_in)
        self.m_out = float(m_out)
        self.lambda_energy = float(lambda_energy)
        self.clamp_logits = float(clamp_logits)

        # ----------------------------
        # Build encoder/backbone
        # IMPORTANT: dobbiamo evitare timm pretrained download
        # Nel tuo vit.py: pretrained = (ckpt_path is None)
        # quindi mettiamo ckpt_path NON None per forzare pretrained=False
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

        # ----------------------------
        # Load .bin (network.*) BEFORE LoRA
        # ----------------------------
        if pretrained_eomt_bin is not None:
            self._load_eomt_bin(pretrained_eomt_bin)

        # ----------------------------
        # LoRA wrap
        # ----------------------------
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

        # ----------------------------
        # Loss / Metrics
        # ----------------------------
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

    def _load_eomt_bin(self, bin_path: str):
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"[LOAD_EOMT_BIN] not found: {bin_path}")

        sd = torch.load(bin_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        if not isinstance(sd, dict):
            raise ValueError("[LOAD_EOMT_BIN] Expected a state_dict-like dict.")

        # tieni solo network.*
        sd = {k: v for k, v in sd.items() if k.startswith("network.")}
        # rimuovi prefisso network.
        sd = {k.replace("network.", "", 1): v for k, v in sd.items()}

        missing, unexpected = self.base_model.load_state_dict(sd, strict=False)
        print(f"[LOAD_EOMT_BIN] loaded from: {bin_path}")
        print(f"[LOAD_EOMT_BIN] missing={len(missing)} unexpected={len(unexpected)}")
        if len(missing) and len(missing) < 30:
            print("[LOAD_EOMT_BIN] missing(sample):", missing[:10])
        if len(unexpected) and len(unexpected) < 30:
            print("[LOAD_EOMT_BIN] unexpected(sample):", unexpected[:10])

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

    def forward(self, x):
        mask_logits_layers, class_logits_layers = self.model(x)
        mask_logits = mask_logits_layers[-1]     # (B,Q,h,w)
        class_logits = class_logits_layers[-1]   # (B,Q,C+1)

        if mask_logits.shape[-2:] != x.shape[-2:]:
            mask_logits = F.interpolate(mask_logits, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if class_logits.shape[-1] == self.num_id_classes + 1:
            class_logits = class_logits[..., :self.num_id_classes]
        else:
            class_logits = class_logits[..., :self.num_id_classes]

        class_logits = class_logits.permute(0, 2, 1)  # (B,C,Q)

        B, Q, H, W = mask_logits.shape
        seg_flat = torch.bmm(
            class_logits.clamp(-self.clamp_logits, self.clamp_logits),
            mask_logits.view(B, Q, -1).clamp(-self.clamp_logits, self.clamp_logits),
        )
        seg = seg_flat.view(B, self.num_id_classes, H, W)

        if self.clamp_logits > 0:
            seg = seg.clamp(-self.clamp_logits, self.clamp_logits)
        return seg

    def energy_map(self, logits):
        return -self.T * torch.logsumexp(logits / self.T, dim=1)

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

    def training_step(self, batch, batch_idx):
        img, mask, source = batch
        source = self._source_to_int(source)

        logits = self(img).float()

        if source == 0:
            loss_ce = self.ce_loss(logits, mask)

            preds = torch.argmax(logits, dim=1)
            valid = (mask != self.ignore_index)
            if valid.any():
                self.train_miou(preds[valid], mask[valid])

            bs = img.shape[0]
            self.log("train/loss_ce", loss_ce, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
            self.log("train/mIoU", self.train_miou, prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
            return loss_ce
        else:
            loss_e, loss_in, loss_out, mean_in, mean_out, sep = self.energy_loss(logits, mask)
            loss = self.lambda_energy * loss_e

            bs = img.shape[0]
            self.log("train/loss_energy", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
            self.log("train/energy_in", mean_in, prog_bar=False, on_step=True, on_epoch=True, batch_size=bs)
            self.log("train/energy_out", mean_out, prog_bar=False, on_step=True, on_epoch=True, batch_size=bs)
            self.log("train/energy_sep", sep, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
            self.log("train/hinge_in", loss_in, prog_bar=False, on_step=True, on_epoch=True, batch_size=bs)
            self.log("train/hinge_out", loss_out, prog_bar=False, on_step=True, on_epoch=True, batch_size=bs)
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