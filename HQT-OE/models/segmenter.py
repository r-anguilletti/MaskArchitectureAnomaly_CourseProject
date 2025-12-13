import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from peft import LoraConfig, get_peft_model
from models.vit import ViT
from models.eomt import EoMT

class AnomalySegmenter(L.LightningModule):
    def __init__(
        self, 
        img_size=(518, 518),
        num_classes=20, 
        lr=1e-4,
        backbone_name="vit_large_patch14_reg4_dinov2",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.num_classes = num_classes
        
        # 1. Backbone & Model (EoMT)
        # Assicurati che patch_size corrisponda al modello (DINOv2 solitamente è 14)
        self.encoder = ViT(img_size=img_size, backbone_name=backbone_name, patch_size=14)
        self.model = EoMT(encoder=self.encoder, num_classes=num_classes, num_q=100, num_blocks=4)
        
        # 2. LoRA Setup
        self._setup_lora()
        
        # 3. CrossEntropy (ignore_index 255 è standard per Cityscapes)
        self.criterion_ce = nn.CrossEntropyLoss(ignore_index=255)

    def _setup_lora(self):
        peft_config = LoraConfig(
            r=8, lora_alpha=16, target_modules=["qkv"], 
            lora_dropout=0.05, bias="none",
            # Salviamo le teste di classificazione e convoluzioni decoder
            modules_to_save=["class_head", "mask_head", "upscale", "q", "norm", "conv1", "conv2"]
        )
        self.model = get_peft_model(self.model, peft_config)

    def forward(self, x):
        """
        Smart Forward:
        - Durante il training (loss calculation), potrebbe servire l'output grezzo.
        - Ma per semplicità e compatibilità con la visualizzazione, qui calcoliamo 
          e restituiamo direttamente i LOGITS DI SEGMENTAZIONE FINALI.
        """
        # 1. Ottieni output grezzi dal trasformatore
        mask_logits_list, class_logits_list = self.model(x)

        # 2. Prendi l'ultimo layer
        final_mask_logits = mask_logits_list[-1]
        final_class_logits = class_logits_list[-1]

        # 3. Ricostruisci la mappa di segmentazione (B, C, H, W)
        # Usiamo le dimensioni dell'input originale x per l'upsample
        target_hw = (x.shape[-2], x.shape[-1])
        seg_logits = self._build_per_class_logits(final_mask_logits, final_class_logits, target_hw)
        
        return seg_logits

    def _build_per_class_logits(self, mask_logits, class_logits, target_hw):
        """
        Combina (B, Q, H, W) e (B, Q, Classi) -> (B, Classi, H, W)
        """
        B, Q, Hp, Wp = mask_logits.shape
        _, _, Cplus1 = class_logits.shape
        C = Cplus1 - 1  # L'ultima classe è solitamente "No Object" / "Void" nell'architettura Mask2Former/EoMT

        # 1. Escludiamo l'ultima classe 'null' dalle predizioni semantiche utili
        class_logits_valid = class_logits[..., :C]   # (B, Q, C)

        # 2. Permutiamo per moltiplicazione matriciale
        class_logits_permuted = class_logits_valid.permute(0, 2, 1).contiguous() # (B, C, Q)

        # 3. Flatten spaziale delle maschere
        flat_mask_logits = mask_logits.view(B, Q, -1) # (B, Q, HWp)

        # 4. Prodotto: Classi x Maschere
        seg_logits_flat = torch.bmm(class_logits_permuted, flat_mask_logits) # (B, C, HWp)

        # 5. Reshape e Upsample
        seg_logits = seg_logits_flat.view(B, C, Hp, Wp)
        
        if (Hp, Wp) != target_hw:
            seg_logits = F.interpolate(seg_logits, size=target_hw, mode='bilinear', align_corners=False)

        return seg_logits

    def training_step(self, batch, batch_idx):
        img, mask = batch
        
        # Chiama forward (che ora restituisce seg_logits pronti)
        seg_logits = self(img)

        # --- LOSS 1: Cross Entropy ---
        loss_ce = self.criterion_ce(seg_logits, mask)

        # --- LOSS 2: Energy Margin Loss (Anomaly Detection) ---
        # Calcola l'Energia: E(x) = -T * logsumexp(logits)
        # Più alto è il logit (confidenza alta), più bassa (negativa) è l'energia.
        T = 1.0
        energy_map = -T * torch.logsumexp(seg_logits / T, dim=1) # (B, H, W)

        # DEFINIZIONE CLASSI:
        # Assumiamo che nel dataset Hybrid:
        # Valori 0-18 = In-Distribution (Cityscapes classes)
        # Valore 19   = Anomaly / Void (da trattare come OOD)
        # Valore 255  = Ignore
        
        # Maschere per i pixel ID (In-Distribution) e OOD (Out-of-Distribution)
        mask_in = (mask != 19) & (mask != 255)
        mask_out = (mask == 19)

        loss_energy = torch.tensor(0.0, device=self.device)
        
        # Margini (Hardcoded o parametri)
        m_in = -7.0  # Vogliamo energia < -7 per ID (molto confidente)
        m_out = -5.0 # Vogliamo energia > -5 per OOD (poco confidente/incerto)

        # Penalizza ID se l'energia è troppo alta (sopra m_in)
        if mask_in.sum() > 0:
            e_in = energy_map[mask_in]
            loss_energy += torch.mean(torch.pow(F.relu(e_in - m_in), 2))

        # Penalizza OOD se l'energia è troppo bassa (sotto m_out)
        if mask_out.sum() > 0:
            e_out = energy_map[mask_out]
            loss_energy += torch.mean(torch.pow(F.relu(m_out - e_out), 2))

        # Somma pesata (0.1 è un fattore comune in letteratura)
        total_loss = loss_ce + 0.1 * loss_energy

        # Log
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("ce_loss", loss_ce, on_step=False, on_epoch=True)
        self.log("energy_loss", loss_energy, on_step=False, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        seg_logits = self(img)
        val_loss = self.criterion_ce(seg_logits, mask)
        
        # Loggare 'val_loss' è essenziale per il ModelCheckpoint nel train.py
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        return val_loss

    def configure_optimizers(self):
        # Ottimizza solo i parametri che richiedono gradiente (LoRA + Heads)
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)