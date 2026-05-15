import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb

# Your codebase
from dataset import LibriMixDataModule   # MUST yield vad_targets: [B,2,T_frames]
from model import SpeakerEncoderDualWrapper
from pvad_model import pVAD_module


# -----------------------------
# Helper: load student checkpoint weights into self.model
# -----------------------------
def strip_dual_model_weights(state_dict):
    """
    Keeps only keys under "model." and strips that prefix.
    Drops any teacher / arcface keys if present.
    """
    new_state = {}
    for k, v in state_dict.items():
        if not k.startswith("model."):
            continue
        k2 = k.replace("model.", "", 1)
        if k2.startswith("single_sp_model.") or k2.startswith("arcface_loss."):
            continue
        new_state[k2] = v
    return new_state




# -----------------------------
# Lightning module: VAD-only training (Option A)
# -----------------------------
class MyVADOnly(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        emb_dim: int = 256,
        ckpt_student_path: str = "",
        vad_hidden: int = 256+80,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ---- Base dual model (student) ----
        self.model = SpeakerEncoderDualWrapper(emb_dim=emb_dim)

        self.pvad = pVAD_module(hidden_dim=vad_hidden)



        # ---- Load pretrained student checkpoint ----
        if ckpt_student_path and os.path.isfile(ckpt_student_path):
            ckpt = torch.load(ckpt_student_path, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            state = strip_dual_model_weights(state)
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            print(f"[Student CKPT] loaded. missing={len(missing)} unexpected={len(unexpected)}")
        else:
            raise FileNotFoundError(f"Student checkpoint not found: {ckpt_student_path}")

        # ---- Freeze base (Option A) ----
        for p in self.model.parameters():
            p.requires_grad = False

        # storage for test aggregation
        self._test_loss_sum = 0.0
        self._test_acc_sum = 0.0
        self._test_f1_sum = 0.0
        self._test_batches = 0

    def forward_features(self, wav):
        """
        Extract per-frame features BEFORE pooling.
        Returns:
          proj1, proj2: [B, emb_dim, T_frames]
        """
        if wav.dim() == 3:  # [B, 1, T]
            wav = wav.squeeze(1)


        emb = self.model(wav) #[B,2,256]

        # emb1, emb2 = emb[:, 0, :], emb[:, 1, :]  # each [B, 256]

        # vad_logits1 = self.pvad(wav, emb1)
        # vad_logits2 = self.pvad(wav, emb2)
        return emb

    @staticmethod
    def _align_T(logits, targets):
        """
        logits:  [B,2,Tm]
        targets: [B,2,Tg]
        -> crop to min T
        """
        Tm = logits.shape[-1]
        Tg = targets.shape[-1]
        if Tm != Tg:
            T = min(Tm, Tg)
            logits = logits[..., :T]
            targets = targets[..., :T]
        return logits, targets

    @staticmethod
    def _pit_best_targets_from_losses(vad_targets, loss_direct, loss_swap):
        """
        vad_targets: [B,2,T]
        loss_direct/loss_swap: [B]
        returns targets_best: [B,2,T], use_swap mask [B]
        """
        use_swap = (loss_swap < loss_direct)  # [B]
        targets_best = vad_targets.clone()
        if use_swap.any():
            targets_best[use_swap] = vad_targets[use_swap][:, [1, 0], :]
        return targets_best, use_swap

    @staticmethod
    def _frame_metrics_from_logits(logits, targets_best):
        """
        logits: [B,2,T]
        targets_best: [B,2,T]
        returns acc, f1 (scalars)
        """
        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).float()

        acc = (pred == targets_best).float().mean()

        tp = (pred * targets_best).sum()
        fp = (pred * (1.0 - targets_best)).sum()
        fn = ((1.0 - pred) * targets_best).sum()
        f1 = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)

        return acc, f1

    def training_step(self, batch, batch_idx):
        """
        Expect batch:
          mix:         [B, T]
          source:      [B, 2, T]         (ignored)
          labels:      [B, 2]            (ignored)
          vad_targets: [B, 2, T_frames]
        """
        if len(batch) != 4:
            raise ValueError("Batch must be (mix, source, labels, vad_targets).")

        mix, _, _, vad_targets = batch
        vad_targets = vad_targets.float()  # [B,2,Tg]

        # ---- Feature extraction (frozen base) ----
        with torch.no_grad():
            embs = self.forward_features(mix)  # [B,2, 256]
        
        emb1, emb2 = embs[:, 0, :], embs[:, 1, :]  # each [B, 256]


        # ---- VAD logits ----
        logit1 = self.pvad(mix, emb1)  # [B, Tm]
        logit2 = self.pvad(mix, emb2)  # [B, Tm]
        logits = torch.stack([logit1, logit2], dim=1)  # [B,2,Tm]

        logits, vad_targets = self._align_T(logits, vad_targets)

        # ---- PIT-BCE (per-sample) ----
        loss_direct = F.binary_cross_entropy_with_logits(
            logits, vad_targets, reduction="none"
        ).mean(dim=(1, 2))  # [B]

        vad_swapped = vad_targets[:, [1, 0], :]
        loss_swap = F.binary_cross_entropy_with_logits(
            logits, vad_swapped, reduction="none"
        ).mean(dim=(1, 2))  # [B]

        loss_per_sample = torch.minimum(loss_direct, loss_swap)
        loss = loss_per_sample.mean()
        swap_rate = (loss_swap < loss_direct).float().mean()

        self.log("train/vad_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=mix.shape[0])
        self.log("train/pit_swap_rate", swap_rate, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=mix.shape[0])

        return loss

    def validation_step(self, batch, batch_idx):
        if len(batch) != 4:
            raise ValueError("Val batch must be (mix, source, labels, vad_targets).")

        mix, _, _, vad_targets = batch
        vad_targets = vad_targets.float()

        with torch.no_grad():
            embs = self.forward_features(mix)  # [B,2, 256]
        
        emb1, emb2 = embs[:, 0, :], embs[:, 1, :]  # each [B, 256]


        # ---- VAD logits ----
        logit1 = self.pvad(mix, emb1)  # [B, Tm]
        logit2 = self.pvad(mix, emb2)  # [B, Tm]
        logits = torch.stack([logit1, logit2], dim=1)  # [B,2,Tm]

        logits, vad_targets = self._align_T(logits, vad_targets)

        loss_direct = F.binary_cross_entropy_with_logits(logits, vad_targets, reduction="none").mean(dim=(1, 2))
        loss_swap   = F.binary_cross_entropy_with_logits(logits, vad_targets[:, [1, 0], :], reduction="none").mean(dim=(1, 2))
        loss = torch.minimum(loss_direct, loss_swap).mean()

        targets_best, use_swap = self._pit_best_targets_from_losses(vad_targets, loss_direct, loss_swap)
        acc, f1 = self._frame_metrics_from_logits(logits, targets_best)
        swap_rate = use_swap.float().mean()

        self.log("val/vad_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=mix.shape[0])
        self.log("val/vad_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=mix.shape[0])
        self.log("val/vad_f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=mix.shape[0])
        self.log("val/pit_swap_rate", swap_rate, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=mix.shape[0])

        return {"val_loss": loss}

    # -----------------------------
    # TEST (requested)
    # -----------------------------
    def on_test_epoch_start(self):
        self._test_loss_sum = 0.0
        self._test_acc_sum = 0.0
        self._test_f1_sum = 0.0
        self._test_batches = 0

    def test_step(self, batch, batch_idx):
        if len(batch) != 4:
            raise ValueError("Test batch must be (mix, source, labels, vad_targets).")

        mix, _, _, vad_targets = batch
        vad_targets = vad_targets.float()

        with torch.no_grad():
            embs = self.forward_features(mix)  # [B,2, 256]
        
        emb1, emb2 = embs[:, 0, :], embs[:, 1, :]  # each [B, 256]


        # ---- VAD logits ----
        logit1 = self.pvad(mix, emb1)  # [B, Tm]
        logit2 = self.pvad(mix, emb2)  # [B, Tm]
        logits = torch.stack([logit1, logit2], dim=1)  # [B,2,Tm]

        logits, vad_targets = self._align_T(logits, vad_targets)

        loss_direct = F.binary_cross_entropy_with_logits(logits, vad_targets, reduction="none").mean(dim=(1, 2))
        loss_swap   = F.binary_cross_entropy_with_logits(logits, vad_targets[:, [1, 0], :], reduction="none").mean(dim=(1, 2))
        loss = torch.minimum(loss_direct, loss_swap).mean()

        targets_best, use_swap = self._pit_best_targets_from_losses(vad_targets, loss_direct, loss_swap)
        acc, f1 = self._frame_metrics_from_logits(logits, targets_best)
        swap_rate = use_swap.float().mean()

        # per-step logs
        self.log("test/vad_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=mix.shape[0])
        self.log("test/vad_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=mix.shape[0])
        self.log("test/vad_f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=mix.shape[0])
        self.log("test/pit_swap_rate", swap_rate, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=mix.shape[0])

        # manual aggregation too (robust even if you change logging later)
        self._test_loss_sum += float(loss.detach().cpu())
        self._test_acc_sum += float(acc.detach().cpu())
        self._test_f1_sum += float(f1.detach().cpu())
        self._test_batches += 1

        return {"test_loss": loss}

    def on_test_epoch_end(self):
        if self._test_batches > 0:
            mean_loss = self._test_loss_sum / self._test_batches
            mean_acc = self._test_acc_sum / self._test_batches
            mean_f1 = self._test_f1_sum / self._test_batches
            print(f"[TEST] mean_loss={mean_loss:.4f} mean_acc={mean_acc:.4f} mean_f1={mean_f1:.4f}")

    def configure_optimizers(self):
        params = list(self.pvad.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.hparams.lr, weight_decay=0.01)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/vad_loss",
                "interval": "epoch",
            },
        }


# ---------------------------------------
# MAIN
# ---------------------------------------
if __name__ == "__main__":
    DATA_ROOT = "/home/sidcs/datasets/LibriMix/LibriMix"
    SPEAKER_MAP = "/home/sidcs/datasets/LibriMix/LibriMix/Libriuni_05_08/Libri2Mix_ovl50to80/wav16k/min/metadata/train360_mapping.json"

    # pretrained dual-embedding checkpoint (student)
    STUDENT_CKPT = "/home/sidcs/model_ckpts/ECAPA_UNMIX_3072_teacher_ECAPA/best-epoch=87-val_separation=0.000.ckpt"

    dm = LibriMixDataModule(
        data_root=DATA_ROOT,
        speaker_map_path=SPEAKER_MAP,
        batch_size=32 * 16,
        num_workers=20,
        num_speakers=2,
    )

    model = MyVADOnly(
        lr=1e-4,
        emb_dim=256,
        ckpt_student_path=STUDENT_CKPT,
        vad_hidden=256+80,
    )

    wandb_logger = WandbLogger(
        project="librispeech-vad-head",
        name="pvad_emb",
        log_model=False,
        save_dir="/home/sidcs/model_ckpts/pvad_emb/wandb_logs",
    )

    ckpt_cb = pl.callbacks.ModelCheckpoint(
        monitor="val/vad_loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch}-{val_vad_loss:.4f}",
        dirpath="/home/sidcs/model_ckpts/pvad_emb/",
    )

    trainer = pl.Trainer(
        strategy="ddp_find_unused_parameters_true",
        accelerator="gpu",
        devices=[0,1,2,3,4,5,6],           # change to [0,1,2,3] when ready
        max_epochs=100,
        logger=wandb_logger,
        callbacks=[ckpt_cb],
        gradient_clip_val=5.0,
        enable_checkpointing=True,
        num_sanity_val_steps=0,  # avoids failing early if your val/test dataset is still being wired
    )

    trainer.fit(model, datamodule=dm, ckpt_path="/home/sidcs/model_ckpts/pvad_emb/best-epoch=47-val_vad_loss=0.0000.ckpt")

    # # If your LibriMixDataModule has test_dataset/test_dataloader, this will run.
    # # Otherwise add it similarly to your val_dataset/val_dataloader.
    # try:
    #     trainer.test(model, datamodule=dm, ckpt_path="/home/sidcs/model_ckpts/VAD_only_on_pretrained_dualemb/best-epoch=30-val_vad_loss=0.0000.ckpt")
    # except Exception as e:
    #     print("[WARN] trainer.test skipped / failed:", repr(e))

    # wandb.finish()