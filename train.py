import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import LibriDataModule          # <-- single-speaker Libri datamodule
from model import SpeakerEncoderWrapper   # <-- your WavLM+ASP encoder
from loss import LossWraper
from metrics import EmbeddingMetrics
import wandb


class MySpEmb(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        finetune_encoder: bool = False,
        emb_dim: int = 256,
        speaker_map_path: str = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/LibriSpeech/train-100_mapping.json",
    ):
        super().__init__()
        self.save_hyperparameters()

        # -----------------------------
        # 1. Speaker Encoder model
        # -----------------------------
        self.model = SpeakerEncoderWrapper(emb_dim=emb_dim)

        # Optionally unfreeze wavlm if finetuning
        if finetune_encoder:
            self.model.wavlm.requires_grad_(True)

        # -----------------------------
        # 2. ArcFace classification head
        # -----------------------------
        with open(speaker_map_path, "r") as f:
            speaker_map = json.load(f)
        num_classes = len(speaker_map)
        print(f"Initializing ArcFace loss with {num_classes} classes.")

        self.arcface_loss = LossWraper(
            num_class=num_classes,
            emb_dim=emb_dim
        )

        # -----------------------------
        # 3. Embedding metrics (for validation)
        # -----------------------------
        self.metrics = EmbeddingMetrics(device="cuda")  # will overwrite device at runtime

    def forward(self, wav):
        """
        wav: [B, T] (or [B, 1, T])
        returns: [B, emb_dim]
        """
        return self.model(wav)

    # -----------------------------
    # TRAINING
    # -----------------------------
    def training_step(self, batch, batch_idx):
        """
        batch: (wav, speaker_label)
          wav: [B, T]
          labels: [B]  (speaker IDs, already mapped to [0..num_classes-1])
        """
        wav, labels = batch
        emb = self.forward(wav)                    # [B, emb_dim]
        loss = self.arcface_loss(emb, labels)      # scalar

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=wav.shape[0],
        )
        return loss

    # -----------------------------
    # VALIDATION (per-batch)
    # -----------------------------

    def on_validation_epoch_start(self):
        self.val_embs = []
        self.val_labels = []

    def validation_step(self, batch, batch_idx):
        """
        For now we just compute arcface loss as a simple val loss.
        The clustering metrics are done in validation_epoch_end
        on the entire validation set.
        """
        wav, labels = batch
        emb = self.forward(wav)

        self.val_embs.append(emb.detach().cpu())
        self.val_labels.append(labels.detach().cpu())


        return {'emb': emb, 'labels': labels}

    # -----------------------------
    # VALIDATION (end of epoch)
    # -----------------------------
    def on_validation_epoch_end(self):
        if not self.trainer.is_global_zero:
            return
        val_embs = torch.cat(self.val_embs, dim=0).to(self.device)
        val_labels = torch.cat(self.val_labels, dim=0).to(self.device)
        if torch.unique(val_labels).shape[0] < 2:
            print("Only one speaker in val set, skipping metrics computation.")
            return
        self.metrics.device = self.device
        results = self.metrics.compute_from_tensors(
            val_embs,
            val_labels,
            # log_tsne=False
        )

        # ---- LOG SCALARS ----
        for k, v in results.items():
            if k in ["tsne_fig"]:
                continue
            self.log(f"val/{k}", v, on_epoch=True, prog_bar=True, logger=True)

        self.val_embs = []
        self.val_labels = []
        # ---- LOG FIGURE TO WANDB ----
        # tsne_fig = results.get("tsne_fig", None)
        # if tsne_fig is not None:
        #     self.logger.experiment.log({
        #         f"tsne/epoch_{self.current_epoch}": wandb.Image(tsne_fig)
        #     })
        #     plt.close(tsne_fig)


    # -----------------------------
    # OPTIMIZER + SCHEDULER
    # -----------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
        # return optimizer

        # monitor one of the embedding metrics, e.g., separation (higher is better)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train/loss",
                "frequency": 1,
                "interval": "epoch",
            },
        }


# ---------------------------------------
# MAIN
# ---------------------------------------
if __name__ == "__main__":
    DATA_ROOT = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/LibriSpeech"

    TRAIN_SPK_MAP = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/LibriSpeech/train-100_mapping.json"

    dm = LibriDataModule(
        data_root=DATA_ROOT,
        train_speaker_map_path=TRAIN_SPK_MAP,
        train_batch_size=32,
        val_batch_size=8,
        num_workers=20,
        sample_rate=16000,
    )

    model = MySpEmb(
        lr=1e-4,
        finetune_encoder=False,
        emb_dim=256,
        speaker_map_path=TRAIN_SPK_MAP,   # ONLY train map here
    )

    wandb_logger = WandbLogger(
        project="librispeech-speaker-encoder",
        name="wavlm_asp_arcface",
        # name='test_run',
        log_model=False,
        save_dir="/mnt/disks/data/model_ckpts/librispeech_asp_wavlm/wandb_logs",
    )

    ckpt = pl.callbacks.ModelCheckpoint(
        monitor="train/loss",
        mode="max",
        save_top_k=10,
        filename="best-{epoch}-{val_separation:.3f}",
        dirpath="/mnt/disks/data/model_ckpts/librispeech_asp_wavlm/"
    )

    trainer = pl.Trainer(
        strategy="ddp_find_unused_parameters_true",
        accelerator="gpu",
        devices=[0, 1],
        max_epochs=1000,
        logger=wandb_logger,
        callbacks=[ckpt],
        gradient_clip_val=5.0,
        enable_checkpointing=True,
    )

    # trainer = pl.Trainer(
    #     accelerator='gpu',
    #     devices=[0],
    #     max_epochs=100,
    #     logger=wandb_logger,
    #     overfit_batches=1,
    #     limit_train_batches=1,
    #     limit_val_batches=1,
    #     num_sanity_val_steps=0,
    #     enable_checkpointing=False,
    # )

    # trainer = pl.Trainer(
    #     accelerator="gpu",
    #     devices=1,
    #     max_epochs=1,
    #     limit_train_batches=1,
    #     limit_val_batches=1,
    #     num_sanity_val_steps=0,
    # )
    trainer.fit(model, datamodule=dm)
    # trainer.validate(model, datamodule=dm)
    wandb.finish()
