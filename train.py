import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import LibriMixDataModule       
from model import SpeakerEncoderDualWrapper   
# from loss import LossWraper
from loss import PITArcFaceLoss
from eval_metrics import compute_clustering_metrics
import wandb
import sys
sys.path.append("/home/sidharth./codebase/")

# from wavlm_single_embedding.model import SpeakerEncoderWrapper as SingleSpeakerEncoderWrapper
import random
random.seed(42)
import warnings
warnings.filterwarnings("ignore")

def strip_model_prefix(state):
    new_state = {}
    for k, v in state.items():
        if k.startswith("model."):
            new_state[k[len("model."):]] = v   # remove "model."
        else:
            new_state[k] = v
    return new_state

class MySpEmb(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        finetune_encoder: bool = False,
        emb_dim: int = 256,
        speaker_map_path: str = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/Libriuni_05_08/Libri2Mix_ovl50to80/wav16k/min/metadata/train360_mapping.json",
    ):
        super().__init__()
        self.save_hyperparameters()

        # -----------------------------
        # 1. Speaker Encoder model
        # -----------------------------
        self.model = SpeakerEncoderDualWrapper(emb_dim=emb_dim, finetune_wavlm=True)

        # Optionally unfreeze wavlm if finetuning
        # if finetune_encoder:
        #     self.model.wavlm.requires_grad_(True)

        # -----------------------------
        # 2. ArcFace classification head
        # -----------------------------
        with open(speaker_map_path, "r") as f:
            speaker_map = json.load(f)


        # self.cosine_loss = LossWraper()
        self.loss_fn = PITArcFaceLoss(num_class=len(speaker_map), emb_dim=emb_dim, s=30, m=0.2)

        # -----------------------------
        # 3. Embedding metrics (for validation)
        # -----------------------------
        # self.metrics = EmbeddingMetrics(device="cuda")  # will overwrite device at runtime

    def forward(self, wav):
        """
        wav: [B, T] (or [B, 1, T])
        returns: [B, 2, emb_dim]
        """
        return self.model(wav)

    # -----------------------------
    # TRAINING
    # -----------------------------
    def training_step(self, batch, batch_idx):
        """
        batch: (wav, speaker_label)
          wav: [B, T]
          labels: [B, 2]  (speaker IDs, already mapped to [0..num_classes-1])
        """
        mix, source, labels = batch
        emb = self.forward(mix)                    # [B, 2, emb_dim]

        loss = self.loss_fn(emb, labels)

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=mix.shape[0],
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
        mix, source, labels = batch
        emb = self.forward(mix)                   # [B, 2, emb_dim]
        #labels : [B, 2]

        self.val_embs.append(emb.detach().cpu())
        self.val_labels.append(labels.detach().cpu())


        return {'emb': emb, 'labels': labels}

    # -----------------------------
    # VALIDATION (end of epoch)
    # -----------------------------
    def on_validation_epoch_end(self):
        if not self.trainer.is_global_zero:
            return

        # [num_batches, B, 2, D] → [total_B, 2, D]
        val_embs = torch.cat(self.val_embs, dim=0)      # [B_total, 2, D]
        val_labels = torch.cat(self.val_labels, dim=0)  # [B_total, 2]

        # flatten: each speaker is a separate point
        B_total, S, D = val_embs.shape                  # S=2
        embs_flat = val_embs.reshape(B_total * S, D)    # [N, D]
        labels_flat = val_labels.reshape(-1)            # [N]

        # skip useless metrics
        if torch.unique(labels_flat).numel() < 2:
            print(labels_flat)
            print("Only one unique speaker in val set, skipping metrics.")
            return

        # compute metrics
        # results = self.metrics.compute_from_tensors(
        #     embs_flat.cpu(),
        #     labels_flat.cpu(),
        # )
        results = compute_clustering_metrics(embs_flat, labels_flat)

        # logging
        for k, v in results.items():
            if k == "tsne_fig":
                continue
            self.log(f"val/{k}", v, on_epoch=True, prog_bar=True, logger=True)

        # clear cache
        self.val_embs = []
        self.val_labels = []



    # -----------------------------
    # OPTIMIZER + SCHEDULER
    # -----------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
        # return optimizer

        # monitor one of the embedding metrics, e.g., separation (higher is better)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train/loss",
                "interval": "epoch",
            },
        }


# ---------------------------------------
# MAIN
# ---------------------------------------
if __name__ == "__main__":
    DATA_ROOT = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix" 
    SPEAKER_MAP = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/Libriuni_05_08/Libri2Mix_ovl50to80/wav16k/min/metadata/train360_mapping.json"


    dm = LibriMixDataModule(
        data_root=DATA_ROOT,
        speaker_map_path=SPEAKER_MAP,
        batch_size=32, 
        num_workers=20, # Set this to your preference
        num_speakers=2
    )

    model = MySpEmb(
        lr=1e-4,
        finetune_encoder=False,
        emb_dim=256,
        speaker_map_path=SPEAKER_MAP,   # ONLY train map here
    )

    wandb_logger = WandbLogger(
        project="librispeech-speaker-encoder",
        name="ft_wavlm_linear_dualemb_noteacher_tr360",
        # name='test_run',
        log_model=False,
        save_dir="/mnt/disks/data/model_ckpts/ft_wavlm_linear_dualemb_noteacher_tr360/wandb_logs",
    )

    ckpt = pl.callbacks.ModelCheckpoint(
        monitor="train/loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch}-{val_separation:.3f}",
        dirpath="/mnt/disks/data/model_ckpts/ft_wavlm_linear_dualemb_noteacher_tr360/"
    )

    trainer = pl.Trainer(
        strategy="ddp_find_unused_parameters_true",
        accelerator="gpu",
        devices=[0, 1, 2, 3],
        max_epochs=50,
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
