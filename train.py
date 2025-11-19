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
from loss import LossWraper
from metrics import EmbeddingMetrics
import wandb
import sys
sys.path.append("/home/sidharth./codebase/")

from wavlm_single_embedding.model import SpeakerEncoderWrapper as SingleSpeakerEncoderWrapper
import random
random.seed(42)


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
        self.model = SpeakerEncoderDualWrapper(emb_dim=emb_dim)

        # Optionally unfreeze wavlm if finetuning
        if finetune_encoder:
            self.model.wavlm.requires_grad_(True)

        # -----------------------------
        # 2. ArcFace classification head
        # -----------------------------
        with open(speaker_map_path, "r") as f:
            speaker_map = json.load(f)


        self.cosine_loss = LossWraper()
        #Get the teacher model
        self.single_sp_model = SingleSpeakerEncoderWrapper(emb_dim=emb_dim)
        teacher_ckpt_path = "/mnt/disks/data/model_ckpts/librispeech_asp_wavlm/best-epoch=47-val_separation=0.000.ckpt"

        ckpt = torch.load(teacher_ckpt_path, map_location="cpu")
        state = ckpt["state_dict"]

        filtered = {}
        for k, v in state.items():
            # only keep model.encoder.* or model.wavlm.*, model.projector.*, model.pooling.*
            if k.startswith("model.") and ("arcface" not in k):
                filtered[k.replace("model.", "", 1)] = v

        print("Loaded teacher keys:", len(filtered))

        self.single_sp_model.load_state_dict(filtered, strict=True)
        self.single_sp_model.eval()
        for param in self.single_sp_model.parameters():
            param.requires_grad = False



        # -----------------------------
        # 3. Embedding metrics (for validation)
        # -----------------------------
        self.metrics = EmbeddingMetrics(device="cuda")  # will overwrite device at runtime

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
        #change here
        with torch.no_grad():
            emb1 = self.single_sp_model(source[:, 0, :])  # [B, emb_dim]
            emb2 = self.single_sp_model(source[:, 1, :])  # [B, emb_dim]
            gt_embs = torch.stack([emb1, emb2], dim=1)  # [B, 2, emb_dim]
        
        loss_dict = self.cosine_loss(emb, gt_embs) #{'total_loss': , 'cosine_loss': , 'ortho_loss': }


        if batch_idx == 0 and self.current_epoch == 0:
            with torch.no_grad():
                cos_gt = F.cosine_similarity(gt_embs[:,0,:], gt_embs[:,1,:], dim=-1).mean()
                cos_pred = F.cosine_similarity(emb[:,0,:], emb[:,1,:], dim=-1).mean()
                print("Mean cos(gt1, gt2) =", cos_gt.item())
                print("Mean cos(pred1, pred2) =", cos_pred.item())

        # self.log(
        #     "train/loss",
        #     loss,
        #     on_step=True,
        #     on_epoch=True,
        #     prog_bar=True,
        #     logger=True,
        #     batch_size=mix.shape[0],
        # )
        for k, v in loss_dict.items():
            self.log(
                f"train/{k}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=mix.shape[0],
            )
        return loss_dict['total_loss']

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
        results = self.metrics.compute_from_tensors(
            embs_flat.cpu(),
            labels_flat.cpu(),
        )

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
                "monitor": "train/total_loss",
                "interval": "epoch",
            },
        }


# ---------------------------------------
# MAIN
# ---------------------------------------
if __name__ == "__main__":
    DATA_ROOT = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix" 
    SPEAKER_MAP = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/Libri2Mix/wav16k/min/metadata/train100_mapping.json"

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
        name="wavlm_asp_dual_embedding-orthogonality",
        # name='test_run',
        log_model=False,
        save_dir="/mnt/disks/data/model_ckpts/librispeech_asp_wavlm_dualemb/wandb_logs",
    )

    ckpt = pl.callbacks.ModelCheckpoint(
        monitor="train/loss",
        mode="min",
        save_top_k=10,
        filename="best-{epoch}-{val_separation:.3f}",
        dirpath="/mnt/disks/data/model_ckpts/librispeech_asp_wavlm_dualemb/"
    )

    trainer = pl.Trainer(
        strategy="ddp_find_unused_parameters_true",
        accelerator="gpu",
        devices=[0, 1],
        max_epochs=100,
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
