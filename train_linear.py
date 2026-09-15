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
from loss import PITArcFaceLoss, CosineSimilarityLoss
from eval_metrics import compute_clustering_metrics
import wandb
import sys
sys.path.append("/home/sidcs.csegpu1/codebase")

from wavlm_single_embedding.model import SpeakerEncoderWrapper as SingleSpeakerEncoderWrapper
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
        speaker_map_path: str = "/home/sidcs.csegpu1/datasets/LibriMix/LibriMix/Libriuni_05_08/Libri2Mix_ovl50to80/wav16k/min/metadata/train360_mapping.json",
    ):
        super().__init__()
        self.save_hyperparameters()

        # -----------------------------
        # 1. Speaker Encoder model
        # -----------------------------
        self.model = SpeakerEncoderDualWrapper(emb_dim=emb_dim, finetune_wavlm=True)
        #load the speaker dual model ckpt
        student_ckpt_path = "/home/sidcs.csegpu1/model_ckpts/ft_wavlm_linear_dualemb_noteacher_tr360/best-epoch=12-val_separation=0.000.ckpt"

        ckpt_st = torch.load(student_ckpt_path, map_location="cpu")
        state_st = ckpt_st["state_dict"]

        filtered_st = {}
        for k, v in state_st.items():
            # only keep model.encoder.* or model.wavlm.*, model.projector.*, model.pooling.*
            if k.startswith("model.") and ("arcface" not in k):
                filtered_st[k.replace("model.", "", 1)] = v
        print("Loaded student keys:", len(filtered_st))

        self.model.load_state_dict(filtered_st, strict=True)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    

        # Teacher model
        self.single_sp_model = SingleSpeakerEncoderWrapper(emb_dim=emb_dim)
        teacher_ckpt_path = "/home/sidcs.csegpu1/model_ckpts/librispeech_asp_wavlm_tr360/best-epoch=62-val_separation=0.000.ckpt"

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



        # Optionally unfreeze wavlm if finetuning
        # if finetune_encoder:
        #     self.model.wavlm.requires_grad_(True)

        # -----------------------------
        # 2. ArcFace classification head
        # -----------------------------
        with open(speaker_map_path, "r") as f:
            speaker_map = json.load(f)


        # self.cosine_loss = LossWraper()
        self.loss_fn = CosineSimilarityLoss()

        # self.linear_map = nn.Linear(256, 256)
        self.mlp = nn.Sequential(nn.Linear(256, 512),
                                nn.ReLU(),
                                nn.Linear(512, 256))
        
        linear_ckpt_path = "/home/sidcs.csegpu1/model_ckpts/ft_wavlm_linear_dualemb_noteacher_tr360_linearmappingtotr/best-epoch=47-val_separation=0.000.ckpt"
        linear_ckpt = torch.load(linear_ckpt_path, map_location="cpu", weights_only=True)
        linear_state = linear_ckpt["state_dict"]

        W = linear_state["linear_map.weight"].to(self.mlp[2].weight)
        b = linear_state["linear_map.bias"].to(self.mlp[2].bias)
        with torch.no_grad():
            I  = torch.eye(256, device=self.mlp[0].weight.device, dtype=self.mlp[0].weight.dtype)

            self.mlp[0].weight.copy_(torch.cat([I, -I], dim=0))
            self.mlp[0].bias.zero_()

            self.mlp[2].weight.copy_(torch.cat([W, -W], dim=1))
            self.mlp[2].bias.copy_(b)
            # Verify equivalence before training.
            x = torch.randn(8, 2, 256, device=W.device, dtype=W.dtype)

            expected = F.linear(x, W, b)
            actual = self.mlp(x)

            torch.testing.assert_close(
                actual, expected,
                rtol=1e-5,
                atol=1e-5,
            )
            print("MLP matches trained linear mapper.")
            print("Maximum error:", (actual - expected).abs().max().item())


    def train(self, mode=True):
        super().train(mode)

        # Frozen encoders always remain in evaluation mode.
        self.model.eval()
        self.single_sp_model.eval()

        return self

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
        with torch.no_grad():
            emb1 = self.single_sp_model(source[:, 0, :])  # [B, emb_dim]
            emb2 = self.single_sp_model(source[:, 1, :])  # [B, emb_dim]
            gt_embs = torch.stack([emb1, emb2], dim=1)  # [B, 2, emb_dim]

        # mapped_emb = self.linear_map(emb)
        mapped_emb = self.mlp(emb)
        # loss = self.loss_fn(emb, labels)
        loss = self.loss_fn(mapped_emb, gt_embs)

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=mix.shape[0],
            sync_dist=True
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
        Mirrors training_step: compute the dual-embedding model's output on the
        mixture and the teacher (single-speaker) embeddings on the clean sources,
        then use the same cosine-similarity loss as the val loss.
        The clustering metrics are done in on_validation_epoch_end
        on the entire validation set.
        """
        mix, source, labels = batch
        emb = self.forward(mix)                    # [B, 2, emb_dim]
        with torch.no_grad():
            emb1 = self.single_sp_model(source[:, 0, :])  # [B, emb_dim]
            emb2 = self.single_sp_model(source[:, 1, :])  # [B, emb_dim]
            gt_embs = torch.stack([emb1, emb2], dim=1)     # [B, 2, emb_dim]

        # mapped_emb = self.linear_map(emb)
        mapped_emb = self.mlp(emb)

        loss = self.loss_fn(mapped_emb, gt_embs)
        #labels : [B, 2]

        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=mix.shape[0],
            sync_dist=True
        )

        self.val_embs.append(mapped_emb.detach().cpu())
        self.val_labels.append(labels.detach().cpu())

        return {'emb': emb, 'labels': labels, 'loss': loss}

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
    DATA_ROOT = "/home/sidcs.csegpu1/datasets/LibriMix/LibriMix" 
    SPEAKER_MAP = "/home/sidcs.csegpu1/datasets/LibriMix/LibriMix/Libriuni_05_08/Libri2Mix_ovl50to80/wav16k/min/metadata/train360_mapping.json"
    import os
    dir_path = "/home/sidcs.csegpu1/model_ckpts/ft_wavlm_linear_dualemb_noteacher_tr360_mlpmappingtotr/linear_init"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


    dm = LibriMixDataModule(
        data_root=DATA_ROOT,
        speaker_map_path=SPEAKER_MAP,
        batch_size=64, 
        num_workers=40, # Set this to your preference
        num_speakers=2
    )

    model = MySpEmb(
        lr=1e-4,
        finetune_encoder=False,
        emb_dim=256,
        speaker_map_path=SPEAKER_MAP,   # ONLY train map here
    )
    # breakpoint()

    wandb_logger = WandbLogger(
        project="librispeech-speaker-encoder",
        name="ft_wavlm_linear_dualemb_noteacher_tr360_mlpmappingtotr",
        # name='test_run',
        log_model=False,
        save_dir=f"{dir_path}/wandb_logs",
    )

    ckpt = pl.callbacks.ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch}-{val_separation:.3f}",
        dirpath=dir_path
    )

    trainer = pl.Trainer(
        strategy="ddp_find_unused_parameters_true",
        accelerator="gpu",
        devices=[0, 1, 2, 3,4,5,6,7],
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
