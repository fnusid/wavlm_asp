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
# from model import SpeakerEncoderDualWrapper   
from loss import LossWraper
from metrics import EmbeddingMetrics
import wandb
import sys
sys.path.append("/home/sidcs.csegpu1/codebase")
from teacher_student_speaker_embedding.model import TeacherStudentSpeakerEmbeddingModel
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

def load_and_verify_teacher(model, checkpoint_path, prefix=""):
    """
    model: TeacherStudentSpeakerEmbeddingModel

    prefix:
      ""         if checkpoint keys are fbank.*, resnet.*, ...
      "teacher." if keys are teacher.fbank.*, teacher.resnet.*, ...
      "model."   if keys are model.fbank.*, model.resnet.*, ...

    Use the actual prefix in YOUR checkpoint.
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    state = checkpoint.get("state_dict", checkpoint)
   
    teacher_state = {
        key[len(prefix):]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }

    if not teacher_state:
        raise RuntimeError(f"No checkpoint keys match prefix {prefix!r}")

    # Fails on missing/extra keys instead of silently accepting a partial load.
    model.teacher.load_state_dict(teacher_state, strict=True)
    model.teacher.requires_grad_(False)
    model.teacher.eval()

    # Verify parameters AND buffers, including BatchNorm running statistics.
    for name, actual in model.teacher.state_dict().items():
        expected = teacher_state[name].to(
            device=actual.device, dtype=actual.dtype
        )
        torch.testing.assert_close(
            actual, expected, rtol=0, atol=0,
            msg=f"Teacher checkpoint mismatch: {name}",
        )

    print(f"Verified teacher checkpoint: {checkpoint_path}")

class MySpEmb(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        finetune_encoder: bool = False,
        emb_dim: int = 256,
        speaker_map_path: str = "/home/sidcs.csegpu1/datasets/LibriMix/LibriMix/Libriuni_03_08/Libri2Mix_ovl30to80/wav16k/min/metadata/train360_mapping.json",
    ):
        super().__init__()
        self.save_hyperparameters()


        # -----------------------------
        # 2. ArcFace classification head
        # -----------------------------
        with open(speaker_map_path, "r") as f:
            speaker_map = json.load(f)

        self.cosine_loss = LossWraper(
            emb_dim=emb_dim,
        )
        self.model = TeacherStudentSpeakerEmbeddingModel(num_speakers=2, emb_dim=emb_dim).to("cuda")
        teacher_ckpt_path = "/home/sidcs.csegpu1/model_ckpts/cord_landwehr_arcface_tr460/best-epoch=58.ckpt"
        load_and_verify_teacher(
            self.model,
            checkpoint_path=teacher_ckpt_path,
            prefix="model.",  # Change to your actual checkpoint prefix.
        )

        self.model.train()

        assert not self.model.teacher.training
        assert all(not p.requires_grad for p in self.model.teacher.parameters())

        # ckpt = torch.load(teacher_ckpt_path, map_location="cuda")
        # state_dict = ckpt.get("state_dict", ckpt)

        # new_state = {}
        # for k, v in state_dict.items():
        #     if k.startswith("model."):
        #         new_state[k.replace("model.", "", 1)] = v
        #     else:
        #         new_state[k] = v
        # self.model.teacher.load_state_dict(new_state, strict=False)
        # self.model.teacher.eval()

        # -----------------------------
        # 3. Embedding metrics (for validation)
        # -----------------------------
        self.metrics = EmbeddingMetrics(device="cuda")  # will overwrite device at runtime

    def forward(self, wav, sources=None):
        """
        wav: [B, T] (or [B, 1, T])
        sources: [B, K, T] optional clean per-speaker sources, needed to
                 compute the teacher targets during training
        returns a dict with:
            d_hat:   [B, K, emb_dim]       student utterance embeddings
            d_hat_t: [B, K, emb_dim, T']    student frame-wise embeddings
            d:       [B, K, emb_dim]       teacher target embeddings
                     (only present if `sources` is given)
        """
        return self.model(wav, sources=sources)

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
        emb_dict = self.forward(mix, sources=source)
        '''
        emb_dict:
            d_hat:   [B, K, emb_dim]       student utterance embeddings
            d_hat_t: [B, K, emb_dim, T']    student frame-wise embeddings
            d:       [B, K, emb_dim]       teacher target embeddings
        '''

        loss_out = self.cosine_loss(emb_dict["d_hat_t"], emb_dict["d"])
        loss = loss_out["loss"]

        if batch_idx == 0 and self.current_epoch == 0:
            with torch.no_grad():
                d_hat, d = emb_dict["d_hat"], emb_dict["d"]
                cos_gt = F.cosine_similarity(d[:, 0, :], d[:, 1, :], dim=-1).mean()
                cos_pred = F.cosine_similarity(d_hat[:, 0, :], d_hat[:, 1, :], dim=-1).mean()
                print("Mean cos(teacher_1, teacher_2) =", cos_gt.item())
                print("Mean cos(student_1, student_2) =", cos_pred.item())

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
        Computes the same teacher-student PIT loss as training_step for a
        val/loss curve. The clustering metrics are done in
        validation_epoch_end on the entire validation set.
        """
        mix, source, labels = batch
        emb_dict = self.forward(mix, sources=source)
        emb = emb_dict["d_hat"]                    # [B, 2, emb_dim]
        #labels : [B, 2]

        loss_out = self.cosine_loss(emb_dict["d_hat_t"], emb_dict["d"])

        self.log(
            "val/loss",
            loss_out["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=mix.shape[0],
        )

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
        name="cord_landwehr_dualemb_tr360_cos",
        # name='test_run',
        log_model=False,
        save_dir="/home/sidcs.csegpu1/model_ckpts/cord_landwehr_dualemb_tr360_cos/wandb_logs",
    )

    ckpt = pl.callbacks.ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=-1,
        filename="best-{epoch}-{val_separation:.3f}",
        dirpath="/home/sidcs.csegpu1/model_ckpts/librispeech_cord_landwehr_dualemb_tr360_cos/"
    )

    trainer = pl.Trainer(
        strategy="ddp_find_unused_parameters_true",
        accelerator="gpu",
        devices=[0, 1, 2, 3, 4, 5, 6, 7],
        max_epochs=60,
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
