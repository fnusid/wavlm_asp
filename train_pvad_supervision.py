import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb

# Your codebase
from dataset import LibriMixDataModule   # yields: mix, source, labels, vad_targets [B,2,T]
from model import SpeakerEncoderDualWrapper
from pvad_model import pVAD_module


# ============================================================
# Helper: load student checkpoint weights
# ============================================================

def strip_dual_model_weights(state_dict):
    """
    Keeps only keys under "model." and strips that prefix.
    Drops teacher / arcface keys if present.
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


# ============================================================
# Utility functions
# ============================================================

def safe_masked_mean(x, mask, eps=1e-8):
    """
    x:    [B,T]
    mask: [B,T]
    """
    return (x * mask).sum() / (mask.sum() + eps)


def compute_frame_metrics(logits, targets, threshold=0.5):
    """
    logits:  [B,2,T]
    targets: [B,2,T]
    """
    probs = torch.sigmoid(logits)
    pred = (probs > threshold).float()

    acc = (pred == targets).float().mean()

    tp = (pred * targets).sum()
    fp = (pred * (1.0 - targets)).sum()
    fn = ((1.0 - pred) * targets).sum()

    f1 = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)

    pred_pos_frac = pred.mean()
    target_pos_frac = targets.mean()
    prob_mean = probs.mean()
    prob_max = probs.max()

    return {
        "acc": acc,
        "f1": f1,
        "pred_pos_frac": pred_pos_frac,
        "target_pos_frac": target_pos_frac,
        "prob_mean": prob_mean,
        "prob_max": prob_max,
    }


def perturb_embedding(e, noise_std=0.01):
    """
    Optional small noise for robustness.
    Keep this small initially.
    """
    if noise_std <= 0:
        return e

    noise = torch.randn_like(e) * noise_std
    return F.normalize(e + noise, dim=-1)


def positive_weighted_bce_with_logits(
    logits,
    targets,
    pos_weight=3.0,
    neg_weight=1.0,
):
    """
    logits:  [B,2,T]
    targets: [B,2,T]

    This prevents the model from solving leakage by suppressing
    all speech probabilities.
    """
    loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )

    weights = neg_weight + (pos_weight - neg_weight) * targets

    return (loss * weights).sum() / (weights.sum() + 1e-8)


def recall_floor_loss(probs, targets, floor=0.65):
    """
    probs:   [B,2,T]
    targets: [B,2,T]

    Penalizes active target frames whose probability is below floor.
    This directly recovers target-speaker recall.
    """
    active = targets
    deficit = F.relu(floor - probs)

    return (deficit * active).sum() / (active.sum() + 1e-8)


def speaker_margin_loss(probs, targets_best, margin=0.25):
    """
    probs:        [B,2,T]
    targets_best: [B,2,T]

    For slot1-only frames:
        p1 should exceed p2 by margin.

    For slot2-only frames:
        p2 should exceed p1 by margin.
    """
    p1 = probs[:, 0, :]
    p2 = probs[:, 1, :]

    t1 = targets_best[:, 0, :]
    t2 = targets_best[:, 1, :]

    s1_only = t1 * (1.0 - t2)
    s2_only = t2 * (1.0 - t1)

    loss_s1 = F.relu(margin - (p1 - p2)) * s1_only
    loss_s2 = F.relu(margin - (p2 - p1)) * s2_only

    denom = s1_only.sum() + s2_only.sum() + 1e-8

    return (loss_s1.sum() + loss_s2.sum()) / denom


# ============================================================
# Lightning module
# ============================================================

class MyVADOnlyRecall(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        emb_dim: int = 256,
        ckpt_student_path: str = "",
        vad_hidden: int = 256 + 80,

        # Small leakage. Previous 0.2 was too suppressive for your diagnostic.
        lambda_leak: float = 0.03,

        # Recall recovery.
        lambda_recall: float = 0.5,
        recall_floor: float = 0.65,

        # Correct-vs-wrong speaker separation.
        lambda_margin: float = 0.3,
        margin_value: float = 0.25,

        # Positive-weighted BCE.
        pos_weight: float = 3.0,
        neg_weight: float = 1.0,

        # Keep off for now.
        lambda_rand: float = 0.0,

        use_embedding_noise: bool = False,
        emb_noise_std: float = 0.01,

        debug_print_every_n_steps: int = 200,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Frozen dual embedding model
        self.model = SpeakerEncoderDualWrapper(emb_dim=emb_dim)

        # Trainable pVAD
        self.pvad = pVAD_module(hidden_dim=vad_hidden)

        # Load pretrained student
        if ckpt_student_path and os.path.isfile(ckpt_student_path):
            ckpt = torch.load(ckpt_student_path, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            state = strip_dual_model_weights(state)

            missing, unexpected = self.model.load_state_dict(state, strict=False)

            print(
                f"[Student CKPT] loaded. "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )
        else:
            raise FileNotFoundError(f"Student checkpoint not found: {ckpt_student_path}")

        # Freeze dual model
        for p in self.model.parameters():
            p.requires_grad = False

        self.model.eval()

        # Test aggregation
        self._test_loss_sum = 0.0
        self._test_acc_sum = 0.0
        self._test_f1_sum = 0.0
        self._test_batches = 0

    # ========================================================
    # Forward helpers
    # ========================================================

    def forward_features(self, wav):
        """
        wav: [B,T] or [B,1,T]
        returns embs: [B,2,256]
        """
        if wav.dim() == 3:
            wav = wav.squeeze(1)

        emb = self.model(wav)
        emb = F.normalize(emb, dim=-1)

        return emb

    @staticmethod
    def _align_T(logits, targets):
        """
        logits:  [B,2,Tm]
        targets: [B,2,Tg]
        """
        Tm = logits.shape[-1]
        Tg = targets.shape[-1]

        if Tm != Tg:
            T = min(Tm, Tg)
            logits = logits[..., :T]
            targets = targets[..., :T]

        return logits, targets

    @staticmethod
    def _align_1d_T(logits, targets):
        """
        logits:  [B,Tm]
        targets: [B,Tg]
        """
        Tm = logits.shape[-1]
        Tg = targets.shape[-1]

        if Tm != Tg:
            T = min(Tm, Tg)
            logits = logits[..., :T]
            targets = targets[..., :T]

        return logits, targets

    # ========================================================
    # Core loss
    # ========================================================

    def compute_loss(self, mix, vad_targets, stage="train"):
        """
        Objective:
          1. Per-sample PIT
          2. Positive-weighted BCE to recover recall
          3. Small leakage suppression
          4. Recall floor loss
          5. Speaker margin loss
          6. Random negative disabled by default
        """
        B = mix.shape[0]
        device = mix.device

        vad_targets = vad_targets.float()

        # ----------------------------------------------------
        # Frozen dual embeddings
        # ----------------------------------------------------
        with torch.no_grad():
            embs = self.forward_features(mix)  # [B,2,256]

        emb1 = embs[:, 0, :]
        emb2 = embs[:, 1, :]

        if self.training and self.hparams.use_embedding_noise:
            emb1 = perturb_embedding(emb1, self.hparams.emb_noise_std)
            emb2 = perturb_embedding(emb2, self.hparams.emb_noise_std)

        # ----------------------------------------------------
        # pVAD logits
        # ----------------------------------------------------
        logit1 = self.pvad(mix, emb1)  # [B,Tm]
        logit2 = self.pvad(mix, emb2)  # [B,Tm]

        logits = torch.stack([logit1, logit2], dim=1)  # [B,2,Tm]
        logits, vad_targets = self._align_T(logits, vad_targets)

        # ----------------------------------------------------
        # Per-sample PIT assignment
        # ----------------------------------------------------
        loss_direct_per = F.binary_cross_entropy_with_logits(
            logits,
            vad_targets,
            reduction="none",
        ).mean(dim=(1, 2))  # [B]

        vad_swapped = vad_targets[:, [1, 0], :]

        loss_swap_per = F.binary_cross_entropy_with_logits(
            logits,
            vad_swapped,
            reduction="none",
        ).mean(dim=(1, 2))  # [B]

        use_swap = loss_swap_per < loss_direct_per  # [B]

        targets_best = vad_targets.clone()

        if use_swap.any():
            targets_best[use_swap] = vad_targets[use_swap][:, [1, 0], :]

        swap_rate = use_swap.float().mean()

        # ----------------------------------------------------
        # Main loss: positive-weighted BCE after PIT alignment
        # This is the main recall recovery change.
        # ----------------------------------------------------
        main_loss = positive_weighted_bce_with_logits(
            logits,
            targets_best,
            pos_weight=self.hparams.pos_weight,
            neg_weight=self.hparams.neg_weight,
        )

        # ----------------------------------------------------
        # Probabilities and region masks
        # ----------------------------------------------------
        probs = torch.sigmoid(logits)

        p1 = probs[:, 0, :]
        p2 = probs[:, 1, :]

        t1 = targets_best[:, 0, :]
        t2 = targets_best[:, 1, :]

        s1_only = t1 * (1.0 - t2)
        s2_only = t2 * (1.0 - t1)
        overlap = t1 * t2
        silence = (1.0 - t1) * (1.0 - t2)

        # ----------------------------------------------------
        # Leakage loss: keep small
        # ----------------------------------------------------
        leak_1_on_2 = safe_masked_mean(p1, s2_only)
        leak_2_on_1 = safe_masked_mean(p2, s1_only)

        loss_leak = leak_1_on_2 + leak_2_on_1

        # ----------------------------------------------------
        # Recall floor loss: force target active frames to fire
        # ----------------------------------------------------
        loss_recall = recall_floor_loss(
            probs=probs,
            targets=targets_best,
            floor=self.hparams.recall_floor,
        )

        # ----------------------------------------------------
        # Margin loss: correct speaker should beat wrong speaker
        # in single-speaker regions
        # ----------------------------------------------------
        loss_margin = speaker_margin_loss(
            probs=probs,
            targets_best=targets_best,
            margin=self.hparams.margin_value,
        )

        # ----------------------------------------------------
        # Optional random negative, disabled by default
        # ----------------------------------------------------
        loss_rand = torch.tensor(0.0, device=device)

        if self.hparams.lambda_rand > 0.0:
            if B > 1:
                perm = torch.randperm(B, device=device)

                rand_emb = emb1[perm].clone()

                use_emb2 = torch.rand(B, device=device) > 0.5
                rand_emb[use_emb2] = emb2[perm][use_emb2]
            else:
                rand_emb = emb2.detach()

            logit_rand = self.pvad(mix, rand_emb)

            vad_any = torch.clamp(t1 + t2, 0.0, 1.0)
            logit_rand, vad_any = self._align_1d_T(logit_rand, vad_any)

            zeros = torch.zeros_like(logit_rand)

            rand_loss_raw = F.binary_cross_entropy_with_logits(
                logit_rand,
                zeros,
                reduction="none",
            )

            rand_weights = 1.0 + 2.0 * vad_any

            loss_rand = (rand_loss_raw * rand_weights).sum() / (
                rand_weights.sum() + 1e-8
            )

        # ----------------------------------------------------
        # Total objective
        # ----------------------------------------------------
        loss = (
            main_loss
            + self.hparams.lambda_leak * loss_leak
            + self.hparams.lambda_recall * loss_recall
            + self.hparams.lambda_margin * loss_margin
            + self.hparams.lambda_rand * loss_rand
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------
        frame_metrics = compute_frame_metrics(logits, targets_best)

        s1_correct = safe_masked_mean(p1, s1_only)
        s1_wrong = safe_masked_mean(p2, s1_only)

        s2_correct = safe_masked_mean(p2, s2_only)
        s2_wrong = safe_masked_mean(p1, s2_only)

        s1_margin = s1_correct - s1_wrong
        s2_margin = s2_correct - s2_wrong

        silence_prob = safe_masked_mean(torch.maximum(p1, p2), silence)

        overlap_p1 = safe_masked_mean(p1, overlap)
        overlap_p2 = safe_masked_mean(p2, overlap)

        target_pos_frac = vad_targets.mean()
        target_best_pos_frac = targets_best.mean()
        pred_pos_frac = frame_metrics["pred_pos_frac"]

        metrics = {
            "loss": loss,
            "main_loss": main_loss.detach(),
            "loss_leak": loss_leak.detach(),
            "loss_recall": loss_recall.detach(),
            "loss_margin": loss_margin.detach(),
            "loss_rand": loss_rand.detach(),

            "swap_rate": swap_rate.detach(),

            "vad_acc": frame_metrics["acc"].detach(),
            "vad_f1": frame_metrics["f1"].detach(),
            "pred_pos_frac": pred_pos_frac.detach(),
            "target_pos_frac": target_pos_frac.detach(),
            "target_best_pos_frac": target_best_pos_frac.detach(),
            "prob_mean": frame_metrics["prob_mean"].detach(),
            "prob_max": frame_metrics["prob_max"].detach(),

            "leak_1_on_2": leak_1_on_2.detach(),
            "leak_2_on_1": leak_2_on_1.detach(),

            "s1_correct": s1_correct.detach(),
            "s1_wrong": s1_wrong.detach(),
            "s2_correct": s2_correct.detach(),
            "s2_wrong": s2_wrong.detach(),
            "s1_margin": s1_margin.detach(),
            "s2_margin": s2_margin.detach(),

            "silence_prob": silence_prob.detach(),
            "overlap_p1": overlap_p1.detach(),
            "overlap_p2": overlap_p2.detach(),
        }

        # ----------------------------------------------------
        # Debug prints
        # ----------------------------------------------------
        if (
            stage in ["train", "val"]
            and self.global_step % self.hparams.debug_print_every_n_steps == 0
            and self.trainer.is_global_zero
        ):
            with torch.no_grad():
                print(
                    f"\n[{stage.upper()} DEBUG step={self.global_step}] "
                    f"loss={loss.detach().item():.6f} "
                    f"main={main_loss.detach().item():.6f} "
                    f"leak={loss_leak.detach().item():.6f} "
                    f"recall={loss_recall.detach().item():.6f} "
                    f"margin_loss={loss_margin.detach().item():.6f} "
                    f"rand={loss_rand.detach().item():.6f} "
                    f"f1={frame_metrics['f1'].detach().item():.6f} "
                    f"acc={frame_metrics['acc'].detach().item():.6f} "
                    f"pred_pos={pred_pos_frac.detach().item():.6f} "
                    f"target_pos={target_pos_frac.detach().item():.6f} "
                    f"prob_mean={frame_metrics['prob_mean'].detach().item():.6f} "
                    f"prob_max={frame_metrics['prob_max'].detach().item():.6f} "
                    f"swap_rate={swap_rate.detach().item():.3f} "
                    f"s1_correct={s1_correct.detach().item():.6f} "
                    f"s1_wrong={s1_wrong.detach().item():.6f} "
                    f"s2_correct={s2_correct.detach().item():.6f} "
                    f"s2_wrong={s2_wrong.detach().item():.6f} "
                    f"s1_margin={s1_margin.detach().item():.6f} "
                    f"s2_margin={s2_margin.detach().item():.6f}"
                )

        return loss, metrics

    # ========================================================
    # Lightning steps
    # ========================================================

    def training_step(self, batch, batch_idx):
        if len(batch) != 4:
            raise ValueError("Batch must be (mix, source, labels, vad_targets).")

        mix, _, _, vad_targets = batch

        loss, metrics = self.compute_loss(
            mix=mix,
            vad_targets=vad_targets,
            stage="train",
        )

        bs = mix.shape[0]

        self.log("train/vad_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=bs)
        self.log("train/main_loss", metrics["main_loss"], on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=bs)
        self.log("train/loss_leak", metrics["loss_leak"], on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=bs)
        self.log("train/loss_recall", metrics["loss_recall"], on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=bs)
        self.log("train/loss_margin", metrics["loss_margin"], on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=bs)
        self.log("train/loss_rand", metrics["loss_rand"], on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=bs)

        self.log("train/vad_acc", metrics["vad_acc"], on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=bs)
        self.log("train/vad_f1", metrics["vad_f1"], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=bs)

        self.log("train/pred_pos_frac", metrics["pred_pos_frac"], on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=bs)
        self.log("train/target_pos_frac", metrics["target_pos_frac"], on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=bs)
        self.log("train/prob_mean", metrics["prob_mean"], on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=bs)
        self.log("train/prob_max", metrics["prob_max"], on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=bs)

        self.log("train/swap_rate", metrics["swap_rate"], on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=bs)

        self.log("train/s1_correct", metrics["s1_correct"], on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=bs)
        self.log("train/s1_wrong", metrics["s1_wrong"], on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=bs)
        self.log("train/s2_correct", metrics["s2_correct"], on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=bs)
        self.log("train/s2_wrong", metrics["s2_wrong"], on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=bs)

        self.log("train/s1_margin", metrics["s1_margin"], on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=bs)
        self.log("train/s2_margin", metrics["s2_margin"], on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=bs)

        self.log("train/leak_1_on_2", metrics["leak_1_on_2"], on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=bs)
        self.log("train/leak_2_on_1", metrics["leak_2_on_1"], on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=bs)

        return loss

    def validation_step(self, batch, batch_idx):
        if len(batch) != 4:
            raise ValueError("Val batch must be (mix, source, labels, vad_targets).")

        mix, _, _, vad_targets = batch

        loss, metrics = self.compute_loss(
            mix=mix,
            vad_targets=vad_targets,
            stage="val",
        )

        bs = mix.shape[0]

        self.log("val/vad_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=bs)
        self.log("val/main_loss", metrics["main_loss"], on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=bs)
        self.log("val/loss_leak", metrics["loss_leak"], on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=bs)
        self.log("val/loss_recall", metrics["loss_recall"], on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=bs)
        self.log("val/loss_margin", metrics["loss_margin"], on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=bs)
        self.log("val/loss_rand", metrics["loss_rand"], on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=bs)

        self.log("val/vad_acc", metrics["vad_acc"], on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=bs)
        self.log("val/vad_f1", metrics["vad_f1"], on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=bs)

        self.log("val/pred_pos_frac", metrics["pred_pos_frac"], on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=bs)
        self.log("val/target_pos_frac", metrics["target_pos_frac"], on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=bs)
        self.log("val/prob_mean", metrics["prob_mean"], on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=bs)
        self.log("val/prob_max", metrics["prob_max"], on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=bs)

        self.log("val/swap_rate", metrics["swap_rate"], on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=bs)

        self.log("val/s1_correct", metrics["s1_correct"], on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=bs)
        self.log("val/s1_wrong", metrics["s1_wrong"], on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=bs)
        self.log("val/s2_correct", metrics["s2_correct"], on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=bs)
        self.log("val/s2_wrong", metrics["s2_wrong"], on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=bs)

        self.log("val/s1_margin", metrics["s1_margin"], on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=bs)
        self.log("val/s2_margin", metrics["s2_margin"], on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=bs)

        self.log("val/leak_1_on_2", metrics["leak_1_on_2"], on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=bs)
        self.log("val/leak_2_on_1", metrics["leak_2_on_1"], on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=bs)

        self.log("val/silence_prob", metrics["silence_prob"], on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=bs)
        self.log("val/overlap_p1", metrics["overlap_p1"], on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=bs)
        self.log("val/overlap_p2", metrics["overlap_p2"], on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=bs)

        return {"val_loss": loss}

    # ========================================================
    # Test
    # ========================================================

    def on_test_epoch_start(self):
        self._test_loss_sum = 0.0
        self._test_acc_sum = 0.0
        self._test_f1_sum = 0.0
        self._test_batches = 0

    def test_step(self, batch, batch_idx):
        if len(batch) != 4:
            raise ValueError("Test batch must be (mix, source, labels, vad_targets).")

        mix, _, _, vad_targets = batch

        loss, metrics = self.compute_loss(
            mix=mix,
            vad_targets=vad_targets,
            stage="test",
        )

        bs = mix.shape[0]

        self.log("test/vad_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=bs)
        self.log("test/vad_acc", metrics["vad_acc"], on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=bs)
        self.log("test/vad_f1", metrics["vad_f1"], on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=bs)
        self.log("test/s1_margin", metrics["s1_margin"], on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=bs)
        self.log("test/s2_margin", metrics["s2_margin"], on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=bs)

        self._test_loss_sum += float(loss.detach().cpu())
        self._test_acc_sum += float(metrics["vad_acc"].detach().cpu())
        self._test_f1_sum += float(metrics["vad_f1"].detach().cpu())
        self._test_batches += 1

        return {"test_loss": loss}

    def on_test_epoch_end(self):
        if self._test_batches > 0:
            mean_loss = self._test_loss_sum / self._test_batches
            mean_acc = self._test_acc_sum / self._test_batches
            mean_f1 = self._test_f1_sum / self._test_batches

            print(
                f"[TEST] mean_loss={mean_loss:.4f} "
                f"mean_acc={mean_acc:.4f} "
                f"mean_f1={mean_f1:.4f}"
            )

    # ========================================================
    # Optimizer
    # ========================================================

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.pvad.parameters(),
            lr=self.hparams.lr,
            weight_decay=0.01,
        )

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
                "monitor": "val/vad_loss",
                "interval": "epoch",
            },
        }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    DATA_ROOT = "/home/sidcs/datasets/LibriMix/LibriMix"

    SPEAKER_MAP = (
        "/home/sidcs/datasets/LibriMix/LibriMix/"
        "Libriuni_05_08/Libri2Mix_ovl50to80/wav16k/min/"
        "metadata/train360_mapping.json"
    )

    STUDENT_CKPT = (
        "/home/sidcs/model_ckpts/ECAPA_UNMIX_3072_teacher_ECAPA/"
        "best-epoch=87-val_separation=0.000.ckpt"
    )

    # Resume from the safer leakage checkpoint if you want continuation,
    # or from the original pVAD checkpoint if you want to recover more speech recall.
    #
    # Recommended first:
    #   use original pVAD checkpoint, because the leakage checkpoint became conservative.
    RESUME_PVAD_CKPT = (
        "/home/sidcs/model_ckpts/pvad_emb/"
        "best-epoch=47-val_vad_loss=0.0000.ckpt"
    )

    SAVE_DIR = "/home/sidcs/model_ckpts/pvad_emb_recall_margin"

    os.makedirs(SAVE_DIR, exist_ok=True)

    dm = LibriMixDataModule(
        data_root=DATA_ROOT,
        speaker_map_path=SPEAKER_MAP,
        batch_size=32 * 16,
        num_workers=20,
        num_speakers=2,
    )

    model = MyVADOnlyRecall(
        lr=1e-4,
        emb_dim=256,
        ckpt_student_path=STUDENT_CKPT,
        vad_hidden=256 + 80,

        # Keep leakage small now.
        lambda_leak=0.03,

        # Recover speech confidence.
        lambda_recall=0.5,
        recall_floor=0.65,

        # Encourage correct speaker > wrong speaker.
        lambda_margin=0.3,
        margin_value=0.25,

        # Positive frames matter more.
        pos_weight=3.0,
        neg_weight=1.0,

        # Still off.
        lambda_rand=0.0,

        use_embedding_noise=False,
        emb_noise_std=0.01,

        debug_print_every_n_steps=200,
    )

    wandb_logger = WandbLogger(
        project="librispeech-vad-head",
        name="pvad_emb_recall_margin_per_sample_pit",
        log_model=False,
        save_dir=os.path.join(SAVE_DIR, "wandb_logs"),
    )

    # Use F1 as primary checkpoint, but watch s2_correct and s2_margin manually.
    ckpt_cb = pl.callbacks.ModelCheckpoint(
        monitor="val/vad_f1",
        mode="max",
        save_top_k=3,
        filename="best-{epoch}-{val_vad_f1:.4f}-{val_vad_loss:.6f}",
        dirpath=SAVE_DIR,
    )

    trainer = pl.Trainer(
        strategy="ddp_find_unused_parameters_true",
        accelerator="gpu",
        devices=[0, 1, 2, 3, 4, 5, 6],
        max_epochs=100,
        logger=wandb_logger,
        callbacks=[ckpt_cb],
        gradient_clip_val=5.0,
        enable_checkpointing=True,
        num_sanity_val_steps=0,
        log_every_n_steps=10,
    )

    trainer.fit(
        model,
        datamodule=dm,
        ckpt_path=RESUME_PVAD_CKPT,
    )

    wandb.finish()