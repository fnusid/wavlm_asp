import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Your codebase
from dataset import LibriMixDataModule            # must yield: (mix, sources, labels, vad_targets[B,2,T])
from model import SpeakerEncoderDualWrapper       # your dual embedding model (ECAPA-based)
from pvad_model import pVAD_module

# # -----------------------------
# # VAD head
# # -----------------------------
# class FrameVADHead(nn.Module):
#     """
#     Input:  [B, D, T]
#     Output: [B, T] logits
#     """
#     def __init__(self, d_in: int, hidden: int = 256):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv1d(d_in, hidden, kernel_size=1),
#             nn.ReLU(),
#             nn.Conv1d(hidden, 1, kernel_size=1),
#         )

#     def forward(self, x):
#         return self.net(x).squeeze(1)  # [B, T]


# -----------------------------
# Wrapper that matches CKPT keys:
#   model.*
#   vad_head1.*
#   vad_head2.*
# -----------------------------
class VADWrapper(nn.Module):
    def __init__(self, emb_dim=256, vad_hidden=256+80):
        super().__init__()
        self.model = SpeakerEncoderDualWrapper(emb_dim=emb_dim)   # MUST be named "model"
        self.pvad = pVAD_module(hidden_dim=vad_hidden)              

    @torch.no_grad()
    def forward_features(self, wav: torch.Tensor):
        """
        """
        if wav.dim() == 3:  # [B,1,T]
            wav = wav.squeeze(1)

        emb = self.model(wav)  # [B, 2, 256]
        return emb

    @torch.no_grad()
    def forward_logits(self, wav: torch.Tensor):
        """
        Returns:
          logits: [B, 2, T_frames]
        """
        emb = self.forward_features(wav)  # [B, 2, 256]
        emb1, emb2 = emb.chunk(2, dim=1)  # [B, 256] each
        if emb1.ndim == 3:
            emb1 = emb1.squeeze(1)
        if emb2.ndim == 3:
            emb2 = emb2.squeeze(1)


        logit1 = self.pvad(wav, emb=emb1)  # [B, T_frames]
        logit2 = self.pvad(wav, emb=emb2)  # [B, T_frames]
        logits = torch.stack([logit1, logit2], dim=1)  # [B, 2, T_frames]

        return logits


# -----------------------------
# PIT utilities
# -----------------------------
def pit_bce_loss(logits: torch.Tensor, targets: torch.Tensor):
    """
    logits:  [B,2,T]
    targets: [B,2,T]
    returns: (loss_scalar, swapped_mask[B], loss_direct[B], loss_swap[B])
    """
    assert logits.shape[:2] == (targets.shape[0], 2)
    assert targets.shape[1] == 2

    # Align time defensively
    T = min(logits.shape[-1], targets.shape[-1])
    logits = logits[..., :T]
    targets = targets[..., :T]

    loss_direct = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    ).mean(dim=(1, 2))  # [B]

    targets_swapped = targets[:, [1, 0], :]
    loss_swap = F.binary_cross_entropy_with_logits(
        logits, targets_swapped, reduction="none"
    ).mean(dim=(1, 2))  # [B]

    swapped = loss_swap < loss_direct
    loss = torch.minimum(loss_direct, loss_swap).mean()

    return loss, swapped, loss_direct, loss_swap, logits, targets


def save_vad_overlay_png(
    wav_1d: torch.Tensor,
    sr: int,
    probs_h0: np.ndarray,
    probs_h1: np.ndarray,
    gt0: np.ndarray,
    gt1: np.ndarray,
    out_path: str,
    hop_s: float = 0.01,   # 10 ms per frame
):
    """
    Simple visualization:
      - waveform
      - head0/head1 prob traces
      - GT as step lines
    """
    wav = wav_1d.detach().cpu().numpy()
    t = np.arange(len(wav)) / sr

    Tf = len(probs_h0)
    tf = np.arange(Tf) * hop_s

    plt.figure(figsize=(14, 6))

    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(t, wav, linewidth=0.6)
    ax1.set_title("Mixture waveform")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("amp")
    ax1.grid(alpha=0.2)

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(tf, probs_h0, label="pred head0", linewidth=2.0,)
    ax2.plot(tf, probs_h1, label="pred head1", linewidth=2.0,)
    ax2.step(tf, gt0[:Tf], where="post", label="gt src0", linewidth=1.4, alpha=0.8, linestyle="--")
    ax2.step(tf, gt1[:Tf], where="post", label="gt src1", linewidth=1.4, alpha=0.8, linestyle="--")
    ax2.set_ylim([-0.05, 1.05])
    ax2.set_title("VAD probs (per head) + GT (per source)")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("prob / label")
    ax2.grid(alpha=0.2)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--speaker_map", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save_png", type=str, default="/home/sidcs/codebase/wavlm_dual_embedding/analysis/pvad/pvad_overlay_debug.png")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) Data
    dm = LibriMixDataModule(
        data_root=args.data_root,
        speaker_map_path=args.speaker_map,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_speakers=2,
    )
    dm.setup("fit")
    dl = dm.val_dataloader()
    batch = next(iter(dl))
    # expected: (mix, sources, labels, vad_targets[B,2,Tf])
    mix, sources, labels, vad_targets = batch
    mix = mix.to(device)
    vad_targets = vad_targets.to(device)

    print("batch shapes:",
          "mix", tuple(mix.shape),
          "sources", tuple(sources.shape),
          "labels", tuple(labels.shape),
          "vad", tuple(vad_targets.shape))

    # 2) Model wrapper
    net = VADWrapper(emb_dim=256, vad_hidden=256+80).to(device)
    net.eval()

    # 3) Load CKPT (NO STRIPPING)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    missing, unexpected = net.load_state_dict(state, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing):
        print("missing sample:", missing[:20])
    if len(unexpected):
        print("unexpected sample:", unexpected[:20])

    # 4) Forward + PIT loss
    with torch.no_grad():
        logits = net.forward_logits(mix)      # [B,2,Tm]
        probs = torch.sigmoid(logits)

    loss, swapped, ld, ls, logits_al, targets_al = pit_bce_loss(logits, vad_targets)

    print(f"[PIT] swapped_rate={(swapped.float().mean().item()):.3f}")
    print(f"[loss] mean={loss.item():.4f}")
    print(f"[loss_direct] mean={ld.mean().item():.4f}  [loss_swap] mean={ls.mean().item():.4f}")

    # print first sample decision
    s0 = 0
    print(f"[sample0] swapped={bool(swapped[s0].item())}  loss_direct={ld[s0].item():.4f}  loss_swap={ls[s0].item():.4f}")

    # 5) Save overlay for sample0
    # Align lengths
    T = min(probs.shape[-1], vad_targets.shape[-1])
    p0 = probs[s0, 0, :T].detach().cpu().numpy()
    p1 = probs[s0, 1, :T].detach().cpu().numpy()
    g0 = vad_targets[s0, 0, :T].detach().cpu().numpy()
    g1 = vad_targets[s0, 1, :T].detach().cpu().numpy()

    # If PIT chose swapped, reorder GT for display so it matches heads
    if swapped[s0]:
        g0, g1 = g1, g0

    # waveform for sample0 (trim to match frames * 160 samples if you want)
    wav0 = mix[s0].detach().cpu()
    save_vad_overlay_png(
        wav_1d=wav0,
        sr=16000,
        probs_h0=p0,
        probs_h1=p1,
        gt0=g0,
        gt1=g1,
        out_path=args.save_png,
        hop_s=0.01,  # 10ms labels
    )
    print(f"[OK] Saved: {args.save_png}")


if __name__ == "__main__":
    main()