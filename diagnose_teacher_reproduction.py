"""
Per-utterance diagnostic: does the student actually reproduce its teacher
targets on the examples it was trained on?

Global clustering accuracy (see eval_metrics.py) conflates several possible
failure modes - it can look fine even if the student has learned to collapse
its two output slots, or if pooling destroys an otherwise-good frame-wise
match. This script isolates those failure modes on a fixed subset of
*training* mixtures:

  1. train_loss_eval        actual training loss (TeacherStudentFrameCosineLoss),
                             forward pass run with model.eval() active
  2. frame_teacher_cos       mean frame-wise cosine to the matched teacher
                             target (= 1 - train_loss_eval, by construction
                             of that loss - reported explicitly as a check
                             that the objective actually measures this)
  3. pooled_teacher_cos      cosine between the *pooled* (TAP-averaged)
                             student embedding and its teacher target, using
                             the SAME per-utterance permutation the frame-wise
                             loss chose - tests whether ordinary pooling
                             preserves what the frame-wise loss optimized
  4. slot_slot_cos           cosine between the two pooled student slots -
                             high values mean the two slots are collapsing
                             towards the same embedding

Run from the wavlm_dual_embedding/ directory (same requirement as
eval_metrics.py / train.py, for the local `dataset`/`loss` imports):

    python diagnose_teacher_reproduction.py [--num-utterances 200] [--no-noise-aug]
"""
import os

# Must be set before numpy/sklearn/torch's BLAS backend is touched - see
# eval_metrics.py for why (256-core host, OpenBLAS compiled for 64 threads).
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "16")

import argparse
import itertools
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append("/home/sidcs.csegpu1/codebase")
from teacher_student_speaker_embedding.model import TeacherStudentSpeakerEmbeddingModel

from dataset import MyLibri2Mix
from loss import LossWraper

DEFAULT_CKPT = "/home/sidcs.csegpu1/model_ckpts/librispeech_cord_landwehr_dualemb_tr360_cos/best-epoch=59-val_separation=0.000-v1.ckpt"
TRAIN_METADATA = "/home/sidcs.csegpu1/datasets/LibriMix/LibriMix/Libriuni_05_08/Libri2Mix_ovl50to80/wav16k/min/metadata/mixture_train-360_mix_clean.csv"
SPEAKER_MAP = "/home/sidcs.csegpu1/datasets/LibriMix/LibriMix/Libriuni_05_08/Libri2Mix_ovl50to80/wav16k/min/metadata/train360_mapping.json"


def load_model(ckpt_path, num_speakers, emb_dim, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["state_dict"]
    filtered = {}
    for k, v in state.items():
        if not k.startswith("model."):
            continue
        if "arcface" in k or "arc_face" in k:
            continue
        filtered[k.replace("model.", "", 1)] = v

    model = TeacherStudentSpeakerEmbeddingModel(num_speakers=num_speakers, emb_dim=emb_dim).to(device)
    model.load_state_dict(filtered, strict=True)
    model.eval()
    assert all(not m.training for m in model.modules())
    return model


@torch.no_grad()
def diagnose_one(model, loss_fn, mix, sources, num_speakers, device):
    """
    mix:     [T]     single mixture waveform
    sources: [K, T]  matching clean sources

    Runs as its own batch of size 1 (no padding) so the diagnostic isn't
    contaminated by zero-padded tail frames from batching utterances of
    different lengths together, unlike the padded batches used at train time.
    """
    mix = mix.unsqueeze(0).to(device)          # [1, T]
    sources = sources.unsqueeze(0).to(device)  # [1, K, T]

    out = model(mix, sources=sources)
    d = out["d"][0]              # [K, E]        teacher targets
    d_hat = out["d_hat"][0]      # [K, E]        pooled student embeddings
    d_hat_t = out["d_hat_t"]     # [1, K, E, T']  frame-wise student embeddings

    # 1) + 2): the actual training loss, run with model.eval() active.
    loss_out = loss_fn(d_hat_t, out["d"])
    train_loss = loss_out["loss"].item()
    frame_teacher_cos = 1.0 - train_loss

    # the exact per-utterance assignment the frame-wise loss picked
    perms = list(itertools.permutations(range(num_speakers)))
    chosen_perm = perms[loss_out["best_perm"][0].item()]

    # 3) pooled teacher-matched cosine, reusing that same assignment
    d_n = F.normalize(d, p=2, dim=-1)
    d_hat_n = F.normalize(d_hat, p=2, dim=-1)
    pooled_cos = torch.stack([
        (d_n[k] * d_hat_n[chosen_perm[k]]).sum() for k in range(num_speakers)
    ]).mean().item()

    # 4) cosine between the two pooled student slots
    slot_cos = F.cosine_similarity(d_hat[0:1], d_hat[1:2], dim=-1).item()

    return {
        "train_loss_eval": train_loss,
        "frame_teacher_cos": frame_teacher_cos,
        "pooled_teacher_cos": pooled_cos,
        "slot_slot_cos": slot_cos,
        "perm": chosen_perm,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--num-utterances", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-speakers", type=int, default=2)
    parser.add_argument("--emb-dim", type=int, default=256)
    parser.add_argument("--no-noise-aug", action="store_true",
                         help="disable the online noise augmentation used at train time "
                              "(default: enabled, to match training conditions)")
    parser.add_argument("--out-csv", default="teacher_reproduction_diagnostics.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = MyLibri2Mix(
        metadata_path=TRAIN_METADATA,
        speaker_map_path=SPEAKER_MAP,
        num_speakers=args.num_speakers,
        split="train",
        add_online_noise=not args.no_noise_aug,
    )
    print(f"Training set has {len(dataset)} mixtures; noise augmentation: {not args.no_noise_aug}")

    indices = sorted(random.sample(range(len(dataset)), min(args.num_utterances, len(dataset))))
    print(f"Diagnosing a fixed subset of {len(indices)} training mixtures (seed={args.seed}).")

    model = load_model(args.ckpt, args.num_speakers, args.emb_dim, device)
    loss_fn = LossWraper(emb_dim=args.emb_dim).to(device)

    rows = []
    for idx in tqdm(indices, desc="Diagnosing"):
        mix, sources, _ = dataset[idx]
        result = diagnose_one(model, loss_fn, mix, sources, args.num_speakers, device)
        result["idx"] = idx
        result["mixture_id"] = dataset.metadata.iloc[idx]["mixture_ID"]
        rows.append(result)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved per-utterance results to {args.out_csv}")

    print(f"\n=== Teacher-reproduction diagnostics ({len(df)} training mixtures) ===")
    for col, desc in [
        ("train_loss_eval", "Actual training loss (eval mode)"),
        ("frame_teacher_cos", "Frame-wise teacher-matched cosine"),
        ("pooled_teacher_cos", "Pooled teacher-matched cosine (same per-utt. assignment)"),
        ("slot_slot_cos", "Cosine between the two pooled student slots"),
    ]:
        vals = df[col]
        print(f"{desc:60s} mean={vals.mean():+.4f}  std={vals.std():.4f}  "
              f"min={vals.min():+.4f}  max={vals.max():+.4f}")

    gap = df["frame_teacher_cos"] - df["pooled_teacher_cos"]
    print(f"\nframe_teacher_cos - pooled_teacher_cos: mean={gap.mean():+.4f}  "
          f"(large positive -> pooling is destroying a frame-wise match that exists)")

    if df["slot_slot_cos"].mean() > 0.8:
        print("\nWARNING: mean cosine between the two student slots is very high - "
              "the two output slots may be collapsing towards the same embedding.")


if __name__ == "__main__":
    main()
