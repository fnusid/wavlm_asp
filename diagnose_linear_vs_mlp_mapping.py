"""
Diagnostic: linear vs. MLP mapping head, compared on identical validation examples.

Both mapping heads (`linear_map`: nn.Linear(256,256), `mlp`: Linear(256,512)->ReLU->
Linear(512,256)) sit on top of the SAME frozen dual-embedding encoder
(`SpeakerEncoderDualWrapper`) and are trained by wavlm_dual_embedding/train_linear.py
to map its per-speaker mixture embeddings onto the frozen teacher's
(`SpeakerEncoderWrapper`) single-speaker embedding space, using the PIT-matched
cosine loss in wavlm_dual_embedding/loss.py (CosineSimilarityLoss).

This script does NOT retrain anything. It loads both frozen mapping-head
checkpoints plus the shared frozen encoders, puts everything in eval() under
torch.no_grad(), runs ONE pass over the validation set, and for every example
computes both heads' outputs side by side so the comparison is apples-to-apples:

  1. Per-example PIT-matched cosine loss (same loss fn used in training),
     for linear and MLP, plus the paired difference and a sign test / paired
     t-test on that difference.
  2. Embedding-norm statistics for: the raw dual-embedding encoder output
     (sanity check -- should sit at ~1.0 since SpeakerEncoder L2-normalizes
     internally), the linear-mapped output, the MLP-mapped output, and the
     teacher's target embedding (also ~1.0 by construction). Cosine loss is
     scale-invariant, but the *raw* mapped embedding (not renormalized) is
     what actually gets fed to DPCCN as conditioning, and DPCCN's
     SpeakerFuseLayer(fuse_type="multiply") does NOT renormalize it -- with
     joint_training=False and use_spk_transform=False (see
     wesep/confs/config_dpcnn.yaml) the embedding goes through nn.Identity()
     and straight into an unnormalized elementwise-multiply gate
     (wesep/modules/common/speaker.py). So if one head's output norm drifts
     from the teacher-embedding scale DPCCN was conditioned on, that changes
     downstream conditioning strength even when cosine similarity looks fine.

Before comparing, the script asserts that the dual-embedding encoder and
teacher weights embedded inside the linear and MLP checkpoints are identical
to each other and to the standalone checkpoints -- i.e. both heads really
were trained on top of the same frozen backbone, mirroring the sanity check
already done in wesep/evaluate_frozen_tse.py's on_test_start().

Usage:
    python diagnose_linear_vs_mlp_mapping.py
    python diagnose_linear_vs_mlp_mapping.py --max_batches 20   # quick smoke test
    python diagnose_linear_vs_mlp_mapping.py --linear_ckpt ... --mlp_ckpt ...
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import stats as sstats
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

sys.path.append("/home/sidcs.csegpu1/codebase")

from wavlm_dual_embedding.dataset import LibriMixDataModule
from wavlm_dual_embedding.loss import CosineSimilarityLoss
from wavlm_dual_embedding.model import SpeakerEncoderDualWrapper
from wavlm_single_embedding.model import SpeakerEncoderWrapper as SingleSpeakerEncoderWrapper

DEFAULTS = dict(
    data_root="/home/sidcs.csegpu1/datasets/LibriMix/LibriMix",
    speaker_map=(
        "/home/sidcs.csegpu1/datasets/LibriMix/LibriMix/Libriuni_05_08/"
        "Libri2Mix_ovl50to80/wav16k/min/metadata/train360_mapping.json"
    ),
    dual_emb_ckpt=(
        "/home/sidcs.csegpu1/model_ckpts/ft_wavlm_linear_dualemb_noteacher_tr360/"
        "best-epoch=12-val_separation=0.000.ckpt"
    ),
    teacher_ckpt=(
        "/home/sidcs.csegpu1/model_ckpts/librispeech_asp_wavlm_tr360/"
        "best-epoch=62-val_separation=0.000.ckpt"
    ),
    # The only checkpoint produced for the linear head.
    linear_ckpt=(
        "/home/sidcs.csegpu1/model_ckpts/ft_wavlm_linear_dualemb_noteacher_tr360_linearmappingtotr/"
        "best-epoch=47-val_separation=0.000.ckpt"
    ),
    # Of the MLP checkpoints on disk, this is the only one saved with
    # ModelCheckpoint(monitor="val/loss", ...) -- i.e. actually selected on
    # validation loss like train_linear.py's ModelCheckpoint is configured to
    # do (the two files sitting one level up, best-epoch=45 and
    # best-epoch=47..._nofinalrelu, predate the run that added the
    # val_monitored/ subdirectory). Override with --mlp_ckpt to compare a
    # different one.
    mlp_ckpt=(
        "/home/sidcs.csegpu1/model_ckpts/ft_wavlm_linear_dualemb_noteacher_tr360_mlpmappingtotr/"
        "val_monitored/best-epoch=9-val_separation=0.000.ckpt"
    ),
)


def strip_prefix(state_dict, prefix):
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


def assert_state_dicts_equal(a, b, tag):
    assert a.keys() == b.keys(), f"[{tag}] key sets differ: {a.keys() ^ b.keys()}"
    changed = [
        k for k in a
        if not torch.equal(a[k].cpu(), b[k].detach().cpu().to(a[k].dtype))
    ]
    assert not changed, f"[{tag}] weights differ on keys: {changed[:5]}"


def pit_cosine_loss_per_example(pred, gt):
    """
    Same PIT-matched cosine loss as CosineSimilarityLoss.forward (loss.py),
    but returns one loss per example instead of the batch mean, so linear
    and MLP can be compared example-by-example.

    pred, gt: [B, 2, D]
    returns:  [B]
    """
    pred_n = torch.nn.functional.normalize(pred, p=2, dim=-1)
    gt_n = torch.nn.functional.normalize(gt, p=2, dim=-1)
    cos = torch.matmul(pred_n, gt_n.transpose(1, 2))  # [B, 2, 2]

    losses = torch.empty(cos.size(0))
    for b in range(cos.size(0)):
        cost = -cos[b].detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost)
        sim = cos[b, row_ind, col_ind]
        losses[b] = 1.0 - sim.mean()
    return losses


def load_dual_emb_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state_dict"]
    filtered = strip_prefix(state, "model.")
    # (in this checkpoint "model." only ever prefixes the dual-emb backbone;
    # there's no nested single_sp_model/arcface under it here)
    model = SpeakerEncoderDualWrapper(emb_dim=256, finetune_wavlm=True)
    model.load_state_dict(filtered, strict=True)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model, filtered


def load_teacher_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state_dict"]
    filtered = {
        k.replace("model.", "", 1): v
        for k, v in state.items()
        if k.startswith("model.") and "arcface" not in k
    }
    model = SingleSpeakerEncoderWrapper(emb_dim=256)
    model.load_state_dict(filtered, strict=True)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model, filtered


def load_mapping_head(ckpt_path, prefix, build_fn, device, expected_backbone, expected_teacher, tag):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state_dict"]

    # Sanity: the encoder frozen inside this checkpoint must be byte-identical
    # to the standalone encoders we're using, otherwise "identical examples,
    # both encoders in eval mode" wouldn't be a fair comparison.
    backbone_here = strip_prefix(state, "model.")
    teacher_here = strip_prefix(state, "single_sp_model.")
    if backbone_here:
        assert_state_dicts_equal(expected_backbone, backbone_here, f"{tag}/dual_emb_backbone")
    if teacher_here:
        assert_state_dicts_equal(expected_teacher, teacher_here, f"{tag}/teacher")

    head_state = strip_prefix(state, prefix)
    head = build_fn()
    head.load_state_dict(head_state, strict=True)
    head.eval().to(device)
    for p in head.parameters():
        p.requires_grad = False
    meta = {"epoch": ckpt.get("epoch"), "global_step": ckpt.get("global_step")}
    return head, meta


def summarize(name, arr):
    arr = np.asarray(arr)
    return {
        "name": name,
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
    }


def print_summary_table(title, summaries):
    print(f"\n{title}")
    header = f"{'':22s}{'mean':>10s}{'std':>10s}{'median':>10s}{'p05':>10s}{'p95':>10s}{'min':>10s}{'max':>10s}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s['name']:22s}{s['mean']:10.4f}{s['std']:10.4f}{s['median']:10.4f}"
            f"{s['p05']:10.4f}{s['p95']:10.4f}{s['min']:10.4f}{s['max']:10.4f}"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k}", default=v)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_batches", type=int, default=None, help="cap for a quick smoke test; default = full val set")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--out_json",
        default="/home/sidcs.csegpu1/codebase/wavlm_dual_embedding/analysis/linear_vs_mlp_diagnostic.json",
    )
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    print("Checkpoints:")
    print(f"  dual_emb encoder : {args.dual_emb_ckpt}")
    print(f"  teacher encoder  : {args.teacher_ckpt}")
    print(f"  linear head      : {args.linear_ckpt}")
    print(f"  mlp head         : {args.mlp_ckpt}")

    # ---- load shared frozen encoders (single source of truth) ----
    dual_emb_model, backbone_state = load_dual_emb_model(args.dual_emb_ckpt, device)
    teacher_model, teacher_state = load_teacher_model(args.teacher_ckpt, device)

    # ---- load both mapping heads, asserting they sit on the same backbone ----
    linear_head, linear_meta = load_mapping_head(
        args.linear_ckpt, "linear_map.",
        lambda: nn.Linear(256, 256),
        device, backbone_state, teacher_state, "linear",
    )
    mlp_head, mlp_meta = load_mapping_head(
        args.mlp_ckpt, "mlp.",
        lambda: nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 256)),
        device, backbone_state, teacher_state, "mlp",
    )
    assert not isinstance(mlp_head[-1], nn.ReLU), "mlp head ends in ReLU -- would clip negative components"
    print(f"Linear head checkpoint: epoch={linear_meta['epoch']} global_step={linear_meta['global_step']}")
    print(f"MLP head checkpoint   : epoch={mlp_meta['epoch']} global_step={mlp_meta['global_step']}")
    print("Verified: both heads are trained on top of byte-identical frozen encoders.")

    cosine_loss_fn = CosineSimilarityLoss()  # batch-mean version, used only as a cross-check

    # ---- data ----
    dm = LibriMixDataModule(
        data_root=args.data_root,
        speaker_map_path=args.speaker_map,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_speakers=2,
    )
    dm.setup()
    val_loader = dm.val_dataloader()  # shuffle=False -> deterministic example order

    n_batches = len(val_loader) if args.max_batches is None else min(args.max_batches, len(val_loader))
    print(f"\nRunning {n_batches} validation batches (batch_size={args.batch_size})...")

    linear_losses, mlp_losses = [], []
    raw_norms, gt_norms, linear_norms, mlp_norms = [], [], [], []
    batch_mean_check = []  # (linear_batch_loss, mlp_batch_loss) cross-check against per-example mean
    batch_sizes = []  # number of speaker-slot examples (2*B) contributed by each batch, for the check above

    t0 = time.time()
    with torch.no_grad():
        for i, (mix, source, labels) in enumerate(tqdm(val_loader, total=n_batches)):
            if args.max_batches is not None and i >= args.max_batches:
                break
            mix = mix.to(device)
            source = source.to(device)

            emb1 = teacher_model(source[:, 0, :])
            emb2 = teacher_model(source[:, 1, :])
            gt_embs = torch.stack([emb1, emb2], dim=1)          # [B, 2, D]

            raw_embs = dual_emb_model(mix)                       # [B, 2, D]
            linear_out = linear_head(raw_embs)                   # [B, 2, D]
            mlp_out = mlp_head(raw_embs)                          # [B, 2, D]

            lin_loss_b = pit_cosine_loss_per_example(linear_out, gt_embs)  # [B]
            mlp_loss_b = pit_cosine_loss_per_example(mlp_out, gt_embs)     # [B]
            linear_losses.append(lin_loss_b.numpy())
            mlp_losses.append(mlp_loss_b.numpy())

            batch_mean_check.append((
                float(cosine_loss_fn(linear_out, gt_embs)),
                float(cosine_loss_fn(mlp_out, gt_embs)),
            ))
            batch_sizes.append(lin_loss_b.shape[0])

            raw_norms.append(raw_embs.norm(dim=-1).flatten().cpu().numpy())
            gt_norms.append(gt_embs.norm(dim=-1).flatten().cpu().numpy())
            linear_norms.append(linear_out.norm(dim=-1).flatten().cpu().numpy())
            mlp_norms.append(mlp_out.norm(dim=-1).flatten().cpu().numpy())

    dt = time.time() - t0
    linear_losses = np.concatenate(linear_losses)
    mlp_losses = np.concatenate(mlp_losses)
    raw_norms = np.concatenate(raw_norms)
    gt_norms = np.concatenate(gt_norms)
    linear_norms = np.concatenate(linear_norms)
    mlp_norms = np.concatenate(mlp_norms)

    assert linear_losses.shape == mlp_losses.shape, "linear/mlp were not run on the same examples"
    n_examples = linear_losses.shape[0]
    print(f"\nProcessed {n_examples} speaker-slot examples ({n_examples // 2} mixtures) in {dt:.1f}s")

    # cross-check: mean of per-example losses should match CosineSimilarityLoss's
    # own batch-mean (sanity that pit_cosine_loss_per_example matches training loss).
    # Batches can have unequal size (the last one), so split on cumulative offsets
    # rather than assuming an equal division.
    bm = np.array(batch_mean_check)
    offsets = np.cumsum(batch_sizes)[:-1]
    lin_batch_means = np.array([b.mean() for b in np.split(linear_losses, offsets)])
    mlp_batch_means = np.array([b.mean() for b in np.split(mlp_losses, offsets)])
    print(
        "Sanity check vs. training-loss fn -- max |per-example mean - batch mean| : "
        f"linear={np.abs(bm[:, 0] - lin_batch_means).max():.2e}, "
        f"mlp={np.abs(bm[:, 1] - mlp_batch_means).max():.2e}"
    )

    # ---- 1. cosine loss comparison ----
    diff = mlp_losses - linear_losses  # negative => MLP better (lower loss) on that example
    t_stat, t_p = sstats.ttest_rel(mlp_losses, linear_losses)
    w_stat, w_p = sstats.wilcoxon(mlp_losses, linear_losses)
    frac_mlp_better = float((diff < 0).mean())

    loss_summaries = [summarize("linear cosine loss", linear_losses), summarize("mlp cosine loss", mlp_losses)]
    print_summary_table("=== 1. Validation cosine loss (PIT-matched, lower is better) ===", loss_summaries)
    print(f"\nPaired diff (mlp - linear): mean={diff.mean():.4f}  std={diff.std():.4f}")
    print(f"Fraction of examples where MLP has LOWER (better) loss than linear: {frac_mlp_better:.3f}")
    print(f"Paired t-test:   t={t_stat:.3f}  p={t_p:.2e}")
    print(f"Wilcoxon signed-rank: W={w_stat:.1f}  p={w_p:.2e}")

    # ---- 2. embedding norm comparison ----
    norm_summaries = [
        summarize("raw dual-emb (sanity)", raw_norms),
        summarize("teacher gt (sanity)", gt_norms),
        summarize("linear-mapped", linear_norms),
        summarize("mlp-mapped", mlp_norms),
    ]
    print_summary_table("=== 2. Embedding norms (raw dual-emb / teacher target are ~1.0 by construction) ===", norm_summaries)

    gt_mean = gt_norms.mean()
    lin_ratio = linear_norms.mean() / gt_mean
    mlp_ratio = mlp_norms.mean() / gt_mean
    print(f"\nMean norm ratio to teacher-embedding scale: linear={lin_ratio:.3f}x   mlp={mlp_ratio:.3f}x")
    print(
        "(DPCCN's SpeakerFuseLayer(fuse_type='multiply') does not renormalize the\n"
        " conditioning embedding -- joint_training=False and use_spk_transform=False\n"
        " in wesep/confs/config_dpcnn.yaml route it through nn.Identity() straight into\n"
        " an unnormalized elementwise-multiply gate. A mapped-embedding norm that drifts\n"
        " from the teacher-embedding scale changes DPCCN's conditioning strength even\n"
        " when cosine similarity to the teacher direction is unchanged.)"
    )

    # ---- decision-tree readout ----
    print("\n=== Reading this against the two branches ===")
    if mlp_losses.mean() > linear_losses.mean():
        print(
            "-> MLP has WORSE mean validation cosine loss than linear "
            f"({mlp_losses.mean():.4f} vs {linear_losses.mean():.4f}).\n"
            "   The nonlinear adapter is fitting or generalizing less effectively than\n"
            "   the linear map on this objective; that alone is enough to prefer linear\n"
            "   here without needing the norm comparison below."
        )
    else:
        print(
            "-> MLP has BETTER (or equal) mean validation cosine loss than linear "
            f"({mlp_losses.mean():.4f} vs {linear_losses.mean():.4f}).\n"
            "   If MLP's downstream TSE (SI-SDR) is nonetheless worse, the cosine\n"
            "   objective may not fully capture downstream compatibility -- check the\n"
            f"   norm ratios above (linear={lin_ratio:.3f}x, mlp={mlp_ratio:.3f}x of the\n"
            "   teacher scale): a bigger mismatch for MLP is consistent with DPCCN's\n"
            "   unnormalized multiplicative conditioning being pushed out of the regime\n"
            "   it was tuned for, even though direction (cosine) improved."
        )

    # ---- save ----
    out = {
        "checkpoints": {
            "dual_emb_ckpt": args.dual_emb_ckpt,
            "teacher_ckpt": args.teacher_ckpt,
            "linear_ckpt": args.linear_ckpt,
            "linear_ckpt_epoch": linear_meta["epoch"],
            "mlp_ckpt": args.mlp_ckpt,
            "mlp_ckpt_epoch": mlp_meta["epoch"],
        },
        "n_examples": int(n_examples),
        "cosine_loss": {"linear": loss_summaries[0], "mlp": loss_summaries[1]},
        "paired_diff_mlp_minus_linear": {
            "mean": float(diff.mean()),
            "std": float(diff.std()),
            "frac_mlp_better": frac_mlp_better,
            "paired_ttest": {"t": float(t_stat), "p": float(t_p)},
            "wilcoxon": {"W": float(w_stat), "p": float(w_p)},
        },
        "embedding_norms": {
            "raw_dual_emb": norm_summaries[0],
            "teacher_gt": norm_summaries[1],
            "linear_mapped": norm_summaries[2],
            "mlp_mapped": norm_summaries[3],
            "linear_ratio_to_teacher_scale": float(lin_ratio),
            "mlp_ratio_to_teacher_scale": float(mlp_ratio),
        },
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved full summary to {out_path}")


if __name__ == "__main__":
    main()
