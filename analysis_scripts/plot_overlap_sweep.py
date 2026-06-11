#!/usr/bin/env python3
"""Plot overlap-sweep metrics and optional shared-space t-SNE frames."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--overlaps", nargs="+", type=int, default=None)
    parser.add_argument("--tsne", action="store_true")
    parser.add_argument("--num-speakers", type=int, default=20)
    parser.add_argument("--max-per-speaker", type=int, default=40)
    parser.add_argument("--tsne-perplexity", type=float, default=20.0)
    parser.add_argument("--tsne-seed", type=int, default=0)
    return parser.parse_args()


def overlap_from_name(path: Path) -> int | None:
    match = re.search(r"ovlp(\d+)", path.name)
    return int(match.group(1)) if match else None


def load_metrics(results_dir: Path, overlaps: list[int] | None) -> dict[int, dict[str, float]]:
    metrics: dict[int, dict[str, float]] = {}
    for path in sorted(results_dir.glob("ovlp*_metrics.json")):
        overlap = overlap_from_name(path)
        if overlap is None:
            continue
        if overlaps is not None and overlap not in overlaps:
            continue
        with path.open() as handle:
            metrics[overlap] = json.load(handle)
    return metrics


def save_line_plot(
    metrics_by_overlap: dict[int, dict[str, float]],
    overlap_order: list[int],
    metric_keys: list[str],
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    for key in metric_keys:
        ys = [metrics_by_overlap[ov][key] for ov in overlap_order]
        plt.plot(overlap_order, ys, marker="o", label=key)
    plt.xlabel("Overlap (%)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def build_style_map(chosen_speakers: np.ndarray) -> dict[int, dict[str, object]]:
    cmap = plt.cm.get_cmap("turbo", len(chosen_speakers))
    markers = ["o", "^", "s", "D", "v", "P", "X", "*", "<", ">", "h", "H", "p", "+", "x"]
    styles = {}
    for idx, spk in enumerate(chosen_speakers):
        styles[int(spk)] = {"color": cmap(idx), "marker": markers[idx % len(markers)]}
    return styles


def choose_common_speakers(
    labels_by_overlap: dict[int, np.ndarray],
    overlap_order: list[int],
    num_speakers: int,
) -> np.ndarray:
    common: set[int] | None = None
    for overlap in overlap_order:
        speaker_ids = set(np.unique(labels_by_overlap[overlap]).astype(int).tolist())
        common = speaker_ids if common is None else (common & speaker_ids)
    if not common:
        raise RuntimeError("No common speakers found across overlaps.")
    common_sorted = np.array(sorted(common), dtype=int)
    return common_sorted[: min(num_speakers, len(common_sorted))]


def make_joint_tsne_coords(
    embs_by_overlap: dict[int, np.ndarray],
    labels_by_overlap: dict[int, np.ndarray],
    overlap_order: list[int],
    chosen_speakers: np.ndarray,
    max_per_speaker: int,
    perplexity: float,
    tsne_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(tsne_seed)
    xs = []
    spks = []
    ovlps = []

    for overlap in overlap_order:
        embs = embs_by_overlap[overlap]
        labels = labels_by_overlap[overlap]
        for speaker in chosen_speakers:
            idx = np.where(labels == speaker)[0]
            if len(idx) == 0:
                continue
            if len(idx) > max_per_speaker:
                idx = rng.choice(idx, size=max_per_speaker, replace=False)
            xs.append(embs[idx])
            spks.append(labels[idx])
            ovlps.append(np.full(len(idx), overlap))

    x_all = np.vstack(xs)
    spk_all = np.concatenate(spks).astype(int)
    ovlp_all = np.concatenate(ovlps).astype(int)
    x_all = x_all / (np.linalg.norm(x_all, axis=1, keepdims=True) + 1e-10)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=tsne_seed,
    )
    coords = tsne.fit_transform(x_all)
    return coords, spk_all, ovlp_all


def render_tsne_frames(
    coords_all: np.ndarray,
    spk_all: np.ndarray,
    ovlp_all: np.ndarray,
    overlap_order: list[int],
    chosen_speakers: np.ndarray,
    out_dir: Path,
) -> None:
    frames_dir = out_dir / "tsne_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    styles = build_style_map(chosen_speakers)
    xmin, xmax = coords_all[:, 0].min(), coords_all[:, 0].max()
    ymin, ymax = coords_all[:, 1].min(), coords_all[:, 1].max()

    for overlap in overlap_order:
        plt.figure(figsize=(10, 8))
        mask = ovlp_all == overlap
        for speaker in chosen_speakers:
            speaker_mask = mask & (spk_all == speaker)
            if not np.any(speaker_mask):
                continue
            style = styles[int(speaker)]
            points = coords_all[speaker_mask]
            plt.scatter(
                points[:, 0],
                points[:, 1],
                s=22,
                c=[style["color"]],
                marker=style["marker"],
                alpha=0.85,
                linewidths=0.3,
                edgecolors="k",
            )

        plt.title(f"Shared-space t-SNE, overlap={overlap}%")
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.tight_layout()
        plt.savefig(frames_dir / f"frame_{overlap:03d}.png", dpi=220)
        plt.close()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_by_overlap = load_metrics(results_dir, args.overlaps)
    if not metrics_by_overlap:
        raise RuntimeError(f"No metric JSON files found in {results_dir}")

    overlap_order = sorted(metrics_by_overlap.keys())
    save_line_plot(
        metrics_by_overlap,
        overlap_order,
        ["same_mean_cos", "diff_mean_cos", "separation"],
        "Cosine similarity stats vs overlap",
        "Cosine / separation",
        out_dir / "metrics_cosine_separation_vs_overlap.png",
    )
    save_line_plot(
        metrics_by_overlap,
        overlap_order,
        ["cluster_acc", "nmi", "ari"],
        "Clustering quality vs overlap",
        "Score",
        out_dir / "metrics_clustering_quality_vs_overlap.png",
    )
    save_line_plot(
        metrics_by_overlap,
        overlap_order,
        ["silhouette"],
        "Silhouette score vs overlap",
        "Silhouette",
        out_dir / "metrics_silhouette_vs_overlap.png",
    )

    summary_path = out_dir / "metrics_summary.json"
    with summary_path.open("w") as handle:
        json.dump({str(k): metrics_by_overlap[k] for k in overlap_order}, handle, indent=2)

    if not args.tsne:
        print(f"[done] wrote plots to {out_dir}")
        return

    embs_by_overlap: dict[int, np.ndarray] = {}
    labels_by_overlap: dict[int, np.ndarray] = {}
    for overlap in overlap_order:
        npz_path = results_dir / f"ovlp{overlap:03d}_results.npz"
        if not npz_path.exists():
            raise RuntimeError(f"Missing t-SNE source file: {npz_path}")
        data = np.load(npz_path)
        embs_by_overlap[overlap] = data["embs"]
        labels_by_overlap[overlap] = data["labels"].astype(int)

    chosen_speakers = choose_common_speakers(labels_by_overlap, overlap_order, args.num_speakers)
    coords_all, spk_all, ovlp_all = make_joint_tsne_coords(
        embs_by_overlap,
        labels_by_overlap,
        overlap_order,
        chosen_speakers,
        args.max_per_speaker,
        args.tsne_perplexity,
        args.tsne_seed,
    )
    render_tsne_frames(coords_all, spk_all, ovlp_all, overlap_order, chosen_speakers, out_dir)
    print(f"[done] wrote plots and t-SNE frames to {out_dir}")


if __name__ == "__main__":
    main()
