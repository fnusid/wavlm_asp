#!/usr/bin/env python3
"""Plot teacher speaker centroids with t-SNE and save as a PNG."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--centroid-pt",
        default="/home/sidcs/datasets/LibriMix/LibriMix/hard_pairs_teacher_centroid/centroids/teacher_centroids.pt",
    )
    parser.add_argument(
        "--output-png",
        default="/home/sidcs/datasets/LibriMix/LibriMix/hard_pairs_teacher_centroid/centroids/teacher_centroids_tsne.png",
    )
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--random-state", type=int, default=44)
    parser.add_argument("--annotate-top-k", type=int, default=60)
    parser.add_argument(
        "--color-by",
        choices=["subset", "sex"],
        default="subset",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    centroid_path = Path(args.centroid_pt)
    output_path = Path(args.output_png)

    payload = torch.load(centroid_path, map_location="cpu")
    speaker_ids = payload["speaker_ids"]
    embeddings = payload["embeddings"].cpu().numpy()
    metadata = payload.get("metadata", {})

    if args.subsets:
        keep_indices = [
            idx for idx, speaker_id in enumerate(speaker_ids)
            if metadata.get(speaker_id, {}).get("subset") in set(args.subsets)
        ]
        speaker_ids = [speaker_ids[idx] for idx in keep_indices]
        embeddings = embeddings[keep_indices]

    n_points = len(speaker_ids)
    if n_points < 2:
        raise ValueError("Need at least 2 centroids for t-SNE.")

    max_valid_perplexity = max(1.0, float(n_points - 1))
    perplexity = min(args.perplexity, max_valid_perplexity)
    if perplexity >= n_points:
        perplexity = max(1.0, float(n_points - 1))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=args.random_state,
    )
    coords = tsne.fit_transform(embeddings)

    labels = []
    for speaker_id in speaker_ids:
        speaker_meta = metadata.get(speaker_id, {})
        labels.append(str(speaker_meta.get(args.color_by, "unknown")))
    labels = np.array(labels)

    unique_labels = sorted(set(labels.tolist()))
    cmap = plt.get_cmap("tab20", max(len(unique_labels), 1))

    plt.figure(figsize=(14, 11))
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=24,
            alpha=0.75,
            color=cmap(idx),
            label=label,
        )

    annotate_count = min(args.annotate_top_k, n_points)
    for idx in range(annotate_count):
        plt.annotate(
            speaker_ids[idx],
            (coords[idx, 0], coords[idx, 1]),
            fontsize=7,
            alpha=0.85,
        )

    plt.title(f"Teacher Speaker Centroids t-SNE ({n_points} speakers)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title=args.color_by, loc="best", fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved t-SNE PNG to {output_path}")


if __name__ == "__main__":
    main()
