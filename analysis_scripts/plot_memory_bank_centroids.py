#!/usr/bin/env python3
"""Plot centroid-evaluation results for the memory-bank ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def save_metric_plot(xs, ys, ylabel: str, title: str, out_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(xs, ys, marker="o", linewidth=2)
    plt.xlabel("Total speakers in conversation")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_dual_cosine_plot(xs, same, diff, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(xs, same, marker="o", linewidth=2, label="Same-speaker centroid cosine")
    plt.plot(xs, diff, marker="s", linewidth=2, label="Different-speaker centroid cosine")
    plt.xlabel("Total speakers in conversation")
    plt.ylabel("Cosine similarity")
    plt.title("Teacher-centroid cosine separation")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_combined_plot(xs, top1, hungarian, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(xs, top1, marker="o", linewidth=2, label="Top-1 event accuracy")
    plt.plot(xs, hungarian, marker="s", linewidth=2, label="Hungarian event accuracy")
    plt.xlabel("Total speakers in conversation")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.05)
    plt.title("Teacher-centroid matching accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_json)
    out_dir = Path(args.out_dir) if args.out_dir else summary_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with summary_path.open() as handle:
        summary = json.load(handle)

    xs = sorted(int(key) for key in summary.keys())
    top1 = [summary[str(x)]["teacher_centroid_top1_event_accuracy_mean"] for x in xs]
    hungarian = [summary[str(x)]["teacher_centroid_hungarian_event_accuracy_mean"] for x in xs]
    same = [summary[str(x)]["teacher_centroid_same_speaker_cosine_mean"] for x in xs]
    diff = [summary[str(x)]["teacher_centroid_different_speaker_cosine_mean"] for x in xs]

    save_metric_plot(
        xs,
        top1,
        "Top-1 event accuracy",
        "Teacher-centroid top-1 accuracy vs global speaker count",
        out_dir / "teacher_centroid_top1_accuracy_vs_total_speakers.png",
    )
    save_metric_plot(
        xs,
        hungarian,
        "Hungarian event accuracy",
        "Teacher-centroid Hungarian accuracy vs global speaker count",
        out_dir / "teacher_centroid_hungarian_accuracy_vs_total_speakers.png",
    )
    save_dual_cosine_plot(
        xs,
        same,
        diff,
        out_dir / "teacher_centroid_cosine_separation_vs_total_speakers.png",
    )
    save_combined_plot(
        xs,
        top1,
        hungarian,
        out_dir / "teacher_centroid_accuracy_main_plot.png",
    )
    print(f"[done] wrote centroid plots to {out_dir}")


if __name__ == "__main__":
    main()
