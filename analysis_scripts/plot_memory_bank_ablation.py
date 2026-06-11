#!/usr/bin/env python3
"""Plot the global-speaker memory-bank ablation results."""

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


def save_combined_plot(xs, primary_acc, primary_label: str, ari, switches, out_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(xs, primary_acc, marker="o", label=primary_label)
    ax1.plot(xs, ari, marker="s", label="ARI")
    ax1.set_xlabel("Total speakers in conversation")
    ax1.set_ylabel("Accuracy / ARI")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(xs, switches, marker="^", linestyle="--", label="ID switches", color="tab:red")
    ax2.set_ylabel("ID switches")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")
    plt.title("Memory-bank scaling with increasing global speakers")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_json)
    out_dir = Path(args.out_dir) if args.out_dir else summary_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with summary_path.open() as handle:
        summary = json.load(handle)

    xs = sorted(int(key) for key in summary.keys())
    reid = [summary[str(x)]["reid_accuracy_mean"] for x in xs]
    majority = [
        summary[str(x)].get("majority_id_accuracy_mean", summary[str(x)]["reid_accuracy_mean"])
        for x in xs
    ]
    ari = [summary[str(x)]["ari_mean"] for x in xs]
    switches = [summary[str(x)]["id_switches_mean"] for x in xs]

    save_metric_plot(
        xs,
        reid,
        "Re-ID accuracy",
        "Re-identification accuracy vs global speaker count",
        out_dir / "reid_accuracy_vs_total_speakers.png",
    )
    save_metric_plot(
        xs,
        majority,
        "Majority-ID accuracy",
        "Majority-ID accuracy vs global speaker count",
        out_dir / "majority_id_accuracy_vs_total_speakers.png",
    )
    save_metric_plot(
        xs,
        ari,
        "ARI",
        "ARI vs global speaker count",
        out_dir / "ari_vs_total_speakers.png",
    )
    save_metric_plot(
        xs,
        switches,
        "ID switches",
        "ID switches vs global speaker count",
        out_dir / "id_switches_vs_total_speakers.png",
    )
    save_combined_plot(
        xs,
        majority,
        "Majority-ID accuracy",
        ari,
        switches,
        out_dir / "memory_bank_ablation_main_plot.png",
    )
    print(f"[done] wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
