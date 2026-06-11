#!/usr/bin/env python3
"""Plot TSE-vs-separator SI-SDR curves from a summary JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def plot_metric(settings, summary, metric_key, ylabel, out_path: Path) -> None:
    tse = [summary[s][f"tse_{metric_key}_mean"] for s in settings]
    sep_match = [summary[s][f"sep_match_{metric_key}_mean"] for s in settings]
    sep_oracle = [summary[s][f"sep_oracle_{metric_key}_mean"] for s in settings]

    plt.figure(figsize=(7, 4.5))
    plt.plot(settings, tse, marker="o", linewidth=2, label="TSE (dual-centroid)")
    plt.plot(settings, sep_match, marker="s", linewidth=2, label="PIT sep + teacher match")
    plt.plot(settings, sep_oracle, marker="^", linewidth=2, label="PIT sep + oracle pick")
    plt.xlabel("Total speakers in long audio")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.summary_json, "r", encoding="utf-8") as handle:
        summary = json.load(handle)

    settings = sorted(summary.keys(), key=int)

    plot_metric(settings, summary, "sisdr", "SI-SDR (dB)", out_dir / "sisdr_vs_total_speakers.png")
    plot_metric(settings, summary, "sisdri", "SI-SDRi (dB)", out_dir / "sisdri_vs_total_speakers.png")


if __name__ == "__main__":
    main()
