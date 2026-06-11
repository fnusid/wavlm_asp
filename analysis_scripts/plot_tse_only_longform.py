#!/usr/bin/env python3
"""Plot only the TSE SI-SDR / SI-SDRi curves from long-form summary JSON."""

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


def plot_one(settings, values, ylabel, title, out_path: Path) -> None:
    plt.figure(figsize=(7, 4.5))
    plt.plot(settings, values, marker="o", linewidth=2, label="TSE (final dual-centroid)")
    plt.xlabel("Total speakers in long audio")
    plt.ylabel(ylabel)
    plt.title(title)
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
    tse_sisdr = [summary[s]["tse_sisdr_mean"] for s in settings]
    tse_sisdri = [summary[s]["tse_sisdri_mean"] for s in settings]

    plot_one(
        settings,
        tse_sisdr,
        "SI-SDR (dB)",
        "TSE Performance vs Total Speakers",
        out_dir / "tse_sisdr_vs_total_speakers.png",
    )
    plot_one(
        settings,
        tse_sisdri,
        "SI-SDRi (dB)",
        "TSE Improvement vs Total Speakers",
        out_dir / "tse_sisdri_vs_total_speakers.png",
    )


if __name__ == "__main__":
    main()
