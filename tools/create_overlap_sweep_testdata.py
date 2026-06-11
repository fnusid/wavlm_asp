#!/usr/bin/env python3
"""Create a compact fixed-overlap Libri2Mix test sweep for quick experiments.

This wrapper samples a shared subset of mixtures from the canonical
`libri2mix_test-clean.csv` metadata and reuses the existing LibriMix generation
script to synthesize one test set per fixed overlap value.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_GENERATOR = (
    "/home/sidcs/datasets/LibriMix/scripts/"
    "create_librimix_uniovlp_li360_from_meta_withid.py"
)
DEFAULT_SOURCE_METADATA = "/home/sidcs/datasets/LibriMix/metadata/Libri2Mix/libri2mix_test-clean.csv"
DEFAULT_LIBRISPEECH_DIR = "/home/sidcs/datasets/LibriMix/LibriMix/LibriSpeech"
DEFAULT_WHAM_DIR = "/home/sidcs/datasets/LibriMix/LibriMix/wham_noise"
DEFAULT_SPEAKER_MAP = (
    "/home/sidcs/datasets/LibriMix/LibriMix/Libriuni_05_08/"
    "Libri2Mix_ovl50to80/wav16k/min/metadata/train360_mapping.json"
)
DEFAULT_OUTPUT_ROOT = "/home/sidcs/datasets/LibriMix/LibriMix/overlap_eval_small"
DEFAULT_WORK_DIR = "/home/sidcs/datasets/LibriMix/metadata/overlap_eval_small"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-script", default=DEFAULT_GENERATOR)
    parser.add_argument("--generator-python", default="/home/sidcs/miniconda3/bin/python3.13")
    parser.add_argument("--source-metadata", default=DEFAULT_SOURCE_METADATA)
    parser.add_argument("--librispeech-dir", default=DEFAULT_LIBRISPEECH_DIR)
    parser.add_argument("--wham-dir", default=DEFAULT_WHAM_DIR)
    parser.add_argument("--speaker-map", default=DEFAULT_SPEAKER_MAP)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    parser.add_argument("--subset-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--overlaps", nargs="+", type=int, default=[0, 25, 50, 75, 100])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_exists(path: str, description: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{description} not found: {path}")


def read_rows(csv_path: str) -> list[dict[str, str]]:
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def write_subset_csv(rows: list[dict[str, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def copy_if_exists(src: str, dst: Path) -> None:
    if src and os.path.exists(src):
        shutil.copy2(src, dst)


def build_generator_cmd(
    args: argparse.Namespace,
    overlap: int,
    metadata_dir: Path,
    output_root: Path,
) -> list[str]:
    return [
        args.generator_python,
        args.generator_script,
        "--librispeech_dir",
        args.librispeech_dir,
        "--wham_dir",
        args.wham_dir,
        "--metadata_dir",
        str(metadata_dir),
        "--librimix_outdir",
        str(output_root),
        "--n_src",
        "2",
        "--freqs",
        "16k",
        "--modes",
        "min",
        "--types",
        "mix_clean",
        "--overlap_min",
        f"{overlap / 100.0:.2f}",
        "--overlap_max",
        f"{overlap / 100.0:.2f}",
        "--seed",
        str(args.seed),
    ]


def main() -> None:
    args = parse_args()
    ensure_exists(args.generator_script, "Generator script")
    ensure_exists(args.source_metadata, "Source metadata")
    ensure_exists(args.librispeech_dir, "LibriSpeech directory")
    ensure_exists(args.wham_dir, "WHAM directory")
    ensure_exists(args.generator_python, "Generator Python")

    rows = read_rows(args.source_metadata)
    if not rows:
        raise RuntimeError(f"No rows found in {args.source_metadata}")
    if args.subset_size <= 0:
        raise ValueError("--subset-size must be positive")

    rng = random.Random(args.seed)
    subset_size = min(args.subset_size, len(rows))
    sampled_rows = rng.sample(rows, subset_size)

    output_root = Path(args.output_root)
    work_dir = Path(args.work_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "source_metadata": args.source_metadata,
        "subset_size": subset_size,
        "seed": args.seed,
        "overlaps": args.overlaps,
        "mixture_ids": [row["mixture_ID"] for row in sampled_rows],
        "datasets": {},
    }

    for overlap in args.overlaps:
        parent_dir = output_root / f"Libri2Mix_{overlap}vlp"
        dataset_dir = parent_dir / f"Libri2Mix_ovl{overlap}to{overlap}"
        metadata_dir = work_dir / f"ovlp_{overlap:03d}"

        if dataset_dir.exists() and args.force:
            shutil.rmtree(dataset_dir)
        if metadata_dir.exists() and args.force:
            shutil.rmtree(metadata_dir)

        if dataset_dir.exists():
            print(f"[skip] overlap={overlap}: dataset already exists at {dataset_dir}")
        else:
            subset_csv = metadata_dir / "libri2mix_test-clean.csv"
            write_subset_csv(sampled_rows, subset_csv)

            cmd = build_generator_cmd(args, overlap, metadata_dir, parent_dir)
            print(f"[run] overlap={overlap}: generating {subset_size} mixtures")
            subprocess.run(cmd, check=True)

            generated_meta = dataset_dir / "wav16k" / "min" / "metadata"
            generated_meta.mkdir(parents=True, exist_ok=True)
            copy_if_exists(args.speaker_map, generated_meta / "train360_mapping.json")

        manifest["datasets"][str(overlap)] = {
            "dataset_root": str(dataset_dir),
            "metadata_csv": str(dataset_dir / "wav16k" / "min" / "metadata" / "mixture_test_mix_clean.csv"),
        }

    manifest_path = output_root / "overlap_eval_small_manifest.json"
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"[done] wrote manifest: {manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"[error] command failed with exit code {exc.returncode}: {' '.join(exc.cmd)}", file=sys.stderr)
        raise
