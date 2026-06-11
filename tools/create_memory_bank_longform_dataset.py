#!/usr/bin/env python3
"""Create long-form conversations from real LibriMix local 2-speaker chunks."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


RATE = 16000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local-metadata-csv",
        default=(
            "/home/sidcs/datasets/LibriMix/LibriMix/Libriuni_06_08/"
            "Libri2Mix_ovl60to80/wav16k/min/metadata/mixture_test_mix_clean.csv"
        ),
    )
    parser.add_argument(
        "--output-root",
        default="/home/sidcs/datasets/LibriMix/LibriMix/memory_bank_longform_eval_v2",
    )
    parser.add_argument("--settings", nargs="+", type=int, default=[2, 4, 6, 8, 10])
    parser.add_argument("--num-conversations", type=int, default=3)
    parser.add_argument("--appearances-per-speaker", type=int, default=4)
    parser.add_argument("--chunk-sec", type=float, default=5.0)
    parser.add_argument("--min-chunk-sec", type=float, default=None)
    parser.add_argument("--gap-sec", type=float, default=0.25)
    parser.add_argument("--extra-random-chunks", type=int, default=2)
    parser.add_argument("--activity-mode", choices=["max2", "exact2"], default="max2")
    parser.add_argument("--unique-speaker-combos", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=200)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_rows(csv_path: str, min_chunk_sec: float | None, chunk_sec: float) -> list[dict[str, str]]:
    min_required_sec = max(chunk_sec, min_chunk_sec or 0.0)
    min_length_samples = int(min_required_sec * RATE)
    with open(csv_path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if int(row["length"]) >= min_length_samples]


def speaker_pair(row: dict[str, str]) -> tuple[str, str]:
    return tuple(sorted((str(row["speaker_1_ID"]), str(row["speaker_2_ID"]))))


def build_indices(rows: list[dict[str, str]]) -> tuple[dict[str, int], dict[str, list[int]]]:
    speaker_counts: dict[str, int] = defaultdict(int)
    speaker_to_rows: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        s1 = str(row["speaker_1_ID"])
        s2 = str(row["speaker_2_ID"])
        speaker_counts[s1] += 1
        speaker_counts[s2] += 1
        speaker_to_rows[s1].append(idx)
        speaker_to_rows[s2].append(idx)
    return dict(speaker_counts), dict(speaker_to_rows)


def build_candidate_events(
    rows: list[dict[str, str]],
    chosen_set: set[str],
    activity_mode: str,
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for idx, row in enumerate(rows):
        s1 = str(row["speaker_1_ID"])
        s2 = str(row["speaker_2_ID"])

        if activity_mode == "max2" and s1 in chosen_set:
            events.append(
                {
                    "event_id": f"{idx}:s1",
                    "row_idx": idx,
                    "mixture_id": f"{row['mixture_ID']}:s1",
                    "active_global_speakers": [s1],
                    "mix_dataset_path": row["source_1_path"],
                    "source_dataset_paths": [row["source_1_path"]],
                    "active_set_key": (s1,),
                }
            )
        if activity_mode == "max2" and s2 in chosen_set:
            events.append(
                {
                    "event_id": f"{idx}:s2",
                    "row_idx": idx,
                    "mixture_id": f"{row['mixture_ID']}:s2",
                    "active_global_speakers": [s2],
                    "mix_dataset_path": row["source_2_path"],
                    "source_dataset_paths": [row["source_2_path"]],
                    "active_set_key": (s2,),
                }
            )
        if s1 in chosen_set and s2 in chosen_set:
            events.append(
                {
                    "event_id": f"{idx}:pair",
                    "row_idx": idx,
                    "mixture_id": row["mixture_ID"],
                    "active_global_speakers": [s1, s2],
                    "mix_dataset_path": row["mixture_path"],
                    "source_dataset_paths": [row["source_1_path"], row["source_2_path"]],
                    "active_set_key": tuple(sorted((s1, s2))),
                }
            )
    return events


def choose_conversation_events(
    rows: list[dict[str, str]],
    speaker_counts: dict[str, int],
    total_speakers: int,
    appearances_per_speaker: int,
    extra_random_chunks: int,
    activity_mode: str,
    rng: random.Random,
    max_attempts: int,
) -> tuple[list[str], list[dict[str, object]]]:
    eligible = [spk for spk, count in speaker_counts.items() if count >= appearances_per_speaker]
    if len(eligible) < total_speakers:
        raise RuntimeError(f"Need {total_speakers} eligible speakers, found {len(eligible)}")

    for _ in range(max_attempts):
        chosen = rng.sample(eligible, total_speakers)
        chosen_set = set(chosen)
        allowed = build_candidate_events(rows, chosen_set, activity_mode)
        if len(allowed) < total_speakers:
            continue

        remaining = {spk: appearances_per_speaker for spk in chosen}
        unused = {event["event_id"]: event for event in allowed}
        selected: list[dict[str, object]] = []

        while any(v > 0 for v in remaining.values()):
            candidates = []
            for event_id, event in unused.items():
                active_speakers = event["active_global_speakers"]
                coverage = sum(int(remaining[speaker] > 0) for speaker in active_speakers)
                if coverage > 0:
                    rarity = sum(remaining[speaker] for speaker in active_speakers)
                    candidates.append((coverage, rarity, rng.random(), event_id))
            if not candidates:
                break
            candidates.sort(reverse=True)
            event_id = candidates[0][3]
            event = unused.pop(event_id)
            selected.append(event)
            for speaker in event["active_global_speakers"]:
                remaining[speaker] = max(0, remaining[speaker] - 1)

        if any(v > 0 for v in remaining.values()):
            continue

        active_sets_seen = {event["active_set_key"] for event in selected}
        extras_pool = [
            event for event in unused.values() if event["active_set_key"] not in active_sets_seen
        ] or list(unused.values())
        rng.shuffle(extras_pool)
        selected.extend(extras_pool[:extra_random_chunks])
        rng.shuffle(selected)
        return chosen, selected

    raise RuntimeError(
        f"Failed to find a valid conversation plan for {total_speakers} speakers after {max_attempts} attempts."
    )


def read_audio(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if sr != RATE:
        raise ValueError(f"Expected {RATE} Hz audio, got {sr} for {path}")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio.astype(np.float32)


def sample_crop_start(audio_length: int, target_length: int, rng: random.Random) -> int:
    if audio_length == target_length:
        return 0
    max_start = audio_length - target_length
    return rng.randint(0, max_start)


def crop_audio(audio: np.ndarray, target_length: int, rng: random.Random) -> np.ndarray:
    start = sample_crop_start(len(audio), target_length, rng)
    end = start + target_length
    return audio[start:end].copy()


def crop_audio_at(audio: np.ndarray, start: int, target_length: int) -> np.ndarray:
    if len(audio) == target_length:
        return audio
    end = start + target_length
    return audio[start:end].copy()


def save_audio(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, RATE)


def build_conversation(
    rows: list[dict[str, str]],
    speaker_counts: dict[str, int],
    total_speakers: int,
    conv_idx: int,
    args: argparse.Namespace,
    rng: random.Random,
) -> dict[str, object]:
    chosen_speakers, selected_events = choose_conversation_events(
        rows=rows,
        speaker_counts=speaker_counts,
        total_speakers=total_speakers,
        appearances_per_speaker=args.appearances_per_speaker,
        extra_random_chunks=args.extra_random_chunks,
        activity_mode=args.activity_mode,
        rng=rng,
        max_attempts=args.max_attempts,
    )

    conv_dir = Path(args.output_root) / f"setting_{total_speakers:02d}sp" / f"conversation_{conv_idx:03d}"
    if conv_dir.exists() and args.force:
        shutil.rmtree(conv_dir)
    conv_dir.mkdir(parents=True, exist_ok=True)

    gap = np.zeros(int(args.gap_sec * RATE), dtype=np.float32)
    target_chunk_length = int(args.chunk_sec * RATE)
    full_mix_parts = []
    metadata_chunks = []
    cursor = 0

    for chunk_index, event in enumerate(selected_events):
        mix_full = read_audio(str(event["mix_dataset_path"]))
        source_dataset_paths = [str(path) for path in event["source_dataset_paths"]]
        source_full_wavs = [read_audio(path) for path in source_dataset_paths]
        crop_start = sample_crop_start(len(mix_full), target_chunk_length, rng)
        mix = crop_audio_at(mix_full, crop_start, target_chunk_length)
        source_wavs = [crop_audio_at(wav, crop_start, target_chunk_length) for wav in source_full_wavs]

        local_mix_path = conv_dir / "chunks" / "mix" / f"chunk_{chunk_index:04d}.wav"
        save_audio(local_mix_path, mix)

        local_source_paths = []
        for source_idx, source_wav in enumerate(source_wavs, start=1):
            local_source_path = (
                conv_dir / "chunks" / f"s{source_idx}" / f"chunk_{chunk_index:04d}.wav"
            )
            save_audio(local_source_path, source_wav)
            local_source_paths.append(str(local_source_path))

        start_sample = cursor
        end_sample = cursor + len(mix)
        full_mix_parts.append(mix)

        metadata_chunks.append(
            {
                "chunk_index": chunk_index,
                "start_sample": start_sample,
                "end_sample": end_sample,
                "start_sec": start_sample / RATE,
                "end_sec": end_sample / RATE,
                "mixture_id": str(event["mixture_id"]),
                "active_global_speakers": list(event["active_global_speakers"]),
                "mix_chunk_path": str(local_mix_path),
                "source_chunk_paths": local_source_paths,
                "source_dataset_paths": source_dataset_paths,
                "local_dataset_mix_path": str(event["mix_dataset_path"]),
                "num_active_speakers": len(event["active_global_speakers"]),
                "length_samples": len(mix),
            }
        )

        cursor = end_sample
        if chunk_index != len(selected_events) - 1 and len(gap) > 0:
            full_mix_parts.append(gap.copy())
            cursor += len(gap)

    long_mix = np.concatenate(full_mix_parts) if full_mix_parts else np.zeros(0, dtype=np.float32)
    save_audio(conv_dir / "mix.wav", long_mix)

    metadata = {
        "setting_total_speakers": total_speakers,
        "max_active_at_once": 2,
        "activity_mode": args.activity_mode,
        "conversation_index": conv_idx,
        "sample_rate": RATE,
        "chunk_sec": args.chunk_sec,
        "gap_sec": args.gap_sec,
        "appearances_per_speaker_target": args.appearances_per_speaker,
        "chosen_global_speakers": chosen_speakers,
        "max_allowed_active_speakers": 2,
        "num_chunks": len(metadata_chunks),
        "duration_sec": len(long_mix) / RATE,
        "local_metadata_csv": args.local_metadata_csv,
        "chunks": metadata_chunks,
    }
    with (conv_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)

    return {
        "setting_total_speakers": total_speakers,
        "conversation_dir": str(conv_dir),
        "num_chunks": len(metadata_chunks),
        "duration_sec": len(long_mix) / RATE,
        "chosen_global_speakers": chosen_speakers,
    }


def main() -> None:
    args = parse_args()
    rows = read_rows(args.local_metadata_csv, args.min_chunk_sec, args.chunk_sec)
    speaker_counts, _ = build_indices(rows)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    manifest = {
        "settings": args.settings,
        "num_conversations_per_setting": args.num_conversations,
        "output_root": args.output_root,
        "local_metadata_csv": args.local_metadata_csv,
        "activity_mode": args.activity_mode,
        "chunk_sec": args.chunk_sec,
        "min_chunk_sec": args.min_chunk_sec,
        "appearances_per_speaker": args.appearances_per_speaker,
        "extra_random_chunks": args.extra_random_chunks,
        "unique_speaker_combos": args.unique_speaker_combos,
        "seed": args.seed,
        "num_eligible_local_chunks": len(rows),
        "conversations": [],
    }

    for total_speakers in args.settings:
        seen_combos: set[tuple[str, ...]] = set()
        conv_idx = 0
        setting_attempts = 0
        max_setting_attempts = max(args.max_attempts, args.num_conversations * 10)
        while conv_idx < args.num_conversations:
            if setting_attempts >= max_setting_attempts:
                raise RuntimeError(
                    f"Failed to build {args.num_conversations} conversations for "
                    f"{total_speakers} speakers after {max_setting_attempts} attempts."
                )
            info = build_conversation(rows, speaker_counts, total_speakers, conv_idx, args, rng)
            combo_key = tuple(sorted(info["chosen_global_speakers"]))
            if args.unique_speaker_combos and combo_key in seen_combos:
                shutil.rmtree(info["conversation_dir"], ignore_errors=True)
                setting_attempts += 1
                continue
            seen_combos.add(combo_key)
            manifest["conversations"].append(info)
            conv_idx += 1
            setting_attempts += 1

    with (output_root / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"[done] wrote dataset manifest to {output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
