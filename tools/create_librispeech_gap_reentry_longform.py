#!/usr/bin/env python3
"""Create LibriSpeech-based long-form meetings with controlled re-entry gaps.

This generator builds synthetic meetings designed for long-form speaker-memory
evaluation. Each meeting contains a fixed pool of true speakers and one target
speaker with a clearly defined:

1. warmup period where all speakers appear multiple times
2. controlled absence gap for the target speaker
3. later target reappearance

The output schema matches the existing long-form evaluators:
  - manifest.csv
  - per-meeting timeline CSV
  - per-meeting speakers CSV

It also writes a per-meeting reentry_plan.csv that explicitly records the target
speaker, the early segment, the re-entry segment, and the realized gap.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


RATE = 16000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--librispeech-root",
        default="/home/sidcs/datasets/LibriMix/LibriMix/LibriSpeech/test-clean",
    )
    parser.add_argument(
        "--output-root",
        default="/home/sidcs/datasets/LibriMix/librispeech_gap_reentry_longform",
    )
    parser.add_argument("--speaker-counts", nargs="+", type=int, default=[8, 10])
    parser.add_argument("--gap-secs", nargs="+", type=float, default=[10, 30, 60, 120, 300])
    parser.add_argument("--meetings-per-setting", type=int, default=3)
    parser.add_argument("--segment-sec", type=float, default=1.0)
    parser.add_argument(
        "--warmup-rounds",
        type=int,
        default=1,
        help="Number of full warmup passes over the speaker list.",
    )
    parser.add_argument(
        "--warmup-segments-per-speaker",
        type=int,
        default=3,
        help="Number of consecutive chunks per speaker within each warmup pass.",
    )
    parser.add_argument(
        "--reentry-segments",
        type=int,
        default=3,
        help="Number of consecutive chunks for the target speaker after re-entry.",
    )
    parser.add_argument("--tail-sec", type=float, default=0.0)
    parser.add_argument("--min-utts-per-speaker", type=int, default=8)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_audio(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32")
    if sr != RATE:
        raise ValueError(f"Expected {RATE} Hz audio, got {sr} for {path}")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio.astype(np.float32)


def save_audio(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, RATE)


def crop_or_pad(audio: np.ndarray, target_len: int, rng: random.Random) -> np.ndarray:
    if len(audio) >= target_len:
        if len(audio) == target_len:
            return audio.copy()
        start = rng.randint(0, len(audio) - target_len)
        return audio[start : start + target_len].copy()

    out = np.zeros(target_len, dtype=np.float32)
    out[: len(audio)] = audio
    return out


def discover_speakers(root: Path, min_utts_per_speaker: int) -> dict[str, list[Path]]:
    speaker_to_files: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*.flac")) + sorted(root.rglob("*.wav")):
        if len(path.parts) < 3:
            continue
        speaker_id = path.parts[-3]
        speaker_to_files.setdefault(speaker_id, []).append(path)
    return {
        spk: files
        for spk, files in speaker_to_files.items()
        if len(files) >= min_utts_per_speaker
    }


@dataclass
class SpeakerSampler:
    speaker_id: str
    files: list[Path]
    rng: random.Random
    _order: list[int] | None = None
    _cursor: int = 0

    def _ensure_order(self) -> None:
        if self._order is None or self._cursor >= len(self._order):
            self._order = list(range(len(self.files)))
            self.rng.shuffle(self._order)
            self._cursor = 0

    def sample_segment(self, target_len: int) -> tuple[np.ndarray, Path]:
        self._ensure_order()
        idx = self._order[self._cursor]
        self._cursor += 1
        src_path = self.files[idx]
        audio = load_audio(src_path)
        return crop_or_pad(audio, target_len, self.rng), src_path


def build_filler_schedule(other_speakers: list[str], num_slots: int, rng: random.Random) -> list[str]:
    if num_slots <= 0:
        return []
    if not other_speakers:
        raise ValueError("Need at least one non-target speaker for filler schedule.")

    out: list[str] = []
    while len(out) < num_slots:
        block = other_speakers[:]
        rng.shuffle(block)
        if out and len(block) > 1 and block[0] == out[-1]:
            block[0], block[1] = block[1], block[0]
        out.extend(block)
    return out[:num_slots]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_meeting(
    out_root: Path,
    chosen_speakers: list[str],
    target_speaker: str,
    gap_sec: float,
    meeting_idx: int,
    segment_sec: float,
    warmup_rounds: int,
    warmup_segments_per_speaker: int,
    reentry_segments: int,
    tail_sec: float,
    samplers: dict[str, SpeakerSampler],
    rng: random.Random,
    force: bool,
) -> dict[str, object]:
    gap_slots = int(round(gap_sec / segment_sec))
    actual_gap_sec = gap_slots * segment_sec
    target_len = int(round(segment_sec * RATE))
    tail_len = int(round(tail_sec * RATE))
    other_speakers = [spk for spk in chosen_speakers if spk != target_speaker]
    filler_schedule = build_filler_schedule(other_speakers, gap_slots, rng)

    meeting_id = f"gap{int(actual_gap_sec):03d}_{len(chosen_speakers):02d}sp_{meeting_idx:04d}"
    meeting_dir = out_root / f"gap_{int(actual_gap_sec):03d}s" / f"{len(chosen_speakers):02d}sp" / meeting_id
    metadata_dir = meeting_dir / "metadata"
    if meeting_dir.exists() and force:
        shutil.rmtree(meeting_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    if warmup_rounds < 1:
        raise ValueError("warmup_rounds must be >= 1")
    if warmup_segments_per_speaker < 1:
        raise ValueError("warmup_segments_per_speaker must be >= 1")
    if reentry_segments < 1:
        raise ValueError("reentry_segments must be >= 1")

    warmup_schedule: list[str] = []
    for _ in range(warmup_rounds):
        round_speakers = chosen_speakers[:]
        rng.shuffle(round_speakers)
        if warmup_schedule and round_speakers[0] == warmup_schedule[-1] and len(round_speakers) > 1:
            round_speakers[0], round_speakers[1] = round_speakers[1], round_speakers[0]
        for speaker_id in round_speakers:
            warmup_schedule.extend([speaker_id] * warmup_segments_per_speaker)

    segment_plan = warmup_schedule + filler_schedule + [target_speaker] * reentry_segments
    if tail_len > 0:
        segment_plan.extend(build_filler_schedule(other_speakers or [target_speaker], int(round(tail_sec / segment_sec)), rng))

    mix_parts: list[np.ndarray] = []
    timeline_rows: list[dict[str, object]] = []
    chunk_rows: list[dict[str, object]] = []
    first_anchor: dict[str, tuple[float, float]] = {}

    cursor = 0
    warmup_end_segment_id = len(warmup_schedule) - 1
    reentry_segment_id = len(warmup_schedule) + gap_slots

    for seg_id, speaker_id in enumerate(segment_plan):
        audio, src_path = samplers[speaker_id].sample_segment(target_len)
        seg_path = meeting_dir / "chunks" / speaker_id / f"segment_{seg_id:04d}.wav"
        save_audio(seg_path, audio)

        start_sec = cursor / RATE
        end_sec = (cursor + len(audio)) / RATE
        if speaker_id not in first_anchor:
            first_anchor[speaker_id] = (start_sec, end_sec)

        mix_parts.append(audio)
        timeline_rows.append(
            {
                "meeting_id": meeting_id,
                "segment_id": seg_id,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "num_active": 1,
                "active_speakers": speaker_id,
                "speaker_1_id": speaker_id,
                "speaker_1_source_path": str(seg_path),
                "speaker_2_id": "",
                "speaker_2_source_path": "",
            }
        )
        chunk_rows.append(
            {
                "meeting_id": meeting_id,
                "segment_id": seg_id,
                "speaker_id": speaker_id,
                "local_chunk_path": str(seg_path),
                "source_dataset_path": str(src_path),
                "start_sec": start_sec,
                "end_sec": end_sec,
            }
        )
        cursor += len(audio)

    mix_audio = np.concatenate(mix_parts) if mix_parts else np.zeros(0, dtype=np.float32)
    mix_path = meeting_dir / "mix.wav"
    save_audio(mix_path, mix_audio)

    speakers_rows = []
    for speaker_id in chosen_speakers:
        anchor_start, anchor_end = first_anchor[speaker_id]
        speakers_rows.append(
            {
                "meeting_id": meeting_id,
                "speaker_id": speaker_id,
                "speaker_ref_path": str(mix_path),
                "anchor_start_sec": anchor_start,
                "anchor_end_sec": anchor_end,
                "anchor_type": "first_occurrence",
            }
        )

    reentry_rows = [
        {
            "meeting_id": meeting_id,
            "target_speaker": target_speaker,
            "warmup_end_segment_id": warmup_end_segment_id,
            "reentry_segment_id": reentry_segment_id,
            "desired_gap_sec": gap_sec,
            "actual_gap_sec": actual_gap_sec,
            "segment_sec": segment_sec,
            "warmup_rounds": warmup_rounds,
            "warmup_segments_per_speaker": warmup_segments_per_speaker,
            "reentry_segments": reentry_segments,
            "num_global_speakers": len(chosen_speakers),
        }
    ]

    timeline_csv = metadata_dir / f"{meeting_id}_timeline.csv"
    speakers_csv = metadata_dir / f"{meeting_id}_speakers.csv"
    reentry_csv = metadata_dir / f"{meeting_id}_reentry_plan.csv"
    chunks_csv = metadata_dir / f"{meeting_id}_chunks.csv"

    write_csv(timeline_csv, list(timeline_rows[0].keys()), timeline_rows)
    write_csv(speakers_csv, list(speakers_rows[0].keys()), speakers_rows)
    write_csv(reentry_csv, list(reentry_rows[0].keys()), reentry_rows)
    write_csv(chunks_csv, list(chunk_rows[0].keys()), chunk_rows)

    return {
        "meeting_id": meeting_id,
        "mix_path": str(mix_path),
        "timeline_csv": str(timeline_csv),
        "speakers_csv": str(speakers_csv),
        "num_global_speakers": len(chosen_speakers),
        "meeting_sec": len(mix_audio) / RATE,
        "segment_sec": segment_sec,
        "max_overlap": 1,
        "target_speaker": target_speaker,
        "desired_gap_sec": gap_sec,
        "actual_gap_sec": actual_gap_sec,
        "num_filler_segments": gap_slots,
        "warmup_rounds": warmup_rounds,
        "warmup_segments_per_speaker": warmup_segments_per_speaker,
        "reentry_segments": reentry_segments,
        "reentry_plan_csv": str(reentry_csv),
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    librispeech_root = Path(args.librispeech_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.segment_sec <= 0:
        raise ValueError("--segment-sec must be positive")
    if args.warmup_rounds < 1:
        raise ValueError("--warmup-rounds must be at least 1")
    if args.warmup_segments_per_speaker < 1:
        raise ValueError("--warmup-segments-per-speaker must be at least 1")
    if args.reentry_segments < 1:
        raise ValueError("--reentry-segments must be at least 1")

    speaker_to_files = discover_speakers(librispeech_root, args.min_utts_per_speaker)
    if not speaker_to_files:
        raise RuntimeError("No eligible LibriSpeech speakers found.")

    manifest_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for speaker_count in args.speaker_counts:
        if speaker_count < 2:
            raise ValueError("--speaker-counts must be at least 2")
        if len(speaker_to_files) < speaker_count:
            raise RuntimeError(
                f"Need {speaker_count} speakers, but only found {len(speaker_to_files)} eligible speakers."
            )

        for gap_sec in args.gap_secs:
            gap_slots = int(round(gap_sec / args.segment_sec))
            actual_gap_sec = gap_slots * args.segment_sec
            if gap_slots < speaker_count - 1:
                raise RuntimeError(
                    f"Gap {gap_sec}s with segment {args.segment_sec}s gives only {gap_slots} filler slots, "
                    f"which is fewer than speaker_count-1={speaker_count - 1}. "
                    "Decrease speaker count or segment_sec, or increase gap_sec."
                )

            for meeting_idx in range(args.meetings_per_setting):
                chosen_speakers = rng.sample(sorted(speaker_to_files.keys()), speaker_count)
                target_speaker = rng.choice(chosen_speakers)
                speaker_rng = random.Random(rng.randint(0, 10**9))
                samplers = {
                    spk: SpeakerSampler(spk, speaker_to_files[spk], random.Random(speaker_rng.randint(0, 10**9)))
                    for spk in chosen_speakers
                }

                row = build_meeting(
                    out_root=output_root,
                    chosen_speakers=chosen_speakers,
                    target_speaker=target_speaker,
                    gap_sec=gap_sec,
                    meeting_idx=len(manifest_rows),
                    segment_sec=args.segment_sec,
                    warmup_rounds=args.warmup_rounds,
                    warmup_segments_per_speaker=args.warmup_segments_per_speaker,
                    reentry_segments=args.reentry_segments,
                    tail_sec=args.tail_sec,
                    samplers=samplers,
                    rng=speaker_rng,
                    force=args.force,
                )
                manifest_rows.append(
                    {
                        "meeting_id": row["meeting_id"],
                        "mix_path": row["mix_path"],
                        "timeline_csv": row["timeline_csv"],
                        "speakers_csv": row["speakers_csv"],
                        "num_global_speakers": row["num_global_speakers"],
                        "meeting_sec": row["meeting_sec"],
                        "segment_sec": row["segment_sec"],
                        "max_overlap": row["max_overlap"],
                    }
                )
                summary_rows.append(row)

    manifest_csv = output_root / "manifest.csv"
    summary_csv = output_root / "build_summary.csv"
    write_csv(manifest_csv, list(manifest_rows[0].keys()), manifest_rows)
    write_csv(summary_csv, list(summary_rows[0].keys()), summary_rows)

    print(f"Built manifest: {manifest_csv}")
    print(f"Built summary:  {summary_csv}")
    print(f"Num meetings:   {len(manifest_rows)}")


if __name__ == "__main__":
    main()
