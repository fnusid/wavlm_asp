import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import soundfile as sf


def load_rttm(path):
    rows = []
    with open(path, "r") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            dur = float(parts[4])
            speaker = parts[7]
            rows.append((start, start + dur, speaker))
    return rows


def build_constant_speaker_segments(rttm_rows):
    boundaries = sorted({t for start, end, _ in rttm_rows for t in (start, end)})
    segments = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end <= start:
            continue
        active = sorted(
            {
                spk
                for seg_start, seg_end, spk in rttm_rows
                if seg_start < end and seg_end > start
            }
        )
        if not active:
            continue
        segments.append((start, end, active))
    return segments


def choose_anchor_segments(rttm_rows):
    speaker_solo_segments = defaultdict(list)
    speaker_all_segments = defaultdict(list)

    constant_segments = build_constant_speaker_segments(rttm_rows)
    for start, end, active in constant_segments:
        dur = end - start
        if dur <= 0:
            continue
        for spk in active:
            speaker_all_segments[spk].append((dur, start, end))
        if len(active) == 1:
            speaker_solo_segments[active[0]].append((dur, start, end))

    anchors = {}
    fallback_count = 0
    for spk, spans in speaker_all_segments.items():
        if speaker_solo_segments[spk]:
            _, start, end = max(speaker_solo_segments[spk], key=lambda x: x[0])
            anchor_type = "solo"
        else:
            _, start, end = max(spans, key=lambda x: x[0])
            anchor_type = "fallback_overlap"
            fallback_count += 1
        anchors[spk] = {
            "anchor_start_sec": start,
            "anchor_end_sec": end,
            "anchor_type": anchor_type,
        }
    return anchors, fallback_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wav_dir",
        default="/home/sidcs/datasets/voxconverse_test_wav",
    )
    parser.add_argument(
        "--rttm_dir",
        default="/home/sidcs/tmp_voxconverse_repo/test",
    )
    parser.add_argument(
        "--out_dir",
        default="/home/sidcs/datasets/voxconverse_longform_test",
    )
    args = parser.parse_args()

    wav_dir = Path(args.wav_dir)
    rttm_dir = Path(args.rttm_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    summary_rows = []

    wav_paths = sorted(wav_dir.glob("*.wav"))
    for wav_path in wav_paths:
        meeting_id = wav_path.stem
        rttm_path = rttm_dir / f"{meeting_id}.rttm"
        if not rttm_path.exists():
            continue

        meeting_dir = out_dir / meeting_id / "metadata"
        meeting_dir.mkdir(parents=True, exist_ok=True)

        rttm_rows = load_rttm(rttm_path)
        constant_segments = build_constant_speaker_segments(rttm_rows)
        anchors, fallback_count = choose_anchor_segments(rttm_rows)

        timeline_rows = []
        skipped_gt2 = 0
        skipped_gt2_dur = 0.0
        kept_dur = 0.0
        max_overlap = 0

        for seg_id, (start, end, active) in enumerate(constant_segments):
            num_active = len(active)
            max_overlap = max(max_overlap, num_active)
            if num_active > 2:
                skipped_gt2 += 1
                skipped_gt2_dur += end - start
                continue

            kept_dur += end - start
            row = {
                "meeting_id": meeting_id,
                "segment_id": len(timeline_rows),
                "start_sec": start,
                "end_sec": end,
                "num_active": num_active,
                "active_speakers": "|".join(active),
                "speaker_1_id": active[0] if num_active >= 1 else "",
                "speaker_1_source_path": str(wav_path),
                "speaker_2_id": active[1] if num_active >= 2 else "",
                "speaker_2_source_path": str(wav_path) if num_active >= 2 else "",
            }
            timeline_rows.append(row)

        timeline_df = pd.DataFrame(timeline_rows)
        timeline_csv = meeting_dir / f"{meeting_id}_timeline.csv"
        timeline_df.to_csv(timeline_csv, index=False)

        speaker_rows = []
        for spk in sorted(anchors.keys()):
            anchor = anchors[spk]
            speaker_rows.append(
                {
                    "meeting_id": meeting_id,
                    "speaker_id": spk,
                    "speaker_ref_path": str(wav_path),
                    "anchor_start_sec": anchor["anchor_start_sec"],
                    "anchor_end_sec": anchor["anchor_end_sec"],
                    "anchor_type": anchor["anchor_type"],
                }
            )
        speakers_df = pd.DataFrame(speaker_rows)
        speakers_csv = meeting_dir / f"{meeting_id}_speakers.csv"
        speakers_df.to_csv(speakers_csv, index=False)

        info = sf.info(str(wav_path))
        duration = info.frames / info.samplerate

        manifest_rows.append(
            {
                "meeting_id": meeting_id,
                "mix_path": str(wav_path),
                "timeline_csv": str(timeline_csv),
                "speakers_csv": str(speakers_csv),
                "num_global_speakers": len(anchors),
                "meeting_sec": duration,
                "segment_sec": "",
                "max_overlap": max_overlap,
            }
        )
        summary_rows.append(
            {
                "meeting_id": meeting_id,
                "num_global_speakers": len(anchors),
                "max_overlap": max_overlap,
                "timeline_rows_kept": len(timeline_rows),
                "kept_duration_sec": kept_dur,
                "skipped_gt2_segments": skipped_gt2,
                "skipped_gt2_duration_sec": skipped_gt2_dur,
                "fallback_anchor_speakers": fallback_count,
            }
        )

    manifest_df = pd.DataFrame(manifest_rows)
    summary_df = pd.DataFrame(summary_rows)

    manifest_path = out_dir / "manifest.csv"
    summary_path = out_dir / "build_summary.csv"
    manifest_df.to_csv(manifest_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Built manifest: {manifest_path}")
    print(f"Built summary:  {summary_path}")
    print(summary_df.mean(numeric_only=True))


if __name__ == "__main__":
    main()
