import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def load_rttm_segments(path):
    segments = []
    with open(path, "r") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            dur = float(parts[4])
            spk = parts[7]
            segments.append((start, start + dur, spk))
    return segments


def load_hyp_segments(assign_df, mode):
    rows = []
    for _, row in assign_df.iterrows():
        pred_memory_id = int(row["pred_memory_id"])
        if mode == "confirmed_only":
            if pred_memory_id < 0:
                continue
            spk = f"mem_{pred_memory_id}"
        elif mode == "all_output":
            if pred_memory_id >= 0:
                spk = f"mem_{pred_memory_id}"
            else:
                pending_id = row["pending_id"]
                if pd.isna(pending_id):
                    continue
                spk = f"pending_{int(pending_id)}"
        else:
            raise ValueError(mode)

        rows.append((float(row["start_sec"]), float(row["end_sec"]), spk))
    return rows


def overlap_duration(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def build_mapping(ref_segments, hyp_segments):
    ref_labels = sorted({spk for _, _, spk in ref_segments})
    hyp_labels = sorted({spk for _, _, spk in hyp_segments})
    if not ref_labels or not hyp_labels:
        return {}

    score = np.zeros((len(hyp_labels), len(ref_labels)), dtype=np.float64)
    hyp_index = {spk: i for i, spk in enumerate(hyp_labels)}
    ref_index = {spk: i for i, spk in enumerate(ref_labels)}

    for hs, he, hspk in hyp_segments:
        hi = hyp_index[hspk]
        for rs, re, rspk in ref_segments:
            overlap = overlap_duration(hs, he, rs, re)
            if overlap > 0:
                ri = ref_index[rspk]
                score[hi, ri] += overlap

    row_ind, col_ind = linear_sum_assignment(-score)
    mapping = {}
    for r, c in zip(row_ind, col_ind):
        if score[r, c] > 0:
            mapping[hyp_labels[r]] = ref_labels[c]
    return mapping


def active_set(segments, start, end):
    return {
        spk
        for seg_start, seg_end, spk in segments
        if seg_start < end and seg_end > start
    }


def compute_der_one(ref_segments, hyp_segments):
    if not ref_segments:
        return {
            "ref_time": 0.0,
            "miss": 0.0,
            "fa": 0.0,
            "conf": 0.0,
            "der": 0.0,
        }

    mapping = build_mapping(ref_segments, hyp_segments)
    boundaries = sorted(
        {
            t
            for start, end, _ in ref_segments + hyp_segments
            for t in (start, end)
        }
    )
    if len(boundaries) < 2:
        return {
            "ref_time": 0.0,
            "miss": 0.0,
            "fa": 0.0,
            "conf": 0.0,
            "der": 0.0,
        }

    ref_time = 0.0
    miss = 0.0
    fa = 0.0
    conf = 0.0

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        dur = end - start
        if dur <= 0:
            continue

        ref_active = active_set(ref_segments, start, end)
        hyp_active = active_set(hyp_segments, start, end)
        mapped_hyp = {mapping[h] for h in hyp_active if h in mapping}

        ref_n = len(ref_active)
        hyp_n = len(hyp_active)
        correct = len(ref_active & mapped_hyp)

        ref_time += ref_n * dur
        miss += max(0, ref_n - hyp_n) * dur
        fa += max(0, hyp_n - ref_n) * dur
        conf += max(0, min(ref_n, hyp_n) - correct) * dur

    der = (miss + fa + conf) / ref_time if ref_time > 0 else 0.0
    return {
        "ref_time": ref_time,
        "miss": miss,
        "fa": fa,
        "conf": conf,
        "der": der,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignments_csv", required=True)
    parser.add_argument("--rttm_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    assign_df = pd.read_csv(args.assignments_csv)
    all_rows = []

    for meeting_id, meeting_df in assign_df.groupby("meeting_id"):
        rttm_path = Path(args.rttm_dir) / f"{meeting_id}.rttm"
        ref_segments = load_rttm_segments(rttm_path)

        out_row = {"meeting_id": meeting_id}
        for mode in ("confirmed_only", "all_output"):
            hyp_segments = load_hyp_segments(meeting_df, mode)
            res = compute_der_one(ref_segments, hyp_segments)
            out_row[f"{mode}_der"] = res["der"]
            out_row[f"{mode}_ref_time"] = res["ref_time"]
            out_row[f"{mode}_miss"] = res["miss"]
            out_row[f"{mode}_fa"] = res["fa"]
            out_row[f"{mode}_conf"] = res["conf"]
        all_rows.append(out_row)

    out_df = pd.DataFrame(all_rows).sort_values("meeting_id")
    out_df.to_csv(args.out_csv, index=False)

    print(f"Saved per-meeting DER to: {args.out_csv}")
    print("\nMacro average DER:")
    print(out_df[["confirmed_only_der", "all_output_der"]].mean())

    print("\nDuration-weighted DER:")
    for mode in ("confirmed_only", "all_output"):
        total_ref = out_df[f"{mode}_ref_time"].sum()
        total_err = (
            out_df[f"{mode}_miss"].sum()
            + out_df[f"{mode}_fa"].sum()
            + out_df[f"{mode}_conf"].sum()
        )
        weighted = total_err / total_ref if total_ref > 0 else 0.0
        print(f"{mode}_der = {weighted:.6f}")


if __name__ == "__main__":
    main()
