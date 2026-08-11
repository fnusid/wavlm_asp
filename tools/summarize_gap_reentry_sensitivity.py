import argparse
from pathlib import Path

import pandas as pd


def load_reentry_plans(manifest_csv: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_csv)
    plans = []
    for row in manifest.itertuples(index=False):
        timeline_path = Path(row.timeline_csv)
        plan_path = timeline_path.with_name(
            timeline_path.name.replace("_timeline.csv", "_reentry_plan.csv")
        )
        plans.append(pd.read_csv(plan_path))
    if not plans:
        raise ValueError(f"No reentry plans found from manifest: {manifest_csv}")
    return pd.concat(plans, ignore_index=True)


def representative_memory(window_df: pd.DataFrame) -> tuple[int, bool]:
    confirmed = window_df[window_df["pred_memory_id"] >= 0]
    if confirmed.empty:
        return -1, False
    mem_id = int(confirmed["pred_memory_id"].mode().iloc[0])
    return mem_id, True


def compute_meeting_metrics(
    assignments_csv: Path,
    plans_df: pd.DataFrame,
    reappearance_chunks: int,
) -> pd.DataFrame:
    assignments = pd.read_csv(assignments_csv)
    rows = []

    for plan in plans_df.itertuples(index=False):
        meeting_df = assignments[assignments["meeting_id"] == plan.meeting_id].copy()
        target_df = meeting_df[meeting_df["true_speaker"] == plan.target_speaker].copy()

        warmup_end_segment_id = int(
            getattr(plan, "warmup_end_segment_id", getattr(plan, "early_segment_id", 0))
        )
        max_reentry = int(
            getattr(plan, "reentry_segments", getattr(plan, "target_appearance_segments", 1))
        )
        reentry_len = min(reappearance_chunks, max_reentry)

        warmup_window = target_df[target_df["segment_id"] <= warmup_end_segment_id].copy()
        reentry_window = target_df[
            (target_df["segment_id"] >= int(plan.reentry_segment_id))
            & (target_df["segment_id"] < int(plan.reentry_segment_id) + reentry_len)
        ].copy()

        early_mem, early_confirmed = representative_memory(warmup_window)
        reentry_mem, reentry_confirmed = representative_memory(reentry_window)
        both_confirmed = early_confirmed and reentry_confirmed
        reid = int(both_confirmed and early_mem == reentry_mem)

        rows.append(
            {
                "meeting_id": plan.meeting_id,
                "actual_gap_sec": float(plan.actual_gap_sec),
                "both_confirmed": both_confirmed,
                "reid": reid,
                "warmup_confirmed": early_confirmed,
                "reentry_confirmed": reentry_confirmed,
            }
        )

    return pd.DataFrame(rows).sort_values(["actual_gap_sec", "meeting_id"])


def summarize_setting(
    meeting_df: pd.DataFrame,
    threshold: float,
    reappearance_chunks: int,
) -> dict:
    confirmed_df = meeting_df[meeting_df["both_confirmed"]].copy()
    reid_all = meeting_df["reid"].astype(float)

    if confirmed_df.empty:
        reid_confirmed = pd.Series(dtype=float)
    else:
        reid_confirmed = confirmed_df["reid"].astype(float)

    def sample_std(series: pd.Series) -> float:
        if len(series) <= 1:
            return 0.0
        return float(series.std(ddof=1))

    def sem(series: pd.Series) -> float:
        if len(series) <= 1:
            return 0.0
        return float(series.std(ddof=1) / (len(series) ** 0.5))

    return {
        "threshold": threshold,
        "reappearance_chunks": reappearance_chunks,
        "n_meetings": int(len(meeting_df)),
        "both_confirmed_rate": float(meeting_df["both_confirmed"].mean()),
        "reid_all_mean": float(reid_all.mean()),
        "reid_all_std": sample_std(reid_all),
        "reid_all_sem": sem(reid_all),
        "n_confirmed": int(len(reid_confirmed)),
        "reid_confirmed_mean": float(reid_confirmed.mean()) if len(reid_confirmed) else float("nan"),
        "reid_confirmed_std": sample_std(reid_confirmed),
        "reid_confirmed_sem": sem(reid_confirmed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--reappearance_chunks", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument(
        "--setting",
        action="append",
        nargs=2,
        metavar=("THRESHOLD", "ASSIGNMENTS_CSV"),
        required=True,
        help="Repeat as: --setting 0.4 /path/to/meeting_memory_assignments.csv",
    )
    args = parser.parse_args()

    plans_df = load_reentry_plans(args.manifest_csv)
    summary_rows = []

    for threshold_str, assignments_path_str in args.setting:
        threshold = float(threshold_str)
        assignments_csv = Path(assignments_path_str)
        for reappearance_chunks in args.reappearance_chunks:
            meeting_df = compute_meeting_metrics(
                assignments_csv=assignments_csv,
                plans_df=plans_df,
                reappearance_chunks=reappearance_chunks,
            )
            summary_rows.append(
                summarize_setting(
                    meeting_df=meeting_df,
                    threshold=threshold,
                    reappearance_chunks=reappearance_chunks,
                )
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["threshold", "reappearance_chunks"]
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.output_csv, index=False)
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {args.output_csv}")


if __name__ == "__main__":
    main()
