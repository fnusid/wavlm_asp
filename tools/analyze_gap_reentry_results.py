import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_reentry_plans(manifest_csv: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_csv)
    plans = []
    for row in manifest.itertuples(index=False):
        timeline_path = Path(row.timeline_csv)
        plan_path = timeline_path.with_name(timeline_path.name.replace("_timeline.csv", "_reentry_plan.csv"))
        plan = pd.read_csv(plan_path)
        plans.append(plan)
    if not plans:
        raise ValueError(f"No reentry plans found from manifest: {manifest_csv}")
    return pd.concat(plans, ignore_index=True)


def compute_metrics(assignments_csv: Path, plans_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignments = pd.read_csv(assignments_csv)
    meeting_rows = []

    for plan in plans_df.itertuples(index=False):
        meeting_df = assignments[assignments["meeting_id"] == plan.meeting_id].copy()
        target_df = meeting_df[meeting_df["true_speaker"] == plan.target_speaker].copy()
        reentry_len = int(getattr(plan, "reentry_segments", getattr(plan, "target_appearance_segments", 1)))
        warmup_end_segment_id = int(getattr(plan, "warmup_end_segment_id", getattr(plan, "early_segment_id", 0)))

        warmup_window = target_df[target_df["segment_id"] <= warmup_end_segment_id].copy()
        reentry_window = meeting_df[
            (meeting_df["segment_id"] >= plan.reentry_segment_id)
            & (meeting_df["segment_id"] < plan.reentry_segment_id + reentry_len)
        ].copy()
        if warmup_window.empty or reentry_window.empty:
            raise ValueError(f"Missing planned windows for meeting {plan.meeting_id}")

        def representative_memory(window_df: pd.DataFrame) -> tuple[int, bool]:
            confirmed = window_df[window_df["pred_memory_id"] >= 0]
            if confirmed.empty:
                return -1, False
            mem_id = int(confirmed["pred_memory_id"].mode().iloc[0])
            return mem_id, True

        early_mem, early_confirmed = representative_memory(warmup_window)
        reentry_mem, reentry_confirmed = representative_memory(reentry_window)
        both_confirmed = early_confirmed and reentry_confirmed
        reid_all = int(both_confirmed and early_mem == reentry_mem)
        switch_all = int(both_confirmed and early_mem != reentry_mem)

        confirmed_target = target_df[target_df["pred_memory_id"] >= 0]
        target_memories = sorted(confirmed_target["pred_memory_id"].unique().tolist())
        fragmentation_all = len(target_memories)

        after_reentry = target_df[target_df["segment_id"] >= plan.reentry_segment_id].copy()
        if reentry_confirmed:
            confirmed_after = after_reentry[after_reentry["pred_memory_id"] >= 0]
            if confirmed_after.empty:
                reentry_switch_rate_confirmed = None
            else:
                reentry_switch_rate_confirmed = float(
                    (confirmed_after["pred_memory_id"] != reentry_mem).mean()
                )
        else:
            reentry_switch_rate_confirmed = None

        meeting_rows.append(
            {
                "meeting_id": plan.meeting_id,
                "target_speaker": int(plan.target_speaker),
                "desired_gap_sec": float(plan.desired_gap_sec),
                "actual_gap_sec": float(plan.actual_gap_sec),
                "warmup_end_segment_id": warmup_end_segment_id,
                "reentry_segment_id": int(plan.reentry_segment_id),
                "reentry_segments": reentry_len,
                "warmup_memory_id": early_mem,
                "reentry_memory_id": reentry_mem,
                "warmup_confirmed": early_confirmed,
                "reentry_confirmed": reentry_confirmed,
                "both_confirmed": both_confirmed,
                "reid_all": reid_all,
                "switch_all": switch_all,
                "fragmentation_all": fragmentation_all,
                "reentry_switch_rate_confirmed": reentry_switch_rate_confirmed,
            }
        )

    meeting_metrics = pd.DataFrame(meeting_rows).sort_values(["actual_gap_sec", "meeting_id"])
    gap_summary = (
        meeting_metrics.groupby("actual_gap_sec", as_index=False)
        .agg(
            num_meetings=("meeting_id", "count"),
            both_confirmed_rate=("both_confirmed", "mean"),
            reid_all=("reid_all", "mean"),
            switch_all=("switch_all", "mean"),
            fragmentation_all=("fragmentation_all", "mean"),
            reentry_switch_rate_confirmed=("reentry_switch_rate_confirmed", "mean"),
        )
        .sort_values("actual_gap_sec")
    )
    confirmed_only = (
        meeting_metrics[meeting_metrics["both_confirmed"]]
        .groupby("actual_gap_sec", as_index=False)
        .agg(reid_confirmed_only=("reid_all", "mean"))
    )
    gap_summary = gap_summary.merge(confirmed_only, on="actual_gap_sec", how="left")
    return meeting_metrics, gap_summary


def make_plot(gap_summary: pd.DataFrame, output_prefix: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(
        gap_summary["actual_gap_sec"],
        gap_summary["reid_all"],
        marker="o",
        linewidth=2,
        label="Re-ID (all meetings)",
    )
    if gap_summary["reid_confirmed_only"].notna().any():
        plt.plot(
            gap_summary["actual_gap_sec"],
            gap_summary["reid_confirmed_only"],
            marker="s",
            linewidth=2,
            label="Re-ID (both confirmed only)",
        )
    plt.xlabel("Absence gap (s)")
    plt.ylabel("Re-ID score")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_prefix.with_suffix(".png"), dpi=200)
    plt.savefig(output_prefix.with_suffix(".pdf"))
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_csv", type=Path, required=True)
    parser.add_argument("--assignments_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plans_df = load_reentry_plans(args.manifest_csv)
    meeting_metrics, gap_summary = compute_metrics(args.assignments_csv, plans_df)

    meeting_path = args.output_dir / "gap_reentry_meeting_metrics.csv"
    gap_path = args.output_dir / "gap_reentry_summary.csv"
    plot_prefix = args.output_dir / "gap_reentry_reid_curve"
    meeting_metrics.to_csv(meeting_path, index=False)
    gap_summary.to_csv(gap_path, index=False)
    make_plot(gap_summary, plot_prefix)

    print("Saved:", meeting_path)
    print("Saved:", gap_path)
    print("Saved:", plot_prefix.with_suffix(".png"))
    print("Saved:", plot_prefix.with_suffix(".pdf"))
    print()
    print(gap_summary.to_string(index=False))


if __name__ == "__main__":
    main()
