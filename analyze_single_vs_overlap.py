import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def summarize_by_overlap(df):
    confirmed = df[df["pred_memory_id"].astype(int) >= 0].copy()

    rows = []
    for num_active, g in confirmed.groupby("num_active"):
        switches = 0
        transitions = 0

        for spk, sg in g.groupby("true_speaker"):
            sg = sg.sort_values("segment_id")
            ids = sg["pred_memory_id"].astype(int).tolist()

            for a, b in zip(ids[:-1], ids[1:]):
                transitions += 1
                if a != b:
                    switches += 1

        rows.append({
            "num_active": num_active,
            "num_records": len(g),
            "num_speakers": g["true_speaker"].nunique(),
            "num_memory_ids": g["pred_memory_id"].nunique(),
            "id_switches": switches,
            "id_transitions": transitions,
            "switch_rate": switches / max(1, transitions),
        })

    return pd.DataFrame(rows)


def plot_lifetime_overlap(df, meeting_id, out_dir):
    mdf = df[df["meeting_id"] == meeting_id].copy()
    mdf = mdf[mdf["pred_memory_id"].astype(int) >= 0]

    if len(mdf) == 0:
        return

    speakers = sorted(mdf["true_speaker"].astype(str).unique())
    n = len(speakers)

    fig, axes = plt.subplots(
        nrows=n,
        ncols=1,
        figsize=(14, max(2, 1.2 * n)),
        sharex=True,
    )

    if n == 1:
        axes = [axes]

    for ax, spk in zip(axes, speakers):
        sdf = mdf[mdf["true_speaker"].astype(str) == spk].sort_values("segment_id")

        single = sdf[sdf["num_active"] == 1]
        overlap = sdf[sdf["num_active"] == 2]

        ax.plot(
            sdf["segment_id"],
            sdf["pred_memory_id"],
            alpha=0.35,
            linewidth=1.2,
        )

        ax.scatter(
            single["segment_id"],
            single["pred_memory_id"],
            s=35,
            marker="o",
            label="single" if spk == speakers[0] else None,
        )

        ax.scatter(
            overlap["segment_id"],
            overlap["pred_memory_id"],
            s=45,
            marker="x",
            label="overlap" if spk == speakers[0] else None,
        )

        ax.set_ylabel(f"Spk {spk}", rotation=0, labelpad=35)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Segment index")
    fig.supylabel("Assigned memory ID")
    fig.suptitle(f"Single vs Overlap Memory Assignment: {meeting_id}")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")

    plt.tight_layout()

    out_path = Path(out_dir) / f"{meeting_id}_single_vs_overlap_lifetime.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved: {out_path}")


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.assignments_csv)

    summary = summarize_by_overlap(df)
    summary_path = out_dir / "single_vs_overlap_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n=== Single vs Overlap Summary ===")
    print(summary)
    print(f"\nSaved summary: {summary_path}")

    if args.meeting_id is not None:
        meeting_ids = [args.meeting_id]
    else:
        meeting_ids = sorted(df["meeting_id"].unique())

    for meeting_id in meeting_ids:
        plot_lifetime_overlap(df, meeting_id, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--assignments_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--meeting_id", type=str, default=None)

    args = parser.parse_args()
    main(args)