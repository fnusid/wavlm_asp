import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

BASE_FONT_SIZE = 8
FONT_SCALE = 2
PLOT_FONT_SIZE = BASE_FONT_SIZE * FONT_SCALE


def timeline_speakers(timeline_csv, meeting_id):
    tdf = pd.read_csv(timeline_csv)
    if "meeting_id" in tdf.columns:
        tdf = tdf[tdf["meeting_id"].astype(str) == str(meeting_id)].copy()

    ordered = []
    seen = set()
    for active in tdf["active_speakers"].astype(str):
        for spk in active.split("|"):
            spk = spk.strip()
            if spk and spk not in seen:
                seen.add(spk)
                ordered.append(spk)
    return ordered


def plot_one_meeting(
    df,
    meeting_id,
    out_dir,
    timeline_csv=None,
    use_time=False,
    max_mem_legend=12,
):
    meeting_df = df[df["meeting_id"] == meeting_id].copy()
    mdf = meeting_df[meeting_df["pred_memory_id"].astype(int) >= 0].copy()

    if len(mdf) == 0:
        print(f"Skipping {meeting_id}: no confirmed assignments.")
        return

    mdf["true_speaker"] = mdf["true_speaker"].astype(str)
    mdf["pred_memory_id"] = mdf["pred_memory_id"].astype(int)
    mdf["num_active"] = mdf["num_active"].astype(int)

    predicted_speakers = (
        mdf.groupby("true_speaker")["segment_id"]
        .min()
        .sort_values()
        .index
        .tolist()
    )
    if timeline_csv is not None:
        speakers = timeline_speakers(timeline_csv, meeting_id)
        if not speakers:
            speakers = predicted_speakers
    else:
        speakers = predicted_speakers

    mem_ids = sorted(mdf["pred_memory_id"].unique())
    cmap = plt.get_cmap("tab20")
    mem_to_color = {mem: cmap(i % 20) for i, mem in enumerate(mem_ids)}

    fig_height = max(3.5, 0.45 * len(speakers) + 1.8)
    fig, ax = plt.subplots(figsize=(20, fig_height))

    y_positions = {spk: i for i, spk in enumerate(speakers)}

    for _, row in mdf.iterrows():
        spk = row["true_speaker"]
        y = y_positions[spk]
        mem_id = row["pred_memory_id"]

        if use_time and {"start_sec", "end_sec"}.issubset(mdf.columns):
            x = float(row["start_sec"])
            width = float(row["end_sec"]) - float(row["start_sec"])
        else:
            x = float(row["segment_id"]) - 0.45
            width = 0.9

        is_overlap = int(row["num_active"]) > 1

        rect = mpatches.Rectangle(
            (x, y - 0.35),
            width,
            0.7,
            facecolor=mem_to_color[mem_id],
            edgecolor="black" if is_overlap else "none",
            linewidth=0.6,
            hatch="//" if is_overlap else None,
            alpha=0.9,
        )
        ax.add_patch(rect)

    if use_time and {"start_sec", "end_sec"}.issubset(mdf.columns):
        x_min = float(mdf["start_sec"].min())
        x_max = float(mdf["end_sec"].max())
        ax.set_xlabel("Time (s)", fontsize=PLOT_FONT_SIZE)
    else:
        x_min = float(mdf["segment_id"].min() - 1)
        x_max = float(mdf["segment_id"].max() + 1)
        ax.set_xlabel("Segment index", fontsize=PLOT_FONT_SIZE)
    ax.set_xlim(x_min, x_max)

    predicted_set = set(predicted_speakers)
    missing_speakers = [spk for spk in speakers if spk not in predicted_set]
    for spk in missing_speakers:
        y = y_positions[spk]
        ax.hlines(
            y,
            xmin=x_min,
            xmax=x_max,
            colors="0.7",
            linestyles="dotted",
            linewidth=1.0,
        )
        ax.text(
            x_max,
            y,
            " no confirmed prediction",
            ha="left",
            va="center",
            fontsize=PLOT_FONT_SIZE,
            color="0.35",
        )

    ax.set_yticks(range(len(speakers)))
    ax.set_yticklabels([f"Spk {s}" for s in speakers], fontsize=PLOT_FONT_SIZE)
    ax.set_ylim(-0.8, len(speakers) - 0.2)
    ax.set_ylabel("Ground-truth speaker", fontsize=PLOT_FONT_SIZE)
    ax.tick_params(axis="x", labelsize=PLOT_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=PLOT_FONT_SIZE)
    ax.grid(axis="x", alpha=0.25)

    legend_items = [
        mpatches.Patch(facecolor="white", edgecolor="none", label="Solid: single-speaker segment"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="//", label="Hatched: overlap segment"),
    ]
    if missing_speakers:
        legend_items.append(
            mpatches.Patch(facecolor="white", edgecolor="0.7", linestyle="dotted", label="No confirmed prediction")
        )

    if len(mem_ids) <= max_mem_legend:
        legend_items += [
            mpatches.Patch(
                facecolor=mem_to_color[mem],
                edgecolor="none",
                label=f"Memory {mem}",
            )
            for mem in mem_ids
        ]

    ax.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=min(5, len(legend_items)),
        fontsize=PLOT_FONT_SIZE,
        frameon=True,
    )

    plt.tight_layout()

    out_path = Path(out_dir) / f"{meeting_id}_speaker_memory_timeline.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.assignments_csv)

    if args.meeting_id is not None:
        meeting_ids = [args.meeting_id]
    else:
        meeting_ids = sorted(df["meeting_id"].unique())

    for meeting_id in meeting_ids:
        plot_one_meeting(
            df=df,
            meeting_id=meeting_id,
            out_dir=out_dir,
            timeline_csv=args.timeline_csv,
            use_time=args.use_time,
            max_mem_legend=args.max_mem_legend,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--assignments_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--meeting_id", type=str, default=None)
    parser.add_argument(
        "--timeline_csv",
        type=str,
        default=None,
        help="Optional timeline CSV used to preserve full ground-truth speaker list.",
    )

    parser.add_argument(
        "--use_time",
        action="store_true",
        help="Use start_sec/end_sec on x-axis instead of segment_id.",
    )

    parser.add_argument(
        "--max_mem_legend",
        type=int,
        default=12,
        help="Only show memory-ID legend if number of memory IDs is <= this.",
    )

    args = parser.parse_args()
    main(args)
