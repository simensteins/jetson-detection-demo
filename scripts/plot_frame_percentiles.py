#!/usr/bin/env python3
"""Frame-time distribution per M15 stage as a bar chart: mean, p50 and p99
of the frame time (p95 is in the CSV only), side by side for each stage, from the profiled
paper-set runs at 25 W (the figure version of the frame-time table in the
paper: the fixed-shape NMS in the graph removes the ~9.5 ms tail that every
torchvision-NMS stage has, so p99 drops to the median).

Frame time = decode start to the next decode start, the first --skip-frames
frames excluded (550 frames remain), computed by
plot_compare_reports.summarize_stage, the same definition as the other
paper figures. Stages 1-3 come from the 25.10 session (paper25w), stages 4-5
from the 27.10 session in the new stage order (paper25w_v2).

Style as the power-mode figures: no title, no numbers on the bars, black
axes and labels, larger text.

Usage (from the repo root, any Python with matplotlib):
    python3 scripts/plot_frame_percentiles.py
    python3 scripts/plot_frame_percentiles.py --outdir reports/figures/paper25w_v2
Writes <outdir>/paper25w_v2_m15_frame_percentiles.png/.pdf and .csv.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_compare_reports import summarize_stage  # noqa: E402
from plot_paper_figures import save, style_axes  # noqa: E402

R = "reports/nsys_paper_reports"
STAGES = [   # label, report
    ("Baseline", f"{R}/paper25w_m15_baseline.sqlite"),
    ("+ CUDA\ngraph", f"{R}/paper25w_m15_graph.sqlite"),
    ("+ GPU\npreproc.", f"{R}/paper25w_m15_gpupre.sqlite"),
    ("+ Fixed\nNMS", f"{R}/paper25w_v2_m15_nms.sqlite"),
    ("+ FP16", f"{R}/paper25w_v2_m15_fp16.sqlite"),
]
STATS = [   # key in summarize_stage's result, legend label, colour
    ("frame_time_ms", "Mean", "#b4b2a9"),        # neutral: not a percentile
    ("frame_p50_ms", "p50", "#86b6ef"),          # percentiles: light -> dark blue
    ("frame_p99_ms", "p99", "#1c4f94"),
]
FS_TICK, FS_LABEL, FS_LEGEND = 12, 13, 11


def main() -> None:
    p = argparse.ArgumentParser(description="M15 frame-time mean and percentiles per stage")
    p.add_argument("--skip-frames", type=int, default=50)
    p.add_argument("--outdir", default="reports/figures/paper25w_v2")
    p.add_argument("--name", default="paper25w_v2_m15_frame_percentiles")
    args = p.parse_args()
    outdir = Path(args.outdir)

    stats = [summarize_stage([path], args.skip_frames) for _, path in STAGES]

    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / f"{args.name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        keys = ["frame_time_ms", "frame_p50_ms", "frame_p95_ms", "frame_p99_ms"]   # p95 in the CSV only
        w.writerow(["stage", "report", "frames", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "p99_minus_p50_ms"])
        for (label, path), st in zip(STAGES, stats):
            w.writerow([label.replace("\n", " "), path, st["frames"]] + [round(st[k], 2) for k in keys]
                       + [round(st["frame_p99_ms"] - st["frame_p50_ms"], 2)])
            print(f"  {label.replace(chr(10), ' '):15s} n={st['frames']}  mean {st['frame_time_ms']:.1f}  p50 {st['frame_p50_ms']:.1f}"
                  f"  p95 {st['frame_p95_ms']:.1f}  p99 {st['frame_p99_ms']:.1f}")

    width = 0.8 / len(STATS)
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    for i, st in enumerate(stats):
        for j, (key, _, color) in enumerate(STATS):
            x = i + (j - (len(STATS) - 1) / 2) * width
            ax.bar(x, st[key], width=width * 0.92, color=color, zorder=3, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(STAGES)))
    ax.set_xticklabels([label for label, _ in STAGES])
    ax.tick_params(axis="x", length=0)
    style_axes(ax, "")
    for spine in ("left", "bottom"):   # black axes, ticks and labels (style_axes makes them grey)
        ax.spines[spine].set_color("black")
    ax.tick_params(colors="black", labelsize=FS_TICK)
    ax.set_ylabel("Frame time (ms)", fontsize=FS_LABEL)
    ax.set_xlabel("Stage", fontsize=FS_LABEL)
    ax.set_ylim(0, max(st["frame_p99_ms"] for st in stats) * 1.05)
    fig.legend(handles=[Patch(facecolor=c, label=lab) for _, lab, c in STATS], frameon=False,
               fontsize=FS_LEGEND, ncol=len(STATS), loc="lower center", columnspacing=1.5,
               handletextpad=0.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, outdir, args.name)


if __name__ == "__main__":
    main()
