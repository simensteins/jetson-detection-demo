#!/usr/bin/env python3
"""Comparison figures for any set of optimization stages, computed directly
from their nsys sqlite exports - the same five figures as
scripts/plot_paper_figures.py (frame time, GPU idle gap, time breakdown, CPU
dispatch time, CPU/GPU swimlane), but for whichever reports are passed in
instead of the hard-coded YOLOv8n numbers there.

Every number is derived with the same definitions plot_paper_figures.py's
DATA uses (checked against it on the YOLOv8n reports):
  frame time      mean decode-start -> next decode-start (the whole loop)
  inference parts scripts/analyze_nsys.py per_frame_breakdown(): pre-kernel,
                  kernel execution, GPU idle gaps, post-kernel, total
  dispatch        all CUDA Runtime API time inside `inference` (CPU side), split
                  into active dispatch and blocked waits (calls > 1 ms), as in
                  the swimlane
All averages skip the first --skip-frames frames (warm-up).

A stage can pool several runs of the same configuration: join the files with
a comma (e.g. the three baseline runs).

Usage (from the repo root):
    python3 scripts/plot_compare_reports.py --name m15 \\
        --reports reports/m15_baseline.sqlite reports/m15_graph.sqlite reports/m15_gpupre.sqlite \\
        --labels "Baseline" "+ CUDA Graph" "+ GPU preprocessing"
    # pooled baseline runs:
    python3 scripts/plot_compare_reports.py --name m15 \\
        --reports reports/m15_baseline_1.sqlite,reports/m15_baseline_2.sqlite reports/m15_graph.sqlite ...

Writes reports/figures/<name>_01_frame_time ... <name>_05_cpu_gpu_swimlane
(.png @300dpi + .pdf) and <name>_summary.csv with the numbers behind them.
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_nsys import check_tables, per_frame_breakdown  # noqa: E402
from plot_paper_figures import (AQUA, BLUE, GRID_COLOR, MUTED, ORANGE, ORD_RAMP,  # noqa: E402
                                YELLOW, extract_frame_timeline, save, style_axes)

WAIT_THRESHOLD_US = 1000.0  # see plot_paper_figures.plot_swimlane: duration, not API name


def ramp(n: int) -> list[str]:
    """n colors from the light->dark ordinal ramp, spread across it."""
    if n == 1:
        return [ORD_RAMP[-1]]
    return [ORD_RAMP[round(i * (len(ORD_RAMP) - 1) / (n - 1))] for i in range(n)] if n <= len(ORD_RAMP) \
        else [ORD_RAMP[min(i, len(ORD_RAMP) - 1)] for i in range(n)]


def summarize_stage(paths: list[str], skip: int) -> dict:
    frames, frame_times, blocked_us = [], [], []
    for path in paths:
        con = sqlite3.connect(path)
        cur = con.cursor()
        check_tables(cur)
        frames += per_frame_breakdown(cur, skip)
        dec = [s for (s,) in cur.execute(
            "SELECT start FROM NVTX_EVENTS WHERE text='decode' ORDER BY start")]
        frame_times += [(b - a) / 1e6 for a, b in zip(dec[skip:], dec[skip + 1:])]
        # The part of the CUDA API time that is the CPU blocked on the GPU
        # (calls longer than WAIT_THRESHOLD_US), same split as the swimlane.
        for s, e in cur.execute("SELECT start, end FROM NVTX_EVENTS WHERE text='inference' "
                                "ORDER BY start").fetchall()[skip:]:
            blocked_us.append(cur.execute(
                "SELECT COALESCE(SUM(end-start),0) FROM CUPTI_ACTIVITY_KIND_RUNTIME "
                "WHERE start >= ? AND end <= ? AND end-start > ?",
                (s, e, WAIT_THRESHOLD_US * 1000)).fetchone()[0] / 1000.0)
        con.close()
    if not frames or not frame_times:
        raise SystemExit(f"No frames after skipping {skip} in {paths}")

    def mean_ms(key: str) -> float:
        return statistics.mean(f[key] for f in frames) / 1000.0

    frame_ms = statistics.mean(frame_times)
    d = dict(
        frame_time_ms=frame_ms,
        fps=1000.0 / frame_ms,
        total_inf_ms=mean_ms("total_us"),
        pre_kernel_ms=mean_ms("pre_kernel_us"),
        kernel_exec_ms=mean_ms("kernel_time_us"),
        idle_gap_ms=mean_ms("idle_gap_us"),
        post_kernel_ms=mean_ms("post_kernel_us"),
        dispatch_ms=mean_ms("runtime_us"),
        kernel_count=statistics.mean(f["kernel_count"] for f in frames),
        frames=len(frames),
    )
    d["idle_gap_pct"] = 100.0 * d["idle_gap_ms"] / d["total_inf_ms"]
    d["blocked_ms"] = statistics.mean(blocked_us) / 1000.0
    d["active_dispatch_ms"] = d["dispatch_ms"] - d["blocked_ms"]
    return d


def plot_dispatch(stats, labels, title, outdir, fname):
    """CUDA API time inside `inference`, split into active dispatch and the CPU
    blocked waiting on the GPU. The stacked total is plot_paper_figures.py's
    dispatch_ms; the split matters once the model dominates (M15: almost all
    of it is waiting)."""
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    x = range(len(labels))
    active = [s["active_dispatch_ms"] for s in stats]
    blocked = [s["blocked_ms"] for s in stats]
    top = max(s["dispatch_ms"] for s in stats)
    ax.bar(x, active, 0.6, color=ORANGE, label="CPU dispatching (active)", zorder=3,
           edgecolor="white", linewidth=0.5)
    ax.bar(x, blocked, 0.6, bottom=active, color="white", edgecolor=MUTED, hatch="////",
           linewidth=0.5, label="CPU blocked, waiting on GPU", zorder=3)
    for i, s in enumerate(stats):
        ax.text(i, s["dispatch_ms"] + top * 0.03, f"{s['active_dispatch_ms']:.2f}ms active",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.text(i, s["dispatch_ms"] + top * 0.10, f"{s['kernel_count']:.0f} kernels",
                ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    style_axes(ax, "CPU time in CUDA API calls (ms)")
    ax.set_ylim(0, top * 1.30)
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    save(fig, outdir, fname)


def bar_chart(stats, labels, key, ylabel, title, sub, outdir, fname, colors):
    vals = [s[key] for s in stats]
    top = max(vals) if max(vals) > 0 else 1.0
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x = range(len(labels))
    bars = ax.bar(x, vals, color=colors, width=0.6, zorder=3, edgecolor="white", linewidth=0.5)
    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + top * 0.03,
                f"{vals[i]:.2f}ms", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + top * 0.10,
                sub(stats[i]), ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    style_axes(ax, ylabel)
    ax.set_ylim(0, top * 1.30)
    ax.set_title(title)
    fig.tight_layout()
    save(fig, outdir, fname)


def plot_time_breakdown(stats, labels, title, outdir, fname):
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    x = range(len(labels))
    segments = [
        ("kernel_exec_ms", "Kernel execution (GPU busy)", BLUE, ""),
        ("idle_gap_ms", "GPU idle gap (waiting on CPU)", ORANGE, "///"),
        ("pre_kernel_ms", "Pre-kernel / setup (CPU)", AQUA, "..."),
        ("post_kernel_ms", "Post-kernel (postprocessing)", YELLOW, "xxx"),
    ]
    top = max(s["total_inf_ms"] for s in stats)
    bottoms = [0.0] * len(labels)
    for key, label, color, hatch in segments:
        vals = [s[key] for s in stats]
        ax.bar(x, vals, 0.55, bottom=bottoms, color=color, label=label,
               edgecolor="white", linewidth=0.5, hatch=hatch, zorder=3)
        for i, v in enumerate(vals):
            if v >= top * 0.05:  # only label segments tall enough to hold text
                ax.text(i, bottoms[i] + v / 2, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if color in (BLUE, ORANGE) else "#0b0b0b", fontweight="bold")
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    for i, s in enumerate(stats):
        ax.text(i, s["total_inf_ms"] + top * 0.02, f"{s['total_inf_ms']:.2f}ms total",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    style_axes(ax, "Time per `inference` range (ms)")
    ax.set_ylim(0, top * 1.15)
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    save(fig, outdir, fname)


def median_frame_index(path: str, skip: int) -> int:
    """Index (into all `inference` ranges) of the frame with the median
    duration after warm-up - a typical frame, not an outlier."""
    con = sqlite3.connect(path)
    durs = [e - s for s, e in con.execute(
        "SELECT start, end FROM NVTX_EVENTS WHERE text='inference' ORDER BY start")]
    con.close()
    order = sorted(range(skip, len(durs)), key=lambda i: durs[i])
    return order[len(order) // 2]


def plot_swimlane(stage_paths, labels, title, frame_offset, skip, outdir, fname):
    panels = []
    for paths, label in zip(stage_paths, labels):
        idx = frame_offset if frame_offset is not None else median_frame_index(paths[0], skip)
        kernels, calls = extract_frame_timeline(paths[0], idx)
        panels.append((kernels, calls, label.replace("\n", " ")))
    x_max = max(k[-1][1] for k, _, _ in panels) * 1.02  # one shared x-axis

    fig, axes = plt.subplots(len(panels), 1, figsize=(7.5, 2.35 * len(panels)), sharex=True,
                             squeeze=False)
    axes = axes[:, 0]
    for ax, (kernels, calls, label) in zip(axes, panels):
        ax.broken_barh([(s, e - s) for s, e in kernels], (0, 8), facecolors=BLUE,
                       edgecolors="none", zorder=3)
        min_width = x_max * 0.0015
        dispatch = [(s, max(e - s, min_width)) for s, e, _ in calls if (e - s) <= WAIT_THRESHOLD_US]
        waiting = [(s, e - s) for s, e, _ in calls if (e - s) > WAIT_THRESHOLD_US]
        ax.broken_barh(dispatch, (14, 8), facecolors=ORANGE, edgecolors="none", zorder=3)
        if waiting:
            ax.broken_barh(waiting, (14, 8), facecolors="none", edgecolors=MUTED,
                           hatch="////", linewidth=0, zorder=2)
        ax.set_yticks([4, 18])
        ax.set_yticklabels(["GPU\n(kernels)", "CPU\n(API calls)"])
        ax.set_ylim(-2, 26)
        ax.set_xlim(0, x_max)
        ax.set_title(f"{label}\n{len(kernels)} kernels across {kernels[-1][1]:.0f}μs",
                     fontsize=9.5, loc="left")
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.tick_params(colors=MUTED, length=3)
    axes[-1].set_xlabel("Time since frame's `inference` NVTX start (μs)")
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=BLUE, edgecolor="none", label="GPU kernel executing"),
        plt.Rectangle((0, 0), 1, 1, facecolor=ORANGE, edgecolor="none", label="CPU dispatching (active)"),
        plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=MUTED, hatch="////",
                      label="CPU blocked, waiting on GPU"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title + " (median frame)" if frame_offset is None else title, fontsize=11, y=1.005)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, outdir, fname)


def main() -> None:
    p = argparse.ArgumentParser(description="Comparison figures for a set of nsys sqlite reports")
    p.add_argument("--reports", nargs="+", required=True,
                   help="One entry per stage, in order; pool runs of a stage with commas")
    p.add_argument("--labels", nargs="+", required=True, help="One label per stage")
    p.add_argument("--name", required=True, help="Prefix for the output files, e.g. m15")
    p.add_argument("--model", default=None, help="Model name for figure titles (default: --name)")
    p.add_argument("--outdir", default="reports/figures")
    p.add_argument("--skip-frames", type=int, default=50, help="Warm-up frames to skip")
    p.add_argument("--frame-offset", type=int, default=None,
                   help="Which frame the swimlane shows (index into `inference` ranges); "
                        "default: each stage's median-duration frame after warm-up")
    args = p.parse_args()

    if len(args.reports) != len(args.labels):
        raise SystemExit(f"{len(args.reports)} --reports entries but {len(args.labels)} --labels")
    stage_paths = [entry.split(",") for entry in args.reports]
    for path in (p_ for paths in stage_paths for p_ in paths):
        if not Path(path).exists():
            raise SystemExit(f"Not found: {path}")
    model = args.model or args.name
    outdir = Path(args.outdir)
    labels = [lbl.replace("\\n", "\n") for lbl in args.labels]
    colors = ramp(len(labels))

    stats = [summarize_stage(paths, args.skip_frames) for paths in stage_paths]

    print(f"{'stage':28s} {'frames':>6s} {'frame ms':>9s} {'FPS':>6s} {'infer ms':>9s} {'pre':>6s} "
          f"{'kernels':>8s} {'idle':>6s} {'post':>6s} {'dispatch':>9s} {'#kern':>6s}")
    for lbl, s in zip(labels, stats):
        print(f"{lbl.replace(chr(10), ' ')[:28]:28s} {s['frames']:6d} {s['frame_time_ms']:9.2f} "
              f"{s['fps']:6.2f} {s['total_inf_ms']:9.2f} {s['pre_kernel_ms']:6.2f} "
              f"{s['kernel_exec_ms']:8.2f} {s['idle_gap_ms']:6.2f} {s['post_kernel_ms']:6.2f} "
              f"{s['dispatch_ms']:9.2f} {s['kernel_count']:6.0f}")

    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"{args.name}_summary.csv"
    with open(csv_path, "w", newline="") as f:
        keys = ["frames", "frame_time_ms", "fps", "total_inf_ms", "pre_kernel_ms", "kernel_exec_ms",
                "idle_gap_ms", "idle_gap_pct", "post_kernel_ms", "dispatch_ms", "active_dispatch_ms",
                "blocked_ms", "kernel_count"]
        w = csv.writer(f)
        w.writerow(["stage", "reports"] + keys)
        for lbl, paths, s in zip(labels, stage_paths, stats):
            w.writerow([lbl.replace("\n", " "), ";".join(paths)] + [round(s[k], 3) for k in keys])
    print(f"  wrote {csv_path}")

    bar_chart(stats, labels, "frame_time_ms", "True end-to-end frame time (ms)",
              f"{model}: total frame time per stage", lambda s: f"{s['fps']:.1f} FPS",
              outdir, f"{args.name}_01_frame_time", colors)
    bar_chart(stats, labels, "idle_gap_ms", "GPU idle time between kernels (ms)",
              f"{model}: GPU idle time between kernels", lambda s: f"{s['idle_gap_pct']:.1f}% of inference",
              outdir, f"{args.name}_02_idle_gap", colors)
    plot_time_breakdown(stats, labels, f"{model}: where does the inference time go?",
                        outdir, f"{args.name}_03_time_breakdown")
    plot_dispatch(stats, labels, f"{model}: CPU time in CUDA API calls",
                  outdir, f"{args.name}_04_dispatch_time")
    plot_swimlane(stage_paths, labels, f"{model}: one real frame per stage, to scale",
                  args.frame_offset, args.skip_frames, outdir, f"{args.name}_05_cpu_gpu_swimlane")
    print(f"\nDone. Figures in {outdir}/")


if __name__ == "__main__":
    main()
