#!/usr/bin/env python3
"""Generates publication-ready figures for a SINGLE nsys sqlite report - the
one-report counterpart to scripts/plot_paper_figures.py (which is fixed to
the four canonical Baseline/Iteration 1-3 reports for the cross-iteration
comparison figures). Use this one for any other report: a one-off test like
reports/custom_nms_display_test.sqlite, a report from a future iteration, or
just re-checking one existing report in isolation.

Reuses the exact same analysis functions as scripts/analyze_nsys.py (the
project's standard text-output analysis tool) and the same drawing helpers
and color tokens as scripts/plot_paper_figures.py, imported directly rather
than reimplemented, so the numbers and the visual style always match the
rest of the project's figures.

Produces four figures, named after the input report's filename stem:
  <stem>_nvtx_ranges.png/pdf   - decode/inference/draw/display means (whichever exist)
  <stem>_breakdown.png/pdf     - single stacked bar: pre-kernel / kernel exec / idle gap / post-kernel
  <stem>_frame_variance.png/pdf - inference time per frame across the whole run (jitter/outliers)
  <stem>_swimlane.png/pdf      - CPU/GPU swimlane for one representative frame, real per-kernel data

Usage:
    python3 scripts/plot_nsys_report.py reports/custom_nms_display_test.sqlite
    python3 scripts/plot_nsys_report.py reports/<report>.sqlite \
        --outdir reports/figures --skip-frames 50 --frame-offset 100 --label "My Report"
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_nsys as an  # noqa: E402 - same dir, reuse its analysis functions as-is
import plot_paper_figures as ppf  # noqa: E402 - reuse its style tokens/helpers as-is

BLUE, ORANGE, AQUA, YELLOW = ppf.BLUE, ppf.ORANGE, ppf.AQUA, ppf.YELLOW
GRID_COLOR, MUTED = ppf.GRID_COLOR, ppf.MUTED
ORD_RAMP = ppf.ORD_RAMP


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot figures for a single nsys sqlite report")
    p.add_argument("report", help="Path to the .sqlite report (from `nsys export --type=sqlite`)")
    p.add_argument("--outdir", default="reports/figures", help="Output directory for PNG/PDF figures")
    p.add_argument("--skip-frames", type=int, default=50,
                   help="Warm-up frames to skip, matching analyze_nsys.py's convention (default 50)")
    p.add_argument("--frame-offset", type=int, default=100,
                   help="Which inference-event index to use as the swimlane's representative frame "
                        "(default 100 - same convention as plot_paper_figures.py)")
    p.add_argument("--label", default=None,
                   help="Human-readable title for the figures (default: derived from the filename)")
    return p.parse_args()


def plot_nvtx_ranges(ranges: dict[str, list[float]], label: str, outdir: Path, stem: str) -> None:
    names = list(ranges.keys())
    means = [an.summarize(ranges[n])["mean"] / 1000.0 for n in names]  # us -> ms
    stdevs = [an.summarize(ranges[n])["stdev"] / 1000.0 for n in names]

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x = range(len(names))
    colors = [ORD_RAMP[i % len(ORD_RAMP)] for i in range(len(names))]
    bars = ax.bar(x, means, yerr=stdevs, capsize=3, color=colors, width=0.55, zorder=3,
                   edgecolor="white", linewidth=0.5, error_kw=dict(ecolor=MUTED, linewidth=1))
    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + stdevs[i] + max(means) * 0.03,
                 f"{means[i]:.2f}ms", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels([n.capitalize() for n in names])
    ax.set_ylabel("Duration (ms)")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.set_ylim(0, max(means) * 1.35)
    ax.set_title(f"{label}\nNVTX range durations (error bars = 1 stdev)")
    fig.tight_layout()
    ppf.save(fig, outdir, f"{stem}_nvtx_ranges")


def plot_breakdown(per_frame: list[dict], label: str, outdir: Path, stem: str) -> None:
    keys = ["kernel_time_us", "idle_gap_us", "pre_kernel_us", "post_kernel_us"]
    means = {k: an.summarize([f[k] for f in per_frame])["mean"] / 1000.0 for k in keys}
    total_ms = sum(means.values())

    segments = [
        ("kernel_time_us", "Kernel execution (GPU busy)", BLUE, ""),
        ("idle_gap_us", "GPU idle gap (waiting on CPU)", ORANGE, "///"),
        ("pre_kernel_us", "Pre-kernel / setup (CPU)", AQUA, "..."),
        ("post_kernel_us", "Post-kernel (postprocessing)", YELLOW, "xxx"),
    ]

    fig, ax = plt.subplots(figsize=(4.2, 4.4))
    bottom = 0.0
    for key, seg_label, color, hatch in segments:
        val = means[key]
        ax.bar(0, val, bottom=bottom, width=0.5, color=color, label=seg_label,
               edgecolor="white", linewidth=0.5, hatch=hatch, zorder=3)
        if val >= total_ms * 0.10:
            text_color = "white" if color in (BLUE, ORANGE) else "#0b0b0b"
            ax.text(0, bottom + val / 2, f"{val:.2f}ms", ha="center", va="center",
                     fontsize=9, fontweight="bold", color=text_color)
        bottom += val

    ax.text(0, total_ms + total_ms * 0.02, f"{total_ms:.2f}ms total", ha="center", va="bottom",
             fontsize=9, fontweight="bold")
    ax.set_xlim(-0.6, 0.6)
    ax.set_xticks([])
    ax.set_ylim(0, total_ms * 1.15)
    ax.set_ylabel("Time (ms)")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.03), frameon=False, fontsize=8, ncol=1)
    ax.set_title(f"{label}\nWhere the frame's time goes")
    fig.tight_layout()
    ppf.save(fig, outdir, f"{stem}_breakdown")


def plot_frame_variance(per_frame: list[dict], label: str, outdir: Path, stem: str) -> None:
    totals_ms = [f["total_us"] / 1000.0 for f in per_frame]
    mean = sum(totals_ms) / len(totals_ms)

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.plot(range(len(totals_ms)), totals_ms, color=BLUE, linewidth=1.2, zorder=3)
    ax.axhline(mean, color=MUTED, linewidth=1, linestyle="--", zorder=2)
    ax.text(len(totals_ms) * 1.0, mean, f" mean {mean:.2f}ms", ha="left", va="center",
            fontsize=8.5, color=MUTED)
    ax.set_xlabel("Frame index (after warm-up)")
    ax.set_ylabel("Inference time (ms)")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.set_title(f"{label}\nInference time per frame across the run")
    fig.tight_layout()
    ppf.save(fig, outdir, f"{stem}_frame_variance")


def plot_single_swimlane(dbname: str, frame_offset: int, label: str, outdir: Path, stem: str) -> None:
    con = sqlite3.connect(dbname)
    cur = con.cursor()
    n_frames = cur.execute("SELECT COUNT(*) FROM NVTX_EVENTS WHERE text='inference'").fetchone()[0]
    con.close()
    if n_frames == 0:
        print("  [skip] swimlane: no 'inference' NVTX events in this report")
        return
    if frame_offset >= n_frames:
        print(f"  [note] --frame-offset {frame_offset} >= {n_frames} available frames; "
              f"using frame {n_frames // 2} instead")
        frame_offset = n_frames // 2

    kernels, calls = ppf.extract_frame_timeline(dbname, frame_offset=frame_offset)
    if not kernels:
        print("  [skip] swimlane: no kernels found in the selected frame")
        return

    x_max = kernels[-1][1] * 1.02
    fig, ax = plt.subplots(figsize=(7.5, 3.1))

    gpu_spans = [(s, e - s) for s, e in kernels]
    ax.broken_barh(gpu_spans, (0, 8), facecolors=BLUE, edgecolors="none", zorder=3)

    # Same duration-threshold classification validated in plot_paper_figures.py:
    # every genuine dispatch-type call across the project's reports finishes
    # under ~660us, every genuine CPU-blocked wait runs past ~1190us - so
    # duration, not the API name, separates "CPU working" from "CPU idle".
    WAIT_THRESHOLD_US = 1000.0
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
    ax.set_xlabel("Time since frame's `inference` NVTX start (μs)")
    ax.set_title(f"{label}\n{len(kernels)} kernels across {kernels[-1][1]:.0f}μs "
                 f"(frame index {frame_offset})", fontsize=10, loc="left")
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=3)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=BLUE, edgecolor="none", label="GPU kernel executing"),
        plt.Rectangle((0, 0), 1, 1, facecolor=ORANGE, edgecolor="none", label="CPU dispatching (active)"),
        plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=MUTED, hatch="////", label="CPU blocked, waiting on GPU"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.35),
              ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    ppf.save(fig, outdir, f"{stem}_swimlane")


def main() -> None:
    args = parse_args()
    report_path = Path(args.report)
    if not report_path.exists():
        raise SystemExit(f"Report not found: {report_path}")
    stem = report_path.stem
    label = args.label or stem.replace("_", " ")
    outdir = Path(args.outdir)

    con = sqlite3.connect(str(report_path))
    cur = con.cursor()
    an.check_tables(cur)

    ranges = an.nvtx_range_stats(cur, args.skip_frames)
    per_frame = an.per_frame_breakdown(cur, args.skip_frames)
    con.close()

    if not ranges:
        raise SystemExit("No decode/inference/draw/display NVTX events found in this report.")
    if not per_frame:
        raise SystemExit("No 'inference' NVTX events found after skipping warm-up.")

    print(f"Plotting {report_path} as '{label}' ({len(per_frame)} frames after skipping {args.skip_frames})...")
    plot_nvtx_ranges(ranges, label, outdir, stem)
    plot_breakdown(per_frame, label, outdir, stem)
    plot_frame_variance(per_frame, label, outdir, stem)
    plot_single_swimlane(str(report_path), args.frame_offset, label, outdir, stem)

    print(f"\nDone. Figures in {outdir}/ ({stem}_*.png + matching .pdf).")


if __name__ == "__main__":
    main()
