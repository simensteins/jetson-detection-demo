#!/usr/bin/env python3
"""Generates publication-ready figures (PNG @300dpi + vector PDF) summarizing
the CUDA Graph / GPU preprocessing / custom NMS optimization work, for use in
a research paper or report.

Two kinds of figures:

1. Summary bar charts (frame time, GPU idle gap, time-breakdown, CPU dispatch
   time) - built from the same verified, cross-checked numbers documented on
   the project's Notion pages and re-confirmed against the raw .sqlite reports
   in this repo (see comments on the DATA dict below for provenance).

2. A CPU/GPU "swimlane" timeline - showing dispatch gaps (baseline) vs. a
   captured CUDA Graph replay (Iteration 1), side by side. This is REAL,
   unmodified per-kernel timing data extracted directly from the nsys sqlite
   exports (not a schematic/illustration) - every rectangle is an actual
   kernel or CUDA API call, drawn to scale. See extract_frame_timeline().

Requires: matplotlib (pip install matplotlib). Reads reports/*.sqlite - run
this from the repo root, after the standard nsys profiling runs have produced
those files (see any iteration script's docstring for the profiling command).

Usage:
    python3 scripts/plot_paper_figures.py [--outdir reports/figures]
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.edgecolor": "#898781",
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

# Categorical palette (colorblind-validated - see dataviz skill palette.md).
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
# Ordinal (magnitude-across-iterations) ramp - single hue, light -> dark.
ORD_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab"]
GRID_COLOR = "#e1e0d9"
MUTED = "#898781"

STAGES = ["Baseline", "Iteration 1\n(CUDA Graph)", "Iteration 2\n(GPU Preprocess)", "Iteration 3\n(Custom NMS)"]

# ---------------------------------------------------------------------------
# Verified summary data. Re-derived in-session from the canonical reports
# with scripts/analyze_nsys.py and cross-checked against each iteration's
# published Notion numbers:
#   Baseline -> reports/baseline_localfile_2.sqlite + _3.sqlite (pooled)
#   Iteration 1 -> reports/cuda_graph_iteration_2.sqlite (the corrected,
#       --cuda-graph-trace=node capture - NOT cuda_graph_iteration_1.sqlite,
#       which is the pre-fix --cuda-graph-trace=graph capture kept only for
#       history; see Iteration 1's Notion page "A measurement bug found and
#       fixed").
#   Iteration 2 -> reports/cuda_graph_gpu_resize_iteration_1.sqlite (step 2b)
#   Iteration 3 -> reports/cuda_graph_custom_nms_2.sqlite (post sync-fix)
# ---------------------------------------------------------------------------
DATA = {
    "frame_time_ms":   [33.45, 24.73, 18.15, 13.47],   # true end-to-end, decode->draw
    "fps":             [29.89, 40.43, 55.10, 74.22],
    "idle_gap_ms":     [7.50, 2.90, 3.00, 0.25],        # GPU idle time within `inference`
    "idle_gap_pct":    [30.1, 15.0, 23.4, 2.4],
    "pre_kernel_ms":   [6.90, 8.27, 0.90, 0.99],
    "kernel_exec_ms":  [10.01, 8.10, 8.85, 8.77],
    "post_kernel_ms":  [0.49, 0.08, 0.08, 0.12],
    "total_inf_ms":    [24.90, 19.35, 12.84, 10.13],
    "dispatch_ms":     [8.10, 8.53, 8.30, 4.71],        # CUDA Runtime API dispatch, CPU-side
    "kernel_count":    [223, 207, 213, 220],
}


def style_axes(ax, ylabel: str) -> None:
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def plot_frame_time(outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x = range(len(STAGES))
    bars = ax.bar(x, DATA["frame_time_ms"], color=ORD_RAMP, width=0.6, zorder=3,
                   edgecolor="white", linewidth=0.5)
    for i, (b, fps) in enumerate(zip(bars, DATA["fps"])):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.9,
                f"{DATA['frame_time_ms'][i]:.2f}ms", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2.6,
                f"{fps:.1f} FPS", ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.set_xticks(list(x))
    ax.set_xticklabels(STAGES)
    style_axes(ax, "True end-to-end frame time (ms)")
    ax.set_ylim(0, max(DATA["frame_time_ms"]) * 1.28)
    ax.set_title("Total frame time, iteration by iteration")
    fig.tight_layout()
    save(fig, outdir, "01_frame_time")


def plot_idle_gap(outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x = range(len(STAGES))
    bars = ax.bar(x, DATA["idle_gap_ms"], color=ORD_RAMP, width=0.6, zorder=3,
                   edgecolor="white", linewidth=0.5)
    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.18,
                f"{DATA['idle_gap_ms'][i]:.2f}ms", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.55,
                f"{DATA['idle_gap_pct'][i]:.1f}% of frame", ha="center", va="bottom",
                fontsize=8, color=MUTED)
    ax.set_xticks(list(x))
    ax.set_xticklabels(STAGES)
    style_axes(ax, "GPU idle time between kernels (ms)")
    ax.set_ylim(0, 8.6)
    ax.set_title("GPU idle time between kernels ($-$97% by Iteration 3)")
    fig.tight_layout()
    save(fig, outdir, "02_idle_gap")


def plot_time_breakdown(outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    x = range(len(STAGES))
    width = 0.55

    segments = [
        ("kernel_exec_ms", "Kernel execution (GPU busy)", BLUE, ""),
        ("idle_gap_ms", "GPU idle gap (waiting on CPU)", ORANGE, "///"),
        ("pre_kernel_ms", "Pre-kernel / setup (CPU)", AQUA, "..."),
        ("post_kernel_ms", "Post-kernel (postprocessing)", YELLOW, "xxx"),
    ]
    bottoms = [0.0] * len(STAGES)
    for key, label, color, hatch in segments:
        vals = DATA[key]
        ax.bar(x, vals, width, bottom=bottoms, color=color, label=label,
               edgecolor="white", linewidth=0.5, hatch=hatch, zorder=3)
        for i, v in enumerate(vals):
            if v >= 1.3:  # only label segments tall enough to hold text
                ax.text(i, bottoms[i] + v / 2, f"{v:.2f}", ha="center", va="center",
                         fontsize=8, color="white" if color in (BLUE, ORANGE) else "#0b0b0b",
                         fontweight="bold")
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    for i, total in enumerate(DATA["total_inf_ms"]):
        ax.text(i, total + 0.5, f"{total:.2f}ms total", ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(STAGES)
    style_axes(ax, "Time (ms)")
    ax.set_ylim(0, max(DATA["total_inf_ms"]) * 1.15)
    ax.set_title("Where does the frame's time go?")
    ax.legend(loc="upper right", frameon=False, ncol=1, fontsize=8)
    fig.tight_layout()
    save(fig, outdir, "03_time_breakdown")


def plot_dispatch_time(outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x = range(len(STAGES))
    bars = ax.bar(x, DATA["dispatch_ms"], color=ORD_RAMP, width=0.6, zorder=3,
                   edgecolor="white", linewidth=0.5)
    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.18,
                f"{DATA['dispatch_ms'][i]:.2f}ms", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.55,
                f"{DATA['kernel_count'][i]} kernels", ha="center", va="bottom",
                fontsize=8, color=MUTED)
    ax.set_xticks(list(x))
    ax.set_xticklabels(STAGES)
    style_axes(ax, "CPU time dispatching CUDA work (ms)")
    ax.set_ylim(0, max(DATA["dispatch_ms"]) * 1.32)
    ax.set_title("CPU time spent dispatching CUDA work")
    fig.tight_layout()
    save(fig, outdir, "04_dispatch_time")


# ---------------------------------------------------------------------------
# CPU/GPU swimlane: real per-kernel timing, extracted directly from the
# sqlite reports (not a schematic).
# ---------------------------------------------------------------------------
def extract_frame_timeline(dbname: str, frame_offset: int = 100):
    """Pulls every GPU kernel and every CPU-side CUDA API call within one
    frame's `inference` NVTX span (the frame at index `frame_offset`, well
    past warm-up), timestamped relative to that NVTX span's own start (not
    the first model kernel) - so any pre-kernel gap (CPU-only preprocessing
    in Baseline/Iteration 1, or the real GPU preprocessing kernels in
    Iteration 2/3) shows up honestly instead of being cropped out. Returns
    (kernels, calls), each a list of (start_us, end_us[, name]) tuples."""
    con = sqlite3.connect(dbname)
    cur = con.cursor()
    cur.execute(
        "SELECT start, end FROM NVTX_EVENTS WHERE text='inference' ORDER BY start LIMIT 1 OFFSET ?",
        (frame_offset,),
    )
    inf_start, inf_end = cur.fetchone()
    t0 = inf_start

    cur.execute(
        """SELECT k.start - ?, k.end - ? FROM CUPTI_ACTIVITY_KIND_KERNEL k
           WHERE k.start >= ? AND k.start <= ? ORDER BY k.start""",
        (t0, t0, inf_start, inf_end),
    )
    kernels = [(s / 1000, e / 1000) for s, e in cur.fetchall()]

    cur.execute(
        """SELECT r.start - ?, r.end - ?, s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r
           JOIN StringIds s ON r.nameId = s.id
           WHERE r.start >= ? AND r.start <= ? ORDER BY r.start""",
        (t0, t0, inf_start, inf_end),
    )
    calls = [(s / 1000, e / 1000, n) for s, e, n in cur.fetchall()]
    con.close()
    return kernels, calls


SWIMLANE_REPORTS = [
    ("baseline_localfile_2.sqlite", "Baseline – individually dispatched (model.predict())"),
    ("cuda_graph_iteration_2.sqlite", "Iteration 1 – + captured CUDA Graph"),
    ("cuda_graph_gpu_resize_iteration_1.sqlite", "Iteration 2 – + GPU preprocessing"),
    ("cuda_graph_custom_nms_2.sqlite", "Iteration 3 – + custom fixed-shape NMS"),
]


def plot_swimlane(outdir: Path, reports_dir: Path) -> None:
    missing = [f for f, _ in SWIMLANE_REPORTS if not (reports_dir / f).exists()]
    if missing:
        print(f"  [skip] swimlane figure needs {missing} in {reports_dir}/")
        return

    panels = []
    for fname, title in SWIMLANE_REPORTS:
        kernels, calls = extract_frame_timeline(str(reports_dir / fname))
        panels.append((kernels, calls, title))

    # One shared x-axis limit across all four panels - this is what makes the
    # compression across iterations honestly comparable at a glance, instead
    # of each panel silently rescaling to its own busiest stretch.
    x_max = max(k[-1][1] for k, _, _ in panels) * 1.02

    fig, axes = plt.subplots(4, 1, figsize=(7.5, 9.4), sharex=True)

    for ax, (kernels, calls, title) in zip(axes, panels):
        gpu_spans = [(s, e - s) for s, e in kernels]
        ax.broken_barh(gpu_spans, (0, 8), facecolors=BLUE, edgecolors="none", zorder=3)

        # Split CPU-side events by what they actually mean. Classifying by
        # call NAME (e.g. "is it a *Synchronize call?") turns out to be
        # wrong: Iteration 3's real blocking wait is a long cudaMemcpyAsync
        # (the .cpu() transfer), not a Synchronize call, while cudaGraphLaunch
        # itself legitimately takes ~500-660us to issue (real, active
        # dispatch work, not a wait). Across all four reports there's a clean
        # gap in observed durations - every genuine dispatch-type call
        # (cudaLaunchKernel, cudaGraphLaunch, a normal H2D memcpy) finishes
        # under ~660us, while every genuine blocking wait (cudaStreamSynchronize
        # in Baseline/1/2, the blocking .cpu() memcpy in Iteration 3) runs
        # >=1190us - so duration, not the API name, is what actually
        # distinguishes "CPU doing work" from "CPU idle, waiting on GPU" here.
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
        span_note = f"{len(kernels)} kernels across {kernels[-1][1]:.0f}μs"
        ax.set_title(f"{title}\n{span_note}", fontsize=9.5, loc="left")
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.tick_params(colors=MUTED, length=3)

    axes[-1].set_xlabel("Time since frame's `inference` NVTX start (μs)")

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=BLUE, edgecolor="none", label="GPU kernel executing"),
        plt.Rectangle((0, 0), 1, 1, facecolor=ORANGE, edgecolor="none", label="CPU dispatching (active)"),
        plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=MUTED, hatch="////", label="CPU blocked, waiting on GPU"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Where the kernel-dispatch gaps go: one real frame, to scale, all four iterations", fontsize=11, y=1.005)
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    save(fig, outdir, "05_cpu_gpu_swimlane")


def save(fig, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    png_path = outdir / f"{name}.png"
    pdf_path = outdir / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {png_path} and {pdf_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate publication-ready figures for the paper/report")
    p.add_argument("--outdir", default="reports/figures", help="Output directory for PNG/PDF figures")
    p.add_argument("--reports-dir", default="reports", help="Directory containing the .sqlite nsys reports")
    args = p.parse_args()

    outdir = Path(args.outdir)
    reports_dir = Path(args.reports_dir)

    print("Generating summary bar charts...")
    plot_frame_time(outdir)
    plot_idle_gap(outdir)
    plot_time_breakdown(outdir)
    plot_dispatch_time(outdir)

    print("Generating CPU/GPU swimlane (real per-kernel data)...")
    plot_swimlane(outdir, reports_dir)

    print(f"\nDone. Figures in {outdir}/ (PNG @300dpi + vector PDF for each).")


if __name__ == "__main__":
    main()
