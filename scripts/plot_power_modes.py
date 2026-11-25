#!/usr/bin/env python3
"""FPS of every stage in every power mode, from scripts/run_power_modes.sh:
one figure per model, grouped by power mode. Each group holds the five stages
side by side (light to dark: baseline to + FP16), so every iteration can be
compared within one power mode, and the groups (15 W, 25 W, MAXN SUPER) show
what the power budget adds.

Only one profiler setting per figure: the runs without a profiler by default
(--profiler plain), or the runs under nsys (--profiler nsys). Each bar is the
median over the rounds, labelled with its FPS; a dotted line marks the target
(default 30 FPS). Bars of cells with at least one throttled run (over-current
counters increased) are hatched.

If reports/<set>/breakdown.csv exists (scripts/summarize_power_modes.py, run
on the Jetson), it also draws where the time goes: per model, for each stage
one stacked bar per power mode with the parts of the `inference` range (GPU
kernel execution, idle gaps between kernels, CPU work before the first kernel,
work after the last), the same segments as plot_compare_reports.py's
time-breakdown figure; means per frame, median over the rounds, from the runs
under nsys.

Usage (from the repo root, after copying reports/<set>/ from the Jetson; any
Python with matplotlib):
    python3 scripts/plot_power_modes.py --set modes_v2
    python3 scripts/plot_power_modes.py --set modes_v2 --target 30 --outdir reports/figures/modes_v2
    python3 scripts/plot_power_modes.py --set modes_v2 --profiler nsys   # the runs under nsys
Writes <outdir>/<set>_<model>_fps.png/.pdf (or _fps_nsys with --profiler nsys) and, with breakdown.csv,
<set>_<model>_breakdown.png/.pdf, one per model.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_compare_reports import ramp  # noqa: E402
from plot_paper_figures import AQUA, BLUE, MUTED, ORANGE, YELLOW, save, style_axes  # noqa: E402

MODELS = {"yolo": "YOLOv8n", "m15": "M15"}
STAGES = ["baseline", "graph", "gpupre", "nms", "fp16"]
LABELS = {"baseline": "Baseline", "graph": "+ CUDA\ngraph", "gpupre": "+ GPU\npreproc.",
          "nms": "+ Fixed\nNMS", "fp16": "+ FP16"}
MODE_ORDER = ["15W", "25W", "MAXN_SUPER"]   # increasing power budget; others go last
MODE_LABELS = {"MAXN_SUPER": "MAXN SUPER", "15W": "15 W", "25W": "25 W"}
TITLES = {"plain": "without profiler", "nsys": "under Nsight Systems"}
# One shade per stage, light to dark in pipeline order (the 4-step ORD_RAMP of
# plot_paper_figures.py with one lighter step in front, same hue).
STAGE_COLORS = ["#b9d5f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab"]
FS_TICK, FS_LABEL, FS_LEGEND = 12, 13, 11   # FPS figure text sizes (pt)
HATCH = "////"
MODE_SHORT = {"15W": "15 W", "25W": "25 W", "MAXN_SUPER": "MAXN"}
SEGMENTS = [   # as plot_compare_reports.plot_time_breakdown
    ("kernel_exec_ms", "Kernel execution (GPU busy)", BLUE, ""),
    ("idle_gap_ms", "GPU idle gap (waiting on CPU)", ORANGE, "///"),
    ("pre_kernel_ms", "Pre-kernel / setup (CPU)", AQUA, "..."),
    ("post_kernel_ms", "Post-kernel (postprocessing)", YELLOW, "xxx"),
]


def plot_breakdown(path: Path, modes, model, set_name, outdir) -> None:
    rows = [r for r in csv.DictReader(open(path)) if r["stage"].startswith(model + "_")]
    if not rows:
        return
    cell = {}
    for r in rows:
        cell.setdefault((r["mode"], r["stage"].split("_", 1)[1]), []).append(r)
    med = {k: {key: statistics.median(float(r[key]) for r in v) for key, *_ in SEGMENTS}
           for k, v in cell.items()}
    present = [m for m in modes if any((m, s) in med for s in STAGES)]
    width = 0.8 / len(present)
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    top, ticks, tick_labels = 0.0, [], []
    for i, stage in enumerate(STAGES):
        for j, mode in enumerate(present):
            vals = med.get((mode, stage))
            if not vals:
                continue
            x = i + (j - (len(present) - 1) / 2) * width
            bottom = 0.0
            for key, label, color, hatch in SEGMENTS:
                ax.bar(x, vals[key], width * 0.9, bottom=bottom, color=color, hatch=hatch,
                       edgecolor="white", linewidth=0.5, zorder=3,
                       label=label if (i, j) == (0, 0) else None)
                bottom += vals[key]
            top = max(top, bottom)
            ticks.append(x)
            tick_labels.append(MODE_SHORT.get(mode, mode))
    style_axes(ax, "Time per inference (ms)")
    ax.set_xticks(ticks)   # first row: the power mode under each bar
    ax.set_xticklabels(tick_labels, fontsize=6.5)
    ax.tick_params(axis="x", length=0, pad=3)
    stages_axis = ax.secondary_xaxis(-0.09)   # second row: the stage under each group
    stages_axis.set_xticks(range(len(STAGES)))
    stages_axis.set_xticklabels([LABELS[s].replace("\n", " ") for s in STAGES], fontsize=8)
    stages_axis.tick_params(axis="x", length=0, colors="black")
    stages_axis.spines["bottom"].set_visible(False)
    ax.set_ylim(0, top * 1.08)
    ax.set_title(f"{MODELS[model]}: inference time per stage and power mode, by component",
                 fontsize=10)
    fig.legend(frameon=False, fontsize=7.5, ncol=2, loc="lower center")
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    save(fig, outdir, f"{set_name}_{model}_breakdown")


def load(path: Path) -> dict[tuple[str, str, str], list[dict]]:
    """(mode, stage, profiler) -> the last successful run of each round."""
    last = {}
    for r in csv.DictReader(open(path)):
        if r["fps"]:
            last[(r["mode"], r["stage"], r["profiler"], r["run"])] = r
    cells: dict[tuple[str, str, str], list[dict]] = {}
    for (m, s, p, _), r in last.items():
        cells.setdefault((m, s, p), []).append(r)
    if not cells:
        raise SystemExit(f"no successful runs in {path}")
    return cells


def plot(cells, modes, model, target, set_name, outdir, profiler="plain") -> None:
    """One group per power mode, one bar per stage inside it."""
    present = [m for m in modes if any((m, f"{model}_{st}", profiler) in cells for st in STAGES)]
    if not present:
        return
    width = 0.8 / len(STAGES)
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    top, throttled_any = 0.0, False
    for i, mode in enumerate(present):
        for j, (stage, color) in enumerate(zip(STAGES, STAGE_COLORS)):
            runs = cells.get((mode, f"{model}_{stage}", profiler))
            if not runs:
                continue
            med = statistics.median(float(r["fps"]) for r in runs)
            x = i + (j - (len(STAGES) - 1) / 2) * width
            throttled = any(r["throttled"] == "1" for r in runs)
            throttled_any |= throttled
            ax.bar(x, med, width=width * 0.92, color=color, zorder=3, edgecolor="white",
                   linewidth=0.5, hatch=HATCH if throttled else None)
            top = max(top, med)
    ax.axhline(target, color="black", linestyle=":", linewidth=1.0, zorder=5)
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([MODE_LABELS.get(m, m) for m in present])
    ax.tick_params(axis="x", length=0)
    style_axes(ax, "Throughput (FPS)")
    for spine in ("left", "bottom"):   # black axes, ticks and labels (style_axes makes them grey)
        ax.spines[spine].set_color("black")
    ax.tick_params(colors="black", labelsize=FS_TICK)
    ax.set_ylabel("Throughput (FPS)", fontsize=FS_LABEL)
    ax.set_xlabel("Power mode", fontsize=FS_LABEL)
    ax.set_ylim(0, max(top, target) * 1.05)
    handles = [Patch(facecolor=c, label=LABELS[st].replace("\n", " "))
               for st, c in zip(STAGES, STAGE_COLORS)]
    if throttled_any:
        handles.append(Patch(facecolor="white", edgecolor=MUTED, hatch=HATCH, label="throttled"))
    handles.append(Line2D([], [], color="black", linestyle=":", linewidth=1.0,
                          label=f"{target:g} FPS target"))
    fig.legend(handles=handles, frameon=False, fontsize=FS_LEGEND, ncol=4, loc="lower center",
               columnspacing=1.2, handletextpad=0.5)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    save(fig, outdir, f"{set_name}_{model}_fps" + ("" if profiler == "plain" else f"_{profiler}"))


def main() -> None:
    p = argparse.ArgumentParser(description="FPS per stage and power mode, with and without nsys")
    p.add_argument("--set", default="modes_v2")
    p.add_argument("--dir", default=None, help="Default: reports/<set>")
    p.add_argument("--outdir", default=None, help="Default: reports/figures/<set>")
    p.add_argument("--target", type=float, default=30.0, help="FPS target line")
    p.add_argument("--profiler", default="plain", choices=["plain", "nsys"],
                   help="Which runs to plot: plain = without a profiler (default), nsys = under nsys")
    args = p.parse_args()
    src = Path(args.dir or f"reports/{args.set}")
    outdir = Path(args.outdir or f"reports/figures/{args.set}")
    cells = load(src / "fps.csv")

    present = {m for m, _, _ in cells}
    modes = [m for m in MODE_ORDER if m in present] + sorted(present - set(MODE_ORDER))
    for model in MODELS:
        if any(s.startswith(model + "_") for _, s, _ in cells):
            plot(cells, modes, model, args.target, args.set, outdir, args.profiler)
            if (src / "breakdown.csv").exists():
                plot_breakdown(src / "breakdown.csv", modes, model, args.set, outdir)


if __name__ == "__main__":
    main()
