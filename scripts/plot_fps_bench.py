#!/usr/bin/env python3
"""Frame-time figure per model from scripts/bench_fps.sh (FPS without a
profiler, N runs per stage): the counterpart of plot_compare_reports.py's
01_frame_time, which comes from nsys reports.

Bar = median frame time over the runs (1000 / median FPS); dots = the single
runs; whisker = their min-max. Labels: median FPS and frame time.

Usage (from the repo root):
    python3 scripts/plot_fps_bench.py                       # reports/fps/fps25w_runs.csv
    python3 scripts/plot_fps_bench.py --set fps25w --outdir reports/figures/fps25w
Writes <outdir>/<set>_<model>_01_frame_time.png/.pdf, one per model in the CSV.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_compare_reports import ramp  # noqa: E402
from plot_paper_figures import MUTED, save, style_axes  # noqa: E402

MODELS = {"yolo": "YOLOv8n", "m15": "M15"}
LABELS = {"baseline": "Baseline", "graph": "+ CUDA Graph", "gpupre": "+ GPU preprocessing",
          "nms": "+ Custom NMS", "fp16": "+ FP16"}


def plot_model(model: str, stages: dict[str, list[float]], set_name: str, outdir: Path) -> None:
    names = list(stages)
    labels = [LABELS.get(n.split("_", 1)[1], n) for n in names]
    median_fps = [statistics.median(stages[n]) for n in names]
    median_ms = [1000.0 / f for f in median_fps]
    runs_ms = [[1000.0 / f for f in stages[n]] for n in names]
    n_runs = min(len(r) for r in runs_ms)
    top = max(max(r) for r in runs_ms)

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x = list(range(len(names)))
    ax.bar(x, median_ms, color=ramp(len(names)), width=0.6, zorder=3, edgecolor="white", linewidth=0.5)
    for i, r in enumerate(runs_ms):
        ax.vlines(i, min(r), max(r), color="black", linewidth=1.0, zorder=4)
        offsets = [(j - (len(r) - 1) / 2) * 0.025 for j in range(len(r))]  # spread the dots a little
        ax.scatter([i + o for o in offsets], r, s=6, color="black", alpha=0.6, zorder=5, linewidths=0)
        ax.text(i, max(r) + top * 0.03, f"{median_ms[i]:.2f}ms", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
        ax.text(i, max(r) + top * 0.10, f"{median_fps[i]:.1f} FPS", ha="center", va="bottom",
                fontsize=8, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([textwrap.fill(l, 14) for l in labels])
    style_axes(ax, "End-to-end frame time (ms)")
    ax.set_ylim(0, top * 1.30)
    ax.set_title(f"{MODELS.get(model, model)}: frame time per stage, no profiler "
                 f"(median of {n_runs} runs)")
    fig.tight_layout()
    save(fig, outdir, f"{set_name}_{model}_01_frame_time")


def main() -> None:
    p = argparse.ArgumentParser(description="Frame-time figures from bench_fps.sh results")
    p.add_argument("--set", default="fps25w", help="Set name used with bench_fps.sh")
    p.add_argument("--runs-csv", default=None, help="Default: reports/fps/<set>_runs.csv")
    p.add_argument("--outdir", default=None, help="Default: reports/figures/<set>")
    args = p.parse_args()
    runs_csv = Path(args.runs_csv or f"reports/fps/{args.set}_runs.csv")
    outdir = Path(args.outdir or f"reports/figures/{args.set}")

    by_model: dict[str, dict[str, list[float]]] = {}
    for row in csv.DictReader(open(runs_csv)):
        if row["fps"]:  # failed runs have no FPS
            by_model.setdefault(row["stage"].split("_")[0], {}).setdefault(row["stage"], []).append(
                float(row["fps"]))
    for model, stages in by_model.items():
        plot_model(model, stages, args.set, outdir)


if __name__ == "__main__":
    main()
