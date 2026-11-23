#!/usr/bin/env python3
"""FPS of every stage in every power mode, from scripts/run_power_modes.sh:
one figure per model, with the runs without a profiler and the runs under
nsys side by side, so the profiler's overhead shows as the drop from the left
panel to the right.

Both panels share one FPS axis, with a dotted target line (default 30 FPS).
Per stage there is one bar per power mode (light to dark: 15 W, 25 W, MAXN
SUPER); the bar is the median over the rounds. Bars of cells with at least one
throttled run (over-current counters increased) are hatched.

Usage (from the repo root, after copying reports/<set>/ from the Jetson; any
Python with matplotlib):
    python3 scripts/plot_power_modes.py --set modes_v2
    python3 scripts/plot_power_modes.py --set modes_v2 --target 30 --outdir reports/figures/modes_v2
Writes <outdir>/<set>_<model>_fps.png/.pdf, one per model.
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
from plot_paper_figures import MUTED, save, style_axes  # noqa: E402

MODELS = {"yolo": "YOLOv8n", "m15": "M15"}
STAGES = ["baseline", "graph", "gpupre", "nms", "fp16"]
LABELS = {"baseline": "Baseline", "graph": "+ CUDA\ngraph", "gpupre": "+ GPU\npreproc.",
          "nms": "+ Fixed\nNMS", "fp16": "+ FP16"}
MODE_ORDER = ["15W", "25W", "MAXN_SUPER"]   # increasing power budget; others go last
MODE_LABELS = {"MAXN_SUPER": "MAXN SUPER", "15W": "15 W", "25W": "25 W"}
TITLES = {"plain": "Without profiler", "nsys": "Under Nsight Systems"}
HATCH = "////"


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


def plot(cells, modes, model, target, set_name, outdir) -> None:
    profilers = [p for p in ("plain", "nsys") if any(p_ == p for _, _, p_ in cells)]
    colors = ramp(len(modes))
    width = 0.8 / len(modes)
    top = max(float(r["fps"]) for (_, s, _), runs in cells.items() if s.startswith(model + "_")
              for r in runs)
    fig, axes = plt.subplots(1, len(profilers), figsize=(7.0, 3.4), sharey=True)
    axes = list(axes) if len(profilers) > 1 else [axes]
    throttled_any = False
    for ax, profiler in zip(axes, profilers):
        for j, (mode, color) in enumerate(zip(modes, colors)):
            for i, stage in enumerate(STAGES):
                runs = cells.get((mode, f"{model}_{stage}", profiler))
                if not runs:
                    continue
                med = statistics.median(float(r["fps"]) for r in runs)
                x = i + (j - (len(modes) - 1) / 2) * width
                throttled = any(r["throttled"] == "1" for r in runs)
                throttled_any |= throttled
                ax.bar(x, med, width=width * 0.92, color=color, zorder=3, edgecolor="white",
                       linewidth=0.5, hatch=HATCH if throttled else None)
        ax.axhline(target, color="black", linestyle=":", linewidth=1.0, zorder=5)
        ax.set_xticks(range(len(STAGES)))
        ax.set_xticklabels([LABELS[s] for s in STAGES], fontsize=7.5)
        style_axes(ax, "Throughput (FPS)" if ax is axes[0] else "")
        if ax is not axes[0]:   # one y-axis per figure: the panels share it
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", left=False)
        ax.set_ylim(0, max(top, target) * 1.08)
        ax.set_title(TITLES[profiler], fontsize=10)
    handles = [Patch(facecolor=c, label=MODE_LABELS.get(m, m)) for m, c in zip(modes, colors)]
    if throttled_any:
        handles.append(Patch(facecolor="white", edgecolor=MUTED, hatch=HATCH, label="throttled"))
    handles.append(Line2D([], [], color="black", linestyle=":", linewidth=1.0,
                          label=f"{target:g} FPS target"))
    fig.legend(handles=handles, frameon=False, fontsize=8, ncol=len(handles), loc="lower center")
    fig.suptitle(f"{MODELS[model]}: throughput per stage and power mode", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, outdir, f"{set_name}_{model}_fps")


def main() -> None:
    p = argparse.ArgumentParser(description="FPS per stage and power mode, with and without nsys")
    p.add_argument("--set", default="modes_v2")
    p.add_argument("--dir", default=None, help="Default: reports/<set>")
    p.add_argument("--outdir", default=None, help="Default: reports/figures/<set>")
    p.add_argument("--target", type=float, default=30.0, help="FPS target line")
    args = p.parse_args()
    src = Path(args.dir or f"reports/{args.set}")
    outdir = Path(args.outdir or f"reports/figures/{args.set}")
    cells = load(src / "fps.csv")

    present = {m for m, _, _ in cells}
    modes = [m for m in MODE_ORDER if m in present] + sorted(present - set(MODE_ORDER))
    for model in MODELS:
        if any(s.startswith(model + "_") for _, s, _ in cells):
            plot(cells, modes, model, args.target, args.set, outdir)


if __name__ == "__main__":
    main()
