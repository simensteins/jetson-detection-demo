#!/usr/bin/env python3
"""All figures for a power-mode set from scripts/run_power_modes.sh.

From reports/<set>/runs.csv (the last successful run of each mode, kind and
stage counts; earlier failed or forced-over rows are ignored):
  <set>_<model>_fps_by_mode      frame time per stage, one bar per power mode,
                                 without the profiler; labels are FPS
  <set>_<model>_energy_by_mode   board energy per frame (VDD_IN power / FPS)
  <set>_summary.csv / .md        every number behind them, per mode and stage
and per mode, from the nsys runs' .sqlite files, the five comparison figures
of scripts/plot_compare_reports.py (frame time, idle gap, time breakdown,
dispatch, swimlane) in <outdir>/<mode>/.

Runs during which the over-current counters increased are kept and marked:
hatched bars, "(throttled)" in the comparison-figure labels, and a column in
the summary. Power modes are ordered by their measured average power, light to
dark.

Usage (from the repo root, after copying reports/<set>/ from the Jetson):
    python3 scripts/plot_power_modes.py --set modes
    python3 scripts/plot_power_modes.py --set modes --skip-nsys   # only the cross-mode figures
"""
from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_compare_reports import ramp  # noqa: E402
from plot_paper_figures import MUTED, save, style_axes  # noqa: E402

MODELS = {"yolo": "YOLOv8n", "m15": "M15"}
STAGES = ["baseline", "graph", "gpupre", "nms", "fp16"]
LABELS = {"baseline": "Baseline", "graph": "+ CUDA Graph", "gpupre": "+ GPU preprocessing",
          "nms": "+ Custom NMS", "fp16": "+ FP16"}
HATCH = "////"


def tag_of(mode: str) -> str:
    return "".join(c for c in mode.lower() if c.isalnum())


def load_runs(path: Path) -> dict[tuple[str, str, str], dict]:
    """(mode, kind, stage) -> last row with an FPS."""
    runs = {}
    for row in csv.DictReader(open(path)):
        if row["fps"]:
            runs[(row["mode"], row["kind"], row["stage"])] = row
    if not runs:
        raise SystemExit(f"no successful runs in {path}")
    return runs


def num(row: dict | None, key: str) -> float | None:
    return float(row[key]) if row and row.get(key) not in (None, "") else None


def grouped_bars(model: str, modes: list[str], runs: dict, value, ylabel: str, title: str,
                 label, set_name: str, outdir: Path, fname: str) -> None:
    stages = [s for s in STAGES if any((m, "fps", f"{model}_{s}") in runs for m in modes)]
    colors = ramp(len(modes))
    width = 0.8 / len(modes)
    fig, ax = plt.subplots(figsize=(max(6.0, 1.5 * len(stages) * len(modes) / 3), 3.8))
    top, throttled_any = 0.0, False
    for j, (mode, color) in enumerate(zip(modes, colors)):
        for i, s in enumerate(stages):
            row = runs.get((mode, "fps", f"{model}_{s}"))
            v = value(row) if row else None
            if v is None:
                continue
            x = i + (j - (len(modes) - 1) / 2) * width
            throttled = row["throttled"] == "1"
            throttled_any |= throttled
            ax.bar(x, v, width=width * 0.92, color=color, zorder=3, edgecolor="white",
                   linewidth=0.5, hatch=HATCH if throttled else None)
            ax.text(x, v, label(row), ha="center", va="bottom", fontsize=6.5, color=MUTED,
                    rotation=90 if len(modes) > 3 else 0)
            top = max(top, v)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels([textwrap.fill(LABELS[s], 14) for s in stages])
    style_axes(ax, ylabel)
    ax.set_ylim(0, top * 1.22)
    handles = [Patch(facecolor=c, label=m) for m, c in zip(modes, colors)]
    if throttled_any:
        handles.append(Patch(facecolor="white", edgecolor=MUTED, hatch=HATCH,
                             label="throttled (over-current)"))
    ax.legend(handles=handles, frameon=False, fontsize=8, ncol=len(handles), loc="upper center",
              bbox_to_anchor=(0.5, -0.17))
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    save(fig, outdir, f"{set_name}_{model}_{fname}")


def main() -> None:
    p = argparse.ArgumentParser(description="Figures for a run_power_modes.sh set")
    p.add_argument("--set", default="modes")
    p.add_argument("--dir", default=None, help="Default: reports/<set>")
    p.add_argument("--outdir", default=None, help="Default: reports/figures/<set>")
    p.add_argument("--skip-nsys", action="store_true", help="Skip the per-mode nsys figures")
    args = p.parse_args()
    src = Path(args.dir or f"reports/{args.set}")
    outdir = Path(args.outdir or f"reports/figures/{args.set}")
    runs = load_runs(src / "runs.csv")

    # Modes ordered by their average board power over the fps runs (unknown power last).
    power = {}
    for (mode, kind, _), row in runs.items():
        if kind == "fps" and num(row, "power_mw") is not None:
            power.setdefault(mode, []).append(num(row, "power_mw"))
    modes = sorted({m for m, _, _ in runs}, key=lambda m: statistics.mean(power[m]) if m in power else 1e9)
    models = [m for m in MODELS if any(s.startswith(m + "_") for _, _, s in runs)]

    # --- summary table -------------------------------------------------------
    keys = ["mode", "stage", "fps", "frame_ms", "power_w", "mj_per_frame", "tj_max_c",
            "gpu_mhz", "throttled", "oc_events", "fps_nsys", "throttled_nsys"]
    rows = []
    for model in models:
        for mode in modes:
            for s in STAGES:
                f = runs.get((mode, "fps", f"{model}_{s}"))
                n = runs.get((mode, "nsys", f"{model}_{s}"))
                if not f and not n:
                    continue
                fps, pw = num(f, "fps"), num(f, "power_mw")
                gmin, gmax = (f or {}).get("gpu_mhz_min", ""), (f or {}).get("gpu_mhz_max", "")
                rows.append({"mode": mode, "stage": f"{model}_{s}", "fps": fps or "",
                             "frame_ms": round(1000 / fps, 2) if fps else "",
                             "power_w": round(pw / 1000, 2) if pw else "",
                             "mj_per_frame": round(pw / fps, 1) if fps and pw else "",
                             "tj_max_c": (f or {}).get("tj_max_c", ""),
                             "gpu_mhz": gmin if gmin == gmax else f"{gmin}-{gmax}",
                             "throttled": (f or {}).get("throttled", ""), "oc_events": (f or {}).get("oc_delta", ""),
                             "fps_nsys": num(n, "fps") or "", "throttled_nsys": (n or {}).get("throttled", "")})
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / f"{args.set}_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    md = ["| Mode | Stage | FPS | Frame | Board power | Energy/frame | Tj max | GPU MHz | FPS under nsys |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        flag = " ⚠ throttled" if r["throttled"] == "1" else ""
        nflag = " ⚠" if r["throttled_nsys"] == "1" else ""
        md.append(f"| {r['mode']} | {r['stage']} | **{r['fps']}**{flag} | {r['frame_ms']} ms | {r['power_w']} W "
                  f"| {r['mj_per_frame']} mJ | {r['tj_max_c']} °C | {r['gpu_mhz']} | {r['fps_nsys']}{nflag} |")
    (outdir / f"{args.set}_summary.md").write_text(
        "Power = board input (VDD_IN, tegrastats) averaged over the measured frames; one run per "
        "mode and stage; ⚠ = the over-current counters increased during the run.\n\n" + "\n".join(md) + "\n")
    print("\n".join(md))

    # --- cross-mode figures ----------------------------------------------------
    for model in models:
        name = MODELS[model]
        grouped_bars(model, modes, runs, lambda r: 1000 / float(r["fps"]), "End-to-end frame time (ms)",
                     f"{name}: frame time per stage and power mode, no profiler (labels: FPS)",
                     lambda r: f"{float(r['fps']):.0f}", args.set, outdir, "fps_by_mode")
        grouped_bars(model, modes, runs,
                     lambda r: num(r, "power_mw") / float(r["fps"]) if num(r, "power_mw") else None,
                     "Board energy per frame (mJ)", f"{name}: energy per frame (VDD_IN / FPS)",
                     lambda r: f"{num(r, 'power_mw') / 1000:.1f}W", args.set, outdir, "energy_by_mode")

    # --- per-mode nsys figures -------------------------------------------------
    if args.skip_nsys:
        return
    for mode in modes:
        tag = tag_of(mode)
        for model in models:
            reports, labels = [], []
            for s in STAGES:
                row = runs.get((mode, "nsys", f"{model}_{s}"))
                sqlite = src / tag / f"{model}_{s}.sqlite"
                if row and sqlite.exists():
                    reports.append(str(sqlite))
                    labels.append(LABELS[s] + (" (throttled)" if row["throttled"] == "1" else ""))
            if not reports:
                continue
            print(f"\n=== {mode} {model}: {len(reports)} stages")
            subprocess.run([sys.executable, str(Path(__file__).parent / "plot_compare_reports.py"),
                            "--name", f"{args.set}_{tag}_{model}", "--model", f"{MODELS[model]} ({mode})",
                            "--outdir", str(outdir / tag), "--reports", *reports, "--labels", *labels],
                           check=True)


if __name__ == "__main__":
    main()
