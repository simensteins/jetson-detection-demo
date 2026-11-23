#!/usr/bin/env python3
"""Effect of the fixed-shape NMS stage on the mean and the slowest frames,
for both models: the stage before it (+ GPU preprocessing) against the stage
with it (+ fixed-shape NMS), from the profiled paper-set runs.

The fixed-shape NMS costs the same on every frame, so it barely moves the mean
on a video where few frames have detections, but it removes the slow frames
where the standard NMS had many candidates to filter. One panel per model
(their frame times differ by ~4x), three groups per panel: mean, p99 and the
worst frame. Frame time = decode start to the next decode start, the first
--skip-frames frames excluded, the same definition as plot_compare_reports.py.

The worst frame leaves out frames in which CUDA loads code (cuLibraryLoad* /
cuModuleLoad* calls): with the standard NMS, the first frame with a detection
loads the detection path's kernels (M15: frame 58, 190 ms, against ~49 ms for
a normal frame). That one-time cost is not the stage's per-frame effect; the
fixed NMS runs all of its kernels during warm-up, so it never shows up there.
The left-out frames are listed in the CSV. Mean and p99 use every frame.

Usage (from the repo root, any Python with matplotlib):
    python3 scripts/plot_nms_gain.py                      # paper-set reports, 25 W
    python3 scripts/plot_nms_gain.py --outdir reports/figures/paper25w_v2
Writes <outdir>/<name>.png/.pdf and <name>.csv with the numbers behind it.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import sqlite3

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_compare_reports import summarize_stage  # noqa: E402
from plot_paper_figures import MUTED, ORD_RAMP, save, style_axes  # noqa: E402

R = "reports/nsys_paper_reports"
# Before = + GPU preprocessing, after = + fixed-shape NMS. M15's NMS stage is
# the 27.10 rerun in the paper's stage order (NMS in the graph, k = 100);
# paper25w_m15_nms is the older order (after FP16, outside the graph, k = 1000).
MODELS = {
    "YOLOv8n": (f"{R}/paper25w_yolo_gpupre.sqlite", f"{R}/paper25w_yolo_nms.sqlite"),
    "M15": (f"{R}/paper25w_m15_gpupre.sqlite", f"{R}/paper25w_v2_m15_nms.sqlite"),
}
STAGE_LABELS = ("+ GPU preprocessing", "+ Fixed-shape NMS")
METRICS = (("frame_time_ms", "Mean"), ("frame_p99_ms", "p99"),
           ("frame_max_clean_ms", "Worst frame"))


def worst_without_loading(path: str, skip: int) -> tuple[float, list[tuple[int, float]]]:
    """Slowest frame time (ms) among frames without CUDA code loading, and the
    frames left out as (frame index, ms)."""
    con = sqlite3.connect(path)
    dec = [s for (s,) in con.execute("SELECT start FROM NVTX_EVENTS WHERE text='decode' ORDER BY start")]
    loads = [s for (s,) in con.execute(
        "SELECT r.start FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds n ON r.nameId = n.id "
        "WHERE n.value LIKE 'cuLibraryLoad%' OR n.value LIKE 'cuModuleLoad%'")]
    con.close()
    clean, left_out = [], []
    for i in range(skip, len(dec) - 1):
        ms = (dec[i + 1] - dec[i]) / 1e6
        if any(dec[i] <= t < dec[i + 1] for t in loads):
            left_out.append((i, round(ms, 1)))
        else:
            clean.append(ms)
    return max(clean), left_out
COLORS = (ORD_RAMP[0], ORD_RAMP[-1])   # light = before, dark = after


def main() -> None:
    p = argparse.ArgumentParser(description="Fixed-shape NMS: mean vs slowest frames")
    p.add_argument("--outdir", default="reports/figures/paper25w_v2")
    p.add_argument("--name", default="nms_gain_mean_worst")
    p.add_argument("--skip-frames", type=int, default=50)
    args = p.parse_args()
    outdir = Path(args.outdir)

    stats = {}
    for m, paths in MODELS.items():
        stats[m] = []
        for path in paths:
            st = summarize_stage([path], args.skip_frames)
            st["frame_max_clean_ms"], st["left_out"] = worst_without_loading(path, args.skip_frames)
            stats[m].append(st)

    fig, axes = plt.subplots(1, len(MODELS), figsize=(7.0, 3.2))
    width = 0.36
    for ax, (model, (before, after)) in zip(axes, stats.items()):
        x = range(len(METRICS))
        top = max(s[k] for s in (before, after) for k, _ in METRICS)
        for j, s in enumerate((before, after)):
            xs = [i + (j - 0.5) * width for i in x]
            vals = [s[k] for k, _ in METRICS]
            ax.bar(xs, vals, width=width, color=COLORS[j], zorder=3, edgecolor="white",
                   linewidth=1.0, label=STAGE_LABELS[j])
            for xi, v in zip(xs, vals):
                ax.text(xi, v + top * 0.02, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)
        for i, (k, _) in enumerate(METRICS):   # change in % above each pair
            change = 100 * (after[k] - before[k]) / before[k]
            ax.text(i, top * 1.13, f"{change:+.0f}%", ha="center", va="bottom", fontsize=8,
                    fontweight="bold", color=MUTED)
        ax.set_xticks(list(x))
        ax.set_xticklabels([label for _, label in METRICS])
        style_axes(ax, "Frame time (ms)" if ax is axes[0] else "")
        ax.set_ylim(0, top * 1.25)
        ax.set_title(model)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, outdir, args.name)

    rows = []
    for model, (before, after) in stats.items():
        for k, label in METRICS:
            rows.append({"model": model, "metric": label,
                         "gpu_preprocessing_ms": round(before[k], 2),
                         "fixed_nms_ms": round(after[k], 2),
                         "change_pct": round(100 * (after[k] - before[k]) / before[k], 1),
                         "left_out_gpu_preprocessing": " ".join(f"{i}:{ms}" for i, ms in before["left_out"])
                         if k == "frame_max_clean_ms" else "",
                         "left_out_fixed_nms": " ".join(f"{i}:{ms}" for i, ms in after["left_out"])
                         if k == "frame_max_clean_ms" else ""})
    csv_path = outdir / f"{args.name}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"  {r['model']:8s} {r['metric']:12s} {r['gpu_preprocessing_ms']:7.2f} -> "
              f"{r['fixed_nms_ms']:7.2f} ms  ({r['change_pct']:+.1f} %)")
    print(f"  wrote {csv_path}")


if __name__ == "__main__":
    main()
