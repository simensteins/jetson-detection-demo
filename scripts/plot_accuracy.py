#!/usr/bin/env python3
"""Detection accuracy per optimization stage, one figure per model: one line
per AP metric across the stages, with the share of the baseline's detections
each stage reproduces written under its name. A control pipeline that is not
a stage can be drawn as unconnected hollow markers (CONFIG "control"); none is
shown now: YOLOv8n's fp16_ultranms (FP16 engine with the standard NMS) only
mattered while the fixed NMS was approximate, and stays in the CSVs. The y-axis starts at zero, so small changes are not inflated.

Inputs are the summary CSVs the evaluations write (one row per pipeline):
  YOLOv8n  reports/eval/yolo_coco_summary.csv (scripts/eval_yolo_coco.py, COCO val2017)
  M15      data/eval_m15/sar_ir_test/summary.csv (the stakeholder's SAR test set;
           private: data/ is gitignored, keep the figure there too)

Usage (from the repo root, any Python with matplotlib):
    python3 scripts/plot_accuracy.py --model yolo
    python3 scripts/plot_accuracy.py --model yolo_person   # reports/eval/yolo_coco_person_summary.csv
    python3 scripts/plot_accuracy.py --model m15
Writes <outdir>/accuracy_<model>.png/.pdf.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_paper_figures import AQUA, BLUE, MUTED, ORANGE, save, style_axes  # noqa: E402

STAGE_LABELS = {"baseline": "Baseline", "gpupre": "+ GPU\npreproc.", "customnms": "+ Fixed\nNMS",
                "nms": "+ Fixed\nNMS", "fp16": "+ FP16",
                "fp16_ultranms": "FP16 with\nstandard NMS"}
CONFIG = {
    "yolo": dict(name="YOLOv8n", dataset="COCO val2017, 600 images",
                 csv="reports/eval/yolo_coco_summary.csv", outdir="reports/figures/eval",
                 stages=["baseline", "gpupre", "customnms", "fp16"],
                 control=None,
                 metrics=[("mAP50", "mAP50", BLUE), ("mAP50_95", "mAP50-95", ORANGE),
                          ("mAP_small", "mAP small objects", AQUA)]),
    "yolo_person": dict(name="YOLOv8n, person only", dataset="COCO val2017, 600 images",
                        csv="reports/eval/yolo_coco_person_summary.csv", outdir="reports/figures/eval",
                        stages=["baseline", "gpupre", "customnms", "fp16"],
                        control=None,
                        metrics=[("mAP50", "AP50", BLUE), ("mAP50_95", "AP50-95", ORANGE),
                                 ("mAP_small", "AP small objects", AQUA)]),
    "m15": dict(name="M15", dataset="SAR test set, 1500 images",
                csv="data/eval_m15/sar_ir_test/summary.csv", outdir="data/eval_m15/figures",
                stages=["baseline", "gpupre", "nms", "fp16"], control=None,
                metrics=[("mAP50", "AP50", BLUE), ("mAP50_95", "AP50-95", ORANGE),
                         ("mAP_small", "AP small objects", AQUA)]),
}


def main() -> None:
    p = argparse.ArgumentParser(description="Accuracy per optimization stage")
    p.add_argument("--model", choices=sorted(CONFIG), required=True)
    p.add_argument("--csv", default=None)
    p.add_argument("--outdir", default=None)
    args = p.parse_args()
    cfg = CONFIG[args.model]
    rows = {r["pipeline"]: r for r in csv.DictReader(open(args.csv or cfg["csv"]))}
    stages = [s for s in cfg["stages"] if s in rows]
    metrics = cfg["metrics"]

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    top = max(float(rows[s][k]) for s in stages for k, *_ in metrics)
    line_stages = [s for s in stages if s != cfg["control"]]
    xs = [stages.index(s) for s in line_stages]
    for key, label, color in metrics:
        ax.plot(xs, [float(rows[s][key]) for s in line_stages], color=color, linewidth=2,
                marker="o", markersize=6, zorder=3, label=label)
        if cfg["control"] in stages:
            ax.plot(stages.index(cfg["control"]), float(rows[cfg["control"]][key]), linestyle="none",
                    marker="o", markersize=7, markerfacecolor="white", markeredgecolor=color,
                    markeredgewidth=2, zorder=3)
    labels = []
    for s in stages:
        kept = rows[s].get("baseline_dets_kept_025") or rows[s].get("baseline_dets_kept") or ""
        labels.append(STAGE_LABELS[s] + (f"\n{float(kept):.1f} % kept" if kept else "\n(reference)"))
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.tick_params(axis="x", length=0)
    style_axes(ax, "Average precision (%)")
    ax.set_ylim(0, top * 1.15)
    ax.set_xlim(-0.4, len(stages) - 0.6)
    ax.set_title(f"{cfg['name']}: detection accuracy per stage ({cfg['dataset']})", fontsize=10)
    handles, _ = ax.get_legend_handles_labels()
    if cfg["control"] in stages:
        handles.append(Line2D([], [], linestyle="none", marker="o", markersize=7,
                              markerfacecolor="white", markeredgecolor=MUTED, markeredgewidth=2,
                              label="control, not a stage"))
    fig.legend(handles=handles, frameon=False, fontsize=8, ncol=len(handles), loc="lower center")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, Path(args.outdir or cfg["outdir"]), f"accuracy_{args.model}")


if __name__ == "__main__":
    main()
