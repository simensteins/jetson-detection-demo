#!/usr/bin/env python3
"""One table from several evaluator metrics.json files (a "coco_bbox" block
and an "operating_point_sweep" block per file): COCO AP and the operating
point at one confidence threshold, per run.

Usage:
    python3 scripts/compare_m15_eval.py --labels Baseline "+ GPU pre" ... \\
        --metrics data/eval_m15/<name>/baseline/metrics.json ... \\
        --out data/eval_m15/<name>/comparison.md
Keep the output under data/ (gitignored): the test results are not ours to publish.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Compare evaluator metrics.json files")
    p.add_argument("--metrics", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--threshold", type=float, default=0.25, help="Operating point to show")
    p.add_argument("--out", default=None, help="Also write the table here (.md)")
    args = p.parse_args()
    if len(args.metrics) != len(args.labels):
        raise SystemExit("one label per metrics file")

    lines = [f"| Run | AP50:95 | AP50 | AP small | AR@300 | P@{args.threshold:g} | R@{args.threshold:g} "
             f"| F1@{args.threshold:g} | FP/image | Detections |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for label, path in zip(args.labels, args.metrics):
        m = json.loads(Path(path).read_text())
        c = m["coco_bbox"]
        op = next((o for o in m["operating_point_sweep"]["metrics_at_confidence_thresholds"]
                   if abs(o["threshold"] - args.threshold) < 1e-9), None)
        if op is None:
            raise SystemExit(f"{path}: no operating point at {args.threshold} "
                             "(pass it to the evaluator with --confidence-thresholds)")
        pct = lambda v: f"{100 * v:.2f} %" if v >= 0 else "–"  # noqa: E731  (-1: no such boxes)
        lines.append(f"| {label} | {pct(c['ap'])} | {pct(c['ap50'])} | {pct(c['ap_small'])} "
                     f"| {pct(c['ar_max_300'])} | {pct(op['precision'])} | {pct(op['recall'])} "
                     f"| {pct(op['f1'])} | {op['fp_per_image']:.3f} | {m['prediction_boxes']} |")
    table = "\n".join(lines)
    print(table)
    if args.out:
        Path(args.out).write_text(table + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
