#!/usr/bin/env python3
"""Score the predictions.coco.json files of scripts/eval_m15.py against the
test annotations, on the Jetson, and write one metrics.json per stage in the
layout scripts/compare_m15_eval.py reads.

Two sets of numbers per stage:

    coco_bbox              COCO AP/AR from pycocotools (the reference
                           implementation), with up to 300 detections per image:
                           AP50:95, AP50, AP75, AP small/medium/large, AR@300.
                           The score floor of the eval_m15.py run decides how
                           much of the curve AP covers (see below).
    operating_point_sweep  At each confidence threshold: detections with a score
                           at or above it are matched to the labels one-to-one,
                           highest score first, each to the free label it
                           overlaps most; IoU >= --match-iou (default 0.5) is a
                           true positive. Gives TP, FP, FN, precision, recall,
                           F1 and false positives per image. Labels marked
                           iscrowd are ignored (a detection on one counts as
                           neither). Also the highest recall at a fixed number
                           of false positives per image (--fppi-targets,
                           default 0.05).

Settings follow the model owner's own M15 test run, so the numbers can be set
next to theirs: predictions kept from confidence 0.25, class-agnostic NMS at
IoU 0.5, up to 300 detections per image (run eval_m15.py with --conf 0.25
--iou 0.5; the baseline stage should then give about as many detections as
their run). Their AP is therefore computed on predictions cut at 0.25. A second
run with --conf 0.01 gives the fuller precision/recall curve. Not known from
their run, and therefore ours: the match IoU at the operating points (0.5).

This is our own scorer, not the model owner's evaluator: use it to compare the
stages with each other. The annotations are only read (they can sit on a
read-only mount); everything written goes next to the predictions, under
data/ (gitignored). The results are not ours to publish.

Usage (on the Jetson, from the repo root; needs `pip install pycocotools`):
    python3 scripts/score_m15_eval.py --annotations <test annotations .json> \\
        --run data/eval_m15/<name>
Writes data/eval_m15/<name>/<stage>/metrics.json for every stage folder that
holds a predictions.coco.json, then prints the compare_m15_eval.py command.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

STAGE_LABELS = {"baseline": "Baseline", "gpupre": "+ GPU pre", "nms": "+ Fixed NMS", "fp16": "+ FP16"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="predictions.coco.json -> metrics.json per stage")
    p.add_argument("--annotations", required=True, help="COCO annotations of the test split")
    p.add_argument("--run", required=True, help="Run folder of eval_m15.py (data/eval_m15/<name>)")
    p.add_argument("--stages", nargs="+", default=None, help="Default: every stage folder in --run")
    p.add_argument("--confidence-thresholds", nargs="+", type=float, default=[0.01, 0.05, 0.1, 0.25, 0.5])
    p.add_argument("--fppi-targets", nargs="+", type=float, default=[0.05],
                   help="Also report the highest recall at these false positives per image")
    p.add_argument("--match-iou", type=float, default=0.5,
                   help="IoU for a true positive at the operating points (default 0.5)")
    p.add_argument("--max-dets", type=int, default=300, help="Detections per image for COCO AP/AR")
    return p.parse_args()


def xyxy(boxes) -> np.ndarray:
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.column_stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]])


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda d: (d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1])  # noqa: E731
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-9)


def match_image(dets: np.ndarray, gts: np.ndarray, crowds: np.ndarray, thr: float) -> np.ndarray:
    """Per detection (xyxy, sorted by score, highest first): 1 true positive,
    0 false positive, -1 ignored (on a crowd label). Each detection takes the
    free label it overlaps most, so the result for the top n detections does
    not depend on the ones below them: one pass serves every threshold."""
    flags = np.zeros(len(dets), dtype=np.int8)
    if len(dets) == 0:
        return flags
    iou = iou_matrix(dets, gts) if len(gts) else np.zeros((len(dets), 0))
    on_crowd = (iou_matrix(dets, crowds) >= thr).any(axis=1) if len(crowds) else np.zeros(len(dets), bool)
    used = np.zeros(len(gts), bool)
    for d in range(len(dets)):
        row = np.where(used, -1.0, iou[d])
        j = int(row.argmax()) if len(gts) else -1
        if j >= 0 and row[j] >= thr:
            used[j], flags[d] = True, 1
        elif on_crowd[d]:
            flags[d] = -1
    return flags


def operating_points(preds, gt_by_image, crowd_by_image, image_ids, thresholds, thr, fppi_targets) -> dict:
    by_image = defaultdict(list)
    for p in preds:
        by_image[p["image_id"]].append(p)
    scores, flags = [np.zeros(0)], [np.zeros(0, dtype=np.int8)]
    for i in image_ids:
        ps = sorted(by_image.get(i, []), key=lambda p: -p["score"])
        scores.append(np.array([p["score"] for p in ps], dtype=np.float64))
        flags.append(match_image(xyxy([p["bbox"] for p in ps]), gt_by_image[i], crowd_by_image[i], thr))
    scores, flags = np.concatenate(scores), np.concatenate(flags)
    order = np.argsort(-scores, kind="stable")
    scores, flags = scores[order], flags[order]
    tps, fps = np.cumsum(flags == 1), np.cumsum(flags == 0)
    n_gt, n_img = sum(len(g) for g in gt_by_image.values()), max(len(image_ids), 1)

    def point(n: int) -> dict:  # the n highest-scoring detections
        tp, fp = (int(tps[n - 1]), int(fps[n - 1])) if n else (0, 0)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / n_gt if n_gt else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"tp": tp, "fp": fp, "fn": n_gt - tp, "precision": precision, "recall": recall,
                "f1": f1, "fp_per_image": fp / n_img}

    at_thresholds = [{"threshold": t, **point(int((scores >= t).sum()))} for t in thresholds]
    # Highest recall with at most `target` false positives per image. A cut is
    # only possible between two different scores.
    cuts = [n for n in range(len(scores) + 1) if n == 0 or n == len(scores) or scores[n] < scores[n - 1]]
    at_fppi = []
    for target in fppi_targets:
        ok = [n for n in cuts if (fps[n - 1] if n else 0) / n_img <= target]
        n = max(ok, key=lambda n: (tps[n - 1] if n else 0, -n))
        at_fppi.append({"fppi_target": target, "threshold": float(scores[n - 1]) if n else None, **point(n)})
    return {"match_iou": thr, "metrics_at_confidence_thresholds": at_thresholds,
            "metrics_at_fppi_targets": at_fppi}


def coco_bbox(gt, preds, image_ids, max_dets: int) -> dict:
    """COCO AP/AR, read from COCOeval's precision/recall arrays (summarize()
    is tied to 100 detections per image). -1 means no labels of that size."""
    from pycocotools.cocoeval import COCOeval

    keys = ["ap", "ap50", "ap75", "ap_small", "ap_medium", "ap_large", f"ar_max_{max_dets}"]
    if not preds:
        return dict.fromkeys(keys, 0.0)
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(gt, gt.loadRes(preds), "bbox")
        E.params.imgIds = image_ids
        E.params.maxDets = [1, 10, max_dets]
        E.evaluate()
        E.accumulate()
    precision, recall = E.eval["precision"], E.eval["recall"]  # [T,R,K,A,M], [T,K,A,M]
    iou_thrs = E.params.iouThrs

    def mean(x: np.ndarray) -> float:
        x = x[x > -1]
        return float(x.mean()) if x.size else -1.0

    def ap(iou: float | None = None, area: int = 0) -> float:
        p = precision[..., area, -1]
        return mean(p if iou is None else p[np.isclose(iou_thrs, iou)])

    return dict(zip(keys, [ap(), ap(0.5), ap(0.75), ap(area=1), ap(area=2), ap(area=3),
                           mean(recall[..., 0, -1])]))


def main() -> None:
    args = parse_args()
    try:
        from pycocotools.coco import COCO
    except ImportError:
        raise SystemExit("pycocotools is missing: pip install pycocotools")

    run = Path(args.run)
    stages = args.stages or [s for s in STAGE_LABELS if (run / s / "predictions.coco.json").exists()]
    stages += [] if args.stages else sorted(
        d.name for d in run.iterdir()
        if d.name not in STAGE_LABELS and (d / "predictions.coco.json").exists())
    if not stages:
        raise SystemExit(f"no <stage>/predictions.coco.json under {run}")

    data = json.loads(Path(args.annotations).read_text())
    for a in data["annotations"]:  # COCOeval needs both fields
        a.setdefault("iscrowd", 0)
        a.setdefault("area", a["bbox"][2] * a["bbox"][3])
    image_ids = sorted(im["id"] for im in data["images"])

    # eval_m15.py --max-images N scores the first N images by id: do the same.
    run_conf = None
    info_path = run / "run_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        run_conf = info.get("conf")
        if info.get("images", len(image_ids)) < len(image_ids):
            image_ids = image_ids[:info["images"]]
            print(f"run covers the first {len(image_ids)} images (run_info.json): scoring only those")
    keep = set(image_ids)

    gt = COCO()
    with contextlib.redirect_stdout(io.StringIO()):
        gt.dataset = data
        gt.createIndex()
    gt_by_image, crowd_by_image, n_gt = {}, {}, 0
    by_image = defaultdict(list)
    for a in data["annotations"]:
        by_image[a["image_id"]].append(a)
    for i in image_ids:
        gt_by_image[i] = xyxy([a["bbox"] for a in by_image[i] if not a["iscrowd"]])
        crowd_by_image[i] = xyxy([a["bbox"] for a in by_image[i] if a["iscrowd"]])
        n_gt += len(gt_by_image[i])

    thresholds = sorted(args.confidence_thresholds)
    if run_conf is not None and any(t < run_conf for t in thresholds):
        print(f"NOTE: the run kept detections with score >= {run_conf}; "
              f"operating points below that are the same as at {run_conf}")
    print(f"{len(image_ids)} images, {n_gt} labels, match IoU {args.match_iou}, stages {stages}")

    for stage in stages:
        preds = [p for p in json.loads((run / stage / "predictions.coco.json").read_text())
                 if p["image_id"] in keep]
        metrics = {
            "scorer": "scripts/score_m15_eval.py (pycocotools + greedy one-to-one matching)",
            "annotations": str(args.annotations),
            "images": len(image_ids),
            "ground_truth_boxes": n_gt,
            "prediction_boxes": len(preds),
            "coco_bbox": coco_bbox(gt, preds, image_ids, args.max_dets),
            "operating_point_sweep": operating_points(
                preds, gt_by_image, crowd_by_image, image_ids, thresholds, args.match_iou,
                args.fppi_targets),
        }
        (run / stage / "metrics.json").write_text(json.dumps(metrics, indent=2))
        c = metrics["coco_bbox"]
        f = metrics["operating_point_sweep"]["metrics_at_fppi_targets"]
        print(f"  {stage:9s} {len(preds):7d} dets  AP50:95 {100 * c['ap']:6.2f} %  AP50 {100 * c['ap50']:6.2f} %"
              + "".join(f"  recall at {x['fppi_target']:g} FP/image {100 * x['recall']:6.2f} %" for x in f))

    labels = " ".join(f'"{STAGE_LABELS.get(s, s)}"' for s in stages)
    files = " ".join(str(run / s / "metrics.json") for s in stages)
    print(f"\nwrote {run}/<stage>/metrics.json. One table:\n"
          f"  python3 scripts/compare_m15_eval.py --labels {labels} \\\n"
          f"      --metrics {files} \\\n      --out {run / 'comparison.md'}")


if __name__ == "__main__":
    main()
