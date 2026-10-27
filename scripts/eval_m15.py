#!/usr/bin/env python3
"""Run every M15 pipeline stage over a COCO-format test split and write its
detections as predictions.coco.json (COCO results format), so each stage can
be scored by the same COCO evaluator.

Every stage reuses the preprocessing and NMS functions of
scripts/infer_m15_trt.py, so this measures the code that was profiled:

    baseline  TF32 engine, CPU preprocessing (cv2), torchvision NMS
              (= the baseline and + CUDA graph stages: the graph replays the
              same kernels, so it cannot change an output)
    gpupre    TF32 engine, GPU preprocessing, torchvision NMS
    nms       TF32 engine, GPU preprocessing, fixed-shape Cluster-NMS (k = 100)
    fp16      FP16 engine, GPU preprocessing (in FP16), fixed-shape NMS

Images can differ in size, so the letterbox is computed per image and the
engine runs eagerly (no CUDA graph).

Defaults are the deployed settings: conf 0.25, NMS IoU from
configs/m15-fast.yaml (0.7), letterbox fill 114. --conf, --iou and
--pad-value change them; --conf 0.001 gives the full precision/recall curve
for AP (the fixed NMS then reports in run_info.json how many images had more
than k candidates).

The test data and the predictions are not ours to publish: the default output
goes to data/eval_m15/ (gitignored). Never commit them.

Usage (on the Jetson, from the repo root):
    python3 scripts/eval_m15.py --annotations <test annotations .json> --images <test image dir>
    python3 scripts/eval_m15.py ... --iou 0.5 --pad-value 0
    python3 scripts/eval_m15.py ... --stages baseline --engine-fp32 <other .engine> --name <run name>
Writes data/eval_m15/<name>/<stage>/predictions.coco.json, plus run_info.json
(settings, detection counts, NMS truncation) and agreement.csv (per stage:
share of the baseline's detections it reproduces, IoU >= 0.5).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import infer_m15_trt as m15  # noqa: E402

STAGES = ["baseline", "gpupre", "nms", "fp16"]
CATEGORY_ID = 1  # the test annotations' single category


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M15 pipeline stages -> predictions.coco.json")
    p.add_argument("--annotations", required=True, help="COCO annotations of the test split")
    p.add_argument("--images", required=True, help="Directory holding the images' file_name paths")
    p.add_argument("--engine-fp32", default="models/m15/m15_fast_fp32-tf32.engine")
    p.add_argument("--engine-fp16", default="models/m15/m15_fast_fp16.engine")
    p.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    p.add_argument("--config", default="configs/m15-fast.yaml")
    p.add_argument("--conf", type=float, default=None, help="Confidence floor (default: config, 0.25)")
    p.add_argument("--iou", type=float, default=None, help="NMS IoU (default: config, 0.7)")
    p.add_argument("--pad-value", type=int, default=m15.LETTERBOX_FILL_U8,
                   help="Letterbox fill, 0-255 (default 114, the Ultralytics fill)")
    p.add_argument("--nms-candidates", type=int, default=m15.NMS_CANDIDATES)
    p.add_argument("--max-images", type=int, default=0, help="0 = all images")
    p.add_argument("--name", default=None,
                   help="Output subfolder (default: conf<conf>_iou<iou>_pad<pad>)")
    p.add_argument("--outdir", default="data/eval_m15")
    return p.parse_args()


class Stage:
    """One pipeline stage: an engine plus the preprocessing and NMS it uses."""

    def __init__(self, name: str, engine_path: str, args, conf: float, iou: float):
        self.name, self.conf, self.iou, self.args = name, conf, iou, args
        self.cpu_pre = name == "baseline"
        self.fixed_nms = name in ("nms", "fp16")
        self.trt = m15.TrtEngine(engine_path)
        self.input = next(iter(self.trt.inputs.values()))
        self.output = next(iter(self.trt.outputs.values()))
        self.gray_w = m15.gray_weights(self.input.dtype, self.input.device)
        self.stream = torch.cuda.Stream()
        self.truncated = self.unconverged = self.max_candidates = 0

    def detect(self, img: np.ndarray) -> np.ndarray:
        """BGR uint8 image -> (M, 5) [x1, y1, x2, y2, score] in image pixels."""
        h0, w0 = img.shape[:2]
        r, new_h, new_w, (pad_l, pad_r, pad_t, pad_b) = m15.letterbox_geometry(h0, w0)
        if self.cpu_pre:
            m15.preprocess_cpu(img, (new_h, new_w), (pad_l, pad_r, pad_t, pad_b), self.input,
                               fill_u8=self.args.pad_value)
        else:
            self.input.fill_(self.args.pad_value / 255.0)
            region = self.input[:, :, pad_t:pad_t + new_h, pad_l:pad_l + new_w]
            m15.preprocess_gpu(torch.from_numpy(img).to(self.input.device), (new_h, new_w), region,
                               self.gray_w)
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.trt.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        if self.fixed_nms:
            pad_xyxy = torch.tensor([pad_l, pad_t, pad_l, pad_t], dtype=torch.float32,
                                    device=self.output.device)
            dets, stats = m15.postprocess_fixed(self.output, self.conf, self.iou, r, pad_xyxy,
                                                (h0, w0), k=self.args.nms_candidates)
            dets, stats = dets.cpu().numpy(), stats.cpu().numpy()
            dets = dets[dets[:, 4] > 0]
            self.max_candidates = max(self.max_candidates, int(stats[0]))
            self.truncated += int(stats[1])
            self.unconverged += int(stats[2])
        else:
            dets = m15.postprocess(self.output, self.conf, self.iou, r, (pad_l, pad_t),
                                   (h0, w0)).cpu().numpy()
        return dets[:, :5]


def to_coco(image_id: int, dets: np.ndarray) -> list[dict]:
    """xyxy -> COCO xywh records; drops boxes left with no area after clipping
    (the evaluator rejects zero width or height)."""
    out = []
    for x1, y1, x2, y2, score in dets.tolist():
        if x2 > x1 and y2 > y1:
            out.append({"image_id": image_id, "category_id": CATEGORY_ID,
                        "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score})
    return out


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda d: (d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1])  # noqa: E731
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-9)


def matched(ref: np.ndarray, other: np.ndarray, thr: float = 0.5) -> int:
    """Greedy one-to-one matches of ref's boxes (highest score first) in other."""
    if len(ref) == 0 or len(other) == 0:
        return 0
    iou = iou_matrix(ref[np.argsort(-ref[:, 4])], other)
    used, n = np.zeros(len(other), bool), 0
    for row in iou:
        row = np.where(used, -1, row)
        j = int(row.argmax())
        if row[j] >= thr:
            used[j], n = True, n + 1
    return n


def main() -> None:
    args = parse_args()
    cfg = m15.load_config(args.config)
    conf = args.conf if args.conf is not None else cfg.get("conf", 0.25)
    iou = args.iou if args.iou is not None else cfg.get("iou", 0.7)
    name = args.name or f"conf{conf:g}_iou{iou:g}_pad{args.pad_value}"
    outdir = Path(args.outdir) / name

    coco = json.loads(Path(args.annotations).read_text())
    images = sorted(coco["images"], key=lambda im: im["id"])
    if args.max_images:
        images = images[:args.max_images]
    print(f"{len(images)} images, stages {args.stages}, conf {conf}, iou {iou}, "
          f"fill {args.pad_value} -> {outdir}")

    all_dets: dict[str, dict[int, np.ndarray]] = {}
    info = {"annotations": str(args.annotations), "images": len(images), "conf": conf,
            "iou": iou, "pad_value": args.pad_value, "stages": {}}
    for stage_name in args.stages:
        engine = args.engine_fp16 if stage_name == "fp16" else args.engine_fp32
        stage = Stage(stage_name, engine, args, conf, iou)
        print(f"\n=== {stage_name} ({engine})")
        records, per_image, t0 = [], {}, time.perf_counter()
        for i, im in enumerate(images):
            img = cv2.imread(str(Path(args.images) / im["file_name"]), cv2.IMREAD_COLOR)
            if img is None:
                raise SystemExit(f"Could not read {im['file_name']}")
            if img.shape[:2] != (im["height"], im["width"]):
                raise SystemExit(f"{im['file_name']}: file is {img.shape[1]}x{img.shape[0]}, "
                                 f"annotations say {im['width']}x{im['height']}")
            dets = stage.detect(img)
            per_image[im["id"]] = dets
            records += to_coco(im["id"], dets)
            if (i + 1) % 250 == 0:
                print(f"  {i + 1}/{len(images)}")
        all_dets[stage_name] = per_image
        (outdir / stage_name).mkdir(parents=True, exist_ok=True)
        (outdir / stage_name / "predictions.coco.json").write_text(json.dumps(records))
        s = {"engine": engine, "detections": len(records), "seconds": round(time.perf_counter() - t0, 1)}
        if stage.fixed_nms:
            s.update(k=args.nms_candidates, max_candidates=stage.max_candidates,
                     images_truncated=stage.truncated, images_unconverged=stage.unconverged)
        info["stages"][stage_name] = s
        print(f"  {s}")
        if stage.fixed_nms and stage.truncated:
            print(f"  WARNING: {stage.truncated} images had >= k={args.nms_candidates} candidates; "
                  "boxes beyond the top k were cut before NMS (raise --nms-candidates to compare exactly)")
        del stage
        torch.cuda.empty_cache()

    # Agreement with the baseline: did a stage change what is detected?
    if "baseline" in all_dets:
        rows = []
        base = all_dets["baseline"]
        n_base = sum(len(d) for d in base.values())
        for stage_name, dets in all_dets.items():
            n = sum(len(d) for d in dets.values())
            kept = sum(matched(base[i], dets[i]) for i in base)
            rows.append({"stage": stage_name, "detections": n,
                         "baseline_dets_reproduced_pct": round(100 * kept / max(n_base, 1), 2),
                         "dets_also_in_baseline_pct": round(100 * kept / max(n, 1), 2)})
        with open(outdir / "agreement.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print("\nAgreement with baseline (IoU >= 0.5):")
        for r in rows:
            print(f"  {r['stage']:9s} {r['detections']:6d} dets  reproduces "
                  f"{r['baseline_dets_reproduced_pct']:6.2f} % of baseline's, "
                  f"{r['dets_also_in_baseline_pct']:6.2f} % of its own are in baseline")
    (outdir / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"\nwrote {outdir}/<stage>/predictions.coco.json")


if __name__ == "__main__":
    main()
