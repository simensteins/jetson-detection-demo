#!/usr/bin/env python3
"""Detection accuracy of the YOLOv8n pipeline stages on COCO val2017.

The speed work changed more than the engine's precision, so each change gets
its own pipeline here. Every pipeline reuses the preprocessing and NMS
functions of the iteration script it stands for, so this measures the code
that was profiled:

    baseline       src/detect.py: Ultralytics predict() - CPU letterbox,
                   Ultralytics NMS (iou 0.7, max 300 detections), FP32 engine
    gpupre         Iteration 2: GPU preprocessing + Ultralytics
                   non_max_suppression() as that script calls it (iou 0.45),
                   FP32 engine
    customnms      Iteration 3: GPU preprocessing + the fixed-shape Fast NMS
                   (top 100 candidates, iou 0.45), FP32 engine
    fp16           Iteration 4 (the final pipeline): FP16 engine with FP16
                   preprocessing + Fast NMS
    fp16_ultranms  Not a stage - the FP16 engine with Iteration 2's Ultralytics
                   NMS, so FP16's own effect is gpupre vs. fp16_ultranms

(Iteration 1, the CUDA graph, replays the same kernels as the baseline and
has no pipeline of its own.)

The CUDA graph is not used here: it replays the same kernels, so it does not
change any output, and evaluation does not need its speed.

Each pipeline runs over every image once with conf 0.001 (the standard for
mAP), so one run gives:
  - COCO mAP@[.5:.95], mAP@.5, mAP small/medium/large (pycocotools)
  - precision / recall / F1 at the deployed threshold (conf 0.25, IoU 0.5,
    same class) - the detections above 0.25 are exactly what a conf-0.25 run
    returns, because lower-scored boxes never suppress higher-scored ones and
    are dropped from the top of the list last
  - agreement with the baseline: of the baseline's detections above 0.25,
    the share this pipeline also finds (same class, IoU >= 0.5), and vice versa

Accuracy does not depend on clocks or power mode, so this can run with the
desktop up. Needs: pip install pycocotools; COCO val2017 in data/coco
(bash scripts/download_coco_val.sh).

Usage (on the Jetson, from the repo root):
    python3 scripts/eval_yolo_coco.py --max-images 200            # quick check
    python3 scripts/eval_yolo_coco.py                               # all 5000 images, all pipelines
    python3 scripts/eval_yolo_coco.py --pipelines baseline fp16
Writes reports/eval/yolo_coco_summary.csv (+ .md) and the raw detections per
pipeline (reports/eval/yolo_coco_<pipeline>_dets.json).
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
sys.path.insert(0, str(ROOT))
import infer_cuda_graph_custom_nms as it3  # noqa: E402
import infer_cuda_graph_fp16 as it4  # noqa: E402
import infer_cuda_graph_gpu_resize as it2  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.data.converter import coco80_to_coco91_class  # noqa: E402

PIPELINES = ["baseline", "gpupre", "customnms", "fp16", "fp16_ultranms"]
IMGSZ = 640
EVAL_CONF = 0.001
DEPLOY_CONF = 0.25
NMS_IOU_ITER = 0.45  # what Iterations 2-4 use (non_max_suppression default / --iou default)
NMS_IOU_PREDICT = 0.7  # Ultralytics predict() default, used by src/detect.py


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="COCO val2017 accuracy of the YOLOv8n pipeline stages")
    p.add_argument("--images", default="data/coco/val2017")
    p.add_argument("--annotations", default="data/coco/annotations/instances_val2017.json")
    p.add_argument("--engine-fp32", default="yolov8n.engine")
    p.add_argument("--engine-fp16", default="yolov8n_fp16_io.engine")
    p.add_argument("--pipelines", nargs="+", default=PIPELINES, choices=PIPELINES)
    p.add_argument("--max-images", type=int, default=0, help="0 = all images")
    p.add_argument("--outdir", default="reports/eval")
    return p.parse_args()


class EngineRunner:
    """Runs an engine the way Iterations 2-4 do: Ultralytics' AutoBackend loads
    it (a raw deserialize fails on this TensorRT version), its execution
    context runs directly on the fixed-address I/O buffers."""

    def __init__(self, path: str):
        self.yolo = YOLO(path)
        self.yolo.predict(np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8), verbose=False)
        backend = self.yolo.predictor.model
        self.context = backend.context
        self.input = backend.bindings["images"].data
        self.output = backend.bindings[backend.output_names[0]].data
        self.context.set_tensor_address("images", self.input.data_ptr())
        self.context.set_tensor_address(backend.output_names[0], self.output.data_ptr())
        self.stream = torch.cuda.Stream()

    def run(self) -> torch.Tensor:
        torch.cuda.current_stream().synchronize()  # preprocessing wrote self.input
        with torch.cuda.stream(self.stream):
            self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return self.output


def ultralytics_nms(raw: torch.Tensor) -> torch.Tensor:
    """Iteration 2's call: non_max_suppression(output, conf_thres=conf), here
    with the eval conf; the time limit is lifted so conf 0.001 is never cut
    short (it only exists to bound worst-case latency)."""
    try:
        out = it2.non_max_suppression(raw.float(), conf_thres=EVAL_CONF, max_time_img=10.0)
    except TypeError:
        out = it2.non_max_suppression(raw.float(), conf_thres=EVAL_CONF)
    return out[0]


def detect(pipeline: str, img: np.ndarray, runners: dict) -> np.ndarray:
    """-> (N, 6) [x1, y1, x2, y2, score, cls80] in original-image pixels."""
    h0, w0 = img.shape[:2]
    if pipeline == "baseline":
        r = runners["predict"].predict(img, conf=EVAL_CONF, iou=NMS_IOU_PREDICT, imgsz=IMGSZ,
                                       max_det=300, verbose=False)[0].boxes
        return torch.cat([r.xyxy, r.conf[:, None], r.cls[:, None]], 1).cpu().numpy()

    scale, new_hw, pad = it2.compute_letterbox_params(h0, w0, IMGSZ)
    gpu_raw = torch.from_numpy(img).cuda()
    if pipeline in ("fp16", "fp16_ultranms"):
        runner = runners["fp16"]
        it4.preprocess_gpu_fp16(gpu_raw, new_hw, pad, runner.input)
    else:
        runner = runners["fp32"]
        it2.preprocess_gpu(gpu_raw, new_hw, pad, runner.input)
    raw = runner.run()

    if pipeline in ("gpupre", "fp16_ultranms"):
        dets = ultralytics_nms(raw)
    elif pipeline == "customnms":
        dets = it3.custom_nms(raw, conf_thres=EVAL_CONF, iou_thres=NMS_IOU_ITER)
    else:  # fp16
        dets = it4.custom_nms(raw, conf_thres=EVAL_CONF, iou_thres=NMS_IOU_ITER)
    dets = dets.float().cpu().numpy()
    dets = dets[dets[:, 4] > 0]  # Fast NMS pads with score 0
    left, top = pad[0], pad[2]
    dets[:, [0, 2]] = ((dets[:, [0, 2]] - left) / scale).clip(0, w0)
    dets[:, [1, 3]] = ((dets[:, [1, 3]] - top) / scale).clip(0, h0)
    return dets


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    area = lambda r: (r[:, 2] - r[:, 0]).clip(0) * (r[:, 3] - r[:, 1]).clip(0)
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-9)


def greedy_matches(pred: np.ndarray, ref: np.ndarray, iou_thr: float = 0.5) -> int:
    """Number of pred boxes matched one-to-one (highest score first) to a ref
    box of the same class with IoU >= iou_thr. pred/ref: (N, >=6) with cls in col 5."""
    if len(pred) == 0 or len(ref) == 0:
        return 0
    pred = pred[np.argsort(-pred[:, 4])]
    ious = iou_matrix(pred[:, :4], ref[:, :4])
    ious[pred[:, 5][:, None] != ref[:, 5][None, :]] = 0
    used = np.zeros(len(ref), bool)
    n = 0
    for i in range(len(pred)):
        cand = np.where(~used & (ious[i] >= iou_thr))[0]
        if len(cand):
            used[cand[np.argmax(ious[i, cand])]] = True
            n += 1
    return n


def main() -> None:
    args = parse_args()
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        raise SystemExit("pip install pycocotools")
    if not Path(args.annotations).exists():
        raise SystemExit(f"{args.annotations} not found - run: bash scripts/download_coco_val.sh")
    gt = COCO(args.annotations)
    img_ids = sorted(gt.getImgIds())
    if args.max_images:
        img_ids = img_ids[:args.max_images]
    cat91 = coco80_to_coco91_class()
    cat80 = {c: i for i, c in enumerate(cat91)}

    # Ground truth per image (non-crowd), in the 80-class index, for P/R at 0.25.
    gt_boxes = {}
    for iid in img_ids:
        anns = [a for a in gt.loadAnns(gt.getAnnIds(imgIds=iid, iscrowd=False))]
        gt_boxes[iid] = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2],
                                   a["bbox"][1] + a["bbox"][3], 1.0, cat80[a["category_id"]]]
                                  for a in anns], dtype=np.float32).reshape(-1, 6)

    runners = {}
    if "baseline" in args.pipelines:
        runners["predict"] = YOLO(args.engine_fp32)
    if any(p in args.pipelines for p in ("gpupre", "customnms")):
        runners["fp32"] = EngineRunner(args.engine_fp32)
    if any(p in args.pipelines for p in ("fp16", "fp16_ultranms")):
        runners["fp16"] = EngineRunner(args.engine_fp16)
        if runners["fp16"].input.dtype != torch.float16 or runners["fp16"].output.dtype != torch.float16:
            raise SystemExit(f"{args.engine_fp16} does not have FP16 I/O - see scripts/export_yolo_fp16.py")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    all_dets: dict[str, dict[int, np.ndarray]] = {}
    rows = []
    for pipeline in args.pipelines:
        t0 = time.perf_counter()
        dets_by_img, coco_dets = {}, []
        for k, iid in enumerate(img_ids):
            info = gt.loadImgs(iid)[0]
            img = cv2.imread(str(Path(args.images) / info["file_name"]))  # BGR, as from cv2.VideoCapture
            d = detect(pipeline, img, runners)
            dets_by_img[iid] = d
            for x1, y1, x2, y2, s, c in d.tolist():
                coco_dets.append({"image_id": iid, "category_id": cat91[int(c)],
                                  "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                                  "score": round(s, 5)})
            if (k + 1) % 500 == 0:
                print(f"  {pipeline}: {k + 1}/{len(img_ids)}", flush=True)
        all_dets[pipeline] = dets_by_img
        (outdir / f"yolo_coco_{pipeline}_dets.json").write_text(json.dumps(coco_dets))

        ev = COCOeval(gt, gt.loadRes(coco_dets) if coco_dets else COCO(), "bbox")
        ev.params.imgIds = img_ids
        ev.evaluate(); ev.accumulate(); ev.summarize()
        s = ev.stats

        tp = fp = fn = 0
        for iid in img_ids:
            p = dets_by_img[iid]; p = p[p[:, 4] >= DEPLOY_CONF]
            m = greedy_matches(p, gt_boxes[iid])
            tp += m; fp += len(p) - m; fn += len(gt_boxes[iid]) - m
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        row = dict(pipeline=pipeline, images=len(img_ids), mAP50_95=100 * s[0], mAP50=100 * s[1],
                   mAP_small=100 * s[3], mAP_medium=100 * s[4], mAP_large=100 * s[5],
                   precision_025=100 * prec, recall_025=100 * rec,
                   f1_025=100 * 2 * prec * rec / max(prec + rec, 1e-9),
                   dets_per_img_025=sum(int((all_dets[pipeline][i][:, 4] >= DEPLOY_CONF).sum())
                                        for i in img_ids) / len(img_ids),
                   seconds=time.perf_counter() - t0)
        if "baseline" in all_dets and pipeline != "baseline":
            kept = found = n_base = n_this = 0
            for iid in img_ids:
                b = all_dets["baseline"][iid]; b = b[b[:, 4] >= DEPLOY_CONF]
                p = dets_by_img[iid]; p = p[p[:, 4] >= DEPLOY_CONF]
                kept += greedy_matches(b, p); found += greedy_matches(p, b)
                n_base += len(b); n_this += len(p)
            row["baseline_dets_kept_025"] = 100 * kept / max(n_base, 1)
            row["dets_also_in_baseline_025"] = 100 * found / max(n_this, 1)
        rows.append(row)
        print(f"  {pipeline}: mAP50-95 {row['mAP50_95']:.2f}  mAP50 {row['mAP50']:.2f}  "
              f"P/R@0.25 {row['precision_025']:.1f}/{row['recall_025']:.1f}  ({row['seconds']:.0f} s)", flush=True)

    keys = ["pipeline", "images", "mAP50_95", "mAP50", "mAP_small", "mAP_medium", "mAP_large",
            "precision_025", "recall_025", "f1_025", "dets_per_img_025",
            "baseline_dets_kept_025", "dets_also_in_baseline_025", "seconds"]
    with open(outdir / "yolo_coco_summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(keys)
        for r in rows:
            w.writerow([round(r[k], 3) if isinstance(r.get(k), float) else r.get(k, "") for k in keys])
    md = ["| Pipeline | mAP50-95 | mAP50 | mAP small | P@0.25 | R@0.25 | F1@0.25 | baseline dets kept |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        kept = f"{r['baseline_dets_kept_025']:.1f} %" if "baseline_dets_kept_025" in r else "–"
        md.append(f"| {r['pipeline']} | {r['mAP50_95']:.2f} | {r['mAP50']:.2f} | {r['mAP_small']:.2f} | "
                  f"{r['precision_025']:.1f} | {r['recall_025']:.1f} | {r['f1_025']:.1f} | {kept} |")
    (outdir / "yolo_coco_summary.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md))
    print(f"\nwrote {outdir}/yolo_coco_summary.csv and .md")


if __name__ == "__main__":
    main()
