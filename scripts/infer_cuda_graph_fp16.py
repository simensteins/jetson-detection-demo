#!/usr/bin/env python3
"""Iteration 4: FP16 precision, applied through the whole pipeline - not just the engine.

Builds on scripts/infer_cuda_graph_custom_nms.py (kept unchanged for comparison).
That iteration left model execution as the single largest remaining cost in the
frame (~55-58% of it, per its Next Steps). See the "FP16 Research - Accuracy vs.
Speed Tradeoff" Notion page for the research behind this iteration: FP16 typically
costs <0.5 mAP (negligible) and, unlike INT8, needs no calibration or retraining -
the standard "free lunch" precision drop on Tensor-Core hardware like this Jetson.

Requires a SEPARATELY exported FP16 .engine (`model.export(format="engine",
half=True, imgsz=640)`) - this script does not export it. Point
configs/default.yaml's `model` at that file, or pass --engine to override per-run
(handy for swapping between the FP32 and FP16 engines without editing the config).

**Why this is a new script rather than just pointing the FP32 pipeline at an FP16
engine**: an earlier, pre-CUDA-Graph attempt at FP16 in this project only changed
the *engine* to FP16 and left everything around it - preprocessing math, NMS math -
written for FP32, unchanged. Every touchpoint between an FP16 engine and FP32
surrounding code needs an implicit cast: TensorRT inserts its own "Reformatting
CopyNode" kernels when a buffer's actual dtype doesn't match what the engine
declares, `.copy_()`-ing an FP32 preprocessed frame into an FP16 input buffer
silently upcasts/downcasts element-by-element every frame, and NMS code written
assuming FP32 forces another cast on the output on the way out. That's several
extra cast kernels every single frame - working directly against everything
Iterations 1-3 did to eliminate exactly this kind of per-kernel overhead.

This version keeps data in FP16 from the moment it lands on the GPU, through
preprocessing, the model, and the start of NMS - zero casts until one deliberate,
necessary exception (see custom_nms below). Ultralytics' AutoBackend already
allocates each binding's tensor in whatever dtype the engine itself declares (see
scripts/inspect_autobackend.py) - so as long as preprocessing computes directly in
FP16 instead of computing in FP32 and casting once at the end, the graph's input
buffer is already the right dtype and no cast is needed at the model boundary at
all.

The one deliberate exception: NMS's area/IoU math. FP16's max representable value
is 65504 - a box spanning most of a 640x640 frame has an area near
640*640 = 409,600, which overflows FP16 and would silently produce inf/nan in the
IoU/union computation. Rather than risk that, the selected MAX_CANDIDATES=100 box
tensor (tiny) is cast to FP32 immediately after `torch.topk` picks it, and the
suppression math runs in FP32 from there - one explicit, understood cast on 100
numbers, not a whole-frame casting problem repeated every frame.

**Correctness check**: run with display first, and separately compare mAP against
the FP32 engine with `model.val()` (see the FP16 Research Notion page) before
trusting any speed number - FP16 accuracy loss is expected to be small but not
zero, and a misconfigured export (half=True silently not taking effect, mirroring
the end2end=True false-positive found in Iteration 3) needs to be ruled out, not
assumed away. This script verifies both I/O buffers are actually torch.float16 at
startup and refuses to run otherwise, for exactly that reason.

Usage (run from anywhere - the project root is added to sys.path below):
    python3 scripts/infer_cuda_graph_fp16.py --config configs/default.yaml \
        --engine yolov8n_fp16.engine --source data/vtest.avi --no-display --max-frames 600
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.profiling import nvtx_range  # noqa: E402
from src.sources import open_source  # noqa: E402

MAX_CANDIDATES = 100  # matches COCO's standard max-100-detections-per-image convention
CLASS_OFFSET = 7680.0  # larger than any plausible box coordinate at imgsz=640


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Iteration 4: FP16 precision through the whole pipeline")
    p.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    p.add_argument("--engine", default=None,
                   help="Override the FP16 .engine path from config (handy for comparing "
                        "against the FP32 engine without editing the config)")
    p.add_argument("--source", default=None,
                   help="Override source: file path, rtsp:// URL, or webcam index")
    p.add_argument("--no-display", action="store_true", help="Run headless (no window)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Override max_frames from config (0 = run to end of source)")
    p.add_argument("--warmup-frames", type=int, default=None,
                   help="Frames to exclude from the FPS timer (in addition to the "
                        "engine/graph warm-up this script always does before the loop)")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def compute_letterbox_params(h0: int, w0: int, imgsz: int):
    """Fixed (source resolution doesn't change) resize scale, target
    (new_h, new_w), and (left, right, top, bottom) padding to letterbox
    h0 x w0 frames to imgsz x imgsz, matching standard YOLO preprocessing."""
    r = min(imgsz / h0, imgsz / w0)
    new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
    dw, dh = (imgsz - new_w) / 2, (imgsz - new_h) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return r, (new_h, new_w), (left, right, top, bottom)


def preprocess_gpu_fp16(gpu_raw: torch.Tensor, new_hw: tuple[int, int],
                         pad_lrtb: tuple[int, int, int, int], dst: torch.Tensor) -> None:
    """Same transform as the FP32 preprocessing (resize, letterbox, BGR->RGB,
    normalize) but computed natively in FP16 throughout - no FP32 intermediate
    tensor, no cast at the end. `dst` (the graph's fixed input buffer) is already
    FP16 because the engine itself is FP16, so this final copy_() is a same-dtype
    copy, not a conversion."""
    img = gpu_raw.flip(-1).half() / 255.0  # BGR -> RGB, cast to FP16, normalize - FP16 from here on
    img = img.permute(2, 0, 1).unsqueeze(0)  # -> (1, 3, H0, W0), still FP16
    img = F.interpolate(img, size=new_hw, mode="bilinear", align_corners=False, antialias=False)
    left, right, top, bottom = pad_lrtb
    img = F.pad(img, (left, right, top, bottom), mode="constant", value=114 / 255)
    dst.copy_(img)  # same dtype (FP16) on both sides - no conversion happening here


def box_iou_matrix(boxes_xyxy: torch.Tensor) -> torch.Tensor:
    """Unchanged from Iteration 3 - runs in FP32 (see custom_nms for why)."""
    area = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]).clamp(min=0) * \
           (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]).clamp(min=0)
    x1 = torch.max(boxes_xyxy[:, 0].unsqueeze(1), boxes_xyxy[:, 0].unsqueeze(0))
    y1 = torch.max(boxes_xyxy[:, 1].unsqueeze(1), boxes_xyxy[:, 1].unsqueeze(0))
    x2 = torch.min(boxes_xyxy[:, 2].unsqueeze(1), boxes_xyxy[:, 2].unsqueeze(0))
    y2 = torch.min(boxes_xyxy[:, 3].unsqueeze(1), boxes_xyxy[:, 3].unsqueeze(0))
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    union = area.unsqueeze(1) + area.unsqueeze(0) - inter
    return inter / union.clamp(min=1e-9)


def custom_nms(raw_output: torch.Tensor, conf_thres: float, iou_thres: float) -> torch.Tensor:
    """raw_output is FP16 (the model's actual output dtype - no cast happened to
    get it there). Candidate selection (max/where/topk) runs directly in FP16:
    every value involved is either a score (0-1) or a raw box coordinate (0-640),
    both safely inside FP16's range.

    The MAX_CANDIDATES=100 selected boxes are then cast to FP32 before any
    area/IoU math - see the module docstring for why (FP16 overflows on box
    areas up to ~409,600 at this resolution). This is the ONE deliberate cast in
    this script, on a 100-row tensor once a frame, not a whole-frame cast
    repeated at every preprocessing step."""
    pred = raw_output[0]  # (84, 8400), FP16
    boxes_cxcywh = pred[:4]  # (4, 8400), FP16
    scores_all, classes_all = pred[4:].max(dim=0)  # FP16 reduction - safe, values are 0-1

    scores_all = torch.where(scores_all >= conf_thres, scores_all, torch.zeros_like(scores_all))

    topk_scores, topk_idx = torch.topk(scores_all, k=MAX_CANDIDATES)  # FP16, fixed (K,), sorted desc
    topk_classes = classes_all[topk_idx]
    topk_cxcywh = boxes_cxcywh[:, topk_idx].T.float()  # <- the one deliberate FP16 -> FP32 cast
    topk_scores = topk_scores.float()

    cx, cy, w, h = topk_cxcywh.unbind(-1)
    boxes_xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)  # FP32

    # Class-aware Fast NMS (see Iteration 3): offset boxes per class, full IoU
    # matrix, suppress a box if it overlaps too much with any earlier
    # (higher-scoring, since topk is sorted descending) box.
    offset = topk_classes.float().unsqueeze(1) * CLASS_OFFSET
    ious = box_iou_matrix(boxes_xyxy + offset)
    ious_earlier_only = torch.triu(ious, diagonal=1)
    max_iou_with_earlier = ious_earlier_only.max(dim=0).values
    keep = (max_iou_with_earlier <= iou_thres) & (topk_scores > 0)
    final_scores = torch.where(keep, topk_scores, torch.zeros_like(topk_scores))

    return torch.cat([boxes_xyxy, final_scores.unsqueeze(1), topk_classes.float().unsqueeze(1)], dim=1)


def draw_detections(frame: np.ndarray, dets_cpu: np.ndarray,
                     scale: float, pad: tuple[int, int]) -> np.ndarray:
    """dets_cpu: (MAX_CANDIDATES, 6) numpy array, already transferred off the GPU
    by the caller (single sync there) - plain CPU/numpy, unchanged from Iteration 3."""
    annotated = frame.copy()
    valid = dets_cpu[dets_cpu[:, 4] > 0]
    if len(valid) == 0:
        return annotated
    boxes = valid[:, :4].copy()
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes /= scale
    for (x1, y1, x2, y2), conf, cls in zip(boxes.tolist(), valid[:, 4].tolist(), valid[:, 5].tolist()):
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(annotated, p1, p2, (56, 56, 255), 2)
        cv2.putText(annotated, f"{int(cls)} {conf:.2f}", (p1[0], max(p1[1] - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (56, 56, 255), 1)
    return annotated


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    source = args.source if args.source is not None else cfg["source"]
    display = cfg.get("display", True) and not args.no_display
    conf = cfg.get("conf", 0.25)
    imgsz = cfg.get("imgsz", 640)
    max_frames = args.max_frames if args.max_frames is not None else cfg.get("max_frames", 0)
    warmup_frames = (args.warmup_frames if args.warmup_frames is not None
                      else cfg.get("warmup_frames", 0))

    engine_path = args.engine if args.engine is not None else cfg["model"]
    if not str(engine_path).endswith(".engine"):
        raise SystemExit(
            f"This script needs a TensorRT .engine, got {engine_path!r}. "
            "Point configs/default.yaml's 'model' (or --engine) at an FP16-exported "
            "engine file (model.export(format='engine', half=True, imgsz=640))."
        )

    print(f"Loading {engine_path} via Ultralytics (known-working path)...")
    yolo = YOLO(engine_path)
    # YOLO() is lazy - force AutoBackend initialization with a dummy predict.
    yolo.predict(np.zeros((imgsz, imgsz, 3), dtype=np.uint8), verbose=False)
    backend = yolo.predictor.model  # ultralytics.nn.autobackend.AutoBackend

    context = backend.context  # already-initialized IExecutionContext
    bindings = backend.bindings  # OrderedDict[name] -> Binding(name, dtype, shape, data)
    input_name = "images"
    output_name = backend.output_names[0]
    input_buf = bindings[input_name].data  # fixed-address CUDA tensor
    output_buf = bindings[output_name].data  # fixed-address CUDA tensor
    print(f"  input  {input_name!r} shape={tuple(input_buf.shape)} dtype={input_buf.dtype}")
    print(f"  output {output_name!r} shape={tuple(output_buf.shape)} dtype={output_buf.dtype}")

    # Verify, don't trust the export flag - this is exactly the kind of silent
    # failure Iteration 3 found with end2end=True (export succeeded, log looked
    # fine, but the flag hadn't actually done what it claimed).
    if input_buf.dtype != torch.float16 or output_buf.dtype != torch.float16:
        raise SystemExit(
            f"Expected an FP16 engine (both I/O tensors torch.float16), but got "
            f"input={input_buf.dtype}, output={output_buf.dtype}. This usually means "
            f"the .engine wasn't actually exported with half=True - re-export with "
            f"model.export(format='engine', half=True, imgsz={imgsz}) and point "
            f"--engine/config at the new file."
        )
    print("  confirmed: engine I/O is FP16 end-to-end")

    cap = open_source(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {source!r}")

    ok, frame0 = cap.read()
    if not ok:
        raise SystemExit("Could not read a frame to warm up the engine.")
    h0, w0 = frame0.shape[:2]
    scale, new_hw, pad_lrtb = compute_letterbox_params(h0, w0, imgsz)
    pad_left_top = (pad_lrtb[0], pad_lrtb[2])  # (left, top), for draw_detections
    print(f"  source frame: {w0}x{h0} -> letterbox scale={scale:.4f} "
          f"new_hw={new_hw} pad_lrtb={pad_lrtb}")

    cpu_staging = torch.empty((h0, w0, 3), dtype=torch.uint8, pin_memory=True)
    gpu_raw = torch.empty((h0, w0, 3), dtype=torch.uint8, device="cuda")

    stream = torch.cuda.Stream()
    context.set_tensor_address(input_name, input_buf.data_ptr())
    context.set_tensor_address(output_name, output_buf.data_ptr())

    # --- warm-up: at least one uncaptured execute is required by TensorRT
    # before graph capture (flushes any deferred setup work) ---
    cpu_staging.copy_(torch.from_numpy(frame0))
    gpu_raw.copy_(cpu_staging, non_blocking=True)
    preprocess_gpu_fp16(gpu_raw, new_hw, pad_lrtb, input_buf)
    for _ in range(3):
        with torch.cuda.stream(stream):
            context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

    # --- capture the forward pass as a CUDA graph, once ---
    print("Capturing CUDA graph...")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        context.execute_async_v3(stream.cuda_stream)
    print("Graph captured. Starting inference loop.")

    frames = 0
    timed_frames = 0
    t0 = time.perf_counter()
    try:
        while True:
            with nvtx_range("decode"):
                ok, frame = cap.read()
            if not ok:
                break

            with nvtx_range("inference"):
                cpu_staging.copy_(torch.from_numpy(frame))
                gpu_raw.copy_(cpu_staging, non_blocking=True)
                preprocess_gpu_fp16(gpu_raw, new_hw, pad_lrtb, input_buf)
                graph.replay()
                stream.synchronize()
                preds = custom_nms(output_buf, conf_thres=conf, iou_thres=args.iou)
                # Single GPU->CPU sync of the fixed (MAX_CANDIDATES, 6) tensor
                # here, inside "inference" - see Iteration 3's draw_detections fix.
                preds_cpu = preds.cpu().numpy()

            with nvtx_range("draw"):
                annotated = draw_detections(frame, preds_cpu, scale, pad_left_top)

            if display:
                with nvtx_range("display"):
                    cv2.imshow("detections (cuda graph, fp16 pipeline)", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            frames += 1
            if frames <= warmup_frames:
                if frames == warmup_frames:
                    t0 = time.perf_counter()
            else:
                timed_frames += 1
            if max_frames and frames >= max_frames:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    dt = time.perf_counter() - t0
    if timed_frames:
        note = f" (after {warmup_frames}-frame warm-up)" if warmup_frames else ""
        print(f"processed {timed_frames} frames in {dt:.1f}s{note}  ->  {timed_frames / dt:.1f} FPS")
    elif frames:
        print(f"processed {frames} frames, all within the {warmup_frames}-frame warm-up  ->  no timed FPS")


if __name__ == "__main__":
    main()
