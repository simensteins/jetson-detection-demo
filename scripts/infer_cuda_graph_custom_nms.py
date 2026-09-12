#!/usr/bin/env python3
"""Iteration 3: fixed-shape, graph-capture-safe NMS.

Builds on scripts/infer_cuda_graph_gpu_resize.py (kept unchanged for
comparison; CUDA graph capture of the model, and GPU-side preprocessing,
are both unchanged here too). That iteration's profiling found the gap
between the model's last kernel and NMS's own work wasn't idle time at
all - it was ~30 small, individually-dispatched kernels from Ultralytics'
non_max_suppression() (~188us of real GPU work spread across ~2.4ms of
wall-clock time), suffering the exact same per-kernel dispatch-overhead
problem CUDA Graph already solved for the model.

Why NMS couldn't just be wrapped in torch.cuda.graph() the same way the
model was: CUDA Graphs require a fixed, deterministic sequence of
operations, but the number of real detections varies frame-to-frame based
on scene content. Ultralytics' non_max_suppression() uses boolean-mask
indexing / torch.nonzero() to filter candidates by confidence, which
changes tensor shape based on data - exactly the anti-pattern for graph
capture (each call forces a GPU->CPU sync to learn how many elements
passed the filter before it can even allocate the output tensor).

Researched before implementing (see chat/session log for full findings):
  - TensorRT's own EfficientNMS plugin (end2end=True export) was the
    other candidate - tested directly and confirmed NOT to work on this
    TensorRT version (10.16.2.10): the ONNX-variant plugin Ultralytics'
    export path depends on was removed in TensorRT 10.16, and end2end=True
    silently built a normal single-output engine with no error at all.
  - CUDA 12+ "conditional graph nodes" solve data-dependent *branching*,
    not variable-size *tensors* - not a fit, and no PyTorch binding exists
    anyway.
  - NVIDIA's own "CUDA Graph Best Practice for PyTorch" guide recommends
    exactly the approach used here: pad to a fixed max size and mask
    invalid entries with shape-preserving ops instead of shape-changing
    ones, so the whole thing stays graph-capturable.

This implementation:
  1. torch.topk(scores, k=MAX_CANDIDATES) - ALWAYS returns exactly
     MAX_CANDIDATES entries regardless of how many real detections exist,
     sorted by score descending. Fixed shape, no data-dependent sizing.
     MAX_CANDIDATES=100 matches COCO's own standard evaluation convention
     (max 100 detections per image), not an arbitrary choice.
  2. "Fast NMS" (Bolya et al., the YOLACT paper) over those K candidates:
     compute the full (K,K) pairwise IoU matrix, then suppress a box if it
     overlaps too much with any *higher-scoring* box, in one fully
     vectorized pass - no sequential/data-dependent loop. This is a known,
     standard approximation of classic greedy NMS (can very rarely
     over-suppress in complex 3+-box overlap chains, since it doesn't
     check whether the higher-scoring box was itself already suppressed)
     - an accepted, well-precedented tradeoff for exactly this class of
     problem.
  3. Class-aware suppression via the standard box-offset trick (offset
     each box spatially by a large multiple of its class index before
     computing IoU, so boxes of different classes never overlap) -
     matches torchvision's own batched_nms approach.
  4. Output stays a fixed (MAX_CANDIDATES, 6) tensor always; suppressed/
     invalid entries have score 0 and are filtered by draw_detections (a
     cheap, fixed-size-K CPU-side check, not a GPU shape-changing op).

Runs eagerly (NOT yet captured into its own CUDA graph) as the first
measured step - correctness and the raw fixed-shape-rewrite win are
checked in isolation before considering graph-capturing this too as a
follow-up, matching the staged approach that worked well for Iteration 2
(GPU reformat measured alone before adding GPU resize on top).

**Correctness is the priority for a from-scratch NMS reimplementation.**
Run with display first and compare detections against
infer_cuda_graph_gpu_resize.py on the same source before trusting any
profiling numbers.

Usage (run from anywhere - the project root is added to sys.path below):
    python3 scripts/infer_cuda_graph_custom_nms.py --config configs/default.yaml \
        --source data/vtest.avi --no-display --max-frames 600
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
    p = argparse.ArgumentParser(description="Iteration 3: fixed-shape graph-capture-safe NMS")
    p.add_argument("--config", default="configs/default.yaml", help="YAML config path")
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


def preprocess_gpu(gpu_raw: torch.Tensor, new_hw: tuple[int, int],
                    pad_lrtb: tuple[int, int, int, int], dst: torch.Tensor) -> None:
    """gpu_raw: (H0, W0, 3) uint8 BGR CUDA tensor holding this frame's raw
    pixels (already transferred). Resizes, letterbox-pads, and normalizes
    entirely on GPU, writing the result into the fixed graph input buffer
    `dst` (shape [1,3,imgsz,imgsz], float32, CUDA)."""
    img = gpu_raw.flip(-1).float() / 255.0  # BGR -> RGB, normalize, still HWC
    img = img.permute(2, 0, 1).unsqueeze(0)  # -> (1, 3, H0, W0)
    img = F.interpolate(img, size=new_hw, mode="bilinear", align_corners=False, antialias=False)
    left, right, top, bottom = pad_lrtb
    img = F.pad(img, (left, right, top, bottom), mode="constant", value=114 / 255)
    dst.copy_(img)


def box_iou_matrix(boxes_xyxy: torch.Tensor) -> torch.Tensor:
    """boxes_xyxy: (K, 4). Returns the full (K, K) pairwise IoU matrix,
    fully vectorized, fixed shape - no loops, no data-dependent sizing."""
    area = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]).clamp(min=0) * \
           (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]).clamp(min=0)  # (K,)
    x1 = torch.max(boxes_xyxy[:, 0].unsqueeze(1), boxes_xyxy[:, 0].unsqueeze(0))  # (K,K)
    y1 = torch.max(boxes_xyxy[:, 1].unsqueeze(1), boxes_xyxy[:, 1].unsqueeze(0))
    x2 = torch.min(boxes_xyxy[:, 2].unsqueeze(1), boxes_xyxy[:, 2].unsqueeze(0))
    y2 = torch.min(boxes_xyxy[:, 3].unsqueeze(1), boxes_xyxy[:, 3].unsqueeze(0))
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)  # (K,K)
    union = area.unsqueeze(1) + area.unsqueeze(0) - inter
    return inter / union.clamp(min=1e-9)


def custom_nms(raw_output: torch.Tensor, conf_thres: float, iou_thres: float) -> torch.Tensor:
    """raw_output: (1, 84, 8400) raw model output (4 box coords + 80 class
    scores per candidate, cxcywh format). Returns a FIXED (MAX_CANDIDATES, 6)
    tensor of [x1, y1, x2, y2, score, cls] - always this shape, regardless
    of how many real detections exist. Entries beyond the real detection
    count have score 0 and must be filtered by the caller."""
    pred = raw_output[0]  # (84, 8400)
    boxes_cxcywh = pred[:4]  # (4, 8400)
    scores_all, classes_all = pred[4:].max(dim=0)  # (8400,), (8400,) - fixed shape reduction

    # Zero out below-threshold scores (shape-preserving - no filtering) so
    # they sort to the bottom and topk naturally excludes them if there
    # are >= MAX_CANDIDATES real detections, or leaves harmless score-0
    # padding entries if there are fewer.
    scores_all = torch.where(scores_all >= conf_thres, scores_all, torch.zeros_like(scores_all))

    topk_scores, topk_idx = torch.topk(scores_all, k=MAX_CANDIDATES)  # fixed (K,), sorted desc
    topk_classes = classes_all[topk_idx]  # (K,)
    topk_cxcywh = boxes_cxcywh[:, topk_idx].T  # (K, 4)

    cx, cy, w, h = topk_cxcywh.unbind(-1)
    boxes_xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)  # (K,4)

    # Class-aware Fast NMS: offset boxes per class so different classes
    # never overlap, then suppress a box if it overlaps too much with any
    # EARLIER (higher-scoring, since topk is sorted descending) box. This
    # is the YOLACT "Fast NMS" approximation - fully vectorized, fixed
    # shape, no data-dependent loop.
    offset = topk_classes.float().unsqueeze(1) * CLASS_OFFSET
    ious = box_iou_matrix(boxes_xyxy + offset)  # (K, K)
    ious_earlier_only = torch.triu(ious, diagonal=1)  # only i < j (i is higher-scored)
    max_iou_with_earlier = ious_earlier_only.max(dim=0).values  # (K,)
    keep = (max_iou_with_earlier <= iou_thres) & (topk_scores > 0)
    final_scores = torch.where(keep, topk_scores, torch.zeros_like(topk_scores))

    return torch.cat([boxes_xyxy, final_scores.unsqueeze(1), topk_classes.float().unsqueeze(1)], dim=1)


def draw_detections(frame: np.ndarray, dets_cpu: np.ndarray,
                     scale: float, pad: tuple[int, int]) -> np.ndarray:
    """dets_cpu: (MAX_CANDIDATES, 6) numpy array, already transferred off
    the GPU by the caller (single sync there) - everything here is plain
    CPU/numpy so it can't reintroduce per-op GPU dispatch overhead."""
    annotated = frame.copy()
    valid = dets_cpu[dets_cpu[:, 4] > 0]  # cheap CPU-side filter, no GPU sync
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

    engine_path = cfg["model"]
    if not str(engine_path).endswith(".engine"):
        raise SystemExit(
            f"This script needs a TensorRT .engine, got {engine_path!r}. "
            "Point configs/default.yaml's 'model' at the exported .engine file."
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
    print(f"  input  {input_name!r} shape={tuple(input_buf.shape)}")
    print(f"  output {output_name!r} shape={tuple(output_buf.shape)}")

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
    preprocess_gpu(gpu_raw, new_hw, pad_lrtb, input_buf)
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
                preprocess_gpu(gpu_raw, new_hw, pad_lrtb, input_buf)
                graph.replay()
                stream.synchronize()
                preds = custom_nms(output_buf, conf_thres=conf, iou_thres=args.iou)
                # Single GPU->CPU sync of the fixed (MAX_CANDIDATES, 6) tensor
                # here, inside "inference" - not one sync per op scattered
                # through draw_detections (see its docstring).
                preds_cpu = preds.cpu().numpy()

            with nvtx_range("draw"):
                annotated = draw_detections(frame, preds_cpu, scale, pad_left_top)

            if display:
                with nvtx_range("display"):
                    cv2.imshow("detections (cuda graph, gpu resize, custom nms)", annotated)
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
