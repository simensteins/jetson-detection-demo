#!/usr/bin/env python3
"""Iteration 1c: CUDA graph + fully GPU-side preprocessing (incl. resize).

Builds on scripts/infer_cuda_graph_gpu_preprocess.py (kept unchanged for
comparison). That iteration moved the reformatting steps (BGR->RGB,
normalize, HWC->CHW) onto the GPU, cutting preprocessing 31.6% - but found
90% of the remaining preprocessing cost (5.07ms of 5.65ms) was still the
cv2.resize/copyMakeBorder letterbox step itself, still running on CPU.

This version moves the resize+pad onto the GPU too, via plain PyTorch
(torch.nn.functional.interpolate + F.pad) - researched against cv2.cuda
(needs a multi-hour OpenCV rebuild from source, not available via apt),
GStreamer's nvvidconv (hardware resize but no letterbox-pad primitive -
that needs the full DeepStream SDK), NVIDIA VPI (available but no
letterbox primitive either, no clear benefit over torch), and NVIDIA DALI
(has a documented letterbox recipe but is designed for batch/multi-stream
throughput, not this single-stream case). Plain torch was the clear choice:
zero new dependencies (already using torch elsewhere in the pipeline), no
missing-capability gap, lowest implementation complexity.

Key assumption this relies on: the source's frame resolution is constant
across the whole run (true for a video file; would break for a source that
changes resolution mid-stream). This lets the resize scale and padding be
computed ONCE at startup from the first frame, rather than every frame -
and lets the CPU staging / GPU raw buffers be pre-allocated at that fixed
native resolution.

Known gotcha from research, applied here: torch's antialias=True resize
(closer to cv2's visual quality) measured ~5x slower in reported
benchmarks - using antialias=False for the real-time path, accepting a
minor quality difference from the cv2-based iterations.

Everything else (graph capture/replay, NMS, NVTX ranges, CLI flags,
correctness-check-first workflow) identical to the prior iterations -
single-variable comparison.

Usage (run from anywhere - the project root is added to sys.path below):
    python3 scripts/infer_cuda_graph_gpu_resize.py --config configs/default.yaml \
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
from ultralytics.utils.nms import non_max_suppression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.profiling import nvtx_range  # noqa: E402
from src.sources import open_source  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Iteration 1c: CUDA graph + fully GPU-side preprocessing")
    p.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    p.add_argument("--source", default=None,
                   help="Override source: file path, rtsp:// URL, or webcam index")
    p.add_argument("--no-display", action="store_true", help="Run headless (no window)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Override max_frames from config (0 = run to end of source)")
    p.add_argument("--warmup-frames", type=int, default=None,
                   help="Frames to exclude from the FPS timer (in addition to the "
                        "engine/graph warm-up this script always does before the loop)")
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


def draw_detections(frame: np.ndarray, dets: torch.Tensor | None,
                     scale: float, pad: tuple[int, int]) -> np.ndarray:
    annotated = frame.copy()
    if dets is None or len(dets) == 0:
        return annotated
    boxes = dets[:, :4].clone()
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes /= scale
    for (x1, y1, x2, y2), conf, cls in zip(boxes.tolist(), dets[:, 4].tolist(), dets[:, 5].tolist()):
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
                preds = non_max_suppression(output_buf, conf_thres=conf)[0]

            with nvtx_range("draw"):
                annotated = draw_detections(frame, preds, scale, pad_left_top)

            if display:
                with nvtx_range("display"):
                    cv2.imshow("detections (cuda graph, gpu resize)", annotated)
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
