#!/usr/bin/env python3
"""Iteration 1: CUDA-graph-accelerated inference.

Same pipeline as src/detect.py (decode -> inference -> draw -> display, same
NVTX ranges, same CLI flags) so it can be profiled and compared against the
baseline with the exact same methodology (nsys + scripts/analyze_nsys.py).

The ONLY thing this changes versus the baseline: how the TensorRT engine
gets invoked. The baseline goes through Ultralytics' model.predict(), which
issues ~223 individual CUDA API calls per frame (measured: ~8.1-8.4ms of
CPU-side dispatch overhead, ~30% GPU idle time between kernels - see the
"Baseline Performance" Notion page). This script captures that same
223-kernel forward pass into a CUDA graph once, then replays it with a
single call per frame, to test whether that collapses the dispatch
overhead and idle gaps.

Everything else - preprocessing (letterbox + normalize) and postprocessing
(NMS) - deliberately reuses Ultralytics' own ops.non_max_suppression rather
than a hand-rolled reimplementation, to keep this a single-variable
comparison against the baseline rather than a from-scratch reimplementation
with its own correctness risk.

Requirements / assumptions - verify these before trusting results:
  - configs/default.yaml's `model` must point at a .engine file (not .pt).
  - The engine must have a STATIC input shape (fixed batch=1, fixed imgsz) -
    true for every export used in this project so far. Dynamic-shape
    engines are not supported by this script.
  - Assumes the engine emits raw (pre-NMS) predictions, i.e. it was NOT
    exported with end2end=True bundling NMS into the engine. This matches
    the baseline engine used for all profiling on the Notion baseline page
    (no NMS-named kernels appear in its ncu/nsys kernel traces). If you
    later profile an end2end=True engine with this script, postprocessing
    here is wrong and needs to be skipped/adjusted.
  - Recommend a first run WITH display (drop --no-display) to visually
    confirm detections look correct before trusting any profiling numbers
    - this isolates "is the raw TensorRT invocation correct" from "does
    the CUDA graph layer change anything," verified independently.

Usage (run from anywhere - the project root is added to sys.path below):
    python3 scripts/infer_cuda_graph.py --config configs/default.yaml \
        --source data/vtest.avi --no-display --max-frames 600
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import torch
import yaml
from ultralytics.utils.nms import non_max_suppression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.profiling import nvtx_range  # noqa: E402
from src.sources import open_source  # noqa: E402

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Iteration 1: CUDA-graph inference")
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


def load_engine(engine_path: str) -> trt.ICudaEngine:
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise SystemExit(f"Failed to load TensorRT engine: {engine_path}")
    return engine


def find_io_tensors(engine: trt.ICudaEngine) -> tuple[str, str]:
    """Return (input_name, output_name), found by IO mode rather than
    assuming index order."""
    input_name = output_name = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            input_name = name
        elif mode == trt.TensorIOMode.OUTPUT:
            output_name = name
    if input_name is None or output_name is None:
        raise SystemExit(
            f"Expected exactly one input and one output tensor, "
            f"found input={input_name!r} output={output_name!r}. "
            "This script doesn't support multi-input/output engines."
        )
    return input_name, output_name


def letterbox(frame: np.ndarray, new_shape: int, color=(114, 114, 114)):
    """Resize + pad to a square, preserving aspect ratio (standard YOLO
    preprocessing). Returns the padded image, the scale factor, and the
    (left, top) padding, needed to map detections back to original coords."""
    h0, w0 = frame.shape[:2]
    r = min(new_shape / h0, new_shape / w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    dw, dh = dw / 2, dh / 2
    resized = cv2.resize(frame, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=color)
    return padded, r, (left, top)


def preprocess(frame: np.ndarray, imgsz: int, dst: torch.Tensor) -> tuple[float, tuple[int, int]]:
    """Letterbox + normalize `frame` directly into the fixed GPU input
    buffer `dst` (shape [1,3,imgsz,imgsz], float32, CUDA)."""
    padded, scale, pad = letterbox(frame, imgsz)
    img = padded[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, HWC -> CHW
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    dst.copy_(torch.from_numpy(img).unsqueeze(0))
    return scale, pad


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

    print(f"Loading engine: {engine_path}")
    engine = load_engine(engine_path)
    context = engine.create_execution_context()
    input_name, output_name = find_io_tensors(engine)
    input_shape = tuple(engine.get_tensor_shape(input_name))
    output_shape = tuple(engine.get_tensor_shape(output_name))
    print(f"  input  {input_name!r} shape={input_shape}")
    print(f"  output {output_name!r} shape={output_shape}")
    if any(d < 0 for d in input_shape):
        raise SystemExit(
            f"Engine has a dynamic input shape {input_shape} - this script "
            "requires a static-shape engine (fixed batch size and imgsz)."
        )

    cap = open_source(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {source!r}")

    stream = torch.cuda.Stream()
    input_buf = torch.zeros(input_shape, dtype=torch.float32, device="cuda")
    output_buf = torch.zeros(output_shape, dtype=torch.float32, device="cuda")
    context.set_tensor_address(input_name, input_buf.data_ptr())
    context.set_tensor_address(output_name, output_buf.data_ptr())

    # --- engine/graph warm-up: at least one uncaptured execute is required
    # by TensorRT before graph capture (flushes any deferred setup work) ---
    ok, frame0 = cap.read()
    if not ok:
        raise SystemExit("Could not read a frame to warm up the engine.")
    preprocess(frame0, imgsz, input_buf)
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
                scale, pad = preprocess(frame, imgsz, input_buf)
                graph.replay()
                stream.synchronize()
                preds = non_max_suppression(output_buf, conf_thres=conf)[0]

            with nvtx_range("draw"):
                annotated = draw_detections(frame, preds, scale, pad)

            if display:
                with nvtx_range("display"):
                    cv2.imshow("detections (cuda graph)", annotated)
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
