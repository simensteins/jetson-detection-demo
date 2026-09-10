#!/usr/bin/env python3
"""Iteration 1b: CUDA graph + GPU-side preprocessing.

Builds on scripts/infer_cuda_graph.py (kept unchanged for comparison). That
iteration confirmed CUDA graph replay eliminates dispatch-induced idle time
between kernels (~30% -> ~0.2% of the model's own execution span), but
revealed preprocessing as the now-largest cost in the frame (~42.7%,
8.27ms) - larger than the model itself.

The only change here: how preprocessing gets the frame onto the GPU and
into the shape/dtype the engine expects. The previous version did all
reformatting (BGR->RGB, HWC->CHW, uint8->float32 cast, /255 normalize) on
the CPU via numpy, then transferred a float32 tensor to the GPU. This
version does only the resize+pad on the CPU (cv2, unavoidable - the frame
starts on the CPU), transfers the *raw* uint8 HWC frame to the GPU via a
pinned staging buffer, then does the reformatting as GPU tensor ops.
Rationale:
  - Transfers 1/4 the data over PCIe (uint8 vs float32 for the same pixels).
  - Pinned (page-locked) host memory lets the CUDA driver DMA the transfer
    directly instead of staging an extra internal copy.
  - The reformatting itself (flip/cast/normalize/permute) runs on the GPU,
    parallel across thousands of cores, instead of single-threaded CPU numpy.

Everything else (graph capture/replay, NMS, NVTX ranges, CLI flags) is
identical to scripts/infer_cuda_graph.py - single-variable comparison.

Usage (run from anywhere - the project root is added to sys.path below):
    python3 scripts/infer_cuda_graph_gpu_preprocess.py --config configs/default.yaml \
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
import yaml
from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.profiling import nvtx_range  # noqa: E402
from src.sources import open_source  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Iteration 1b: CUDA graph + GPU-side preprocessing")
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


def letterbox(frame: np.ndarray, new_shape: int, color=(114, 114, 114)):
    """Resize + pad to a square, preserving aspect ratio (standard YOLO
    preprocessing). Returns the padded uint8 HWC BGR image, the scale
    factor, and the (left, top) padding, needed to map detections back to
    original coordinates. CPU/cv2 - unavoidable, the frame starts here."""
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


def preprocess(frame: np.ndarray, imgsz: int, cpu_staging: torch.Tensor,
                gpu_raw: torch.Tensor, dst: torch.Tensor) -> tuple[float, tuple[int, int]]:
    """Letterbox on CPU (uint8, HWC, BGR - unavoidable), then transfer that
    raw uint8 frame to the GPU and do all reformatting (channel flip,
    normalize, HWC->CHW) there, writing the result into the fixed GPU
    input buffer `dst` (shape [1,3,imgsz,imgsz], float32, CUDA)."""
    padded, scale, pad = letterbox(frame, imgsz)
    cpu_staging.copy_(torch.from_numpy(padded))  # numpy -> pinned CPU tensor
    gpu_raw.copy_(cpu_staging, non_blocking=True)  # pinned CPU -> GPU (uint8, small)
    rgb = gpu_raw.flip(-1).float() / 255.0  # BGR->RGB + normalize, still HWC, on GPU
    dst.copy_(rgb.permute(2, 0, 1).unsqueeze(0))  # HWC->CHW, add batch dim
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

    cpu_staging = torch.empty((imgsz, imgsz, 3), dtype=torch.uint8, pin_memory=True)
    gpu_raw = torch.empty((imgsz, imgsz, 3), dtype=torch.uint8, device="cuda")

    cap = open_source(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {source!r}")

    stream = torch.cuda.Stream()
    context.set_tensor_address(input_name, input_buf.data_ptr())
    context.set_tensor_address(output_name, output_buf.data_ptr())

    # --- warm-up: at least one uncaptured execute is required by TensorRT
    # before graph capture (flushes any deferred setup work) ---
    ok, frame0 = cap.read()
    if not ok:
        raise SystemExit("Could not read a frame to warm up the engine.")
    preprocess(frame0, imgsz, cpu_staging, gpu_raw, input_buf)
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
                scale, pad = preprocess(frame, imgsz, cpu_staging, gpu_raw, input_buf)
                graph.replay()
                stream.synchronize()
                preds = non_max_suppression(output_buf, conf_thres=conf)[0]

            with nvtx_range("draw"):
                annotated = draw_detections(frame, preds, scale, pad)

            if display:
                with nvtx_range("display"):
                    cv2.imshow("detections (cuda graph, gpu preprocess)", annotated)
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
