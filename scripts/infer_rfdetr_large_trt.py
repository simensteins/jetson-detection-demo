#!/usr/bin/env python3
"""RF-DETR Large through TensorRT, with and without a CUDA graph.

A separate script from scripts/infer_rfdetr_trt.py on purpose: that one is the
nano experiment exactly as it was run (including its --preprocess cpu baseline,
see README "RF-DETR nano experiments - note on the baseline"), and stays
unchanged for traceability.

The baseline here is rfdetr's own default inference path (RFDETR.predict(),
source-checked in rfdetr 1.11.1), with only the model swapped from PyTorch to
TensorRT:

- Preprocessing on the GPU, as predict() does it: the uint8 frame is uploaded
  through pinned memory, then converted to float / 255, resized to the model's
  square input (bilinear, antialias=False) and ImageNet-normalized on the GPU.
  There is no CPU-preprocessing option - rfdetr itself never preprocesses on
  the CPU, so it would not be a baseline. The only extra step is BGR -> RGB
  (OpenCV decodes BGR, predict() expects RGB), done on the GPU.
- Postprocessing mirrors rfdetr's PostProcess: sigmoid, then top-k over every
  (query, class) pair with num_select = 300 (RFDETRLargeConfig), no NMS.
- No CUDA graph (predict() uses none unless optimize_for_inference() is called).

The single variable this script changes is --no-graph:

    --no-graph    TensorRT baseline: engine executed eagerly every frame
    (no flags)    + CUDA graph (Iteration 1): the forward pass captured once,
                  replayed with one call per frame

The engine is built by scripts/export_rfdetr.py --size large (strict FP32 by
default). Its input resolution (704x704 for Large) is read from the engine.

Usage (run from anywhere - the project root is added to sys.path below):
    python3 scripts/infer_rfdetr_large_trt.py --engine models/rfdetr-large_fp32-strict.engine --no-graph \
        --no-display --max-frames 600 --warmup-frames 50
    python3 scripts/infer_rfdetr_large_trt.py --engine models/rfdetr-large_fp32-strict.engine \
        --no-display --max-frames 600 --warmup-frames 50
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
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.profiling import nvtx_range  # noqa: E402
from src.sources import open_source  # noqa: E402

NUM_SELECT = 300  # rfdetr's num_select for RF-DETR Large (RFDETRLargeConfig)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RF-DETR Large: TensorRT baseline vs. + CUDA graph")
    p.add_argument("--config", default="configs/rfdetr-large.yaml", help="YAML config path")
    p.add_argument("--engine", default=None,
                   help="TensorRT .engine from scripts/export_rfdetr.py --size large "
                        "(overrides config 'model')")
    p.add_argument("--source", default=None,
                   help="Override source: file path, rtsp:// URL, or webcam index")
    p.add_argument("--no-display", action="store_true", help="Run headless (no window)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Override max_frames from config (0 = run to end of source)")
    p.add_argument("--warmup-frames", type=int, default=None,
                   help="Frames to exclude from the FPS timer (in addition to the "
                        "engine/graph warm-up this script always does before the loop)")
    p.add_argument("--no-graph", action="store_true",
                   help="TensorRT baseline: execute the engine eagerly every frame "
                        "instead of replaying a CUDA graph")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def trt_to_torch_dtype(dt) -> torch.dtype:
    return torch.from_numpy(np.empty(0, dtype=trt.nptype(dt))).dtype


class TrtEngine:
    """Deserialized engine + execution context + one fixed-address CUDA tensor
    per I/O binding (fixed addresses are what make CUDA graph capture valid)."""

    def __init__(self, path: str):
        logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(logger)  # must outlive the engine
        with open(path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise SystemExit(f"Could not deserialize {path!r} - was it built on this device "
                             "with this TensorRT version? (rebuild with scripts/export_rfdetr.py)")
        self.context = self.engine.create_execution_context()
        self.inputs: dict[str, torch.Tensor] = {}
        self.outputs: dict[str, torch.Tensor] = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                raise SystemExit(f"Tensor {name!r} has a dynamic shape {shape}; "
                                 "export a static batch-1 engine.")
            buf = torch.empty(shape, dtype=trt_to_torch_dtype(self.engine.get_tensor_dtype(name)),
                              device="cuda")
            self.context.set_tensor_address(name, buf.data_ptr())
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            (self.inputs if is_input else self.outputs)[name] = buf


def split_outputs(outputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Identify (boxes, logits) by shape rather than by name: boxes are (1, Q, 4),
    class logits are (1, Q, num_classes)."""
    if len(outputs) != 2:
        raise SystemExit(f"Expected 2 engine outputs (boxes, logits), got {list(outputs)}")
    boxes = [t for t in outputs.values() if t.shape[-1] == 4]
    logits = [t for t in outputs.values() if t.shape[-1] != 4]
    if len(boxes) != 1 or len(logits) != 1:
        raise SystemExit(f"Can't tell boxes from logits: "
                         f"{ {n: tuple(t.shape) for n, t in outputs.items()} }")
    return boxes[0], logits[0]


def preprocess(gpu_raw: torch.Tensor, res_hw: tuple[int, int], mean: torch.Tensor,
               std: torch.Tensor, dst: torch.Tensor) -> None:
    """rfdetr predict()'s GPU steps, in its order: uint8 -> float / 255, resize
    (bilinear, antialias=False), ImageNet-normalize. The BGR -> RGB flip comes
    first because OpenCV decodes BGR."""
    img = gpu_raw.flip(-1).permute(2, 0, 1).unsqueeze(0).to(dst.dtype) / 255.0  # (1, 3, H0, W0)
    img = F.interpolate(img, size=res_hw, mode="bilinear", align_corners=False, antialias=False)
    dst.copy_((img - mean) / std)


def postprocess(boxes_out: torch.Tensor, logits_out: torch.Tensor, conf_thres: float,
                scale_xyxy: torch.Tensor) -> torch.Tensor:
    """Fixed-shape (NUM_SELECT, 6) [x1, y1, x2, y2, score, cls] in source-frame
    pixels, as rfdetr's PostProcess: sigmoid, then top-k over every (query, class)
    pair, boxes (cx, cy, w, h) normalized to [0, 1] scaled to the frame. Rows
    below conf_thres get score 0 (filtered when drawing, not by a shape change)."""
    prob = logits_out[0].float().sigmoid()  # (Q, C)
    num_classes = prob.shape[1]
    topk_scores, topk_idx = torch.topk(prob.reshape(-1), k=NUM_SELECT)  # sorted desc
    query_idx = torch.div(topk_idx, num_classes, rounding_mode="floor")
    classes = topk_idx % num_classes

    cx, cy, w, h = boxes_out[0][query_idx].float().unbind(-1)
    boxes_xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1) * scale_xyxy
    scores = torch.where(topk_scores >= conf_thres, topk_scores, torch.zeros_like(topk_scores))
    return torch.cat([boxes_xyxy, scores.unsqueeze(1), classes.float().unsqueeze(1)], dim=1)


def draw_detections(frame: np.ndarray, dets_cpu: np.ndarray, class_names: dict) -> np.ndarray:
    """dets_cpu: (NUM_SELECT, 6), already in source-frame pixels (no letterbox to undo)."""
    annotated = frame.copy()
    for x1, y1, x2, y2, conf, cls in dets_cpu[dets_cpu[:, 4] > 0].tolist():
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        label = f"{class_names.get(int(cls), int(cls))} {conf:.2f}"
        cv2.rectangle(annotated, p1, p2, (56, 56, 255), 2)
        cv2.putText(annotated, label, (p1[0], max(p1[1] - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (56, 56, 255), 1)
    return annotated


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    source = args.source if args.source is not None else cfg["source"]
    display = cfg.get("display", True) and not args.no_display
    conf = cfg.get("conf", 0.25)
    max_frames = args.max_frames if args.max_frames is not None else cfg.get("max_frames", 0)
    warmup_frames = (args.warmup_frames if args.warmup_frames is not None
                      else cfg.get("warmup_frames", 0))

    engine_path = args.engine if args.engine is not None else cfg["model"]
    if not str(engine_path).endswith(".engine"):
        raise SystemExit(
            f"This script needs a TensorRT .engine, got {engine_path!r}. Build one with "
            "scripts/export_rfdetr.py --size large and pass it with --engine."
        )

    # Class names are cosmetic - the script runs without rfdetr installed.
    try:
        from rfdetr.util.coco_classes import COCO_CLASSES as class_names
    except ImportError:
        class_names = {}

    print(f"Loading {engine_path} ...")
    trt_engine = TrtEngine(engine_path)
    context = trt_engine.context
    if len(trt_engine.inputs) != 1:
        raise SystemExit(f"Expected 1 engine input, got {list(trt_engine.inputs)}")
    input_buf = next(iter(trt_engine.inputs.values()))
    boxes_buf, logits_buf = split_outputs(trt_engine.outputs)
    for name, t in {**trt_engine.inputs, **trt_engine.outputs}.items():
        print(f"  {name!r} shape={tuple(t.shape)} dtype={t.dtype}")
    res_hw = tuple(input_buf.shape[2:])
    print(f"  model input resolution: {res_hw[1]}x{res_hw[0]}, engine I/O dtype: {input_buf.dtype}")
    if res_hw != (704, 704):
        print(f"  WARNING: RF-DETR Large is 704x704 - is {engine_path!r} really a Large engine?")
    if boxes_buf.shape[1] < NUM_SELECT:
        raise SystemExit(f"Engine has {boxes_buf.shape[1]} queries, fewer than NUM_SELECT={NUM_SELECT}")

    cap = open_source(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {source!r}")

    ok, frame0 = cap.read()
    if not ok:
        raise SystemExit("Could not read a frame to warm up the engine.")
    h0, w0 = frame0.shape[:2]
    print(f"  source frame: {w0}x{h0} -> plain resize to {res_hw[1]}x{res_hw[0]}")

    cpu_staging = torch.empty((h0, w0, 3), dtype=torch.uint8, pin_memory=True)
    gpu_raw = torch.empty((h0, w0, 3), dtype=torch.uint8, device="cuda")
    mean = torch.tensor(IMAGENET_MEAN, dtype=input_buf.dtype, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=input_buf.dtype, device="cuda").view(1, 3, 1, 1)
    scale_xyxy = torch.tensor([w0, h0, w0, h0], dtype=torch.float32, device="cuda")

    def upload_and_preprocess(frame: np.ndarray) -> None:
        cpu_staging.copy_(torch.from_numpy(frame))
        gpu_raw.copy_(cpu_staging, non_blocking=True)
        preprocess(gpu_raw, res_hw, mean, std, input_buf)

    stage = f"graph={'off' if args.no_graph else 'on'}"
    print(f"  stage: {stage} (preprocessing on the GPU, as rfdetr's predict())")

    stream = torch.cuda.Stream()

    # --- warm-up: at least one uncaptured execute is required by TensorRT
    # before graph capture (flushes any deferred setup work) ---
    upload_and_preprocess(frame0)
    torch.cuda.synchronize()
    for _ in range(3):
        with torch.cuda.stream(stream):
            context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

    graph = None
    if not args.no_graph:
        # --- capture the forward pass as a CUDA graph, once ---
        print("Capturing CUDA graph...")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            context.execute_async_v3(stream.cuda_stream)
        print("Graph captured.")
    print("Starting inference loop.")

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
                upload_and_preprocess(frame)
                if graph is None:
                    # Preprocessing ran on the default stream, so make the
                    # engine's stream wait for it before executing.
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        context.execute_async_v3(stream.cuda_stream)
                else:
                    graph.replay()
                stream.synchronize()
                preds = postprocess(boxes_buf, logits_buf, conf, scale_xyxy)
                # Single GPU->CPU sync of the fixed (NUM_SELECT, 6) tensor,
                # inside "inference" - same placement as the YOLO iterations.
                preds_cpu = preds.cpu().numpy()

            with nvtx_range("draw"):
                annotated = draw_detections(frame, preds_cpu, class_names)

            if display:
                with nvtx_range("display"):
                    cv2.imshow(f"detections (rf-detr large trt, {stage})", annotated)
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
