#!/usr/bin/env python3
"""RF-DETR through the same optimized pipeline as the YOLO iterations.

The RF-DETR counterpart of scripts/infer_cuda_graph_custom_nms.py (Iteration 3)
and scripts/infer_cuda_graph_fp16.py (Iteration 4), so the two models can be
compared with the same optimizations applied, not just baseline-vs-baseline:

- TensorRT engine executed through a captured CUDA graph (Iteration 1)
- fully GPU-side preprocessing, including resize (Iterations 1b/1c)
- fixed-shape, graph-safe GPU postprocessing with a single GPU->CPU sync (Iteration 3)
- data kept in whatever dtype the engine declares - FP16 end-to-end for an
  FP16-I/O engine, with one deliberate cast on the tiny selected tensor (Iteration 4)

What is structurally different from YOLO, and worth looking for in the nsys
timeline:

- **No NMS.** RF-DETR is a DETR-style detector: a fixed set of object queries,
  trained with one-to-one (Hungarian) matching, so each object is predicted
  once. Postprocessing is just sigmoid + top-k over (query, class) pairs - no
  IoU matrix, no suppression. Iteration 3's whole custom-NMS stage disappears.
- **No letterbox.** RF-DETR is trained on a plain (aspect-distorting) resize to
  a square input, and predicts boxes normalized to [0, 1]. Scaling back to the
  source frame is a multiply by (w0, h0, w0, h0) - no padding to subtract.
- **ImageNet normalization** ((x - mean) / std) instead of YOLO's plain /255,
  because the backbone is a DINOv2 ViT.
- **Class ids are COCO category ids** (1..90, with gaps), not YOLO's contiguous
  0..79 - the drawn numbers are not comparable between the two models.

The engine is built by scripts/export_rfdetr.py (ONNX -> trtexec). It's
deserialized directly with the TensorRT Python API rather than through
Ultralytics' AutoBackend, since Ultralytics doesn't load RF-DETR.

Stage flags - so each step can be measured on its own (defaults = everything on).
The baseline follows what rfdetr's own predict() does (source-checked, rfdetr
1.11.1): it already preprocesses on the GPU (uint8 upload to pinned memory,
then float / resize without antialias / ImageNet-normalize on the GPU) and uses
no CUDA graph. So, unlike YOLO (Ultralytics letterboxes on the CPU), GPU
preprocessing is part of RF-DETR's starting point, not an optimization:

    --no-graph --preprocess gpu   TensorRT baseline: rfdetr's default inference path,
                                  with the model run by TensorRT instead of PyTorch
    (no flags)                    + CUDA graph (Iteration 1)
    --preprocess cpu              controlled experiment only: YOLO's starting point
                                  (cv2 resize + numpy normalize on the CPU). Not
                                  something rfdetr itself does.

Postprocessing is the same fixed-shape GPU top-k in every variant (there is no
NMS to vary, see above).

Usage (run from anywhere - the project root is added to sys.path below):
    # TensorRT baseline vs. + CUDA graph (Iteration 1 A/B), strict-FP32 engine:
    python3 scripts/infer_rfdetr_trt.py --engine models/rfdetr-nano_fp32-strict.engine --no-graph --preprocess gpu \
        --no-display --max-frames 600 --warmup-frames 50
    python3 scripts/infer_rfdetr_trt.py --engine models/rfdetr-nano_fp32-strict.engine \
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

MAX_CANDIDATES = 100  # same fixed output size as the YOLO iterations (COCO max-100 convention)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RF-DETR: TensorRT + CUDA graph + GPU pre/postprocessing")
    p.add_argument("--config", default="configs/rfdetr.yaml", help="YAML config path")
    p.add_argument("--engine", default=None,
                   help="TensorRT .engine from scripts/export_rfdetr.py (overrides config 'model')")
    p.add_argument("--source", default=None,
                   help="Override source: file path, rtsp:// URL, or webcam index")
    p.add_argument("--no-display", action="store_true", help="Run headless (no window)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Override max_frames from config (0 = run to end of source)")
    p.add_argument("--warmup-frames", type=int, default=None,
                   help="Frames to exclude from the FPS timer (in addition to the "
                        "engine/graph warm-up this script always does before the loop)")
    p.add_argument("--no-graph", action="store_true",
                   help="Execute the engine eagerly every frame instead of replaying a "
                        "CUDA graph (with --preprocess gpu: the TensorRT baseline)")
    p.add_argument("--preprocess", choices=["cpu", "gpu"], default="gpu",
                   help="gpu: uint8 upload + GPU resize/normalize, as rfdetr's own predict() "
                        "does (baseline); cpu: cv2 resize + numpy normalize (controlled "
                        "experiment with YOLO's starting point only)")
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
    """Identify (boxes, logits) by shape rather than by name, so it doesn't
    depend on what a given rfdetr version calls them: boxes are (1, Q, 4),
    class logits are (1, Q, num_classes)."""
    if len(outputs) != 2:
        raise SystemExit(f"Expected 2 engine outputs (boxes, logits), got {list(outputs)}")
    boxes = [t for t in outputs.values() if t.shape[-1] == 4]
    logits = [t for t in outputs.values() if t.shape[-1] != 4]
    if len(boxes) != 1 or len(logits) != 1:
        raise SystemExit(f"Can't tell boxes from logits: "
                         f"{ {n: tuple(t.shape) for n, t in outputs.items()} }")
    return boxes[0], logits[0]


def preprocess_gpu(gpu_raw: torch.Tensor, res_hw: tuple[int, int], mean: torch.Tensor,
                    std: torch.Tensor, dst: torch.Tensor) -> None:
    """BGR uint8 HWC -> RGB, [0,1], plain resize to the model's square input,
    ImageNet-normalize - computed directly in dst's dtype (FP16 for an FP16-I/O
    engine), so the final copy_() into the graph's input buffer is same-dtype.
    antialias=False matches the YOLO iterations' resize."""
    img = gpu_raw.flip(-1).to(dst.dtype) / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)  # -> (1, 3, H0, W0)
    img = F.interpolate(img, size=res_hw, mode="bilinear", align_corners=False, antialias=False)
    dst.copy_((img - mean) / std)


def preprocess_cpu(frame: np.ndarray, res_hw: tuple[int, int], dst: torch.Tensor) -> None:
    """Same math as preprocess_gpu(), done on the CPU the way the baseline does:
    cv2 resize, numpy BGR->RGB / [0,1] / ImageNet-normalize / HWC->CHW in float32,
    then one upload - copy_() also casts to the engine's input dtype."""
    img = cv2.resize(frame, (res_hw[1], res_hw[0]), interpolation=cv2.INTER_LINEAR)
    img = img[:, :, ::-1].astype(np.float32) / 255.0
    img = (img - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    img = np.ascontiguousarray(img.transpose(2, 0, 1))[None]  # -> (1, 3, H, W)
    dst.copy_(torch.from_numpy(img))


def postprocess(boxes_out: torch.Tensor, logits_out: torch.Tensor, conf_thres: float,
                 scale_xyxy: torch.Tensor) -> torch.Tensor:
    """Fixed-shape (MAX_CANDIDATES, 6) [x1, y1, x2, y2, score, cls] in source-frame
    pixels; rows below conf_thres get score 0 (same convention as Iteration 3).

    Mirrors rfdetr's own PostProcess: sigmoid, then top-k over every
    (query, class) pair - no NMS. Sigmoid and top-k run in the engine's output
    dtype (scores are 0-1, safe in FP16). The selected boxes are cast to FP32
    before scaling to pixels: boxes are normalized to [0, 1], and FP16 has only
    ~3 significant digits, so scaling in FP16 would cost ~0.5 px at 768 wide."""
    prob = logits_out[0].sigmoid()  # (Q, C)
    num_classes = prob.shape[1]
    topk_scores, topk_idx = torch.topk(prob.reshape(-1), k=MAX_CANDIDATES)  # sorted desc
    query_idx = torch.div(topk_idx, num_classes, rounding_mode="floor")
    classes = topk_idx % num_classes

    cx, cy, w, h = boxes_out[0][query_idx].float().unbind(-1)  # <- the one deliberate cast
    boxes_xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1) * scale_xyxy
    scores = topk_scores.float()
    scores = torch.where(scores >= conf_thres, scores, torch.zeros_like(scores))
    return torch.cat([boxes_xyxy, scores.unsqueeze(1), classes.float().unsqueeze(1)], dim=1)


def draw_detections(frame: np.ndarray, dets_cpu: np.ndarray, class_names: dict) -> np.ndarray:
    """dets_cpu: (MAX_CANDIDATES, 6), already in source-frame pixels (no letterbox to undo)."""
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
            f"This script needs a TensorRT .engine, got {engine_path!r}. "
            "Build one with scripts/export_rfdetr.py and pass it with --engine."
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
    input_name, input_buf = next(iter(trt_engine.inputs.items()))
    boxes_buf, logits_buf = split_outputs(trt_engine.outputs)
    for name, t in {**trt_engine.inputs, **trt_engine.outputs}.items():
        print(f"  {name!r} shape={tuple(t.shape)} dtype={t.dtype}")
    res_hw = tuple(input_buf.shape[2:])
    print(f"  model input resolution: {res_hw[1]}x{res_hw[0]}, engine I/O dtype: {input_buf.dtype}")

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

    def preprocess(frame: np.ndarray) -> None:
        if args.preprocess == "cpu":
            preprocess_cpu(frame, res_hw, input_buf)
        else:
            cpu_staging.copy_(torch.from_numpy(frame))
            gpu_raw.copy_(cpu_staging, non_blocking=True)
            preprocess_gpu(gpu_raw, res_hw, mean, std, input_buf)

    stage = f"graph={'off' if args.no_graph else 'on'} preprocess={args.preprocess}"
    print(f"  stages: {stage}")

    stream = torch.cuda.Stream()

    # --- warm-up: at least one uncaptured execute is required by TensorRT
    # before graph capture (flushes any deferred setup work) ---
    preprocess(frame0)
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
                preprocess(frame)
                if graph is None:
                    # Eager baseline: preprocessing ran on the default stream, so
                    # make the engine's stream wait for it before executing.
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        context.execute_async_v3(stream.cuda_stream)
                else:
                    graph.replay()
                stream.synchronize()
                preds = postprocess(boxes_buf, logits_buf, conf, scale_xyxy)
                # Single GPU->CPU sync of the fixed (MAX_CANDIDATES, 6) tensor,
                # inside "inference" - same placement as the YOLO iterations.
                preds_cpu = preds.cpu().numpy()

            with nvtx_range("draw"):
                annotated = draw_detections(frame, preds_cpu, class_names)

            if display:
                with nvtx_range("display"):
                    cv2.imshow(f"detections (rf-detr trt, {stage})", annotated)
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
