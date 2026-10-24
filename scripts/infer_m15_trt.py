#!/usr/bin/env python3
"""m15_fast_detector through TensorRT, with a CUDA graph.

The m15 counterpart of scripts/infer_rfdetr_large_trt.py, with the same
structure (same TrtEngine, NVTX ranges, warm-up/timing loop) so the numbers
are comparable. The model is a yolo11s with an added P2 detection branch
(strides 4/8/16/32, see models/m15/yolo11s-p2-m15.yaml), 1-channel grayscale
SAR/IR input, one class, trained at 960x960. Weights: best.pt, EMA
(rebuilt by scripts/build_m15_pt.py, exported by scripts/export_m15.py).

The engine input is static (1, 1, 960, 960), FP32 or FP16 (an fp16 engine
from scripts/export_m15.py --precision fp16 has FP16 I/O); its output (1, 4+nc, N)
holds DFL-decoded xywh boxes in 960x960 input pixels and sigmoid class
scores over the N anchors of the four strides - no NMS in the graph.

Per-frame path:
- Preprocessing (GPU, default): the BGR frame is uploaded through pinned memory,
  resized, converted to grayscale (BT.601 weights, as cv2.cvtColor) and divided
  by 255, and written into the 960x960 letterbox (Ultralytics' letterbox
  semantics: scale to fit, centered pad, 114 fill; the pad is filled once). Ultralytics' own predict() letterboxes on the CPU;
  here it runs on the GPU, mirroring the RF-DETR Large script's placement.
  --preprocess cpu runs the same pipeline in cv2/numpy on the CPU instead -
  the Ultralytics predict() placement - with a single upload per frame.
- Postprocessing, --nms torchvision (default): confidence filter,
  class-agnostic NMS (the model has one class; torchvision.ops.nms on the
  GPU, iou from config), top max_det=300 by score, letterbox undone to
  source-frame pixels. The boolean filter and keep.any() make the CPU wait
  on the GPU (~2-3 syncs per frame), and the number of kernels depends on
  how many candidates pass the threshold.
- Postprocessing, --nms fixed: fixed-shape NMS as in the YOLOv8n Iteration 3
  (scripts/infer_cuda_graph_custom_nms.py) - top-K by score, K x K IoU
  matrix, no data-dependent shapes, so the only sync left is the final
  .cpu() - but Cluster-NMS instead of Iteration 3's Fast NMS: Fast NMS lets
  an already-suppressed box suppress the boxes below it and so drops boxes
  standard NMS keeps; a fixed number of Cluster-NMS passes gives the same
  detections as torchvision.ops.nms.
- The surviving detections are moved to the CPU in one sync, inside
  "inference" - same placement as the RF-DETR Large iterations.

The stage flags are --no-graph, --preprocess and --nms (defaults: graph on,
GPU preprocessing, torchvision NMS - so the stage commands already used for
the profiling runs keep meaning the same thing):

    (no flags)        + CUDA graph: the forward pass captured once, replayed
                      with one call per frame
    --preprocess cpu  + CUDA graph, CPU (cv2/numpy) preprocessing - the
                      Ultralytics predict() placement
    --no-graph        TensorRT baseline: engine executed eagerly every frame
    --nms fixed       + fixed-shape NMS (on top of the defaults)

Usage (run from anywhere - the project root is added to sys.path below):
    python3 scripts/infer_m15_trt.py --engine models/m15/m15_fast_fp32-tf32.engine \
        --no-display --max-frames 600 --warmup-frames 50
    python3 scripts/infer_m15_trt.py --engine models/m15/m15_fast_fp32-tf32.engine \
        --max-frames 600 --warmup-frames 50   # with display, press q to quit
    # FP16 engine - the same script, the dtypes are read from the engine:
    python3 scripts/infer_m15_trt.py --engine models/m15/m15_fast_fp16.engine \
        --no-display --max-frames 600 --warmup-frames 50
    # + fixed-shape NMS (stage 5, here on the FP16 engine):
    python3 scripts/infer_m15_trt.py --engine models/m15/m15_fast_fp16.engine \
        --nms fixed --no-display --max-frames 600 --warmup-frames 50
    # CPU preprocessing instead of GPU (same math, Ultralytics' placement):
    python3 scripts/infer_m15_trt.py --engine models/m15/m15_fast_fp32-tf32.engine \
        --preprocess cpu --no-display --max-frames 600 --warmup-frames 50
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
from torchvision.ops import nms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.profiling import nvtx_range  # noqa: E402
from src.sinks import open_sink  # noqa: E402
from src.sources import open_source  # noqa: E402

IMGSZ = 960  # the engine input resolution (m15 training image size)
GRAY_BGR_WEIGHTS = (0.114, 0.587, 0.299)  # BT.601 luma, OpenCV BGR order
LETTERBOX_FILL_U8 = 114  # Ultralytics letterbox fill, uint8 (CPU path, pre-normalization)
LETTERBOX_FILL = 114 / 255.0  # Ultralytics letterbox fill, already normalized
MAX_DET = 300  # Ultralytics predict() default
# Default fixed top-K for --nms fixed (--nms-candidates). The torchvision path
# runs NMS on EVERY box above the threshold, so K must exceed the candidates a
# frame can have, or low-scored boxes are cut before NMS (each person yields
# many anchors above threshold with the stride-4 head). K x K is the IoU
# matrix, so K also sets the cost. A run reports the most candidates any frame
# had and how many frames filled all K - use that to pick K.
NMS_CANDIDATES = 1000
NMS_ITERS = 10  # Cluster-NMS passes for --nms fixed (see postprocess_fixed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="m15 detector: TensorRT + CUDA graph")
    p.add_argument("--config", default="configs/m15-fast.yaml", help="YAML config path")
    p.add_argument("--engine", default=None,
                   help="TensorRT .engine from scripts/export_m15.py (overrides config 'model')")
    p.add_argument("--source", default=None,
                   help="Override source: file path, rtsp:// URL, or webcam index")
    p.add_argument("--no-display", action="store_true", help="Run headless (no window)")
    p.add_argument("--sink", default=None,
                   help="Send annotated video on, e.g. rtp://<host-ip>:5000 (see src/sinks.py)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Override max_frames from config (0 = run to end of source)")
    p.add_argument("--warmup-frames", type=int, default=None,
                   help="Frames to exclude from the FPS timer (in addition to the "
                        "engine/graph warm-up this script always does before the loop)")
    p.add_argument("--no-graph", action="store_true",
                   help="TensorRT baseline: execute the engine eagerly every frame "
                        "instead of replaying a CUDA graph")
    p.add_argument("--nms", choices=["torchvision", "fixed"], default="torchvision",
                   help="'torchvision': confidence filter + torchvision.ops.nms (default); "
                        "'fixed': fixed-shape Cluster-NMS, one GPU->CPU sync per frame")
    p.add_argument("--nms-candidates", type=int, default=NMS_CANDIDATES,
                   help=f"--nms fixed: top-K boxes kept for NMS (default {NMS_CANDIDATES})")
    p.add_argument("--preprocess", choices=["cpu", "gpu"], default="gpu",
                   help="Preprocessing placement: 'gpu' (default) uploads the frame "
                        "once and does grayscale/letterbox/normalize on the GPU; "
                        "'cpu' does the same pipeline in cv2/numpy, as Ultralytics' "
                        "predict() does, with one upload per frame")
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
                             "with this TensorRT version? (rebuild with scripts/export_m15.py)")
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


def preprocess_cpu(frame: np.ndarray, new_hw: tuple[int, int],
                   pad_lrtb: tuple[int, int, int, int], dst: torch.Tensor) -> None:
    """The GPU path's pipeline on the CPU, the way Ultralytics' predict() does
    it: BT.601 grayscale (cv2.cvtColor), cv2.resize, letterbox pad with the 114
    fill, /255 - then one upload into the engine's input buffer (copy_() also
    casts to the engine's input dtype)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # BT.601 luma, uint8
    img = cv2.resize(gray, (new_hw[1], new_hw[0]), interpolation=cv2.INTER_LINEAR)
    pad_l, pad_r, pad_t, pad_b = pad_lrtb
    img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r,
                             cv2.BORDER_CONSTANT, value=LETTERBOX_FILL_U8)
    img = img.astype(np.float32)[None, None] / 255.0  # -> (1, 1, 960, 960)
    dst.copy_(torch.from_numpy(img))


def postprocess(pred_out: torch.Tensor, conf: float, iou: float,
                gain: float, pad_lt: tuple[float, float],
                frame_hw: tuple[int, int]) -> torch.Tensor:
    """(1, 4+nc, N) engine output -> (M, 6) [x1, y1, x2, y2, score, cls] in
    source-frame pixels: confidence filter, class-agnostic NMS (nc=1),
    top MAX_DET by score, letterbox undone and clipped to the frame.

    An FP16 engine output is cast to FP32 first (one small cast, 5 x N):
    box areas reach 960*960 = 921,600, beyond FP16's max of 65,504, and NMS
    computes areas and IoU in the tensor's dtype."""
    pred_out = pred_out.float()  # no-op for an FP32 engine
    boxes_xywh = pred_out[0, :4].T  # (N, 4), input-pixel xywh
    scores = pred_out[0, 4]  # (N,), sigmoid already applied in the graph
    keep = scores > conf
    if not keep.any():
        return torch.zeros((0, 6), device=pred_out.device)
    boxes_xyxy = torch.empty_like(boxes_xywh[keep])
    boxes_xyxy[:, 0] = boxes_xywh[keep][:, 0] - boxes_xywh[keep][:, 2] / 2
    boxes_xyxy[:, 1] = boxes_xywh[keep][:, 1] - boxes_xywh[keep][:, 3] / 2
    boxes_xyxy[:, 2] = boxes_xywh[keep][:, 0] + boxes_xywh[keep][:, 2] / 2
    boxes_xyxy[:, 3] = boxes_xywh[keep][:, 1] + boxes_xywh[keep][:, 3] / 2
    keep_idx = nms(boxes_xyxy, scores[keep], iou)[:MAX_DET]  # sorted by score desc
    boxes_xyxy = (boxes_xyxy[keep_idx] - torch.tensor([pad_lt[0], pad_lt[1],
                                                       pad_lt[0], pad_lt[1]],
                                                      device=pred_out.device)) / gain
    h0, w0 = frame_hw
    boxes_xyxy[:, 0::2] = boxes_xyxy[:, 0::2].clamp_(0, w0 - 1)
    boxes_xyxy[:, 1::2] = boxes_xyxy[:, 1::2].clamp_(0, h0 - 1)
    cls = torch.zeros((boxes_xyxy.shape[0], 1), device=pred_out.device)  # single class
    return torch.cat([boxes_xyxy, scores[keep][keep_idx, None], cls], dim=1)


def box_iou_matrix(boxes_xyxy: torch.Tensor) -> torch.Tensor:
    """(K, 4) xyxy -> (K, K) pairwise IoU, fixed shape (as in
    scripts/infer_cuda_graph_custom_nms.py)."""
    area = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]).clamp(min=0) * \
           (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]).clamp(min=0)  # (K,)
    x1 = torch.max(boxes_xyxy[:, 0].unsqueeze(1), boxes_xyxy[:, 0].unsqueeze(0))  # (K, K)
    y1 = torch.max(boxes_xyxy[:, 1].unsqueeze(1), boxes_xyxy[:, 1].unsqueeze(0))
    x2 = torch.min(boxes_xyxy[:, 2].unsqueeze(1), boxes_xyxy[:, 2].unsqueeze(0))
    y2 = torch.min(boxes_xyxy[:, 3].unsqueeze(1), boxes_xyxy[:, 3].unsqueeze(0))
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)  # (K, K)
    union = area.unsqueeze(1) + area.unsqueeze(0) - inter
    return inter / union.clamp(min=1e-9)


def postprocess_fixed(pred_out: torch.Tensor, conf: float, iou: float,
                      gain: float, pad_lt: tuple[float, float],
                      frame_hw: tuple[int, int], k: int = NMS_CANDIDATES
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """(1, 4+nc, N) engine output -> FIXED (k, 6)
    [x1, y1, x2, y2, score, cls] in source-frame pixels, every frame the
    same shape and the same kernels: no boolean filtering, no .any(), so the
    CPU never waits on the GPU here. Rows that are not detections have
    score 0 - the caller filters them on the CPU after the single .cpu().
    Also returns `stats`, a (3,) FP32 tensor, so a run can report (without
    extra syncs) whether the fixed sizes were ever too small:
      [0] candidates   boxes above the threshold among the top k (= k means
                       more may have been cut before NMS)
      [1] truncated    1 if all k top boxes were above the threshold
      [2] unconverged  1 if NMS_ITERS passes were not enough (one more pass
                       would still change the result)

    Cluster-NMS (one class, so no class offset): top-K by score, then a box
    is dropped if its IoU with a higher-scored KEPT box is above `iou` -
    the same result as torchvision.ops.nms. topk runs in the engine's dtype; the K selected rows are cast to
    FP32 for the box maths (areas reach 921,600, beyond FP16's 65,504)."""
    topk_scores, topk_idx = torch.topk(pred_out[0, 4], k=k)  # (K,), sorted desc
    topk_scores = topk_scores.float()
    cx, cy, w, h = pred_out[0, :4, topk_idx].float().unbind(0)  # each (K,), input pixels
    boxes_xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)  # (K, 4)

    # suppresses[i, j]: higher-scored box i (i < j) overlaps box j above `iou`.
    suppresses = torch.triu(box_iou_matrix(boxes_xyxy) > iou, diagonal=1)
    # Cluster-NMS: a box is dropped only if a higher-scored box that is ITSELF
    # kept overlaps it. One pass of this (from "all kept") is Fast NMS, which
    # over-suppresses; iterating converges to exactly standard NMS. A fixed
    # number of passes keeps the shapes and kernels fixed (no sync to test
    # for convergence) - NMS_ITERS was checked against torchvision.ops.nms.
    candidate = topk_scores > conf
    keep = candidate
    for _ in range(NMS_ITERS):
        keep = candidate & ~(suppresses & keep.unsqueeze(1)).any(dim=0)
    unconverged = (keep != (candidate & ~(suppresses & keep.unsqueeze(1)).any(dim=0))).any()
    keep = keep & (keep.cumsum(0) <= MAX_DET)  # at most MAX_DET, highest scores first
    scores = torch.where(keep, topk_scores, torch.zeros_like(topk_scores))

    boxes_xyxy = (boxes_xyxy - torch.tensor([pad_lt[0], pad_lt[1], pad_lt[0], pad_lt[1]],
                                            device=pred_out.device)) / gain
    h0, w0 = frame_hw
    boxes_xyxy[:, 0::2] = boxes_xyxy[:, 0::2].clamp_(0, w0 - 1)
    boxes_xyxy[:, 1::2] = boxes_xyxy[:, 1::2].clamp_(0, h0 - 1)
    cls = torch.zeros((k, 1), device=pred_out.device)  # single class
    stats = torch.stack([candidate.sum(), candidate[-1], unconverged]).float()
    return torch.cat([boxes_xyxy, scores.unsqueeze(1), cls], dim=1), stats


def draw_detections(frame: np.ndarray, dets_cpu: np.ndarray, class_names: dict) -> np.ndarray:
    """dets_cpu: (M, 6), already in source-frame pixels (letterbox undone).
    Rows with score 0 are --nms fixed padding, filtered here on the CPU."""
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
    sink_spec = args.sink if args.sink is not None else cfg.get("sink")
    conf = cfg.get("conf", 0.25)
    iou = cfg.get("iou", 0.7)
    max_frames = args.max_frames if args.max_frames is not None else cfg.get("max_frames", 0)
    warmup_frames = (args.warmup_frames if args.warmup_frames is not None
                     else cfg.get("warmup_frames", 0))

    engine_path = args.engine if args.engine is not None else cfg["model"]
    if not str(engine_path).endswith(".engine"):
        raise SystemExit(
            f"This script needs a TensorRT .engine, got {engine_path!r}. Build one with "
            "scripts/export_m15.py and pass it with --engine."
        )

    # Single-class model; the name is cosmetic (the checkpoint carries none).
    class_names = {0: "target"}

    print(f"Loading {engine_path} ...")
    trt_engine = TrtEngine(engine_path)
    context = trt_engine.context
    if len(trt_engine.inputs) != 1:
        raise SystemExit(f"Expected 1 engine input, got {list(trt_engine.inputs)}")
    input_buf = next(iter(trt_engine.inputs.values()))
    if len(trt_engine.outputs) != 1:
        raise SystemExit(f"Expected 1 engine output, got {list(trt_engine.outputs)}")
    pred_buf = next(iter(trt_engine.outputs.values()))
    for name, t in {**trt_engine.inputs, **trt_engine.outputs}.items():
        print(f"  {name!r} shape={tuple(t.shape)} dtype={t.dtype}")
    if tuple(input_buf.shape) != (1, 1, IMGSZ, IMGSZ):
        raise SystemExit(f"Expected input (1, 1, {IMGSZ}, {IMGSZ}), got {tuple(input_buf.shape)}")
    if pred_buf.shape[1] != 5:  # 4 box + 1 class
        raise SystemExit(f"Expected output (1, 5, N) for the single-class head, "
                         f"got {tuple(pred_buf.shape)}")
    print(f"  anchors per frame: {pred_buf.shape[2]} (strides 4/8/16/32 at {IMGSZ}x{IMGSZ})")

    cap = open_source(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {source!r}")

    ok, frame0 = cap.read()
    if not ok:
        raise SystemExit("Could not read a frame to warm up the engine.")
    h0, w0 = frame0.shape[:2]

    # Letterbox geometry (Ultralytics semantics): scale to fit 960x960, then
    # center-pad the remainder. Constant per source - computed once from frame0.
    r = min(IMGSZ / w0, IMGSZ / h0)
    new_w, new_h = round(w0 * r), round(h0 * r)
    dw, dh = IMGSZ - new_w, IMGSZ - new_h
    pad_l, pad_r = dw // 2, dw - dw // 2
    pad_t, pad_b = dh // 2, dh - dh // 2
    print(f"  source frame: {w0}x{h0} -> grayscale + letterbox "
          f"(x{r:.4g}) to {new_w}x{new_h}, pad l/r t/b = {pad_l}/{pad_r} {pad_t}/{pad_b}")

    cpu_staging = torch.empty((h0, w0, 3), dtype=torch.uint8, pin_memory=True)
    gpu_raw = torch.empty((h0, w0, 3), dtype=torch.uint8, device="cuda")
    # The GPU path computes in the engine's input dtype (FP16 for an fp16
    # engine, so nothing is cast at the engine boundary). 0-255 pixel values
    # are exact in FP16, and the weighted sum stays <= 1.0.
    pre_dtype = input_buf.dtype
    # BT.601 weights with the /255 folded in, shaped to weight the C axis of (1, 3, h, w).
    gray_w = (torch.tensor(GRAY_BGR_WEIGHTS, dtype=torch.float32, device="cuda")
              / 255.0).to(pre_dtype).view(1, 3, 1, 1)
    # The letterbox border is the same every frame: fill the whole input once,
    # then each frame writes only the image region (no per-frame pad + copy).
    input_buf.fill_(LETTERBOX_FILL)
    input_region = input_buf[:, :, pad_t:pad_t + new_h, pad_l:pad_l + new_w]

    def preprocess(frame: np.ndarray) -> None:
        if args.preprocess == "cpu":
            preprocess_cpu(frame, (new_h, new_w), (pad_l, pad_r, pad_t, pad_b), input_buf)
        else:
            cpu_staging.copy_(torch.from_numpy(frame))
            gpu_raw.copy_(cpu_staging, non_blocking=True)
            # Resize first, then grayscale. Both are linear, so the order doesn't
            # change the result, but for a source larger than the letterbox
            # (1920x1080 -> 960x540) the weighted sum runs on 4x fewer pixels.
            # The weighted sum is a broadcast multiply + sum, not a matmul
            # (`@ gray_w` ran as a cuBLAS gemv: ~4 ms per frame at 1920x1080).
            img = gpu_raw.permute(2, 0, 1)[None].to(pre_dtype)  # (1, 3, H0, W0), channels-last memory
            img = F.interpolate(img, size=(new_h, new_w), mode="bilinear",
                                align_corners=False, antialias=False)  # as cv2.resize
            input_region.copy_((img * gray_w).sum(dim=1, keepdim=True))

    stage = (f"graph={'off' if args.no_graph else 'on'} preprocess={args.preprocess} "
             f"nms={args.nms}" + (f" k={args.nms_candidates}" if args.nms == "fixed" else ""))
    print(f"  stage: {stage}")

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

    sink = None
    truncated_frames = unconverged_frames = max_candidates = 0
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
                    # Preprocessing ran on the default stream, so make the
                    # engine's stream wait for it before executing.
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        context.execute_async_v3(stream.cuda_stream)
                else:
                    graph.replay()
                stream.synchronize()
                # Single GPU->CPU sync of the detections, inside "inference" -
                # same placement as the RF-DETR iterations.
                if args.nms == "fixed":
                    dets, stats = postprocess_fixed(pred_buf, conf, iou, r, (pad_l, pad_t),
                                                    (h0, w0), k=args.nms_candidates)
                    flat = torch.cat([dets.flatten(), stats]).cpu().numpy()
                    preds_cpu = flat[:-3].reshape(-1, 6)
                    max_candidates = max(max_candidates, int(flat[-3]))
                    truncated_frames += int(flat[-2])
                    unconverged_frames += int(flat[-1])
                else:
                    preds = postprocess(pred_buf, conf, iou, r, (pad_l, pad_t), (h0, w0))
                    preds_cpu = preds.cpu().numpy()

            with nvtx_range("draw"):
                annotated = draw_detections(frame, preds_cpu, class_names)

            if display:
                with nvtx_range("display"):
                    cv2.imshow(f"detections (m15 trt, {stage})", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            if sink_spec:
                if sink is None:  # opened on the first frame, once its size is known
                    sink = open_sink(sink_spec, w0, h0, cap.get(cv2.CAP_PROP_FPS))
                with nvtx_range("output"):
                    sink.write(annotated)

            frames += 1
            if frames <= warmup_frames:
                if frames == warmup_frames:
                    t0 = time.perf_counter()  # discard warm-up from the FPS timer
            else:
                timed_frames += 1
            if max_frames and frames >= max_frames:
                break
    finally:
        cap.release()
        if sink is not None:
            sink.release()
        cv2.destroyAllWindows()

    dt = time.perf_counter() - t0
    if timed_frames:
        note = f" (after {warmup_frames}-frame warm-up)" if warmup_frames else ""
        print(f"processed {timed_frames} frames in {dt:.1f}s{note}  ->  {timed_frames / dt:.1f} FPS")
    elif frames:
        print(f"processed {frames} frames, all within the {warmup_frames}-frame warm-up  ->  no timed FPS")
    if args.nms == "fixed":
        print(f"--nms fixed (k={args.nms_candidates}): most candidates above conf={conf} in a "
              f"frame: {max_candidates}; {truncated_frames} of {frames} frames filled all k, "
              f"{unconverged_frames} needed more than {NMS_ITERS} NMS passes "
              f"(0 and 0 = identical to torchvision.ops.nms on every frame)")


if __name__ == "__main__":
    main()
