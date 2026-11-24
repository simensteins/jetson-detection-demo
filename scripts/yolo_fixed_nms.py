"""Fixed-shape, class-aware Cluster-NMS for YOLOv8n's raw output - the same
algorithm as M15's postprocess_fixed (scripts/infer_m15_trt.py), extended to
80 classes. Used by Iterations 3 and 4 (infer_cuda_graph_custom_nms.py,
infer_cuda_graph_fp16.py) and by scripts/eval_yolo_coco.py.

Every call issues the same operations on tensors of the same shapes,
whatever the image contains: no boolean filtering, no .nonzero(), nothing
that makes the CPU wait for the GPU. Rows that are not detections have score
0 and are dropped by the caller after its single copy to the CPU.

  1. Per candidate the best class and its score (as Ultralytics' NMS with
     multi_label=False), then the k highest scores (torch.topk, sorted).
     k = 300 by default, Ultralytics' max_det. `classes` (a fixed index
     tensor, e.g. [0] for person) restricts this to those classes, as
     Ultralytics' classes= does, so all k slots go to them.
  2. Class-aware suppression with the box-offset trick Ultralytics and
     torchvision use: each box is shifted by class * CLASS_OFFSET, so boxes
     of different classes never overlap.
  3. Cluster-NMS (Zheng et al.): suppresses[i, j] says a higher-scored box i
     overlaps box j above the IoU threshold. A box is dropped only if a
     higher-scored box that is ITSELF kept suppresses it. One pass from "all
     kept" is Fast NMS (YOLACT), which over-suppresses chains of overlapping
     boxes; repeating the pass converges to exactly standard (greedy) NMS.
     A fixed number of passes keeps the work fixed (no check for
     convergence, which would need a sync); `iters` passes are exact for
     chains of up to `iters` boxes.

topk runs in the output's dtype (FP16 for Iteration 4); the k selected rows
are cast to FP32 before any box maths, because box areas at 640x640 reach
409,600, beyond FP16's 65,504.
"""
from __future__ import annotations

import torch

CANDIDATES = 300   # Ultralytics' max_det
ITERS = 10         # Cluster-NMS passes, as M15's NMS_ITERS
MAX_DET = 300      # at most this many detections, highest scores first
CLASS_OFFSET = 7680.0  # Ultralytics' max_wh: larger than any box coordinate at 640x640


def box_iou_matrix(boxes_xyxy: torch.Tensor) -> torch.Tensor:
    """(K, 4) xyxy -> (K, K) pairwise IoU."""
    area = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
    x1 = torch.max(boxes_xyxy[:, None, 0], boxes_xyxy[None, :, 0])
    y1 = torch.max(boxes_xyxy[:, None, 1], boxes_xyxy[None, :, 1])
    x2 = torch.min(boxes_xyxy[:, None, 2], boxes_xyxy[None, :, 2])
    y2 = torch.min(boxes_xyxy[:, None, 3], boxes_xyxy[None, :, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    return inter / (area[:, None] + area[None, :] - inter).clamp(min=1e-9)


def cluster_nms(raw_output: torch.Tensor, conf_thres: float, iou_thres: float,
                k: int = CANDIDATES, iters: int = ITERS, max_det: int = MAX_DET,
                classes: torch.Tensor | None = None) -> torch.Tensor:
    """raw_output: (1, 4 + nc, N) YOLOv8 output, cxcywh boxes in input pixels.
    classes: optional 1-D long tensor of class indices to keep (on the same
    device). -> FIXED (k, 6) [x1, y1, x2, y2, score, cls], unused rows score 0."""
    pred = raw_output[0]
    if classes is None:
        scores_all, classes_all = pred[4:].max(dim=0)           # (N,), (N,)
    else:   # fixed index tensor: same shapes and kernels every call
        scores_all, idx = pred[4:].index_select(0, classes).max(dim=0)
        classes_all = classes[idx]
    topk_scores, topk_idx = torch.topk(scores_all, k=k)        # sorted descending
    topk_scores = topk_scores.float()
    topk_classes = classes_all[topk_idx].float()
    cx, cy, w, h = pred[:4, topk_idx].float().unbind(0)        # each (K,)
    boxes_xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)

    offset = (topk_classes * CLASS_OFFSET).unsqueeze(1)
    suppresses = torch.triu(box_iou_matrix(boxes_xyxy + offset) > iou_thres, diagonal=1)
    candidate = topk_scores > conf_thres
    keep = candidate
    for _ in range(iters):
        keep = candidate & ~(suppresses & keep.unsqueeze(1)).any(dim=0)
    keep = keep & (keep.cumsum(0) <= max_det)
    scores = torch.where(keep, topk_scores, torch.zeros_like(topk_scores))
    return torch.cat([boxes_xyxy, scores.unsqueeze(1), topk_classes.unsqueeze(1)], dim=1)
