# FP16 Research - Accuracy vs. Speed Tradeoff
*Exported from Notion (page last edited 2026-09-16).*

Research done ahead of a potential Iteration 4 (FP16 precision), prompted by model execution now being the single largest remaining cost in the frame (\~55-58% of it, per Iteration 3's Next Steps). Covers: expected accuracy loss, whether retraining/QAT is relevant to FP16 specifically, an evaluation methodology, and which already-annotated datasets are available for it. No implementation yet - this is the research pass before deciding whether to build it.

---

## Expected accuracy loss from FP16

Consistently reported across sources as **negligible - typically under 0.5 mAP**, sometimes unmeasurable. NVIDIA's own TensorRT guidance treats this as close to a free lunch: *"if you lose more than 0.5 mAP on FP16, you should re-export with more workspace, a larger calibration set, or fall back one precision level"* - i.e. losing more than that is treated as a red flag, not an expected outcome. One study did find a \~1.9% drop on a specific dataset, so it isn't literally risk-free - which is exactly why this needs measuring on our own model/data rather than trusted on a general number.

### Why FP16 is a fundamentally lower-risk change than INT8

|  | FP16 | INT8 |
| --- | --- | --- |
| Calibration required? | No - dynamic range (±65,504) comfortably covers normalized image data and typical YOLO weight distributions | Yes - narrow integer range requires a calibration dataset to set per-layer scale/zero-point |
| Typical accuracy loss | \~0-0.5 mAP (negligible) | \~3-7% mAP50-95 on clean data |
| Typical speedup | Real, via Tensor Cores + half the memory bandwidth | Larger (\~1.5-3.3x), but with the larger accuracy cost above |

---

## Does retraining (QAT) apply here?

**Short answer: not really, and Ultralytics' own tooling reflects that.**

Quantization-Aware Training (QAT) fine-tunes a model with "fake quantization" in the training loop so its weights learn to tolerate reduced precision. Ultralytics added QAT support via NVIDIA's Model Optimizer - `model.train(..., quantize=8)`. The `8` is the tell: **this feature is built specifically for INT8, not FP16.**

This isn't an arbitrary gap - a recent benchmark found QAT recovers only **0.008-0.017 mAP50-95** over plain post-training INT8 quantization across model sizes, and **"buys almost nothing on the smallest model"** (nano-sized, like `yolov8n`) since simple calibration already works well there. If QAT barely moves the needle for INT8 on a nano model, it has essentially nothing to recover for FP16, where the starting loss is already near zero.

**Conclusion**: FP16 on `yolov8n` doesn't need retraining - that's the actual appeal of FP16 over INT8. QAT only becomes relevant if INT8 is pursued later as a separate, heavier follow-up (calibration dataset + fine-tuning run), not something to bundle into the FP16 experiment.

---

## Evaluation methodology

Ultralytics' built-in `model.val()` runs a model (any format - `.pt`, `.engine`, any precision) against an annotated dataset and reports standard COCO metrics (mAP50, mAP50-95) directly:

```py
from ultralytics import YOLO
model = YOLO("yolov8n_fp16.engine")
metrics = model.val(data="coco128.yaml")  # or coco.yaml
print(metrics.box.map, metrics.box.map50)
```

Plan: run once against the current FP32 engine, once against a new FP16 engine, same dataset - the mAP delta between the two runs is the accuracy cost, directly comparable against the throughput numbers already measured in Iteration 3.

---

## Already-annotated datasets available (zero setup)

|  | COCO128 (`coco128.yaml`) | Full COCO val2017 (`coco.yaml`) |
| --- | --- | --- |
| Size | 128 images, 7MB | 5,000 images, 36,781 boxes, 80 classes (20.3GB combined train+val download) |
| Auto-downloads? | Yes, on first use | Yes, on first use |
| Purpose | Fast sanity check / pipeline smoke test - not a rigorous benchmark | The actual standard benchmark - YOLOv8n's published 37.3 mAP is measured against this |

**Recommended order**: COCO128 first as a quick go/no-go check (minutes, catches anything badly broken), then a full COCO val2017 pass for the real, citable mAP-vs-speed number if COCO128 looks healthy.

---

## Recommended plan (not yet started)

1. Re-export `yolov8n` with `half=True` to get a separate FP16 `.engine`.
2. **Verify, don't trust the export log** (same discipline as the `end2end=True` false-positive found in Iteration 3) - confirm via kernel names that FP16 (`f16f16`) kernels are actually being used, not just TF32/FP32 ones.
3. Run `model.val()` on both the existing FP32 engine and the new FP16 engine against COCO128, then COCO val2017 - record the mAP delta.
4. If the delta is within the expected \~0.5 mAP: swap the FP16 engine into `infer_cuda_graph_custom_nms.py` (no other code changes needed - CUDA Graph, GPU preprocessing, and NMS are all precision-agnostic) and profile with the standard `nsys` methodology, same as every other iteration.
5. If the delta is larger than expected: investigate before proceeding (per NVIDIA's own guidance) rather than accepting it - re-export with more workspace, or treat it as a real finding worth documenting rather than assuming FP16 was misconfigured.

## Sources

- [What is Quantization-Aware Training (QAT)? \| Ultralytics](https://www.ultralytics.com/glossary/quantization-aware-training-qat)
- [Support Quantization Aware Training with NVIDIA Model Optimizer · PR #22198 · ultralytics/ultralytics](https://github.com/ultralytics/ultralytics/pull/22198)
- [Understanding FP32, FP16, and INT8 Precision in Deep Learning Models](https://medium.com/@vishalindev/understanding-fp32-fp16-and-int8-precision-in-deep-learning-models-why-int8-calibration-is-5406b1c815a8)
- [Accuracy Considerations — NVIDIA TensorRT Docs](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html)
- [COCO128 Detection Dataset \| Ultralytics Docs](https://docs.ultralytics.com/datasets/detect/coco128)
- [COCO Detection Dataset \| Ultralytics Docs](https://docs.ultralytics.com/datasets/detect/coco)
- [Ultralytics YOLO Evolution overview (arXiv)](https://arxiv.org/pdf/2510.09653)
