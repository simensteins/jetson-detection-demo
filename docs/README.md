# Documentation

Optimizing object-detection pipelines on a Jetson Orin Nano Super with Nsight Systems and Nsight Compute. This folder replaces the project's Notion workspace (exported pages are marked "Exported from Notion").

## Start here
- **[Results: paper set, 25 W](results-paper25w.md)** – final numbers and figures for both models
- [Measurement protocol](measurement-protocol.md) – how to measure so numbers are comparable, and the pitfalls we hit
- [Scripts and tools](tooling.md)

## YOLOv8n – iterations
| | |
|---|---|
| [Baseline](yolov8n/00-baseline.md) | Ultralytics `predict()` on a TensorRT engine |
| [Iteration 1 – CUDA Graph](yolov8n/01-cuda-graph.md) | |
| [Iteration 2 – GPU preprocessing](yolov8n/02-gpu-preprocessing.md) | |
| [Iteration 3 – Custom fixed-shape NMS](yolov8n/03-custom-nms.md) | |
| [Iteration 4 – FP16](yolov8n/04-fp16.md) | |

Iteration pages 0–3 hold the September measurements on `vtest.avi`; the comparable numbers are in the results page.

## Other models
- [M15](m15.md) – the stakeholder's YOLO11-based detector: pipeline, results, findings
- [RF-DETR Large](rfdetr-large.md) – transformer detector, power/throttling tests

## Research and guides
- [Nsight Systems & Compute – beginner's guide](research/nsight-guide.md) · [Replica setup](research/replica-setup.md)
- [Nsight Compute on YOLO](research/nsight-compute-yolo.md)
- [FP16 research – accuracy vs. speed](research/fp16-research.md)
- [Transformer vs. CNN](research/transformer-vs-cnn.md)

## Paper
- [Paper](paper/paper.md) · [Remaining work before the paper](paper/remaining-work.md)

## Meetings
[26.08](meetings/2026-08-26.md) · [02.09](meetings/2026-09-02.md) · [09.09](meetings/2026-09-09.md) · [16.09](meetings/2026-09-16.md) · [14.10](meetings/2026-10-14.md)

## Not included
The repository is public. VM access details and the stakeholder's internal documents (model cards, training reports, accuracy on their test data) are deliberately not in this folder.
