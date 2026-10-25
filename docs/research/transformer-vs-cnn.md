# TRANSFORMER vs CNN
*Exported from Notion (page last edited 2026-10-01).*

# RF-DETR nano vs. yolov8n: TensorRT baselines

**Goal:** check whether the Iteration 1–4 fixes that sped up YOLO also help RF-DETR.\
This session: set up RF-DETR nano on the Jetson and measure the TensorRT baselines and more importantly run iteration 1 and 2 (CUDA graph + GPU preprocessing) on RF-DETR to see if the gains are transferable to the latest architecture of the models trained on SAR data.

01.10 → Just found out RF-DETR never preprocess on CPU so Claude Code fuckup

## Setup

- Branch: `add-rf-detr-model-to-experiments`
- Models: `yolov8n.engine` (Ultralytics, 640×640) vs. `rfdetr-nano` (rfdetr 1.11.1, fixed 384×384), both COCO-pretrained
- YOLO run through `src/detect.py` with the library's own `predict()`: TensorRT, FP32,\
  **no CUDA graph / GPU preprocessing** (the TensorRT-engine baseline the YOLO iterations started from).
- RF-DETR baseline run through `scripts/infer_rfdetr_trt.py --no-graph --preprocess cpu`
- Source: data/vtest.avi, conf 0.25, 600 frames, first 50 excluded as warm-up
- `jetson_clocks` on

## Results (average of 3 runs, shuffled order)

| Model | Headless | With display |
| --- | --- | --- |
| yolov8n (TensorRT) | 11 s → \~49 FPS (\~20 ms/frame) | 13 s → \~41 FPS (\~24 ms/frame) |
| rfdetr-nano (TensorRT) | 18 s → \~30 FPS (\~33 ms/frame) | 20 s → \~27 FPS (\~37 ms/frame) |

## Findings

- Showing the video adds a roughly fixed \~4 ms per frame, whichever model runs.\
  That's a small share now, but it will be a large share once the iterations make inference faster.
- RF-DETR nano is \~1.6× slower than yolov8n with TensorRT with equal conditions, despite a smaller input (384 vs. 640). This means that RF-DETR nano's network is heavier to compute than YOLOv8n's on this Jetson in FP32. Accuracy isn't included so this isn't a general statement about RF-DETR vs. YOLO.
- Considering the latest model used in the VM is called RF-DETR Large, it will likely be considerably slower on the Jetson than the nano numbers here, but the idea of this experiment is to compare a CNN-based objection detection model with transformer based model and see if same optimization could be done on the small version (not exact replica of the SAR project’s latest model)

## RF-DETR (TensorRT FP32) – nsys iteration results

**Note:** The iteration 3 is not applicable to RF-DETR nano model (it is for the actual latest project model) and FP16 can be explored but lower precision typically has more effect on accuracy in Transformer-based models vs. CNN (YOLO.

#### Setup:

`jetson_clocks` on, `data/vtest.avi`, `nsys profile --trace=cuda,nvtx --cuda-graph-trace=node`, 600 frames headless, `scripts/analyze_nsys.py` with 50-frame warm-up skip (n=550 analyzed). Reports: `reports/rfdetr_trt_baseline`, `rfdetr_trt_graph`, `rfdetr_trt_graph_gpupre`.

### Pipeline stages

| Stage | Baseline | + CUDA graph | + GPU preprocessing |
| --- | --- | --- | --- |
| decode | 1.41 ms | 1.38 ms | 1.35 ms |
| **inference** | **32.86 ms** | **28.44 ms (−13.4%)** | **20.11 ms (−29.3% vs. graph, −38.8% vs. baseline)** |
| draw | 0.88 ms | 0.88 ms | 0.64 ms |
| **Total per frame** | 35.2 ms (\~28 FPS) | 30.7 ms (\~33 FPS) | 22.1 ms (\~45 FPS) |

### Inside "inference"

| Component | Baseline | + CUDA graph | + GPU preprocessing |
| --- | --- | --- | --- |
| CPU preprocessing (pre-kernel) | 10.35 ms | 10.45 ms | 0.54 ms |
| GPU kernel execution | 18.71 ms | 17.97 ms | 18.54 ms |
| GPU idle gaps between kernels | 3.54 ms (10.8%) | ≈ 0 ms | 0.91 ms |
| Postprocessing (post-kernel) | 0.26 ms | 0.14 ms | 0.13 ms |
| Kernels per frame | 286 | 286 | 292 |

### Findings: CUDA graph and GPU preprocessing, YOLO vs. RF-DETR

**Caveat: YOLO's Iteration 1 changed more than the graph.**\
The YOLO baseline went through Ultralytics' `predict()`, and Iteration 1 replaced that with a simpler hand-written pipeline (own letterbox, direct engine call). Part of YOLO's −22% probably comes from removing Ultralytics' Python overhead, not from the graph itself. RF-DETR's baseline already ran in a lean script (`scripts/infer_rfdetr_trt.py --no-graph --preprocess cpu`), so its −13% is a cleaner measure of the graph effect alone.

**GPU preprocessing works at least as well for RF-DETR as for YOLO.**\
RF-DETR saves more in absolute terms (8.3 vs. 6.5 ms), and the relative gain is almost the same (−29% vs. −34%).

**RF-DETR had more to gain from GPU preprocessing, because its CPU preprocessing was more expensive.**\
It does ImageNet normalization, subtracting a mean and dividing by a standard deviation per channel, on float arrays in numpy, while YOLO only divides by 255. That extra arithmetic is exactly what the GPU does much faster
