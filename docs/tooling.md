# Scripts and tools

Run everything from the repo root. Inference and profiling run on the Jetson (venv active: `source .venv/bin/activate`); analysis and plotting run on the Mac.

## Inference – YOLOv8n (one script per iteration, each kept unchanged for comparison)

| Script | Iteration |
|---|---|
| `python3 -m src.detect` | Baseline: Ultralytics `model.predict()` on a TensorRT engine |
| `scripts/infer_cuda_graph.py` | 1 – forward pass captured as a CUDA graph |
| `scripts/infer_cuda_graph_gpu_preprocess.py` | 1b – + partial GPU preprocessing (superseded by 1c) |
| `scripts/infer_cuda_graph_gpu_resize.py` | 2 (1c) – + full GPU preprocessing incl. resize |
| `scripts/infer_cuda_graph_custom_nms.py` | 3 – + fixed-shape NMS, one GPU→CPU sync |
| `scripts/infer_cuda_graph_fp16.py` | 4 – + FP16 end to end (`--engine yolov8n_fp16_io.engine`) |

Common flags: `--config configs/default.yaml --source <file|rtsp://...> --no-display --max-frames N --warmup-frames N`.

## Inference – M15
`scripts/infer_m15_trt.py --engine <file.engine>` with stage flags `--no-graph`, `--preprocess cpu|gpu`, `--nms torchvision|fixed`, `--nms-candidates k`. See [m15.md](m15.md).

## Building engines (on the Jetson)

| Script | What |
|---|---|
| `scripts/export_yolo_fp16.py` | YOLOv8n FP16 engine with FP16 I/O from Ultralytics' ONNX |
| `scripts/build_m15_pt.py` | M15 training checkpoint → Ultralytics `.pt` |
| `scripts/export_m15.py` | M15 `.pt` → ONNX → TensorRT (`--precision fp32-tf32 \| fp32-strict \| fp16`) |
| `scripts/inspect_engine.py <file.engine> ...` | Print an engine's inputs/outputs (name, shape, dtype) |
| `scripts/inspect_autobackend.py` | Diagnostic: Ultralytics AutoBackend internals |

## Profiling (on the Jetson)

| Script | What |
|---|---|
| `scripts/profile_paper_set.sh [set-name] [all\|yolo\|m15]` | **All stages of both models (or one) in one controlled session.** See [measurement-protocol.md](measurement-protocol.md). |
| `scripts/profile.sh` | Old v1 wrapper (NVTX only, YOLO demo); kept for history |
| `scripts/nvtx_smoketest.py` | Checks that NVTX ranges reach nsys |

## Analysis and figures (on the Mac)

| Script | What |
|---|---|
| `scripts/analyze_nsys.py <report.sqlite> [skip]` | Per-frame breakdown of one report |
| `scripts/plot_compare_reports.py --name ... --reports ... --labels ...` | Five comparison figures + summary CSV for any set of stages |
| `scripts/plot_nsys_report.py <report.sqlite>` | Figures for a single report |
| `scripts/plot_paper_figures.py` | The original YOLOv8n figure set (hard-coded September numbers) |

Figures live in `reports/figures/<set>/`; see [`reports/figures/README.md`](../reports/figures/README.md).

## Accuracy evaluation (on the Jetson)

| Script | What |
|---|---|
| `bash scripts/download_coco_val.sh` | COCO val2017 images + annotations into `data/coco/` (~1 GB download) |
| `scripts/eval_yolo_coco.py [--max-images N] [--max-objects N] [--pipelines ...]` | mAP, P/R and false alarms per image at conf 0.25, and agreement with the baseline for each YOLOv8n stage (baseline, gpupre, customnms, fp16, plus fp16_ultranms to isolate FP16), overall and split by objects per image (0, 1–3, 4–10, >10). `--max-objects 10` keeps only images with at most 10 objects (the production scenario). Needs `pip install pycocotools`. Results: `reports/eval/yolo_coco[_max10obj]_summary.md` and `_buckets.md` (+ .csv) |

## Sources and sinks
`src/sources.py` (files, webcams, `rtsp://` via hardware decoding), `src/sinks.py` / `src/gstreamer.py` (`--sink rtp://...` / `rtsp://...` sends the annotated video on), `scripts/rtsp_server_mac.sh` (stream a file from the Mac).

## What is in git
Code, configs, architecture YAMLs, figures, summary CSVs and run logs. **Not** in git (`.gitignore`): engines, weights, ONNX files, videos, `.sqlite` / `.nsys-rep` / `.ncu-rep` profiler output. A `.sqlite` can always be regenerated from its `.nsys-rep` with `nsys export --type=sqlite`.
