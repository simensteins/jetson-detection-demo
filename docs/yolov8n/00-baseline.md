# Baseline Performance - YOLOv8n FP32 .engine

*Exported from Notion (page last edited 2026-09-13).*

> These numbers were measured on `data/vtest.avi` (768×576) in September 2026, before the measurement protocol was fixed (power mode and clocks not recorded). The final, comparable numbers are in [results-paper25w.md](../results-paper25w.md).

Reference baseline (`yolov8n.pt` COCO -> TensorRT `.engine`, unmodified Ultralytics `model.predict()`) that every optimization iteration is compared against. All numbers measured directly from `nsys` sqlite exports, not estimated.

**Methodology**: local file source (`data/vtest.avi`), not RTSP - deterministic frame sequence, no network jitter. `sudo jetson_clocks` locked before every capture (see reproducibility check below for why this matters).

## Config

| Base model | `yolov8n.pt` (COCO-pretrained) |
| --- | --- |
| Export | TensorRT `.engine`, FP32/TF32 (confirmed via kernel names: `f32f32_tf32f32_f32`, no `f16f16`) |
| Input resolution | 640x640 |
| Confidence threshold | 0.25 |
| Source | `data/vtest.avi` (local file) |
| Hardware | Jetson Orin Nano, GPU "Orin" (GA10B), 8 SMs, \~7.3GB, JetPack 7.2.1 |
| GPU clocks | Locked via `jetson_clocks` |

## Reproducibility check

Same config captured 3x back-to-back. Run 1 (cold start, before `jetson_clocks` finding) was a real outlier; runs 2-3 agree to within 0.03%.

| Run | Inference mean | Stdev |
| --- | --- | --- |
| 1 (cold) | 25758.69us | 684.41us |
| 2 | 24896.74us | 359.50us |
| 3 | 24903.23us | 386.83us |

Run 1's gap was concentrated in GPU kernel execution (+500us) and idle gaps (+250-300us), not CPU dispatch - pointing at GPU clock ramp-up (Jetson's default DVFS governor doesn't start at max clock). **Fix: `sudo jetson_clocks` before every profiling run, from here on.**

## Official baseline numbers (pooled runs 2+3, n=1100 frames)

### Pipeline stages

| Stage | Mean | Median | Stdev | Min | Max |
| --- | --- | --- | --- | --- | --- |
| decode | 2.83ms | 2.78ms | 0.52ms | 2.49ms | 15.77ms |
| **inference** | **24.90ms** | 24.81ms | 0.37ms | 24.41ms | 28.28ms |
| draw | 2.75ms | 2.78ms | 0.40ms | 1.73ms | 4.46ms |

### Inference breakdown

| Component | Mean | Share |
| --- | --- | --- |
| Pre-kernel (Python/setup) | 6.90ms | 27.7% |
| GPU kernel execution | 10.01ms | 40.2% |
| GPU idle gaps between kernels | 7.50ms | 30.1% |
| Post-kernel (Python/postproc) | 0.49ms | 2.0% |
| **Total** | **24.90ms** | **100.0%** |

| Overlapping CUPTI detail (informational, not additive) | Mean | % of inference |
| --- | --- | --- |
| CUDA Runtime API dispatch (CPU-side) | 8.10ms | 32.5% |
| Synchronization | 1.78ms | 7.1% |
| Memcpy | 0.10ms | 0.4% |
| Memset | 0.01ms | 0.0% |

Kernel count: exactly 223 on every one of the 1100 pooled frames (stdev = 0.00) - the compute graph is fully static regardless of frame content.

## Reproducing this

Script: `scripts/analyze_nsys.py` (repo, commit `a318bec`)

```bash
sudo jetson_clocks
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia/nsight-systems/<version>/target-linux-tegra-armv8
export DEBUGINFOD_URLS=""
nsys profile --trace=cuda,nvtx -o reports/<name> python -m src.detect --config configs/default.yaml --source data/vtest.avi --no-display --max-frames 600
nsys export --type=sqlite -o reports/<name>.sqlite reports/<name>.nsys-rep
python3 scripts/analyze_nsys.py reports/<name>.sqlite 50
```

Use `--max-frames 600`, `skip_frames 50`, local file source, `jetson_clocks` for every future iteration so results stay directly comparable.

## Next iterations to compare against this baseline

- CUDA Graph capture (target: the 7.50ms idle-gap + most of the 8.10ms dispatch overhead)
- FP16 precision
- Combinations of the above

---

## Appendix: original RTSP-based measurement (superseded)

Single run, live RTSP stream, GStreamer hardware decode, no `jetson_clocks`, n=550. Kept for reference only - local file is now the standard methodology (see top of page).

| Stage | Mean | Stdev | Max |
| --- | --- | --- | --- |
| decode | 4.38ms | 6.15ms | 135.10ms (wifi jitter) |
| inference | 26.20ms | 1.35ms | 38.38ms |
| draw | 2.82ms | 0.40ms | 4.33ms |

Inference mean (26.20ms) is within \~5% of the local-file baseline (24.90ms) - confirms `inference` is largely independent of source/frame content; `decode` is the only stage meaningfully affected by the network.
