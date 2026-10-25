# WIP: Trying out RF-DETR large
*Exported from Notion (page last edited 2026-10-02).*

## Setup

- Model: RF-DETR Large, COCO-pretrained (`rf-detr-large-2026.pth`, rfdetr 1.11.1), input 704×704 – same architecture as the project's SAR/IR detector, but not its fine-tuned weights
- Engine: TensorRT, strict FP32 (`scripts/export_rfdetr.py --size large` → `models/rfdetr-large_fp32-strict.engine`)
- Script: `scripts/infer_rfdetr_large_trt.py` – GPU preprocessing as in rfdetr's own `predict()`; only variable is `-no-graph` (TensorRT baseline vs. + CUDA graph)
- Platform: Jetson Orin Nano, power mode **25 W**, `jetson_clocks` on
- Source: data/vtest.avi, 600 frames, first 50 excluded as warm-up

## Findings so far

- **Over-current throttling:** running the Large engine on 600 frames triggered the system warning *"nvpmodel: system throttled due to over-current"*.
  - The Jetson lowers its clocks in hardware when the current draw exceeds the limit for the power budget; `jetson_clocks` cannot override this.
  - YOLOv8n and RF-DETR nano did not trigger it under the same settings – Large loads the GPU considerably more.
- **Consequence:** the throttled run is **not valid as a measurement** (part of it ran at reduced clocks). No Large FPS numbers recorded yet.
- **Relevance for the hardware evaluation:** at 25 W with locked clocks, the Orin Nano cannot run RF-DETR Large continuously without hitting its current limit. This is a power-budget constraint, not only a compute constraint – relevant for power supply and heat in the helicopter installation.

## Open questions / next steps

1. Check the power supply: official 19 V barrel adapter, or something weaker / USB-C?
2. Quantify the throttling: log `tegrastats` during a run and check whether the GPU frequency drops briefly or stays down.
3. Decide the power mode:
   - No or momentary throttling at 25 W → keep 25 W (earlier YOLO/nano results stay valid), include the log as documentation.
   - Clear throttling at 25 W → switch to 15 W and **re-run all models** (YOLO, nano, Large) in that mode, so no model is throttled and results remain comparable.
4. Then: TensorRT baseline vs. + CUDA graph for Large (3 runs each, shuffled), plus nsys reports.

## Update 13.10 (written after the Notion export limit)

### 13.10 – Power-mode test

**Power supply:** stock NVIDIA 19 V / 2.37 A adapter (≈45 W). This is *not* the cause. 45 W covers the whole dev kit (carrier board, fan, SSD, USB devices, conversion losses, margin). The module itself is limited by its power mode and by its built-in over-current protection, whatever adapter is used.

**Test:** RF-DETR Large + CUDA graph, strict FP32 engine, `jetson_clocks` on, `data/vtest.avi`, 600 frames headless, first 50 excluded. One run per mode, with `tegrastats` and the over-current counters logged (`reports/power_test/` on the Jetson, not yet analysed).

| Power mode | FPS | Throttled (over-current popup) | Valid measurement |
|---|---|---|---|
| 15 W | 5.4 | No | **Yes** |
| 25 W | 7.8 | Yes | No |
| MAXN SUPER | 8.4 | Yes | No |

**Engine rebuilt on the Jetson** (in 15 W): the TensorRT warning *"Using an engine plan file across different models of devices"* is gone. Speed unchanged (5.3 FPS vs. 5.4), so the old engine was mismatched but didn't hurt performance. GPU reports all 8 SMs in each mode, so the power modes don't switch off part of the GPU.

#### Why the lower power mode "works"

- The model runs in every mode. What differs is whether the clocks stay where `jetson_clocks` puts them.
- At 25 W and MAXN SUPER, Large's continuous GPU load at the pinned top clocks draws more current than the module's hardware protection allows, so the chip cuts its own clocks partway through the run. Even MAXN SUPER (no fixed power budget) throttles, so Large hits the module's hard over-current ceiling.
- At 15 W the top clock is lower, and lower clocks also run at lower voltage. Power falls faster than speed, so the current stays under the limit and the clocks never get cut.
- **Throttled modes are still faster** (8.4 and 7.8 vs. 5.4 FPS). 15 W is the right mode for **measuring** (stable, reproducible clocks), not necessarily for **deploying**. In the helicopter, MAXN SUPER without `jetson_clocks` (the chip managing its own clocks) would likely be faster even with throttling.

#### Is RF-DETR Large too big for the Orin Nano?

In its current form (strict FP32, batch 1): **yes, by a wide margin.**

- The model always takes a 704×704 input. The source is resized to that, so one inference costs the same at 768×576 or full HD.
- Full HD costs more because of **tiling**: the SAR pipeline splits each 1920×1080 frame into ~8–13 overlapping 704×704 tiles plus one downscaled full view, so roughly 9–14 inferences per video frame.

| | Per 704×704 inference | Full-HD frame (~11 inferences) |
|---|---|---|
| 15 W (clean) | ~190 ms (5.3 FPS) | ~2 s → ~0.5 FPS |
| MAXN SUPER (throttled) | ~120 ms (8.4 FPS) | ~1.3 s → ~0.8 FPS |
| Target: 30 FPS | 33 ms | 33 ms |

A single tile at 30 FPS is 4–6× away; the real tiled pipeline is roughly 40–60× away.

**What could close part of the gap**, roughly easiest first:

1. **Lower precision.** Strict FP32 leaves the tensor cores idle for the main matmuls. TF32 uses them; FP16 is typically faster again. The model was trained with bfloat16 AMP, so FP16 accuracy has a fair chance of holding, but transformers are precision-sensitive, so accuracy must be checked. INT8 goes further, with more accuracy risk.
2. **Batch the tiles.** The RF-DETR nano kernel analysis showed the GPU is underused at batch 1; running all tiles as one batch should lower cost per tile.
3. **Fewer inferences per frame.** Less tile overlap, skip the full view, or detect every Nth frame and track in between.
4. **A smaller model**, fine-tuned on SAR data (see *[removed: internal]*).
5. **Bigger hardware** (Orin NX, AGX Orin).

**Open question:** what does "real-time" have to mean for SAR? 30 FPS on full-HD tiled video with Large on an Orin Nano isn't realistic even with all the above. A few detections per second plus tracking may be enough. The answer decides whether the question is "make Large fast enough" or "which model and hardware meet the target".

#### Other notes

- **PyTorch warning:** the venv has a generic pip PyTorch (`2.13.0+cu130`), not NVIDIA's Jetson build ("No published PyTorch CUDA builds … support this GPU", CC 8.7). It runs, but its GPU code isn't built for this chip. This affects only the PyTorch pre/postprocessing, not the TensorRT model. Worth mentioning in the paper's setup section.
- **Comparability:** the YOLO engine computes in TF32 (see [Nsight Compute YOLO](research/nsight-compute-yolo.md)), the Large engine is strict FP32. Build `--precision fp32-tf32` for Large when comparing against YOLO/nano.

#### Status of the earlier next steps

- [x] Check the power supply – stock 45 W adapter, not the cause
- [x] Quantify the throttling – popup at 25 W and MAXN SUPER, none at 15 W. `tegrastats` logs captured, not yet analysed
- [x] Decide the power mode – **15 W for benchmarking**
- [ ] TensorRT baseline vs. + CUDA graph for Large at 15 W (3 runs each, shuffled) + nsys reports
- [ ] Build the `fp32-tf32` Large engine for comparison with YOLO/nano
- [ ] Re-run YOLO and nano at 15 W so all three models are comparable
- [ ] Short FP16 test of Large (speed + accuracy)
