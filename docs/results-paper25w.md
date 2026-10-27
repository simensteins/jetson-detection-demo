# Results: paper set, 25 W (YOLOv8n + M15)

Both models, every optimization stage, one profiling session on 25.10.2026. These are the numbers for the paper; they replace the earlier per-iteration measurements. M15's last two stages were re-run on 27.10.2026 in the same order as YOLOv8n (custom NMS before FP16), under the same conditions (set `paper25w_v2`).

> **Measurement conditions**
> - Jetson Orin Nano Super, **25 W**, `sudo jetson_clocks` (GPU 918 MHz, CPU 1344 MHz, EMC 3199 MHz, all pinned)
> - **Desktop stopped** (`systemctl isolate multi-user.target`), nothing else running; see [measurement-protocol.md](measurement-protocol.md)
> - Video `data/testdata.mkv`, 1920×1080, 600 frames per run, first 50 skipped as warm-up
> - **No throttling:** over-current counters 0 before and after
> - Script: `scripts/profile_paper_set.sh` (git `73c9f3b`; M15's last two stages: `scripts/profile_m15_nms_fp16.sh`, git `834c784`), logs in `reports/nsys_paper_reports/logfiles/`

## Summary

| Stage | YOLOv8n FPS | M15 FPS |
|---|---|---|
| Baseline | 49.5 | 19.1 |
| + CUDA Graph | 56.6 | 19.6 |
| + GPU preprocessing | 69.5 | 20.2 |
| + Custom NMS | 72.7 | 20.3 |
| + FP16 | **86.7** | **40.9** |
| **Total speed-up** | **1.75×** | **2.14×** |

FPS = 1000 / mean frame time (decode start to next decode start).

## Key findings
- **Amdahl in action.** YOLOv8n's model is small, so removing overhead (graph, preprocessing, NMS) gives 1.47×. M15's model is 83 % of the frame, so the same steps give only 1.06×.
- **FP16 is the big lever for both:** model GPU time 9.2 → 5.0 ms (YOLO) and 44.4 → 19.6 ms (M15).
- **Custom NMS on YOLO evens out frame times:** median barely changes, but p95 drops 18.6 → 14.1 ms. Ultralytics' NMS costs more on frames with more detections.
- **Custom NMS on M15: same mean, no slow frames.** Captured in the CUDA graph and sized to 100 candidates, it costs ~0.5 ms of GPU time on every frame, but frames with detections no longer cost extra: p99 58.4 → 50.5 ms, worst frame 190 → 56 ms, 16 CUDA API calls on every frame (torchvision: 19 to 242). The first attempt (run outside the graph, sized for 1000 candidates) was slower: 40.3 → 34.6 FPS.
- **YOLO FP16 is now CPU-bound**, not GPU-bound: further gains need the remaining per-op Python launches (preprocessing, NMS) inside the CUDA graph.

## YOLOv8n

| Stage | Frame (ms) | FPS | Median / p95 (ms) | Pre-kernel CPU | GPU kernels | GPU idle | CPU dispatch (active) |
|---|---|---|---|---|---|---|---|
| Baseline | 20.19 | 49.5 | 19.2 / 23.0 | 3.73 | 8.38 | 3.58 | 4.19 |
| + CUDA Graph | 17.66 | 56.6 | 16.6 / 21.8 | 5.90 | 7.74 | 0.71 | 2.00 |
| + GPU preprocessing | 14.39 | 69.5 | 13.4 / 18.6 | 1.49 | 8.85 | 0.85 | 1.55 |
| + Custom NMS | 13.75 | 72.7 | 13.7 / 14.1 | 1.66 | 9.16 | 0.07 | 2.49 |
| + FP16 | 11.53 | 86.7 | 11.5 / 12.1 | 1.66 | 5.01 | 1.90 | 2.55 |

All times in ms per frame. GPU kernels include the GPU preprocessing kernels from stage 3 on.

![Frame time per stage. 49.5 → 86.7 FPS.](../reports/figures/paper25w/paper25w_yolo_01_frame_time.png)
*Frame time per stage. 49.5 → 86.7 FPS.*

![GPU idle time between kernels. The CUDA graph removes most of it; in FP16 it grows again because the GPU now waits on the CPU.](../reports/figures/paper25w/paper25w_yolo_02_idle_gap.png)
*GPU idle time between kernels. The CUDA graph removes most of it; in FP16 it grows again because the GPU now waits on the CPU.*

![Where the inference time goes. The CUDA graph removes idle gaps, GPU preprocessing removes CPU pre-kernel time, FP16 halves the model.](../reports/figures/paper25w/paper25w_yolo_03_time_breakdown.png)
*Where the inference time goes. The CUDA graph removes idle gaps, GPU preprocessing removes CPU pre-kernel time, FP16 halves the model.*

![CPU time in CUDA API calls, split into active dispatch and waiting on the GPU.](../reports/figures/paper25w/paper25w_yolo_04_dispatch_time.png)
*CPU time in CUDA API calls, split into active dispatch and waiting on the GPU.*

![One real (median) frame per stage, to scale: GPU kernels vs CPU API calls.](../reports/figures/paper25w/paper25w_yolo_05_cpu_gpu_swimlane.png)
*One real (median) frame per stage, to scale: GPU kernels vs CPU API calls.*


## M15

Stages 1–3 from the 25.10 session, stages 4–5 from the 27.10 session (same protocol, desktop stopped, no throttling).

| Stage | Frame (ms) | FPS | Median / p95 / p99 (ms) | Worst frame (ms) | Pre-kernel CPU | GPU kernels | GPU idle | CPU dispatch (active) |
|---|---|---|---|---|---|---|---|---|
| Baseline | 52.43 | 19.1 | 51.6 / 54.2 / 61.3 | 195.4 | 3.97 | 43.81 | 1.63 | 5.59 |
| + CUDA Graph | 50.93 | 19.6 | 50.2 / 52.0 / 60.0 | 191.7 | 4.30 | 43.03 | 0.61 | 1.71 |
| + GPU preprocessing | 49.59 | 20.2 | 48.8 / 51.2 / 58.4 | 190.3 | 1.61 | 44.38 | 0.68 | 1.41 |
| + Custom NMS (in graph, k = 100) | 49.34 | 20.3 | 49.3 / 49.8 / 50.5 | 55.7 | 1.59 | 44.91 | 0.03 | 0.43 |
| + FP16 | 24.44 | 40.9 | 24.4 / 24.7 / 25.9 | 29.9 | 1.56 | 20.05 | 0.02 | 1.13 |

- **Custom NMS:** the mean barely moves, because the NMS kernels add ~0.5 ms of GPU time to every frame while the torchvision path only cost extra on the 25 of 550 frames with a detection. What changes is the spread: the slow frames are gone (std 6.5 → 0.4 ms), the GPU never waits for the CPU (idle 0.68 → 0.03 ms), and every frame makes the same 16 CUDA API calls.
- **FP16:** halves the model (44.9 → 20.0 ms of GPU time), as in the earlier order.
- **Production relevance:** in the earlier order, FP16 with torchvision NMS took ~30 ms on a frame with a detection against 24 ms on an empty one. Now every frame takes 24.4 ms, whatever it contains. On a video with targets in every frame, the gain should be much larger than on this test video (not measured yet).

![Frame time per stage.](../reports/figures/paper25w_v2/paper25w_v2_m15_01_frame_time.png)
*Frame time per stage.*

![GPU idle time between kernels: small in every stage, because M15 keeps the GPU busy; with the NMS in the graph it is nearly zero.](../reports/figures/paper25w_v2/paper25w_v2_m15_02_idle_gap.png)
*GPU idle time between kernels: small in every stage, because M15 keeps the GPU busy; with the NMS in the graph it is nearly zero.*

![Where the inference time goes: GPU kernel execution dominates every stage.](../reports/figures/paper25w_v2/paper25w_v2_m15_03_time_breakdown.png)
*Where the inference time goes: GPU kernel execution dominates every stage.*

![CPU time in CUDA API calls: almost all of it is the CPU waiting for the GPU.](../reports/figures/paper25w_v2/paper25w_v2_m15_04_dispatch_time.png)
*CPU time in CUDA API calls: almost all of it is the CPU waiting for the GPU.*

![One real (median) frame per stage, to scale.](../reports/figures/paper25w_v2/paper25w_v2_m15_05_cpu_gpu_swimlane.png)
*One real (median) frame per stage, to scale.*

### Earlier order (25.10): FP16 before custom NMS

| Stage | Frame (ms) | FPS | Median / p95 / p99 (ms) | Worst frame (ms) | GPU kernels |
|---|---|---|---|---|---|
| + FP16 (torchvision NMS) | 24.81 | 40.3 | 24.0 / 26.4 / 33.6 | 186.4 | 19.59 |
| + Custom NMS (eager, k = 1000) | 28.90 | 34.6 | 28.8 / 29.2 / 30.5 | 37.6 | 22.55 |

The NMS run outside the graph and sized for 1000 candidates (1000×1000 IoU matrix, 112 kernels, 3 ms GPU) was slower, although the video never has more than 3 candidates per frame. Figures: [`reports/figures/paper25w/`](../reports/figures/paper25w/) (`paper25w_m15_*`).

## Reproduce
- Jetson (desktop stopped, 25 W, `sudo jetson_clocks`): `bash scripts/profile_paper_set.sh`
- Jetson, M15's last two stages only: `bash scripts/profile_m15_nms_fp16.sh`
- Mac: `python3 scripts/plot_compare_reports.py --name paper25w_yolo --outdir reports/figures/paper25w --reports <5 yolo .sqlite> --labels ...`; M15: `--name paper25w_v2_m15 --outdir reports/figures/paper25w_v2` with `paper25w_m15_{baseline,graph,gpupre}` and `paper25w_v2_m15_{nms,fp16}`. The summary CSV includes the frame-time median, p95, p99, worst frame and std.
- Figures, summary CSVs and logs are in git: `reports/figures/paper25w/`, `reports/figures/paper25w_v2/`, `reports/nsys_paper_reports/logfiles/`.

## Related
- How the numbers were measured and the pitfalls found on the way: [measurement-protocol.md](measurement-protocol.md)
- YOLOv8n iterations: [yolov8n/](yolov8n/) · M15: [m15.md](m15.md)
- Per-run console output and the session log (clocks, power mode, over-current counters): [`reports/nsys_paper_reports/logfiles/`](../reports/nsys_paper_reports/logfiles/)
