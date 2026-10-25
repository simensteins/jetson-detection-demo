# Iteration 2 - GPU Preprocessing

*Exported from Notion (page last edited 2026-09-12).*

> These numbers were measured on `data/vtest.avi` (768×576) in September 2026, before the measurement protocol was fixed (power mode and clocks not recorded). The final, comparable numbers are in [results-paper25w.md](../results-paper25w.md).

**This iteration builds on and keeps Iteration 1 (CUDA Graph) in place - it is not a replacement or an independent change.** The engine is still invoked via captured graph replay exactly as in Iteration 1; this iteration additionally moves preprocessing (previously hand-written CPU code: `cv2.resize`, letterbox padding, BGR->RGB flip, HWC->CHW transpose, normalize) onto the GPU. Iteration 1 found preprocessing was the single largest cost in the frame (42.7%, 8.27ms) - larger than the model itself - with CUDA Graph's own idle-gap problem otherwise fully solved. This iteration addresses that.

Done in two steps, both measured and reported below: **2a** moved the reformatting operations (flip/normalize/transpose) to the GPU; **2b** moved the resize/letterbox-pad step to the GPU too. Both steps keep CUDA Graph from Iteration 1 unchanged.

---

## Why GPU preprocessing is so much faster on this exact hardware

### The hardware, confirmed for this specific board

This Jetson Orin Nano's GPU clock, captured directly from `nsys`'s own `TARGET_INFO_GPU` record during baseline profiling (1020 MHz), matches NVIDIA's **"Super" mode** specification exactly - not the original Orin Nano's factory specs. This board is running the boosted power/clock profile (a free JetPack software update NVIDIA shipped in late 2024). Specs below are for that mode, sourced from NVIDIA's own announcement:

|  | CPU (6-core Arm Cortex-A78AE) | GPU (Ampere, 1024 CUDA cores + 32 Tensor Cores) |
| --- | --- | --- |
| Clock | 1.7 GHz | 1020 MHz |
| Parallel execution units | 6 cores | 1024 CUDA cores |
| Memory bandwidth | 102 GB/s (shared with GPU - see below) | 102 GB/s (shared with CPU) |

Source: [NVIDIA - Jetson Orin Nano Developer Kit Gets a Super Boost](https://developer.nvidia.com/blog/nvidia-jetson-orin-nano-developer-kit-gets-a-super-boost/) (original, pre-Super specs for comparison: CPU 1.5GHz, GPU 635MHz, 68GB/s bandwidth - the Super update raised all three).

### The core-count gap is the real story, not clock speed

The CPU is actually clocked *higher* per-core than the GPU (1.7GHz vs 1.02GHz) - clock speed isn't where the CPU loses. It loses on raw parallelism: **1024 GPU cores vs 6 CPU cores, a \~170x difference in how many independent execution units can work simultaneously.** Each CPU core is individually more sophisticated (out-of-order execution, branch prediction, large caches) - good for complex, branchy, sequential code - but there are only 6 of them.

### Why this specific workload is exactly what that gap rewards

Preprocessing (resize, BGR->RGB flip, normalize, HWC->CHW transpose) is **embarrassingly parallel**: every output pixel's value depends only on itself (or a small local neighborhood, for resize) - there's no sequential dependency between computing pixel (0,0) and pixel (639,639). A 640x640x3 frame is \~1.23 million independent values needing near-identical treatment. This is precisely the workload class GPUs are architecturally built for (SIMT - thousands of simple cores each doing the same operation on different data simultaneously) and precisely the workload class where a CPU's sophistication (branch prediction, out-of-order execution) buys nothing, because there's no branching or sequential logic to exploit - just raw throughput on identical, independent operations. The 170x core-count gap translates almost directly into real speedup here, in a way it wouldn't for, say, running Python's own interpreter loop.

### Why the CPU->GPU handoff itself isn't a bottleneck here

Jetson uses a **unified memory architecture** - the CPU and GPU share the same physical LPDDR5 memory pool (102GB/s), unlike a desktop system where a discrete GPU has its own separate VRAM connected via PCIe. There's no PCIe bus to bottleneck the "transfer" - moving data from CPU-accessible to GPU-accessible memory happens over the SoC's internal memory fabric, which is fast and shares that same bandwidth pool. This is also why the pinned-memory staging buffer trick (Step 2a) works well here specifically - it lets the DMA engine move data directly through that fast internal path instead of staging an extra copy.

### Connecting this to what was actually measured

Total preprocessing dropped 8268us (CPU) -> 1641us (GPU), a \~5x realized speedup - real and substantial, but much smaller than the raw \~170x core-count ratio would suggest. That gap between theoretical and realized speedup is expected and worth being honest about: `cv2.resize` isn't naive scalar code, OpenCV's CPU implementation already uses SIMD instructions internally, so the CPU baseline wasn't as slow as a worst-case comparison would imply. There's also fixed overhead on the GPU side that doesn't shrink with more cores - kernel launch cost, dispatch overhead - which matters more for a workload this small (1.23M elements is enough to saturate a fraction of the GPU's capacity, but not enough to fully hide every fixed cost). The \~5x realized gain is the honest, measured number; \~170x is the raw hardware ratio that explains *why* GPU parallelism helps here at all, not a promise of what any given workload will actually achieve.

---

## Step 2a - GPU-side reformatting

### Implementation

Script: `scripts/infer_cuda_graph_gpu_preprocess.py` (repo). `scripts/infer_cuda_graph.py` (Iteration 1) kept unchanged for comparison - only preprocessing differs, everything else (including CUDA Graph capture/replay) identical.

Previous preprocessing did all reformatting on the CPU via numpy, then transferred a float32 tensor to the GPU. This version: `cv2.resize`/pad still runs on CPU (unavoidable - the frame originates there), but the *raw* uint8 HWC frame is transferred to the GPU via a pinned staging buffer immediately after, and all reformatting happens as GPU tensor ops. Rationale: transfers 1/4 the data (uint8 vs float32), pinned memory allows direct DMA transfer, and the reformatting itself is trivially parallel on GPU vs single-threaded CPU numpy.

**Correctness check**: ran with display, visually confirmed detections still look correct.

### A measurement correction made during analysis

Initial boundary detection (using "first kernel of any kind" as the model-execution start) showed model busy time increasing slightly (8437us vs Iteration 1's 7848us) - suspicious, since the same engine should cost the same regardless of how the input got there. Investigated directly: the new GPU-side preprocessing kernels (`index_elementwise_kernel`, `unrolled_elementwise_kernel`, `vectorized_elementwise_kernel`, `elementwise_kernel` - the flip/normalize/permute ops) run as ordinary, un-graph-captured kernels immediately *before* the graph's first real `_trt` kernel, and were being miscounted as model time. Fixed by using the first `_trt`-suffixed kernel specifically as the true model-span boundary. Corrected model busy time (7856.96us) matches Iteration 1 almost exactly, as it should - confirming CUDA Graph's contribution is untouched by this change.

### Results vs. Baseline and vs. Iteration 1 (n=550 frames, `--cuda-graph-trace=node`, corrected boundaries)

| Component | Step 2a (GPU reformat) | Iteration 1 (CUDA Graph, CPU preprocess) | Baseline (no CUDA Graph) |
| --- | --- | --- | --- |
| Preprocessing, total | 5654.45us | 8268.14us | (bundled inside `model.predict()`, not separable) |
| — CPU-side (cv2 resize/pad, dispatch) | 5074.35us | 8268.14us | — |
| — GPU-side kernels (flip/normalize/permute) | 580.10us | — | — |
| Model execution (busy) | 7856.96us | 7847.90us | \~10,000-11,000us (individually dispatched) |
| Model idle gap | 15.90us (0.20%) | 19.47us (0.2%) | \~7500-8000us (\~30%) |
| **Total inference** | **16725.30us** | 19348.85us | **24900us** |
| **Change vs. Baseline** | **-32.8%** | -22.3% | — |

(Sum of parts: 16716.09us vs measured 16725.30us - matches to within 9us.)

**Interpretation**: preprocessing dropped 31.6% vs. Iteration 1 (real, but smaller than hoped) - 90% of the remaining preprocessing time (5074us of 5654us) is still CPU-side `cv2.resize`/`copyMakeBorder`, untouched by this step. Model execution and CUDA Graph's idle-gap win both correctly unchanged, confirming this step only touched preprocessing.

---

## Step 2b - GPU-side resize (full preprocessing now on GPU)

### Research before implementing

Before writing code, researched the real options for GPU-side resize on this platform:

| Option | Available now? | Letterbox support | Verdict |
| --- | --- | --- | --- |
| `cv2.cuda` | No - needs 2-3hr from-source OpenCV rebuild, real risk of silent CPU fallback if `CUDA_ARCH_BIN` wrong for Orin (8.7) | Manual compose | Rejected - high complexity, high risk |
| GStreamer `nvvidconv` | Yes | **No** • genuine gap, even NVIDIA's own DeepStream `nvdspreprocess` plugin lacks it; real letterbox needs the full DeepStream SDK (`nvstreammux`) | Rejected - architecture change, not a tweak |
| **PyTorch `F.interpolate` • `F.pad`** | **Yes - already using torch** | Compose manually, well-documented pattern | **Chosen** • zero new dependencies, lowest complexity |
| NVIDIA VPI | Yes (JetPack 7.2 ships 4.1.3, Python bindings) | No built-in pad primitive either | Rejected - no found benefit over torch |
| NVIDIA DALI | Yes (aarch64 wheels exist) | Yes, documented exact recipe (Ultralytics' own guide) | Rejected - designed for batch/multi-stream throughput, this is single-stream batch=1 |

Known gotcha applied: `antialias=True` (closer to cv2's visual quality) measured \~5x slower in reported PyTorch benchmarks - used `antialias=False` for the real-time path.

### Implementation

Script: `scripts/infer_cuda_graph_gpu_resize.py` (repo). Prior scripts kept unchanged for comparison. CUDA Graph capture/replay from Iteration 1 still unchanged here too.

Relies on source frame resolution being constant across the run (true for a video file) - lets the resize scale and padding be computed **once at startup** rather than per-frame. Full-resolution raw frame (uint8, HWC, BGR) transferred to GPU via pinned staging buffer; resize (`F.interpolate`, bilinear, no antialias), letterbox pad (`F.pad`, constant 114/255), channel-flip, and normalize all happen as GPU tensor ops.

**Correctness check**: ran with display, visually confirmed detection boxes still correctly positioned - most correctness-sensitive change so far (resize math + coordinate-unletterbox logic in `draw_detections`), verified carefully.

### Results vs. Baseline and vs. Step 2a (n=550 frames, `--cuda-graph-trace=node`, corrected boundaries)

| Component | Step 2b (GPU resize, full GPU preprocess) | Step 2a (GPU reformat only) | Baseline (no CUDA Graph, no GPU preprocess) |
| --- | --- | --- | --- |
| Preprocessing, total | 1640.99us | 5654.45us | (bundled, not separable) |
| — CPU-side | 903.59us | 5074.35us | — |
| — GPU-side kernels | 737.40us | 580.10us | — |
| Model execution (busy) | 7861.98us | 7856.96us | \~10,000-11,000us |
| Model idle gap | 18.95us (0.24%) | 15.90us (0.20%) | \~7500-8000us (\~30%) |
| Gap (NMS setup) | 2462.95us | 2460.06us | n/a |
| NMS execution | 648.55us | 649.23us | n/a |
| Post-kernel | 79.93us | 79.50us | n/a |
| **Total inference** | **12836.91us** | 16725.30us | **24900us** |
| **Change vs. Baseline** | **-48.4%** | -32.8% | — |

(Sum of parts: 12713.35us vs measured 12836.91us - within \~1%.)

**Interpretation**: preprocessing dropped a further 71% (5654us -> 1641us) from moving resize onto the GPU - **80.2% total reduction from the original CPU-only implementation** (8268us -> 1641us). Model execution and CUDA Graph's idle-gap win remain correctly unchanged, confirming this step - like 2a - only touched preprocessing.

---

## Full picture, Baseline -> Iteration 1 -> Iteration 2

|  | Total inference | Change vs. Baseline |
| --- | --- | --- |
| Baseline (no CUDA Graph, CPU preprocess) | 24.90ms | — |
| Iteration 1 (+ CUDA Graph) | 19.35ms | -22.3% |
| Iteration 2, step 2a (+ GPU reformat) | 16.73ms | -32.8% |
| **Iteration 2, step 2b (+ GPU resize)** | **12.84ms** | **-48.4%** |

**Inference time has been essentially cut in half from the original baseline**, via two independent, individually-verified changes: CUDA Graph (Iteration 1, solved dispatch overhead/idle gaps) and GPU-side preprocessing (Iteration 2, solved the preprocessing cost that CUDA Graph's own fix made newly visible).

## The new bottleneck: NMS setup gap

With preprocessing down to 1.64ms, the 2.46ms gap between the model's last kernel and NMS's first kernel - present since Iteration 1, never addressed by either iteration - is now the clear second-largest cost (19.2% of the frame), bigger than preprocessing itself. NMS's actual GPU computation remains cheap (0.65ms); the cost is in getting there.

## Next steps

- **Investigate the NMS setup gap** (2.46ms, now the top unaddressed cost) - likely Python-side confidence thresholding/indexing in `non_max_suppression` before its kernels launch.
- Consider whether NMS itself is worth graph-capturing.
- Model execution (7.86ms, 61% of the frame) is now the single largest cost by a wide margin - future gains beyond this point would need to target the model itself (e.g. revisiting FP16 now that dispatch/preprocessing overhead is mostly eliminated, changing the calculus from the earlier FP16 test which found it not worthwhile).
- Investigate why kernel count differs from baseline (166-ish in the traced model span vs. 223 in the baseline) - carried over from Iteration 1, still open.

**Continued in [Iteration 3 - Custom Fixed-Shape NMS](03-custom-nms.md)**, which addresses the NMS setup gap directly.
