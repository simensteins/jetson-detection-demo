# Iteration 1 - CUDA Graph

*Exported from Notion (page last edited 2026-09-30).*

> These numbers were measured on `data/vtest.avi` (768×576) in September 2026, before the measurement protocol was fixed (power mode and clocks not recorded). The final, comparable numbers are in [results-paper25w.md](../results-paper25w.md).

Tests the theory from baseline profiling: \~223 individually-dispatched CUDA API calls per frame (\~8.1-8.4ms CPU-side overhead) and \~30% GPU idle time between kernels should collapse by capturing the TensorRT engine's forward pass into a CUDA Graph and replaying it with one call per frame, instead of Ultralytics' `model.predict()` re-dispatching every kernel every frame. **Only the engine invocation mechanism changes here** - preprocessing (hand-written CPU letterbox/normalize) and NMS (Ultralytics' own `non_max_suppression`) are untouched.

## Implementation

Script: `scripts/infer_cuda_graph.py` (repo). Same pipeline shape as `src/detect.py` (decode -> inference -> draw -> display, same NVTX ranges/CLI flags) for direct comparability.

Reuses Ultralytics' own `AutoBackend` object (`context`, `bindings`) rather than re-deserializing the `.engine` independently. One uncaptured warm-up `execute_async_v3()` call (required by TensorRT before capture), then `torch.cuda.graph()` records that call once; every subsequent frame calls `graph.replay()`.

**Debugging note**: an earlier version deserialized the `.engine` directly via `trt.Runtime().deserialize_cuda_engine()`, which failed with `"incompatible serialization version"` - root cause was a pip-installed `tensorrt_cu13` package shadowing the correct JetPack system TensorRT inside the venv, routing the call through an incompatible "dispatch" runtime variant. Fixed by reusing Ultralytics' already-working `AutoBackend.context`/`.bindings` instead (see `scripts/inspect_autobackend.py`).

**Correctness check**: ran with display, visually confirmed detection boxes look correct.

## Profiling methodology

Same as baseline: `jetson_clocks` locked, local file source (`data/vtest.avi`), `nsys profile --trace=cuda,nvtx --max-frames 600`, `scripts/analyze_nsys.py` with 50-frame warm-up skip (n=550 analyzed).

## A measurement bug found and fixed

First profiling pass showed an implausible drop: `kernel_count` 223 -> 34, `kernel_time` \~10ms -> \~0.22ms. Cause: `nsys profile`'s `--cuda-graph-trace` defaults to `graph` mode ("trace whole CUDA graphs without node activities"), which doesn't break a captured graph's replay into individual kernel events - so almost all model kernels were invisible to the trace, even though the graph itself replayed correctly. Fixed by re-capturing with `--cuda-graph-trace=node`. All results below use that fix.

## Results vs. Baseline (n=550 frames, `--cuda-graph-trace=node`)

### Pipeline stages

| Stage | Iteration 1 | Baseline | Change |
| --- | --- | --- | --- |
| decode | 2.67ms | 2.83ms | -5.7% (noise) |
| **inference** | **19.35ms** | **24.90ms** | **-22.3%** |
| draw | 2.63ms | 2.75ms | -4.4% (noise) |

### Model idle-gap, kernel-name-anchored (`_trt` suffix = model kernel, `nms_kernel_impl` = NMS)

|  | Iteration 1 (CUDA Graph) | Baseline (dispatched per-kernel) |
| --- | --- | --- |
| Kernel count | 166 | 223 |
| Kernel busy time | 7847.90us | \~10,000-11,000us |
| **Idle gap between kernels** | **19.47us (0.2%)** | **\~7500-8000us (\~30%)** |

**Headline result: the dispatch-overhead/idle-gap problem from the baseline is solved.** 166 kernels now execute back-to-back with almost no gap. (Kernel count dropping 223 -> 166 is a real, separate difference - not yet explained, doesn't affect this finding.)

### What "inference" (19.35ms) is made of

| Component | Time | Share |
| --- | --- | --- |
| Pre-kernel (preprocessing, hand-written, CPU) | 8268.14us | 42.7% |
| Model/graph kernel execution (busy) | 7847.90us | 40.6% |
| Model/graph idle gap | 19.47us | 0.1% |
| Gap (NMS setup, before its own kernels start) | 2491.55us | 12.9% |
| NMS kernel execution | 637.92us | 3.3% |
| Post-kernel | 83.86us | 0.4% |
| **Total** | **19348.85us** | **100.0%** |

(Baseline's `model.predict()` bundles this into one opaque call - no baseline column here; the NVTX table above is the baseline comparison that matters.)

Two costs invisible in the baseline's coarser measurement are now the dominant ones: preprocessing (42.7%, larger than the model itself) and the NMS setup gap (12.9%, vs 3.3% NMS compute itself).

## Interpretation

- Core theory confirmed cleanly: CUDA Graph replay eliminates dispatch-induced idle time between kernels (30% -> 0.2%).
- The 22.3% net `inference` improvement is real but smaller than the idle-gap elimination alone would suggest, because two costs invisible in the baseline - preprocessing (42.7%) and NMS setup overhead (12.9%) - are now the dominant remaining costs.
- `nsys` cannot see inside a CUDA graph's replay without `--cuda-graph-trace=node` - worth remembering for any future graph-based capture.

## Next steps

- Optimize preprocessing - now the single largest cost in the pipeline (42.7%).
- Investigate the 2.49ms pre-NMS setup gap.
- Consider whether NMS itself is worth graph-capturing, now that the model portion is solved.
- Investigate why kernel count differs (166 in the traced model span vs. 223 in the baseline).
