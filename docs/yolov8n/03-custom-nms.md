# Iteration 3 - Custom Fixed-Shape NMS

*Exported from Notion (page last edited 2026-09-12).*

> These numbers were measured on `data/vtest.avi` (768×576) in September 2026, before the measurement protocol was fixed (power mode and clocks not recorded). The final, comparable numbers are in [results-paper25w.md](../results-paper25w.md).

**This iteration builds on and keeps Iterations 1 (CUDA Graph) and 2 (GPU Preprocessing) in place.** The model is still invoked via captured graph replay exactly as in Iteration 1; preprocessing is still fully GPU-side exactly as in Iteration 2 (step 2b). The only thing this iteration changes is how NMS (non-max suppression) is computed.

---

## Motivation

Iteration 2 closed with preprocessing down to 1.64ms and model execution unchanged at \~7.86ms (61% of the frame) - and flagged a **2.46ms gap between the model's last kernel and NMS's first kernel** (19.2% of the frame, bigger than preprocessing itself) as the clear next bottleneck. NMS's own GPU computation was cheap (0.65ms); the cost was entirely in *getting there*.

Direct investigation of the raw kernel trace confirmed why: Ultralytics' `non_max_suppression()` uses confidence-threshold boolean-mask filtering and `torch.nonzero()`-style dynamic-shape indexing before it can even launch its \~30 real GPU kernels. Each of those triggers a GPU->CPU sync to learn how many candidates passed the filter before the next op can be sized - the same per-dispatch-overhead problem CUDA Graph already solved for the model, reappearing in postprocessing. And because the *shape* of that work is data-dependent, it can't simply be wrapped in `torch.cuda.graph()` the way the model was - CUDA Graphs require a fixed, deterministic operation sequence.

---

## Research before implementing

Two real candidates were investigated and tested before writing any code:

| Approach | Verdict |
| --- | --- |
| TensorRT's own EfficientNMS plugin (`end2end=True` export) | **Tested directly and confirmed not to work on this stack.** Export completed with no error, but `scripts/inspect_autobackend.py` showed `output_names` stayed `['output0']` - a single raw output, not the 4 fixed-shape outputs EfficientNMS produces. Root cause: the `EfficientNMS_ONNX_TRT` plugin variant Ultralytics' ONNX export path depends on was removed in TensorRT 10.16, and this system runs 10.16.2.10. This also retroactively explains an earlier, pre-CUDA-Graph FP16 test that found no measurable difference between `end2end=True/False` - the flag was likely never doing anything for this export path/version combination. |
| CUDA 12+ conditional graph nodes | Rejected - these solve data-dependent *branching* inside an already-captured graph, not variable-size *tensors*, and have no PyTorch binding regardless. |

The approach actually used follows NVIDIA's own published "CUDA Graph Best Practice for PyTorch" pattern: pad to a **fixed maximum size** and mask invalid entries with shape-preserving ops, instead of ops that change tensor shape based on data. Concretely:

- `torch.topk(scores, k=100)` **always** returns exactly 100 entries, sorted by score descending, regardless of how many real detections exist in a frame - fixed shape, no GPU->CPU sync needed to size anything. K=100 matches COCO's own standard evaluation convention (max 100 detections per image), not an arbitrary choice.
- Suppression uses **"Fast NMS"** (Bolya et al., the YOLACT paper): compute the full 100x100 pairwise IoU matrix over the top-K candidates, then suppress a box if it overlaps too much with any *higher-scoring* box, in one fully vectorized pass (upper-triangular IoU matrix, no sequential loop). This is a well-known, standard **approximation** of classic greedy NMS - it doesn't check whether the higher-scoring box was itself already suppressed, so it can very rarely over-suppress in complex 3+-box overlap chains. Accepted, precedented tradeoff for exactly this class of problem, not a bug.
- Class-aware suppression via the standard box-offset trick (each box shifted spatially by a large multiple of its class index before computing IoU, so different classes never overlap) - same approach torchvision's own `batched_nms` uses.
- Output is always a fixed `(100, 6)` tensor `[x1, y1, x2, y2, score, cls]`; suppressed/unused slots simply have `score == 0`.

---

## Implementation

Script: `scripts/infer_cuda_graph_custom_nms.py` (repo). Prior iteration scripts kept unchanged for comparison. CUDA Graph capture/replay and GPU-side preprocessing from Iterations 1-2 are untouched here - only the `non_max_suppression()` call is replaced, by `custom_nms()`.

Runs **eagerly** (not yet captured into its own CUDA graph) as the first measured step, to isolate and measure the fixed-shape rewrite's own effect before considering graph-capturing NMS too as a possible follow-up - the same staged approach that worked well for Iteration 2 (2a measured alone before 2b added on top).

**Correctness check**: ran with display, visually confirmed detection boxes match the prior iteration's output on the same source - the most correctness-sensitive change so far, since this is the first iteration reimplementing detection logic from scratch rather than reusing Ultralytics' code path.

### A correction made during analysis

The first profiling pass (n=550) already showed a large win - but the raw kernel trace revealed `draw_detections()` was quietly reintroducing the exact problem this iteration was built to solve. It filtered the GPU-resident output tensor with `dets[dets[:, 4] > 0]` (a boolean-mask compaction implemented via `cub::DeviceSelect`, which forces a sync), then called `.tolist()` three separate times - each one its own additional CPU\<->GPU round trip. The kernel trace showed a string of tiny kernels spaced \~200-300us apart at the tail of the `draw` NVTX range, the same per-dispatch-overhead signature seen everywhere else in this project.

**Fix**: transfer the fixed `(100, 6)` tensor to CPU exactly once, immediately after `custom_nms()`, still inside the `inference` NVTX range (consistent with prior iterations, where `non_max_suppression()`'s sync was also counted inside `inference`). `draw_detections()` now receives a plain numpy array and does all filtering/coordinate math on CPU, where it can't reintroduce GPU dispatch overhead. Re-profiled after the fix; all numbers below are post-fix.

---

## Results vs. Iteration 2 and Baseline (n=550 frames, `--cuda-graph-trace=node`, kernel-name-anchored boundaries)

| Component | Iteration 3 (custom NMS, final) | Iteration 2, step 2b (GPU resize) | Baseline |
| --- | --- | --- | --- |
| Preprocessing (pre-kernel + GPU kernels) | 2110.49us | 1764.55us | (bundled, not separable) |
| Model execution (busy) | 7390.17us | 7880.94us | \~10,000-11,000us |
| NMS setup gap + NMS execution (combined) | **503.96us** | 3111.49us (2462.95 gap + 648.55 exec) | n/a |
| **Total inference (NVTX range)** | **10127.98us** | 12836.91us | \~24,900us |
| `draw` (postprocessing, drawing boxes) | **643.32us** | 2603.75us | n/a |
| Blocking GPU sync (CUPTI, overlapping category) | 19.98us | 6046.98us | n/a |

Model execution shows a small (\~6%) apparent drop that is **not attributed to this change** - the model and its CUDA Graph capture are byte-for-byte identical to Iteration 2's; this is most likely ordinary run-to-run thermal/clock variance (the project has previously measured \~3.5% cold-start variance even with `jetson_clocks` locked), not a real effect.

The NMS setup gap - the entire target of this iteration - dropped from 2462.95us to effectively nothing (the equivalent combined bucket, gap + execution, fell from 3111.49us to 503.96us, an 83.8% reduction). This is the direct, expected result of replacing dynamic-shape filtering with a fixed-`(100,6)`-shape computation: there's no GPU->CPU sync required to learn how many detections exist before NMS's kernels can launch.

### Why NVTX-bucket comparisons alone aren't enough here

Comparing `Total inference` alone would be misleading: it went **up** slightly after the sync-relocation fix (7728.97us -> 10127.98us in the first-pass-vs-final comparison) purely because the single GPU->CPU sync now happens *inside* the `inference` range instead of leaking into `draw` - moving where a cost is counted isn't the same as changing it. The metric that actually matters is **true end-to-end frame time** (`decode` start to `draw` end, measured directly from raw NVTX timestamps, 50-frame warm-up skipped):

| Version | Frame time | FPS | vs. Baseline |
| --- | --- | --- | --- |
| Baseline | 33452us | 29.89 | — |
| Iteration 1 (CUDA Graph) | 24732us | 40.43 | +35.3% |
| Iteration 2 (+ GPU preprocessing) | 18147us | 55.10 | +84.3% |
| Iteration 3, first pass (+ custom NMS, before the draw-sync fix) | 15384us | 65.00 | +117.5% |
| **Iteration 3, final (+ single-sync fix)** | **13474us** | **74.22** | **+148.4%** |

**Interpretation**: the custom NMS rewrite alone (first pass) delivered an 18% real throughput gain over Iteration 2 (55.10 -> 65.00 FPS). Fixing the incidental sync-scattering bug in `draw_detections()` delivered a further, larger 14.2% gain on top of that (65.00 -> 74.22 FPS) - a reminder that a correct fixed-shape rewrite can still be undermined by careless GPU/CPU handoff in the code around it, and that per-NVTX-range bucket totals need to be checked against true wall-clock time before trusting them.

---

## Full picture, Baseline -> Iteration 1 -> Iteration 2 -> Iteration 3

|  | True end-to-end frame time | FPS | Change vs. Baseline |
| --- | --- | --- | --- |
| Baseline (no CUDA Graph, CPU preprocess, Ultralytics NMS) | 33.45ms | 29.89 | — |
| Iteration 1 (+ CUDA Graph) | 24.73ms | 40.43 | +35.3% |
| Iteration 2 (+ GPU preprocessing) | 18.15ms | 55.10 | +84.3% |
| **Iteration 3 (+ custom fixed-shape NMS)** | **13.47ms** | **74.22** | **+148.4%** |

The pipeline now runs at roughly **2.5x** the original baseline's throughput, via three independent, individually-verified changes: CUDA Graph (dispatch overhead/idle gaps), GPU-side preprocessing (the cost CUDA Graph's own fix made newly visible), and fixed-shape NMS (the cost that became the top bottleneck once preprocessing was fixed).

## Next steps

- Model execution (\~7.4-7.9ms) is now an even larger share of the frame (\~55-58%) and the single largest remaining cost by a wide margin. Revisiting FP16 is the natural next candidate, now that dispatch, preprocessing, and NMS overhead are all mostly eliminated - changing the calculus from the earlier FP16 test (done before any of these iterations existed, when overhead dominated the frame and could have hidden a real FP16 win).
- Consider whether the now-fixed-shape NMS is itself worth graph-capturing - the 503.96us remaining bucket is small in absolute terms, but the staged 2a->2b pattern from Iteration 2 suggests checking is cheap.
- The still-open kernel-count discrepancy from Iteration 1 (traced model span kernel count vs. baseline's 223) remains unaddressed.
