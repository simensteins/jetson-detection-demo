# Nsight Systems & Compute - Beginner's Guide
*Exported from Notion (page last edited 2026-10-21).*

A beginner's guide to reading the Nsight Systems timeline and the Nsight Compute Details page, grounded in NVIDIA's own documentation. Written around the actual YOLOv8n/TensorRT profiling work done this session, so examples reference real numbers from our own reports where possible.

Sources: [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html), [Nsight Compute Kernel Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html), CUDA Programming Guide, [Modal GPU Glossary](https://modal.com/gpu-glossary/perf).

---

## 1. The big picture: which tool answers which question

- **Nsight Systems (`nsys`)** - the *timeline* tool. Answers "where does time go across my whole pipeline, over time?" Shows CPU threads, CUDA API calls, GPU kernel execution, and our own NVTX ranges (`decode`/`inference`/`draw`/`display`), all laid out on one shared timeline so you can see how they overlap and depend on each other.
- **Nsight Compute (`ncu`)** - the *microscope* tool. Answers "why is this *one specific kernel* slow?" It picks a single kernel launch, replays it multiple times to gather detailed hardware counters, and gives you a deep breakdown of occupancy, memory throughput, warp stalls, etc.

Use `nsys` first to find *which* kernels/stages matter, then use `ncu` to zoom into the ones that do.

---

## 2. Core terminology (the vocabulary both tools assume you know)

| Term | Meaning |
| --- | --- |
| **Host / Device** | Host = the CPU and its memory/code. Device = the GPU and its memory/code. "Host↔device transfer" = a memory copy between CPU RAM and GPU VRAM (what the `MEMCPY` events we queried represent). |
| **Kernel** | A function launched to run on the GPU. Each row in our `ncu` CSVs and each block in `nsys`'s CUDA HW row is one kernel launch. |
| **Thread** | A single execution unit running the kernel's code. |
| **Thread block** | A group of threads (up to 1024) that all run on the *same* SM and can share fast on-chip memory. |
| **Grid** | The full set of thread blocks launched by one kernel call, spread across all SMs in use. |
| **Warp** | A group of 32 threads within a block that execute together, instruction-by-instruction, in lockstep (SIMT). The GPU's actual unit of scheduling. |
| **SM (Streaming Multiprocessor)** | The GPU's core compute unit. A GPU contains many SMs; the hardware scheduler assigns thread blocks to whichever SMs are free. |
| **Stream** | An ordered queue of GPU operations - work in one stream executes in the order it was issued, but different streams can overlap/run concurrently. Relevant to the `cudaStreamIsCapturing`/`cudaStreamSynchronize` calls we saw in our data. |
| **Context** | The CUDA driver's per-process container of state on a GPU - allocated memory, loaded modules, etc. Roughly "the session" a process has open with the GPU. |
| **Occupancy** | The ratio of active warps per SM to the maximum possible warps per SM. Limited by registers-per-thread, shared memory per block, or block size - whichever runs out first. **This is exactly what we found capping some of our conv kernels**: 128-154 registers/thread left little room for multiple resident blocks per SM. |
| **Compute-bound** | A kernel limited by raw arithmetic throughput (how fast the SM's math pipelines can churn) rather than by data movement. |
| **Memory-bound** | A kernel limited by how fast data can move between VRAM/cache and the SMs, not by the math itself. |
| **Latency-bound** | Warps are mostly just *waiting* (for data to arrive, for a dependency to resolve) rather than being capped by either compute or memory bandwidth ceilings. **Our low-utilization Myelin elementwise kernels (5-35% compute, 8-28% memory, but still costing real time) look like this case**  • neither ceiling is close to full, so something else (small tensor size, launch overhead) is the real limiter. |

---

## 3. Reading the Nsight Systems timeline

### Row structure

When you expand a profiled process in the timeline, you typically see, top to bottom:

1. **Thread rows** - CPU-side activity: which thread is doing what, including OS-level blocking/sync if you traced `osrt`.
2. **CUDA API row** - every CUDA API call the host made (`cudaLaunchKernel`, `cuLaunchKernel`, `cudaMemcpyAsync`, etc.), shown as a bar for however long that *call* took on the CPU.
3. **CUDA HW row** (expandable into **Kernel** and **Memory** sub-rows) - what's actually executing *on the GPU itself*, i.e. real kernel execution and real memory transfers.
4. **NVTX row** - our own custom range markers (`decode`, `inference`, `draw`, `display`), shown both on the CPU thread that pushed them and *projected* onto the GPU timeline so you can see what GPU work happened inside each named range.

### The single most important thing to understand

The **CUDA API row is not the same as the CUDA HW row**, and they often don't line up in time. Kernel launches are *asynchronous*: the CPU calls `cudaLaunchKernel`, that call returns almost immediately (the CPU-side bar is short), and the *actual* GPU execution (the HW row bar) can start a little later and take a completely different amount of time. This is exactly what our own investigation quantified this session: the CUDA API row time (\~8.3ms of dispatch overhead) and the CUDA HW row time (\~10.75ms of real kernel execution) are two separate costs that both eat into the same 28.5ms `inference` NVTX range, for different reasons.

### Navigating the timeline

- **Zoom in**: highlight a region with your mouse, then right-click → "zoom to selection."
- **Events View**: right-click any row → "show in events view" to populate the table at the bottom of the screen with every event in that row, sortable and filterable - this is the GUI equivalent of the SQL queries we ran by hand against the exported `.sqlite` files all session (e.g. `SELECT text, COUNT(*), AVG(end-start) FROM NVTX_EVENTS ...`). If you're comfortable with the timeline, the Events View is often faster for getting exact numbers than eyeballing bar widths.
- **NVTX ranges as landmarks**: since our `decode`/`inference`/`draw`/`display` ranges repeat every frame, they're good visual landmarks for scrolling to "the next frame" or comparing frame-to-frame consistency at a glance.

---

## 4. Reading the Nsight Compute Details page

When you open a single kernel's result in `ncu`, the Details page has several sections, roughly in order of "zoom level":

### GPU Speed of Light (SOL)

The headline summary: achieved compute throughput % and achieved memory throughput % versus the hardware's theoretical maximum. **This is exactly the "Compute Throughput \[%\]" and "Memory Throughput \[%\]" columns in our CSVs all session.** Also includes a **roofline chart** - plots the kernel's arithmetic intensity (operations per byte moved) against two ceiling lines (a compute ceiling and a memory-bandwidth ceiling). Sitting near the compute ceiling = compute-bound; near the memory ceiling = memory-bound; sitting well below *both* ceilings = latency-bound (neither resource is actually saturated - matches what we saw in the Myelin kernels).

### Compute Workload Analysis

Breaks down utilization per compute pipeline (different math units on the SM) and achieved instructions-per-clock. High utilization spread across pipelines is good (means the SM is well-fed); one pipeline pinned near 100% while others sit idle means *that* pipeline specifically is your bottleneck.

### Memory Workload Analysis

A heatmap showing which level of the memory hierarchy (L1/shared memory, L2 cache, or DRAM) is the actual limiter, via three possible causes: hardware units fully busy, bandwidth exhausted, or the GPU running out of memory-instruction issue slots.

### Scheduler Statistics

Each SM has warp schedulers, each holding a pool of resident warps. Key numbers: **Theoretical Warps** (the upper bound your launch configuration allows), **Active Warps** (how many are actually resident), **Eligible Warps** (active *and* ready to issue an instruction right now, i.e. not stalled). Lots of "skipped issue slots" (cycles where no warp was eligible) means poor latency hiding.

### Warp State Statistics

Shows *why* warps aren't issuing when they're not - e.g. waiting on a memory fetch, waiting on a dependency, waiting on synchronization. NVIDIA's own docs give an important caveat worth remembering: **stalls aren't automatically bad or fixable** - only chase a specific stall reason if the scheduler is actually failing to issue an instruction most cycles.

### Occupancy

Theoretical occupancy (computed from your launch configuration + what the hardware allows) versus achieved occupancy (what actually happened). The three usual limiters: registers per thread, shared memory per block, or block size. **This is the section that would show, directly, whether our high-register-count kernels (128-154 registers/thread) are actually occupancy-capped** rather than us just inferring it from the register count alone. NVIDIA's own caveat here too: higher occupancy doesn't automatically mean higher performance, but *very low* occupancy reliably hurts, since there's not enough parallel work available to hide memory/instruction latency.

### Source Counters

Maps branch efficiency and sampled stall reasons back to your actual source code / loop structure, where available.

### Baseline comparison

Click "Compare → Add Baseline" on any result to pin it. Every other result you then open (in this report or another) shows both its own numbers *and* the baseline's, side by side - this is the built-in GUI way to do exactly the FP32-vs-FP16 kernel comparison we did by hand today via CSV diffing.

---

## 5. Quick mental model for this project specifically

- **`nsys` timeline** → confirms *which pipeline stage* (`decode`/`inference`/`draw`) dominates, and splits `inference` further into GPU-kernel-execution vs. CPU-dispatch-overhead vs. unaccounted Python time (our 37.7% / 29.2% / 30.1% split).
- **`ncu` Details page** → for one specific kernel inside `inference` (e.g. one of the low-utilization Myelin elementwise kernels), tells you *why* it's slow: Occupancy section for register/shared-memory limits, Warp State Statistics for what it's actually stalled on, SOL/roofline for whether it's compute-, memory-, or latency-bound.
- Use `nsys` to decide *where* to point `ncu`, not the other way around - `ncu`'s replay-based profiling is too slow to run across a whole pipeline.

**Child page:** [Replica setup](replica-setup.md)
