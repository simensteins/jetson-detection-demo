# Kernel-level analysis – RF-DETR nano

## Goal
After the system-level optimizations (TensorRT, CUDA graph, GPU preprocessing), ~90% of RF-DETR's inference time is GPU kernel execution. This analysis explains that remaining time: **which parts of the model** it goes to, and **what limits it on the Jetson Orin Nano** (compute, memory or parallelism), to point to the next optimizations.

## Setup
- Model/engine: RF-DETR nano, TensorRT FP32 with TensorRT's default TF32 (`models/rfdetr-nano_fp32.engine`), CUDA graph + GPU preprocessing (`scripts/infer_rfdetr_trt.py --preprocess gpu`)
- Jetson Orin Nano (8 SMs), 25 W, `jetson_clocks`
- Source report: `reports/rfdetr_trt_graph_gpupre` (nsys, 600 frames, first 50 skipped)

## Method
1. **Find the heaviest kernels** – `scripts/top_kernels.py`: GPU kernels inside the `inference` NVTX range, after warm-up, ranked by time.
2. **Select** – top 5 kernels = **53% of GPU kernel time** inside `inference` (selection criterion: ≥ 50%).
3. **Profile with Nsight Compute** – `scripts/ncu_rfdetr_kernels.py`: one full frame of each kernel's launches, `--set full`, `--clock-control none` (clocks as in the nsys runs). Kernel times agree with nsys within 3–8%.
4. **Map kernels to layers** – nsys recording of the same engine without CUDA graph (`rfdetr_nano_layers`), TensorRT's per-layer NVTX ranges, linked via each launch's correlation ID. All launches mapped.

## Results
| Kernel | Layer / operation | Share | Bound by | Key numbers |
|---|---|---|---|---|
| k1 – TF32 GEMM | MLP fc1 (12 layers) + Q/K/V projection (9 layers) | 23.3% | Latency (low occupancy) | occupancy 16.7%, compute 48%, memory 52%, 3.75 waves on 8 SMs |
| k2 – TF32 GEMM | MLP fc2 + attention output projection (12 layers each) | 14.1% | Near L2 bandwidth | memory 81%, tensor pipe 65%, some launches < 1 wave |
| k3 – FP32 FFMA GEMM | Attention scores Q·Kᵀ (9 layers) | 6.1% | Balanced, well utilized | issue slots 74%, compute 71%, memory 79% |
| k4 – fused GELU | GELU in the MLP (12 layers) | 5.2% | Memory | memory 82% vs compute 47%, long-scoreboard stall 11.7 |
| k5 – FP32 FFMA GEMM | Attention × V (9 layers) | 4.3% | Balanced, well utilized | issue slots 78%, compute 71%, memory 77% |

Shares are of GPU kernel time inside `inference`. Throughputs are % of peak, duration-weighted over all instances in one frame.

## Findings
1. **The time is in the image backbone** (DINOv2 vision transformer) – only ~1% of the selected launches are in the detection decoder.
2. **The MLP blocks cost more than attention** – MLP (fc1 + GELU + fc2) ≈ 5.5 ms/frame vs. attention ≈ 4.4 ms/frame (selected kernels, no-graph layer recording).
3. **The heaviest kernel is limited by too little parallel work at batch 1** – large-tile GEMM with one block per SM (occupancy 16.7%, limited by registers and shared memory), and partial waves on the 8 SMs.
4. **Large linear layers use TF32 Tensor Cores; the attention matrix multiplications run in plain FP32** – TensorRT chose FFMA kernels for the many small per-window matmuls, and these are the best-utilized kernels.
5. **GELU is memory-bound** – a separate element-wise kernel that re-reads and re-writes the full MLP tensor.
6. **Layers 3, 6, 9 use different attention kernels** – consistent with RF-DETR's mix of windowed and global attention (not yet verified against the model config).

## Pointers for the master's thesis (all FP32)
- **Batching** – more work per launch for the low-occupancy GEMMs; natural for RF-DETR Large with ~8–13 tiles per image.
- **Fusing GELU into the fc1 GEMM** – removes a full read/write of the MLP tensor.
- **Precision cost** – measure what strict FP32 (`--noTF32`) costs for k1/k2, which currently rely on TF32 Tensor Cores.

## Limitations
- DRAM counters are not available in Nsight Compute on this Jetson (reported as 0) – memory throughput refers to L1/L2.
- Nsight Compute flushes caches between passes by default – memory numbers may be somewhat pessimistic.
- RF-DETR nano with COCO weights at 384×384, not the project's RF-DETR Large at 704 – patterns transfer, absolute numbers do not.
- The layer-mapping analysis is not yet a script in the repo (needed for reproducibility).

## Data
The reports are not in the repository (size); they are stored locally:
- Nsight Compute: `ncu_rfdetr_nano_k1`–`k5` (`.ncu-rep`, `.csv`, `.cmd.txt`), `ncu_rfdetr_nano_selection.txt`, `top_kernels_rfdetr_nano.txt`
- Layer mapping: `rfdetr_nano_layers` (`.nsys-rep`, `.sqlite`), `layers_rfdetr_nano_fp32.json`/`.log`, `layers_rfdetr_nano_detailed.json`

## Implication for the hardware evaluation: the GPU is not saturated
The heaviest kernel (k1) is not limited by the Orin Nano running out of compute or memory capacity – compute (48%) and memory (52%) are both at about half of peak. It is limited by **too little parallel work at batch 1**: one thread block of 8 warps per SM (occupancy 16.7%), and only 30 blocks for 8 SMs (3.75 waves, so 2 SMs idle in the last wave). With few warps to switch between, the SMs stall while waiting for memory.

This means there is capacity left on the chip. **Batching tiles** – running several 704×704 tiles of the RF-DETR Large pipeline through the engine in one pass instead of one at a time – gives each GEMM more blocks to spread over the SMs, and can extract more from the same hardware before it is actually maxed out.

It also means a larger GPU does not by itself remove this bottleneck: with more SMs, the same batch-1 GEMMs would leave an even larger part of the GPU idle. Organising the work to fill the GPU (batching) is needed regardless of platform. Whether the Orin Nano meets the operational requirement therefore cannot be concluded from batch-1 measurements alone; it should be evaluated with batched tiles, against the required frame rate and the power budget (see the over-current throttling observed with RF-DETR Large at 25 W).
