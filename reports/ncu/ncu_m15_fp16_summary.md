# Nsight Compute summary: m15_fp16

Durations are from Nsight Compute's replays (caches flushed per pass): compare launches with them, take real times from nsys. Limited by: a unit above 60 % of peak, else latency (neither compute nor memory near peak).

## k1: `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize128x128x32_stage4_warpsize2x2x1_g1_tensor16x8x16_simple_t1r1s1_linkable_execute_kernel_trt`

18 launches, 3.13 ms under ncu. Time-weighted: SM 38 %, memory 63 %, tensor pipe 9 %, achieved occupancy 17 %, L2 hit 85 %. Time by limiter: memory 66 %, latency 34 %.

| # | µs | Grid | Waves/SM | Regs | Occ. theor./achieved | SM % | Mem % | Limited by | Top pipe | Tensor % | Top stall | L2 hit | Mem read MB (GB/s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 456.7 | 450 | 28.12 | 199 | 16.7 / 17.3 % | 36.1 | 63.3 | memory | alu 19% | 6.1 | wait 1.64 | 84.8 % | 11.14 (24.4) |
| 2 | 141.4 | 113 | 7.06 | 199 | 16.7 / 16.7 % | 30.3 | 58.4 | latency | alu 16% | 6.5 | wait 1.69 | 82.8 % | 3.78 (26.7) |
| 3 | 103.6 | 58 | 3.62 | 199 | 16.7 / 16.4 % | 34.5 | 50.2 | latency | alu 14% | 9.4 | wait 1.84 | 88.8 % | 2.03 (19.6) |
| 4 | 95.3 | 32 | 2.0 | 199 | 16.7 / 16.8 % | 60.2 | 63.1 | memory | tensor_op_hmma 16% | 15.8 | wait 2.17 | 90.4 % | 2.39 (25.1) |
| 5 | 190.1 | 58 | 3.62 | 199 | 16.7 / 16.7 % | 54.6 | 68.6 | memory | tensor_op_hmma 15% | 15.1 | wait 2.17 | 86.3 % | 6.57 (34.6) |
| 6 | 123.3 | 58 | 3.62 | 199 | 16.7 / 16.2 % | 42.3 | 57.6 | latency | lsu 13% | 11.5 | wait 1.96 | 88.3 % | 3.02 (24.5) |
| 7 | 171.3 | 113 | 7.06 | 199 | 16.7 / 17.1 % | 29.9 | 63.9 | memory | alu 14% | 8.0 | wait 1.78 | 80.8 % | 5.64 (32.9) |
| 8 | 123.8 | 58 | 3.62 | 199 | 16.7 / 16.2 % | 42.2 | 57.4 | latency | lsu 13% | 11.4 | wait 1.96 | 88.3 % | 3.02 (24.4) |
| 9 | 125.7 | 58 | 3.62 | 199 | 16.7 / 16.6 % | 42.3 | 56.2 | latency | lsu 13% | 11.5 | wait 1.96 | 88.3 % | 3.02 (24.1) |
| 10 | 94.8 | 32 | 2.0 | 199 | 16.7 / 16.8 % | 60.1 | 63.8 | memory | tensor_op_hmma 16% | 15.7 | wait 2.17 | 90.4 % | 2.41 (25.4) |
| 11 | 96.3 | 32 | 2.0 | 199 | 16.7 / 17.0 % | 60.2 | 63.0 | memory | tensor_op_hmma 16% | 15.8 | wait 2.17 | 90.4 % | 2.4 (24.9) |
| 12 | 36.1 | 8 | 0.5 | 199 | 16.7 / 9.1 % | 28.2 | 48.3 | latency | lsu 8% | 7.7 | wait 1.99 | 74.1 % | 1.11 (30.8) |
| 13 | 63.1 | 29 | 1.81 | 199 | 16.7 / 16.7 % | 29.1 | 53.8 | latency | alu 12% | 8.0 | wait 1.83 | 78.6 % | 1.97 (31.2) |
| 14 | 50.0 | 29 | 1.81 | 199 | 16.7 / 17.2 % | 24.0 | 42.2 | latency | alu 13% | 5.4 | wait 1.66 | 81.2 % | 1.01 (20.3) |
| 15 | 142.5 | 113 | 7.06 | 199 | 16.7 / 16.8 % | 30.5 | 58.1 | latency | alu 16% | 6.5 | wait 1.69 | 82.8 % | 3.78 (26.5) |
| 16 | 142.8 | 113 | 7.06 | 199 | 16.7 / 16.9 % | 29.5 | 58.5 | latency | alu 16% | 6.5 | wait 1.69 | 82.8 % | 3.78 (26.5) |
| 17 | 487.4 | 450 | 28.12 | 199 | 16.7 / 17.4 % | 36.1 | 66.1 | memory | alu 19% | 7.5 | wait 1.69 | 83.2 % | 14.84 (30.4) |
| 18 | 484.9 | 450 | 28.12 | 199 | 16.7 / 16.8 % | 35.8 | 69.7 | memory | alu 18% | 7.4 | wait 1.69 | 83.2 % | 14.84 (30.6) |

Nsight Compute warnings (launches raising each; example from the longest of them):

- **Issue Slot Utilization** (18/18): Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 2.8 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per scheduler, this workload allocates an average of 1.94 active warps per scheduler, but only an average of 0.44 warps were eligible per cycle. Eligible warps are the subset of active warps that are ready to issue their next inst
- **FP32/64 Instructions** (18/18): This workload executes 230400 fused and 460800 non-fused FP32 instructions. By converting pairs of non-fused instructions to their fused:https://docs.nvidia.com/cuda/floating-point/#cuda-and-floating-point@, higher-throughput equivalent, the achieved FP32 performance could be increased by up to 33% (relative to its current performance).
- **Theoretical Occupancy** (18/18): The 2.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the hardware maximum of 12. This kernel's theoretical occupancy (16.7%) is limited by the number of required registers, and the required amount of shared memory.
- **Uncoalesced Shared Accesses** (18/18): This workload has uncoalesced shared accesses resulting in a total of 230176 excessive wavefronts (17% of the total 1382176 wavefronts). Check the L1 Wavefronts Shared Excessive table for the primary source locations. The CUDA Best Practices Guide:https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c-ab@ has an example on optimizing shared memory accesses.
- **Bottleneck** (15/18): Memory is more heavily utilized than Compute: Look at the MemoryWorkloadAnalysis:Memory Workload Analysis@ section to identify the L2 bottleneck. Check memory replay (coalescing) metrics to make sure you're efficiently utilizing the bytes transferred. Also consider whether it is possible to do more work per memory access (kernel fusion) or whether there are values you can (re)compute.
- **Shared Memory Conflicts** (14/18): The memory access pattern for shared stores might not be optimal and causes on average a 3.1 - way bank conflict across all 14848 shared store requests.This results in 5626 bank conflicts,  which represent 12.23% of the overall 46002 wavefronts for shared stores. Check the SourceCounters:Source Counters@ section for uncoalesced shared stores.
- **Warp Stall** (8/18): On average, each warp of this workload spends 1.7 cycles being stalled waiting on a fixed latency execution dependency. Typically, this stall reason should be very low and only shows up as a top contributor in already highly optimized kernels. Try to hide the corresponding instruction latencies by increasing the number of active warps, restructuring the code or unrolling loops. Furthermore, consider switching to lower-latency instructions, e.g. by making use of fast math compiler options. This s
- **Launch Configuration** (8/18): A wave of thread blocks is defined as the maximum number of blocks that can be executed in parallel on the target GPU. The number of blocks in a wave depends on the number of multiprocessors and the theoretical occupancy of the kernel. This kernel launch results in 3 full waves and a partial wave of 10 thread blocks. Under the assumption of a uniform execution duration of all thread blocks, this partial wave may account for up to 25.0% of the total runtime of this kernel. Try launching a grid wi
- **High Pipe Utilization** (4/18): Tensor is the highest-utilized pipeline (60.2%) based on elapsed cycles in the workload, taking into account the rates of its different instructions. It is the logical aggregation of individual tensor pipelines. It's dominated by its Tensor (FP) sub-pipeline. The pipeline is well-utilized, but might become a bottleneck if more work is added. Based on the number of executed instructions, the highest utilized pipeline (15.2%) is Tensor (FP). It executes 16-bit floating point tensor operations. Com
- **Workload Imbalance** (1/18): One or more SMs have a much higher number of active cycles than the average number of active cycles. Maximum instance value is 6.40% above the average, while the minimum instance value is 2.74% below the average.

## k2: `permutationKernelPLC3`

41 launches, 3.10 ms under ncu. Time-weighted: SM 64 %, memory 60 %, tensor pipe 0 %, achieved occupancy 95 %, L2 hit 52 %. Time by limiter: compute 69 %, memory 22 %, latency 9 %.

| # | µs | Grid | Waves/SM | Regs | Occ. theor./achieved | SM % | Mem % | Limited by | Top pipe | Tensor % | Top stall | L2 hit | Mem read MB (GB/s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 175.3 | 1800 | 18.75 | 30 | 100.0 / 97.3 % | 84.2 | 59.8 | compute | alu 78% | 0.0 | not_selected 4.40 | 79.9 % | 1.86 (10.6) |
| 2 | 118.8 | 1800 | 18.75 | 40 | 100.0 / 105.4 % | 67.7 | 60.9 | compute | alu 66% | 0.0 | long_scoreboard 6.18 | 49.9 % | 3.7 (31.1) |
| 3 | 123.3 | 1800 | 18.75 | 40 | 100.0 / 108.1 % | 66.5 | 58.9 | compute | alu 65% | 0.0 | long_scoreboard 6.55 | 49.9 % | 3.7 (30.0) |
| 4 | 116.0 | 1800 | 18.75 | 40 | 100.0 / 103.5 % | 68.2 | 59.0 | compute | alu 66% | 0.0 | long_scoreboard 6.26 | 49.9 % | 3.7 (31.9) |
| 5 | 118.8 | 1800 | 18.75 | 40 | 100.0 / 106.1 % | 68.6 | 57.7 | compute | alu 66% | 0.0 | long_scoreboard 6.04 | 49.9 % | 3.7 (31.2) |
| 6 | 62.7 | 900 | 9.38 | 40 | 100.0 / 102.9 % | 64.4 | 64.0 | compute | alu 64% | 0.0 | long_scoreboard 5.80 | 49.9 % | 1.86 (29.6) |
| 7 | 63.7 | 900 | 9.38 | 40 | 100.0 / 104.3 % | 65.1 | 63.0 | compute | alu 65% | 0.0 | long_scoreboard 5.77 | 49.9 % | 1.86 (29.2) |
| 8 | 62.8 | 900 | 9.38 | 40 | 100.0 / 103.1 % | 65.2 | 57.9 | compute | alu 65% | 0.0 | long_scoreboard 5.71 | 49.9 % | 1.86 (29.6) |
| 9 | 63.2 | 900 | 9.38 | 40 | 100.0 / 102.8 % | 64.4 | 57.5 | compute | alu 64% | 0.0 | long_scoreboard 5.72 | 49.9 % | 1.86 (29.4) |
| 10 | 23.0 | 226 | 2.35 | 40 | 100.0 / 97.3 % | 48.2 | 40.0 | latency | alu 53% | 0.0 | long_scoreboard 5.87 | 49.6 % | 0.47 (20.6) |
| 11 | 35.0 | 452 | 4.71 | 40 | 100.0 / 104.7 % | 60.1 | 55.9 | compute | alu 62% | 0.0 | long_scoreboard 5.64 | 49.8 % | 0.94 (26.7) |
| 12 | 35.1 | 452 | 4.71 | 40 | 100.0 / 104.6 % | 60.5 | 55.9 | compute | alu 62% | 0.0 | long_scoreboard 5.72 | 49.8 % | 0.94 (26.7) |
| 13 | 22.8 | 232 | 2.42 | 40 | 100.0 / 99.3 % | 49.9 | 44.2 | latency | alu 54% | 0.0 | long_scoreboard 5.57 | 49.6 % | 0.47 (20.8) |
| 14 | 23.0 | 232 | 2.42 | 40 | 100.0 / 99.8 % | 51.0 | 43.9 | latency | alu 54% | 0.0 | long_scoreboard 5.58 | 49.6 % | 0.47 (20.6) |
| 15 | 23.6 | 232 | 2.42 | 40 | 100.0 / 102.3 % | 49.2 | 39.1 | latency | alu 54% | 0.0 | long_scoreboard 5.69 | 49.6 % | 0.47 (20.1) |
| 16 | 23.1 | 232 | 2.42 | 40 | 100.0 / 99.1 % | 49.2 | 40.0 | latency | alu 54% | 0.0 | long_scoreboard 5.62 | 49.6 % | 0.47 (20.6) |
| 17 | 24.3 | 232 | 2.42 | 40 | 100.0 / 103.6 % | 50.2 | 38.0 | latency | alu 54% | 0.0 | long_scoreboard 5.70 | 49.6 % | 0.47 (19.5) |
| 18 | 30.7 | 232 | 2.42 | 40 | 100.0 / 97.3 % | 57.8 | 66.3 | memory | alu 53% | 0.0 | long_scoreboard 4.12 | 69.4 % | 0.47 (15.5) |
| 19 | 22.1 | 232 | 2.42 | 40 | 100.0 / 95.3 % | 61.4 | 46.1 | compute | alu 63% | 0.0 | long_scoreboard 3.33 | 49.6 % | 0.47 (21.4) |
| 20 | 23.1 | 232 | 2.42 | 40 | 100.0 / 99.0 % | 49.4 | 40.0 | latency | alu 55% | 0.0 | long_scoreboard 5.58 | 49.6 % | 0.47 (20.6) |
| 21 | 115.8 | 1808 | 18.83 | 40 | 100.0 / 103.3 % | 68.4 | 69.1 | memory | alu 66% | 0.0 | long_scoreboard 6.05 | 49.9 % | 3.7 (32.0) |
| 22 | 33.9 | 452 | 4.71 | 40 | 100.0 / 101.2 % | 60.7 | 54.0 | compute | alu 62% | 0.0 | long_scoreboard 5.63 | 49.8 % | 0.94 (27.6) |
| 23 | 35.4 | 452 | 4.71 | 40 | 100.0 / 104.6 % | 60.3 | 51.5 | compute | alu 62% | 0.0 | long_scoreboard 5.72 | 49.8 % | 0.94 (26.4) |
| 24 | 35.6 | 452 | 4.71 | 40 | 100.0 / 106.3 % | 60.8 | 56.5 | compute | alu 63% | 0.0 | long_scoreboard 5.75 | 49.8 % | 0.94 (26.3) |
| 25 | 34.5 | 452 | 4.71 | 40 | 100.0 / 103.2 % | 60.9 | 53.1 | compute | alu 63% | 0.0 | long_scoreboard 5.64 | 49.8 % | 0.94 (27.1) |
| 26 | 217.6 | 3600 | 37.5 | 40 | 100.0 / 101.2 % | 69.2 | 66.4 | compute | alu 67% | 0.0 | long_scoreboard 6.01 | 50.0 % | 7.39 (33.9) |
| 27 | 61.6 | 900 | 9.38 | 40 | 100.0 / 101.8 % | 65.5 | 58.9 | compute | alu 65% | 0.0 | long_scoreboard 5.72 | 49.9 % | 1.86 (30.1) |
| 28 | 64.1 | 900 | 9.38 | 40 | 100.0 / 105.3 % | 65.3 | 56.8 | compute | alu 65% | 0.0 | long_scoreboard 5.66 | 49.9 % | 1.86 (29.0) |
| 29 | 62.9 | 900 | 9.38 | 40 | 100.0 / 102.1 % | 65.0 | 57.7 | compute | alu 64% | 0.0 | long_scoreboard 5.78 | 49.9 % | 1.86 (29.5) |
| 30 | 63.3 | 900 | 9.38 | 40 | 100.0 / 104.2 % | 64.1 | 57.3 | compute | alu 64% | 0.0 | long_scoreboard 5.96 | 49.9 % | 1.86 (29.3) |
| 31 | 63.5 | 904 | 9.42 | 40 | 100.0 / 102.9 % | 64.8 | 61.5 | compute | alu 64% | 0.0 | long_scoreboard 5.89 | 49.9 % | 1.86 (29.3) |
| 32 | 33.1 | 452 | 4.71 | 40 | 100.0 / 99.2 % | 59.7 | 59.2 | latency | alu 62% | 0.0 | long_scoreboard 5.74 | 49.8 % | 0.94 (28.3) |
| 33 | 35.4 | 452 | 4.71 | 40 | 100.0 / 105.2 % | 60.2 | 55.4 | compute | alu 62% | 0.0 | long_scoreboard 5.63 | 49.8 % | 0.94 (26.4) |
| 34 | 35.4 | 452 | 4.71 | 40 | 100.0 / 105.7 % | 60.1 | 51.6 | compute | alu 63% | 0.0 | long_scoreboard 5.56 | 49.8 % | 0.94 (26.4) |
| 35 | 35.7 | 452 | 4.71 | 40 | 100.0 / 107.7 % | 60.0 | 51.0 | compute | alu 63% | 0.0 | long_scoreboard 5.61 | 49.8 % | 0.94 (26.2) |
| 36 | 35.6 | 464 | 4.83 | 40 | 100.0 / 102.2 % | 59.9 | 56.4 | latency | alu 62% | 0.0 | long_scoreboard 5.48 | 49.8 % | 0.94 (26.3) |
| 37 | 23.1 | 232 | 2.42 | 40 | 100.0 / 104.6 % | 49.6 | 43.6 | latency | alu 54% | 0.0 | long_scoreboard 5.74 | 49.6 % | 0.47 (20.5) |
| 38 | 23.8 | 232 | 2.42 | 40 | 100.0 / 104.4 % | 50.6 | 42.3 | latency | alu 54% | 0.0 | long_scoreboard 5.59 | 49.6 % | 0.47 (19.9) |
| 39 | 297.1 | 9563 | 74.71 | 38 | 66.7 / 67.5 % | 57.4 | 60.8 | memory | alu 58% | 0.0 | long_scoreboard 6.81 | 50.0 % | 9.8 (33.0) |
| 40 | 352.2 | 9600 | 75.0 | 38 | 66.7 / 65.7 % | 64.3 | 61.4 | compute | alu 63% | 0.0 | long_scoreboard 4.74 | 52.6 % | 9.8 (27.8) |
| 41 | 249.0 | 2391 | 24.91 | 40 | 100.0 / 103.4 % | 64.1 | 77.7 | memory | alu 65% | 0.0 | long_scoreboard 7.53 | 50.0 % | 9.81 (39.4) |

Nsight Compute warnings (launches raising each; example from the longest of them):

- **Warp Stall** (38/41): On average, each warp of this workload spends 4.7 cycles being stalled waiting for a scoreboard dependency on a L1TEX (local, global, surface, texture) operation. Find the instruction producing the data being waited upon to identify the culprit. To reduce the number of cycles waiting on L1TEX data accesses verify the memory access patterns are optimal for the target architecture, attempt to increase cache hit rates by increasing data locality (coalescing), or by changing the cache configuration.
- **Shared Memory Conflicts** (23/41): The memory access pattern for shared loads might not be optimal and causes on average a 2.1 - way bank conflict across all 153024 shared load requests.This results in 168363 bank conflicts,  which represent 51.89% of the overall 324434 wavefronts for shared loads. Check the SourceCounters:Source Counters@ section for uncoalesced shared loads.
- **Launch Configuration** (22/41): A wave of thread blocks is defined as the maximum number of blocks that can be executed in parallel on the target GPU. The number of blocks in a wave depends on the number of multiprocessors and the theoretical occupancy of the kernel. This kernel launch results in 4 full waves and a partial wave of 68 thread blocks. Under the assumption of a uniform execution duration of all thread blocks, this partial wave may account for up to 20.0% of the total runtime of this kernel. Try launching a grid wi
- **High Pipe Utilization** (18/41): ALU is the highest-utilized pipeline (62.8%) based on elapsed cycles in the workload, taking into account the rates of its different instructions. It executes integer and logic operations. The pipeline is well-utilized, but might become a bottleneck if more work is added. Based on the number of executed instructions, the highest utilized pipeline (62.8%) is ALU. It executes integer and logic operations. Comparing the two, the overall pipeline utilization appears to be caused by frequent, low-lat
- **Bottleneck** (14/41): Memory is more heavily utilized than Compute: Look at the MemoryWorkloadAnalysis:Memory Workload Analysis@ section to identify the L2 bottleneck. Check memory replay (coalescing) metrics to make sure you're efficiently utilizing the bytes transferred. Also consider whether it is possible to do more work per memory access (kernel fusion) or whether there are values you can (re)compute.
- **Issue Slot Utilization** (11/41): Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 1.7 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per scheduler, this workload allocates an average of 7.44 active warps per scheduler, but only an average of 1.14 warps were eligible per cycle. Eligible warps are the subset of active warps that are ready to issue their next inst
- **Memory Cache Access Pattern** (3/41): The memory access pattern for global loads from DRAM might not be optimal. On average, only 29.3 of the 32 bytes transmitted per sector are utilized by each thread. This applies to the 97.9% of sectors missed in L2. This could possibly be caused by a stride between threads. Check the SourceCounters:Source Counters@ section for uncoalesced global loads.
- **Uncoalesced Global Accesses** (3/41): This workload has uncoalesced global accesses resulting in a total of 57312 excessive sectors (9% of the total 669408 sectors). Check the L2 Theoretical Sectors Global Excessive table for the primary source locations. The CUDA Programming Guide:https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses@ has additional information on reducing uncoalesced device memory accesses.
- **Uncoalesced Shared Accesses** (2/41): This workload has uncoalesced shared accesses resulting in a total of 612096 excessive wavefronts (67% of the total 918144 wavefronts). Check the L1 Wavefronts Shared Excessive table for the primary source locations. The CUDA Best Practices Guide:https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c-ab@ has an example on optimizing shared memory accesses.
- **Theoretical Occupancy** (2/41): The 8.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the hardware maximum of 12. This kernel's theoretical occupancy (66.7%) is limited by the number of blocks that can fit on the SM, and the required amount of shared memory.

## k3: `sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize256x64x32_stage3_warpsize4x1x1_g1_tensor16x8x16_t1r3s3_linkable_execute_kernel_trt`

10 launches, 2.34 ms under ncu. Time-weighted: SM 56 %, memory 75 %, tensor pipe 15 %, achieved occupancy 16 %, L2 hit 95 %. Time by limiter: memory 93 %, latency 7 %.

| # | µs | Grid | Waves/SM | Regs | Occ. theor./achieved | SM % | Mem % | Limited by | Top pipe | Tensor % | Top stall | L2 hit | Mem read MB (GB/s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 50.6 | 15 | 0.94 | 202 | 16.7 / 16.4 % | 41.0 | 50.4 | latency | alu 15% | 11.3 | wait 1.72 | 94.1 % | 0.6 (11.9) |
| 2 | 51.2 | 15 | 0.94 | 202 | 16.7 / 16.5 % | 41.1 | 49.9 | latency | alu 15% | 11.3 | wait 1.72 | 94.1 % | 0.6 (11.7) |
| 3 | 78.8 | 15 | 0.94 | 202 | 16.7 / 16.1 % | 51.9 | 63.5 | memory | tensor_op_hmma 14% | 14.1 | wait 1.85 | 94.5 % | 1.14 (14.4) |
| 4 | 76.6 | 15 | 0.94 | 202 | 16.7 / 16.0 % | 52.2 | 65.4 | memory | tensor_op_hmma 14% | 14.2 | long_scoreboard 1.95 | 94.5 % | 1.14 (14.8) |
| 5 | 130.9 | 15 | 0.94 | 202 | 16.7 / 16.0 % | 61.4 | 74.5 | memory | tensor_op_hmma 16% | 16.4 | wait 1.95 | 94.6 % | 2.21 (16.8) |
| 6 | 51.1 | 15 | 0.94 | 202 | 16.7 / 16.5 % | 40.0 | 49.9 | latency | alu 15% | 11.3 | wait 1.72 | 94.1 % | 0.6 (11.8) |
| 7 | 270.8 | 57 | 3.56 | 202 | 16.7 / 15.7 % | 54.5 | 71.7 | memory | tensor_op_hmma 15% | 14.6 | wait 1.86 | 95.2 % | 4.1 (15.1) |
| 8 | 164.2 | 57 | 3.56 | 202 | 16.7 / 16.0 % | 45.6 | 61.0 | memory | alu 16% | 12.4 | wait 1.73 | 95.4 % | 1.98 (12.1) |
| 9 | 911.6 | 225 | 14.06 | 202 | 16.7 / 16.3 % | 63.2 | 85.1 | memory | tensor_op_hmma 16% | 16.0 | wait 1.87 | 95.6 % | 15.13 (16.6) |
| 10 | 549.9 | 225 | 14.06 | 202 | 16.7 / 16.1 % | 52.5 | 73.6 | memory | alu 18% | 13.5 | wait 1.75 | 95.8 % | 7.51 (13.7) |

Nsight Compute warnings (launches raising each; example from the longest of them):

- **Issue Slot Utilization** (10/10): Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 3.4 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per scheduler, this workload allocates an average of 1.96 active warps per scheduler, but only an average of 0.35 warps were eligible per cycle. Eligible warps are the subset of active warps that are ready to issue their next inst
- **FP32/64 Instructions** (10/10): This workload executes 115200 fused and 230400 non-fused FP32 instructions. By converting pairs of non-fused instructions to their fused:https://docs.nvidia.com/cuda/floating-point/#cuda-and-floating-point@, higher-throughput equivalent, the achieved FP32 performance could be increased by up to 33% (relative to its current performance).
- **Shared Memory Conflicts** (10/10): The memory access pattern for shared stores might not be optimal and causes on average a 2.9 - way bank conflict across all 57600 shared store requests.This results in 30569 bank conflicts,  which represent 18.58% of the overall 164495 wavefronts for shared stores. Check the SourceCounters:Source Counters@ section for uncoalesced shared stores.
- **Theoretical Occupancy** (10/10): The 2.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the hardware maximum of 12. This kernel's theoretical occupancy (16.7%) is limited by the number of required registers, and the required amount of shared memory.
- **Uncoalesced Shared Accesses** (10/10): This workload has uncoalesced shared accesses resulting in a total of 1303318 excessive wavefronts (27% of the total 4911866 wavefronts). Check the L1 Wavefronts Shared Excessive table for the primary source locations. The CUDA Best Practices Guide:https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c-ab@ has an example on optimizing shared memory accesses.
- **Bottleneck** (9/10): Memory is more heavily utilized than Compute: Look at the MemoryWorkloadAnalysis:Memory Workload Analysis@ section to identify the L2 bottleneck. Check memory replay (coalescing) metrics to make sure you're efficiently utilizing the bytes transferred. Also consider whether it is possible to do more work per memory access (kernel fusion) or whether there are values you can (re)compute.
- **Launch Configuration** (8/10): A wave of thread blocks is defined as the maximum number of blocks that can be executed in parallel on the target GPU. The number of blocks in a wave depends on the number of multiprocessors and the theoretical occupancy of the kernel. This kernel launch results in 3 full waves and a partial wave of 9 thread blocks. Under the assumption of a uniform execution duration of all thread blocks, this partial wave may account for up to 25.0% of the total runtime of this kernel. Try launching a grid wit
- **High Pipe Utilization** (2/10): Tensor is the highest-utilized pipeline (63.2%) based on elapsed cycles in the workload, taking into account the rates of its different instructions. It is the logical aggregation of individual tensor pipelines. It's dominated by its Tensor (FP) sub-pipeline. The pipeline is well-utilized, but might become a bottleneck if more work is added. Based on the number of executed instructions, the highest utilized pipeline (15.9%) is Tensor (FP). It executes 16-bit floating point tensor operations. Com
- **Workload Imbalance** (1/10): One or more SMs have a much lower number of active cycles than the average number of active cycles. Maximum instance value is 6.00% above the average, while the minimum instance value is 14.30% below the average.

## k4: `generatedNativePointwise`

23 launches, 1.65 ms under ncu. Time-weighted: SM 60 %, memory 79 %, tensor pipe 0 %, achieved occupancy 106 %, L2 hit 52 %. Time by limiter: memory 82 %, latency 18 %.

| # | µs | Grid | Waves/SM | Regs | Occ. theor./achieved | SM % | Mem % | Limited by | Top pipe | Tensor % | Top stall | L2 hit | Mem read MB (GB/s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 331.1 | 7200 | 75.0 | 27 | 100.0 / 96.7 % | 69.3 | 90.5 | memory | alu 65% | 0.0 | long_scoreboard 7.78 | 50.0 % | 14.75 (44.6) |
| 2 | 71.7 | 900 | 9.38 | 28 | 100.0 / 112.2 % | 42.6 | 70.1 | memory | alu 31% | 0.9 | long_scoreboard 16.69 | 66.6 % | 1.85 (25.8) |
| 3 | 32.5 | 225 | 2.34 | 27 | 100.0 / 170.5 % | 32.2 | 38.9 | latency | xu 23% | 0.4 | long_scoreboard 16.44 | 66.4 % | 0.47 (14.4) |
| 4 | 32.2 | 225 | 2.34 | 27 | 100.0 / 167.0 % | 31.6 | 39.2 | latency | xu 23% | 0.4 | long_scoreboard 16.44 | 66.4 % | 0.47 (14.6) |
| 5 | 26.2 | 113 | 1.18 | 27 | 100.0 / 196.3 % | 25.4 | 24.3 | latency | xu 19% | 0.3 | long_scoreboard 14.06 | 66.1 % | 0.24 (9.1) |
| 6 | 25.7 | 113 | 1.18 | 27 | 100.0 / 205.2 % | 25.7 | 24.8 | latency | xu 20% | 0.4 | long_scoreboard 13.97 | 66.1 % | 0.24 (9.3) |
| 7 | 25.8 | 113 | 1.18 | 27 | 100.0 / 201.3 % | 23.2 | 24.7 | latency | xu 20% | 0.4 | long_scoreboard 13.94 | 66.1 % | 0.24 (9.2) |
| 8 | 26.0 | 113 | 1.18 | 27 | 100.0 / 200.7 % | 25.2 | 24.5 | latency | xu 20% | 0.4 | long_scoreboard 13.88 | 66.1 % | 0.24 (9.2) |
| 9 | 18.0 | 57 | 0.59 | 28 | 100.0 / 136.2 % | 19.2 | 17.9 | latency | alu 18% | 0.5 | long_scoreboard 10.43 | 66.0 % | 0.12 (6.7) |
| 10 | 26.8 | 450 | 4.69 | 27 | 100.0 / 94.1 % | 56.0 | 71.1 | memory | alu 57% | 0.0 | long_scoreboard 8.53 | 49.9 % | 0.93 (34.6) |
| 11 | 12.7 | 113 | 1.18 | 27 | 100.0 / 98.6 % | 35.4 | 38.4 | latency | alu 42% | 0.0 | long_scoreboard 7.21 | 49.7 % | 0.24 (18.8) |
| 12 | 46.8 | 900 | 9.38 | 27 | 100.0 / 94.3 % | 62.3 | 80.8 | memory | alu 61% | 0.0 | long_scoreboard 8.41 | 50.0 % | 1.85 (39.5) |
| 13 | 26.9 | 450 | 4.69 | 27 | 100.0 / 94.6 % | 55.7 | 70.9 | memory | alu 58% | 0.0 | long_scoreboard 8.65 | 49.9 % | 0.93 (34.5) |
| 14 | 86.7 | 1800 | 18.75 | 27 | 100.0 / 93.4 % | 65.9 | 87.0 | memory | alu 63% | 0.0 | long_scoreboard 8.04 | 50.0 % | 3.69 (42.6) |
| 15 | 87.0 | 1800 | 18.75 | 27 | 100.0 / 94.9 % | 66.6 | 86.7 | memory | alu 63% | 0.0 | long_scoreboard 8.23 | 50.0 % | 3.69 (42.5) |
| 16 | 18.8 | 150 | 1.56 | 40 | 100.0 / 90.4 % | 38.4 | 55.6 | latency | alu 41% | 0.0 | long_scoreboard 9.81 | 33.3 % | 0.62 (33.1) |
| 17 | 18.7 | 150 | 1.56 | 40 | 100.0 / 92.1 % | 37.3 | 57.7 | latency | alu 40% | 0.0 | long_scoreboard 10.12 | 33.5 % | 0.62 (33.3) |
| 18 | 19.3 | 150 | 1.56 | 40 | 100.0 / 92.5 % | 36.2 | 59.8 | latency | alu 37% | 0.0 | long_scoreboard 12.88 | 35.9 % | 0.62 (32.2) |
| 19 | 23.6 | 150 | 1.56 | 24 | 100.0 / 89.3 % | 35.9 | 45.3 | latency | alu 24% | 0.0 | long_scoreboard 14.38 | 33.4 % | 0.62 (26.2) |
| 20 | 27.7 | 299 | 3.11 | 40 | 100.0 / 95.7 % | 57.0 | 66.1 | memory | alu 53% | 0.0 | long_scoreboard 6.21 | 58.6 % | 0.77 (28.0) |
| 21 | 329.4 | 7200 | 75.0 | 27 | 100.0 / 96.0 % | 69.5 | 91.0 | memory | alu 65% | 0.0 | long_scoreboard 7.96 | 50.0 % | 14.75 (44.8) |
| 22 | 327.1 | 7200 | 75.0 | 27 | 100.0 / 95.5 % | 70.0 | 91.8 | memory | alu 65% | 0.0 | long_scoreboard 8.07 | 50.0 % | 14.75 (45.1) |
| 23 | 10.2 | 75 | 0.78 | 26 | 100.0 / 82.0 % | 29.8 | 32.1 | latency | alu 35% | 0.0 | long_scoreboard 4.93 | 49.2 % | 0.16 (16.0) |

Nsight Compute warnings (launches raising each; example from the longest of them):

- **Warp Stall** (22/23): On average, each warp of this workload spends 7.8 cycles being stalled waiting for a scoreboard dependency on a L1TEX (local, global, surface, texture) operation. Find the instruction producing the data being waited upon to identify the culprit. To reduce the number of cycles waiting on L1TEX data accesses verify the memory access patterns are optimal for the target architecture, attempt to increase cache hit rates by increasing data locality (coalescing), or by changing the cache configuration.
- **FP32/64 Instructions** (22/23): This workload executes 230400 fused and 460800 non-fused FP32 instructions. By converting pairs of non-fused instructions to their fused:https://docs.nvidia.com/cuda/floating-point/#cuda-and-floating-point@, higher-throughput equivalent, the achieved FP32 performance could be increased by up to 33% (relative to its current performance).
- **Bottleneck** (16/23): Memory is more heavily utilized than Compute: Look at the MemoryWorkloadAnalysis:Memory Workload Analysis@ section to identify the L2 bottleneck. Check memory replay (coalescing) metrics to make sure you're efficiently utilizing the bytes transferred. Also consider whether it is possible to do more work per memory access (kernel fusion) or whether there are values you can (re)compute.
- **Issue Slot Utilization** (14/23): Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 2.2 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per scheduler, this workload allocates an average of 10.64 active warps per scheduler, but only an average of 0.78 warps were eligible per cycle. Eligible warps are the subset of active warps that are ready to issue their next ins
- **Memory Cache Access Pattern** (14/23): The memory access pattern for local loads from DRAM might not be optimal. On average, only 0.5 of the 32 bytes transmitted per sector are utilized by each thread. This applies to the 100.0% of sectors missed in L2. This could possibly be caused by a stride between threads. Check the SourceCounters:Source Counters@ section for uncoalesced local loads.
- **Launch Configuration** (14/23): A wave of thread blocks is defined as the maximum number of blocks that can be executed in parallel on the target GPU. The number of blocks in a wave depends on the number of multiprocessors and the theoretical occupancy of the kernel. This kernel launch results in 2 full waves and a partial wave of 33 thread blocks. Under the assumption of a uniform execution duration of all thread blocks, this partial wave may account for up to 33.3% of the total runtime of this kernel. Try launching a grid wi
- **High Pipe Utilization** (12/23): ALU is the highest-utilized pipeline (63.8%) based on elapsed cycles in the workload, taking into account the rates of its different instructions. It executes integer and logic operations. The pipeline is well-utilized, but might become a bottleneck if more work is added. Based on the number of executed instructions, the highest utilized pipeline (63.8%) is ALU. It executes integer and logic operations. Comparing the two, the overall pipeline utilization appears to be caused by frequent, low-lat
- **Local Memory Usage** (8/23): This workload accesses local memory:https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses@, accounting for 55.56% of all sectors requested in L1TEX. Local memory refers to a memory space that is private to each thread. Although it is logically distinct, local memory resides in global memory. This can lead to higher memory latency compared to the usage of registers or shared memory, a decrease in available bandwidth to global memory, and an increase in executed i
- **Uncoalesced Global Accesses** (3/23): This workload has uncoalesced global accesses resulting in a total of 14340 excessive sectors (20% of the total 71724 sectors). Check the L2 Theoretical Sectors Global Excessive table for the primary source locations. The CUDA Programming Guide:https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses@ has additional information on reducing uncoalesced device memory accesses.
- **Achieved Occupancy** (2/23): The difference between calculated theoretical (100.0%) and measured achieved occupancy (89.3%) can be the result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices Guide:https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy@ for more details on optimizing occupancy.

## k5: `sm80_xmma_depthwise_convolution_fprop_tiling_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize1x8x8_1x3x3_1x1x1_1x1x1_stage4_warpsize4x1x1_g64_betaFalse_execute_kernel_trt`

5 launches, 1.03 ms under ncu. Time-weighted: SM 42 %, memory 73 %, tensor pipe 0 %, achieved occupancy 9 %, L2 hit 60 %. Time by limiter: memory 93 %, latency 7 %.

| # | µs | Grid | Waves/SM | Regs | Occ. theor./achieved | SM % | Mem % | Limited by | Top pipe | Tensor % | Top stall | L2 hit | Mem read MB (GB/s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 67.1 | 8 | 0.5 | 132 | 16.7 / 10.0 % | 38.2 | 53.2 | latency | fma_type_fp16 24% | 0.0 | wait 0.94 | 60.0 % | 1.88 (28.0) |
| 2 | 105.8 | 8 | 0.5 | 132 | 16.7 / 9.4 % | 40.2 | 68.0 | memory | fma_type_fp16 25% | 0.0 | wait 0.97 | 60.2 % | 3.72 (35.2) |
| 3 | 105.0 | 8 | 0.5 | 132 | 16.7 / 9.3 % | 40.4 | 68.5 | memory | fma_type_fp16 25% | 0.0 | wait 0.97 | 60.2 % | 3.72 (35.4) |
| 4 | 375.6 | 8 | 0.5 | 132 | 16.7 / 9.2 % | 42.7 | 76.8 | memory | fma_type_fp16 26% | 0.0 | wait 0.98 | 60.6 % | 14.78 (39.3) |
| 5 | 377.5 | 8 | 0.5 | 132 | 16.7 / 9.3 % | 42.1 | 76.2 | memory | fma_type_fp16 26% | 0.0 | wait 0.98 | 60.6 % | 14.79 (39.2) |

Nsight Compute warnings (launches raising each; example from the longest of them):

- **Bottleneck** (5/5): Memory is more heavily utilized than Compute: Look at the MemoryWorkloadAnalysis:Memory Workload Analysis@ section to identify the L2 bottleneck. Check memory replay (coalescing) metrics to make sure you're efficiently utilizing the bytes transferred. Also consider whether it is possible to do more work per memory access (kernel fusion) or whether there are values you can (re)compute.
- **Issue Slot Utilization** (5/5): Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 2.3 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per scheduler, this workload allocates an average of 1.00 active warps per scheduler, but only an average of 0.43 warps were eligible per cycle. Eligible warps are the subset of active warps that are ready to issue their next inst
- **Warp Stall** (5/5): On average, each warp of this workload spends 1.0 cycles being stalled waiting on a fixed latency execution dependency. Typically, this stall reason should be very low and only shows up as a top contributor in already highly optimized kernels. Try to hide the corresponding instruction latencies by increasing the number of active warps, restructuring the code or unrolling loops. Furthermore, consider switching to lower-latency instructions, e.g. by making use of fast math compiler options. This s
- **Launch Configuration** (5/5): If you execute __syncthreads() to synchronize the threads of a block, it is recommended to have at least two blocks per multiprocessor (compared to the currently executed 1.0 blocks) This way, blocks that aren't waiting for __syncthreads() can keep the hardware busy.
- **Shared Memory Conflicts** (5/5): The memory access pattern for shared stores might not be optimal and causes on average a 1.6 - way bank conflict across all 115200 shared store requests.This results in 43545 bank conflicts,  which represent 23.45% of the overall 185691 wavefronts for shared stores. Check the SourceCounters:Source Counters@ section for uncoalesced shared stores.
- **Theoretical Occupancy** (5/5): The 2.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the hardware maximum of 12. This kernel's theoretical occupancy (16.7%) is limited by the required amount of shared memory.

