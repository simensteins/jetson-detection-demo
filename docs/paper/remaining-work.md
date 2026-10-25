# Remaining work before paper
*Exported from Notion (page last edited 2026-10-02).*

**Remaining stuff for the Nsight Systems part of the project:**

- [ ] Test the RF-DETR large (704) and do CUDA graph (transferrable results)
- [ ] Produce some good figures from the Nsight Systems timeline
- [ ] Dig a bit deeper into the numbers behind and why we tried the different stuff
- [ ] Possibly compare time vs. accuracy?

**Since GPU kernel execution dominates inference time after optimizations, we need to have some nsight compute analysis in the paper:**

- [ ] Identify the heaviest kernels (`nsys stats —report cuda_gpu_kern_sum` )
- [ ] Select the 3-5 kernels that together account for most of the GPU time. Map kernel names to model layers using TensorRT’s per-layer NVTX ranges and per-layer timings
- [ ] Profile the selected kernels with Nsight Compute: Restricted to kernels inside the *inference* NVTX range, after warm-up, a few instances per kernel (`—set full` ). Clock control set to match the Nsight Systems runs, so kernel times are comparable
- [ ] Characterize each kernel. Suggestions: Compute throughput and memory throughput as % of peak, roofline position (compute-bound vs. memory-bound), achieved occupancy, Tensor Core usage (perhaps FP32 vs. TF32 engine)
- [ ] Interpret the results: What limits the GPU work on the Jetson Orin Nano. Can point to future iterations for the master’s thesis.

[Nsight Compute YOLO](../research/nsight-compute-yolo.md)
