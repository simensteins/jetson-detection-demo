# Nsight Compute YOLO
*Exported from Notion (page last edited 2026-10-06).*

Inspecting the `custom_nms_display_test.nsys-rep` (YOLO Iteration 3):

- 8.90 ms GPU kernel time per frame, 220 launches per frame, 38 distinct kernel names

## The heaviest kernels:

- **#: 1**\
  **Share (cumulative):** 29.7% (29.7%)\
  **ms/frame:** 2.64\
  **Launches/frame:** 24\
  **Kernel:** `sm80_xmma_fprop_implicit_gemm_f32f32_tf32f32_f32_...t1r3s3...`\
  **Type:** 3×3 convolution (TensorRT)
- **#: 2**\
  **Share (cumulative):** 16.2% (45.8%)\
  **ms/frame:** 1.44\
  **Launches/frame:** 63\
  **Kernel:** `generatedNativePointwise`\
  **Type:** fused element-wise ops, e.g. SiLU activations (TensorRT)
- **#: 3**\
  **Share (cumulative):** 5.8% (51.6%)\
  **ms/frame:** 0.52\
  **Launches/frame:** 16\
  **Kernel:** `copyPackedKernel`\
  **Type:** data copying / layout (TensorRT)
- **#: 4–5**\
  **Share (cumulative):** 9.7% (61.3%)\
  **ms/frame:** 0.86\
  **Launches/frame:** 10\
  **Kernel:** other `implicit_gemm_indexed` variants\
  **Type:** convolutions

### Findings:

1. The “FP32” YOLO engine actually computes in TF32
2. The heaviest kernel is probably low-occupancy because of tiny grids. These two signs are estimtates, need to be comfirmed in Nsight Compute:
   1. Grid Size: the same kernel runs grids of only 4, 8, 13, 26, 50 and 100 thread blocks. Most launches are the small ones: per frame, about 8 at 4 blocks and 14 at 13 blocks.
   2. Registers: 238 registers per thread x 128 thread per block probably gives room for only \~2 blocks per SM
   3. About 11% of the time is pure data shuffling

Suggested Nsight Compute targets for YOLO: The top 3 cover 51.6%, which fits the “≥ 50%” criterion, and they give nice spread: compute-heavy, element-wise and pure copying.

**For the paper, state the hypotheses up front and test them, e.g.:**

- **H1:** The heaviest kernels in both models are TF32 GEMMs (confirmed for YOLO by the kernel name).
- **H2:** YOLO's heaviest kernels underutilise the GPU because of small grids at batch 1. RF-DETR's GEMMs over many tokens fill it better.
- **H3:** RF-DETR has a larger share of memory-bound kernels (LayerNorm, softmax, deformable attention) than YOLO.
