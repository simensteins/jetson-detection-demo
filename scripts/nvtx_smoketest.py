#!/usr/bin/env python3
"""Minimal NVTX capture test.

Isolates whether torch.cuda.nvtx events are visible to nsys on this
GPU/torch build, independent of detect.py's GStreamer/TensorRT pipeline.

Run:
    nsys profile --trace=nvtx -o reports/nvtx_smoketest python3 scripts/nvtx_smoketest.py
"""
import time

import torch

print("CUDA available:", torch.cuda.is_available())

for i in range(50):
    torch.cuda.nvtx.range_push(f"iter_{i}")
    time.sleep(0.05)
    torch.cuda.nvtx.range_pop()

print("done")
