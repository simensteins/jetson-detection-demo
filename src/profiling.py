"""NVTX range helper.

Uses torch.cuda.nvtx when a CUDA build of torch is present; otherwise the
context manager is a no-op, so the same code runs on a laptop or the Jetson.
"""
from __future__ import annotations

from contextlib import contextmanager

try:
    import torch
    _NVTX = torch.cuda.is_available()
except Exception:
    _NVTX = False


@contextmanager
def nvtx_range(name: str):
    if _NVTX:
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield
