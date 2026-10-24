#!/usr/bin/env python3
"""Print a TensorRT engine's I/O tensors (name, mode, shape, dtype) - to tell
engine files apart without trusting their names, e.g. whether an "fp16"
engine really has FP16 I/O, or whether an export added NMS (an end2end
engine's outputs are detections, not the raw (1, 84, 8400) predictions).

Usage (on the Jetson - engines only load on the device/TensorRT they were built for):
    python3 scripts/inspect_engine.py yolov8n_fp16.engine yolov8n_fp16_e2e.engine
"""
from __future__ import annotations

import sys

import tensorrt as trt


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: inspect_engine.py <file.engine> [...]")
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    for path in sys.argv[1:]:
        with open(path, "rb") as f:
            data = f.read()
        # Ultralytics prefixes its engines with a 4-byte length + JSON metadata.
        engine = runtime.deserialize_cuda_engine(data)
        if engine is None and len(data) > 4:
            meta_len = int.from_bytes(data[:4], byteorder="little")
            engine = runtime.deserialize_cuda_engine(data[4 + meta_len:])
            if engine is not None:
                print(f"{path}: Ultralytics metadata: {data[4:4 + meta_len].decode(errors='replace')}")
        if engine is None:
            print(f"{path}: could not deserialize")
            continue
        print(f"{path}:")
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            print(f"  {engine.get_tensor_mode(name).name:6s} {name!r:12s} "
                  f"shape={tuple(engine.get_tensor_shape(name))} dtype={engine.get_tensor_dtype(name).name}")


if __name__ == "__main__":
    main()
