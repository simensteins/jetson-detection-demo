#!/usr/bin/env python3
"""Diagnostic: inspect Ultralytics' AutoBackend TensorRT object after it
successfully loads an engine, to find the real attribute names (context,
engine, bindings, etc.) for reusing its working execution context instead
of re-deserializing the engine ourselves (which hits a dispatch-runtime
mismatch on this TensorRT version - see scripts/infer_cuda_graph.py).

Usage:
    python3 scripts/inspect_autobackend.py [path/to/model.engine]
"""
import sys

import numpy as np
from ultralytics import YOLO

engine_path = sys.argv[1] if len(sys.argv) > 1 else "yolov8n.engine"

print(f"Loading {engine_path} via Ultralytics (known-working path)...")
m = YOLO(engine_path)

# YOLO() is lazy - m.model is just the path string until a predict() call
# actually triggers backend initialization.
m.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)

backend = m.predictor.model
print(f"\nbackend type: {type(backend)}")
print(f"\nbackend attributes:\n{[a for a in dir(backend) if not a.startswith('_')]}")

for name in ("context", "engine", "bindings", "model", "runtime", "output_names", "input_names"):
    if hasattr(backend, name):
        val = getattr(backend, name)
        print(f"\n{name}: {type(val)} = {val!r:.200}")
