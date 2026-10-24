#!/usr/bin/env python3
"""Build the YOLOv8n FP16 engine with FP16 I/O for scripts/infer_cuda_graph_fp16.py.

Ultralytics' own export (`YOLO(...).export(format="engine", half=True)`) builds
FP16 layers but keeps the engine's input/output FP32, so TensorRT adds
reformat kernels at both ends every frame - and infer_cuda_graph_fp16.py
refuses such an engine. This script builds the engine from the ONNX that
Ultralytics' export leaves next to the .pt, with trtexec --fp16 and FP16 I/O
(--inputIOFormats/--outputIOFormats fp16:chw), the same way scripts/export_m15.py
builds the M15 FP16 engine.

The result keeps Ultralytics' engine file format - a 4-byte length + JSON
metadata header in front of the TensorRT plan, copied from the Ultralytics
engine (--meta-from) - so Ultralytics' AutoBackend, which the YOLO scripts
load engines through, reads it as before (class names, imgsz, stride).

Must run on the Jetson (engines are specific to the GPU and TensorRT version).

Usage (from the repo root, after the Ultralytics half=True export has written
yolov8n_fp16.onnx and yolov8n_fp16.engine):
    python3 scripts/export_yolo_fp16.py
    python3 scripts/infer_cuda_graph_fp16.py --config configs/default.yaml \\
        --engine yolov8n_fp16_io.engine --source data/vtest.avi --no-display --max-frames 600
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import tensorrt as trt

TRTEXEC_DEFAULT = "/usr/src/tensorrt/bin/trtexec"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLOv8n ONNX -> FP16 engine with FP16 I/O")
    p.add_argument("--onnx", default="yolov8n_fp16.onnx", help="ONNX from the Ultralytics export")
    p.add_argument("--meta-from", default="yolov8n_fp16.engine",
                   help="Ultralytics-exported engine whose metadata header is copied")
    p.add_argument("--out", default="yolov8n_fp16_io.engine", help="Output engine")
    p.add_argument("--trtexec", default=None, help=f"trtexec path (default: PATH, then {TRTEXEC_DEFAULT})")
    p.add_argument("--mem-pool-mb", type=int, default=None, help="Cap the builder workspace (MiB)")
    return p.parse_args()


def read_ultralytics_meta(path: Path) -> dict:
    data = path.read_bytes()
    meta_len = int.from_bytes(data[:4], byteorder="little", signed=True)
    try:
        return json.loads(data[4:4 + meta_len].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise SystemExit(f"{path} has no Ultralytics metadata header ({e}) - pass an "
                         "Ultralytics-exported engine with --meta-from") from e


def find_trtexec(override: str | None) -> str:
    for cand in (override, shutil.which("trtexec"), TRTEXEC_DEFAULT):
        if cand and Path(cand).exists():
            return cand
    raise SystemExit(f"trtexec not found - pass --trtexec (usually {TRTEXEC_DEFAULT}).")


def main() -> None:
    args = parse_args()
    onnx_path, meta_path, out_path = Path(args.onnx), Path(args.meta_from), Path(args.out)
    for path in (onnx_path, meta_path):
        if not path.exists():
            raise SystemExit(f"Not found: {path} - run the Ultralytics half=True export first")
    meta = read_ultralytics_meta(meta_path)
    meta.setdefault("args", {})["fp16_io"] = True  # record how this file differs

    with tempfile.TemporaryDirectory() as tmp:
        plan_path = Path(tmp) / "plan.engine"
        cmd = [find_trtexec(args.trtexec), f"--onnx={onnx_path}", f"--saveEngine={plan_path}",
               "--fp16", "--inputIOFormats=fp16:chw", "--outputIOFormats=fp16:chw"]
        if args.mem_pool_mb:
            cmd.append(f"--memPoolSize=workspace:{args.mem_pool_mb}")
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        plan = plan_path.read_bytes()

    # Verify, don't trust the flags: the plan must load here and have FP16 I/O.
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(plan)
    if engine is None:
        raise SystemExit("The built engine does not deserialize with this TensorRT")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        dtype = engine.get_tensor_dtype(name)
        print(f"  {engine.get_tensor_mode(name).name:6s} {name!r} "
              f"shape={tuple(engine.get_tensor_shape(name))} dtype={dtype.name}")
        if dtype != trt.DataType.HALF:
            raise SystemExit(f"{name!r} is {dtype.name}, expected HALF - not writing {out_path}")

    header = json.dumps(meta).encode("utf-8")
    out_path.write_bytes(len(header).to_bytes(4, byteorder="little", signed=True) + header + plan)
    print(f"wrote {out_path} (Ultralytics header from {meta_path} + FP16-I/O plan)")


if __name__ == "__main__":
    main()
