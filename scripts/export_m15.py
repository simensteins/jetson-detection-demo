#!/usr/bin/env python3
"""Export the m15 model to ONNX, then build a TensorRT engine from it with trtexec.

The m15 counterpart of scripts/export_rfdetr.py: Ultralytics does have a
one-call engine export, but going ONNX -> trtexec explicitly keeps the same
steps, flags and file naming as the RF-DETR experiments, so results trace to
one engine file.

1. ONNX export via Ultralytics YOLO(...).export(format="onnx"): static
   batch-1 input at the training image size (960x960, 1 channel), no NMS in
   the graph. The output is (1, 4+nc, N): DFL-decoded xywh boxes in input
   pixels and sigmoid class scores, over the N = 76,590 anchors of the four
   strides (240^2 + 120^2 + 60^2 + 30^2 at 960x960).
2. Engine build via trtexec (ships with JetPack at /usr/src/tensorrt/bin).
   This step MUST run on the Jetson itself - TensorRT engines are specific to
   the GPU and TensorRT version they were built on.

The precision is named in the engine file so every result traces to one
(same convention as scripts/export_rfdetr.py):

    --precision fp32-tf32 (default)  FP32, but TensorRT may run matmuls/
                                     convolutions in TF32 (10-bit mantissa) -
                                     TensorRT's own default on the Orin
                                     -> m15_fast_fp32-tf32.engine
    --precision fp32-strict         every layer in full FP32 (trtexec --noTF32)
                                     -> m15_fast_fp32-strict.engine
    --precision fp16                 FP16 layers (trtexec --fp16) AND FP16 I/O
                                     (--inputIOFormats/--outputIOFormats fp16:chw),
                                     so the pipeline around the engine stays
                                     FP16 too - FP32 I/O around an FP16 engine
                                     makes TensorRT add reformat kernels at both
                                     ends every frame (as in Iteration 4,
                                     scripts/infer_cuda_graph_fp16.py)
                                     -> m15_fast_fp16.engine

The Orin Nano shares its memory between CPU and GPU - --mem-pool-mb caps
TensorRT's builder workspace (e.g. 2048).

Usage (from the repo root):
    python3 scripts/export_m15.py --precision fp32-tf32 --mem-pool-mb 2048
    python3 scripts/export_m15.py --onnx models/m15/m15_fast_960.onnx --precision fp16 --mem-pool-mb 2048
    # ONNX without building an engine:
    python3 scripts/export_m15.py --onnx-only
    # Engine from an existing ONNX:
    python3 scripts/export_m15.py --onnx models/m15/m15_fast_960.onnx
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRTEXEC_DEFAULT = "/usr/src/tensorrt/bin/trtexec"
IMGSZ = 960  # the m15 training image size; the engine input is fixed at 960x960


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export m15 to ONNX + TensorRT engine")
    p.add_argument("--pt", default=str(ROOT / "models" / "m15" / "m15_fast_best_ema.pt"),
                   help="Ultralytics .pt from scripts/build_m15_pt.py")
    p.add_argument("--precision", default="fp32-tf32", choices=["fp32-strict", "fp32-tf32", "fp16"],
                   help="fp32-tf32: FP32 with TensorRT's default TF32 matmuls (default); "
                        "fp32-strict: every layer in full FP32 (trtexec --noTF32); "
                        "fp16: FP16 layers and FP16 engine input/output")
    p.add_argument("--outdir", default=str(ROOT / "models" / "m15"),
                   help="Where the .onnx/.engine are written")
    p.add_argument("--onnx", default=None, help="Skip ONNX export and build from this .onnx")
    p.add_argument("--onnx-only", action="store_true", help="Export ONNX, don't build an engine")
    p.add_argument("--trtexec", default=None,
                   help=f"trtexec path (default: PATH, then {TRTEXEC_DEFAULT})")
    p.add_argument("--mem-pool-mb", type=int, default=None,
                   help="Cap the TensorRT builder workspace (MiB), e.g. 2048 on the Orin Nano")
    return p.parse_args()


def export_onnx(pt_path: Path, outdir: Path) -> Path:
    from ultralytics import YOLO

    print(f"Exporting {pt_path} to ONNX ({IMGSZ}x{IMGSZ}, 1 channel, static batch 1) ...")
    exported = YOLO(str(pt_path)).export(format="onnx", imgsz=IMGSZ, dynamic=False,
                                         simplify=True, device="cpu")
    src = Path(exported)
    if not src.exists():
        raise SystemExit(f"ONNX export produced no file at {src}")
    dst = outdir / "m15_fast_960.onnx"
    shutil.copyfile(src, dst)
    print(f"ONNX: {dst}")
    return dst


def find_trtexec(override: str | None) -> str:
    for cand in (override, shutil.which("trtexec"), TRTEXEC_DEFAULT):
        if cand and Path(cand).exists():
            return cand
    raise SystemExit("trtexec not found - pass --trtexec (on the Jetson it's usually "
                     f"{TRTEXEC_DEFAULT}).")


def build_engine(onnx_path: Path, engine_path: Path, precision: str, trtexec: str,
                 mem_pool_mb: int | None = None) -> None:
    cmd = [trtexec, f"--onnx={onnx_path}", f"--saveEngine={engine_path}"]
    if precision == "fp32-strict":
        cmd.append("--noTF32")
    if precision == "fp16":
        # The m15 ONNX has one input ('images') and one output ('output0').
        cmd += ["--fp16", "--inputIOFormats=fp16:chw", "--outputIOFormats=fp16:chw"]
    if mem_pool_mb:
        cmd.append(f"--memPoolSize=workspace:{mem_pool_mb}")
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    onnx_path = (Path(args.onnx) if args.onnx
                 else export_onnx(Path(args.pt), outdir))
    if args.onnx_only:
        return

    engine_path = outdir / f"m15_fast_{args.precision}.engine"
    build_engine(onnx_path, engine_path, args.precision, find_trtexec(args.trtexec),
                 mem_pool_mb=args.mem_pool_mb)
    print(f"wrote {engine_path}")
    print(f"next: python3 scripts/infer_m15_trt.py --engine {engine_path}")


if __name__ == "__main__":
    main()
