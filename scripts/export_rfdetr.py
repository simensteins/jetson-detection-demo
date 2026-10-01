#!/usr/bin/env python3
"""Export RF-DETR to ONNX, then build a TensorRT engine from it with trtexec.

The RF-DETR counterpart of `YOLO(...).export(format="engine")`: rfdetr has no
one-call TensorRT export, so this does the two steps explicitly.

1. ONNX export via rfdetr's own `model.export()` (needs `pip install
   "rfdetr[onnxexport]"`). This step does not need the GPU - if installing the
   export extras on the Jetson is painful, run it on a laptop with
   --onnx-only and copy the .onnx over.
2. Engine build via trtexec (ships with JetPack at /usr/src/tensorrt/bin).
   This step MUST run on the Jetson itself - TensorRT engines are specific to
   the GPU and TensorRT version they were built on.

--precision fp16 builds with --fp16 AND declares the engine's I/O tensors as
FP16 (--inputIOFormats/--outputIOFormats), so the whole pipeline in
scripts/infer_rfdetr_trt.py stays FP16 end-to-end - the same reasoning as
Iteration 4 (scripts/infer_cuda_graph_fp16.py): FP32 I/O around an FP16 engine
means TensorRT inserts reformatting kernels at both boundaries every frame.
Transformers are more sensitive to FP16 than CNNs (LayerNorm/softmax ranges),
so check detections visually and compare against the FP32 engine before
trusting an FP16 speed number.

--precision fp32 is "FP32" in TensorRT's default sense: on Ampere GPUs (the
Jetson Orin included) TensorRT may run matmuls/convolutions in TF32 (10-bit
mantissa) unless told otherwise. --no-tf32 passes --noTF32 for strict FP32;
the engine then gets a "_notf32" suffix so the two can't be mixed up.

The Orin Nano shares its memory between CPU and GPU, and building the Large
engine (704x704) can run out of it - --mem-pool-mb caps TensorRT's builder
workspace (e.g. 2048).

Usage (from the repo root):
    python3 scripts/export_rfdetr.py --size nano --precision fp32
    python3 scripts/export_rfdetr.py --size nano --precision fp16
    # ONNX on another machine, engine on the Jetson:
    python3 scripts/export_rfdetr.py --size nano --onnx-only
    python3 scripts/export_rfdetr.py --size nano --onnx models/rfdetr-nano.onnx --precision fp16
    # Large on the Orin Nano, default FP32 (TF32 allowed) and strict FP32 from the same ONNX:
    python3 scripts/export_rfdetr.py --size large --precision fp32 --mem-pool-mb 2048
    python3 scripts/export_rfdetr.py --size large --precision fp32 --no-tf32 --mem-pool-mb 2048 \
        --onnx models/rfdetr-large.onnx
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.detectors import RFDETR_CLASSES, load_rfdetr  # noqa: E402

TRTEXEC_DEFAULT = "/usr/src/tensorrt/bin/trtexec"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export RF-DETR to ONNX + TensorRT engine")
    p.add_argument("--size", default="nano", choices=sorted(RFDETR_CLASSES), help="RF-DETR model size")
    p.add_argument("--weights", default=None, help="Custom-trained .pth (default: COCO-pretrained)")
    p.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    p.add_argument("--outdir", default="models", help="Where the .onnx/.engine are written")
    p.add_argument("--onnx", default=None, help="Skip ONNX export and build from this .onnx")
    p.add_argument("--onnx-only", action="store_true", help="Export ONNX, don't build an engine")
    p.add_argument("--trtexec", default=None, help=f"trtexec path (default: PATH, then {TRTEXEC_DEFAULT})")
    p.add_argument("--no-tf32", action="store_true",
                   help="Strict FP32: pass --noTF32 to trtexec (engine name gets a _notf32 suffix)")
    p.add_argument("--mem-pool-mb", type=int, default=None,
                   help="Cap the TensorRT builder workspace (MiB), e.g. 2048 on the Orin Nano")
    return p.parse_args()


def export_onnx(size: str, weights: str | None, outdir: Path) -> Path:
    name = f"rfdetr-{size}" + (f":{weights}" if weights else "")
    model = load_rfdetr(name)
    export_dir = outdir / f"rfdetr-{size}_onnx"
    print(f"Exporting {name} to ONNX in {export_dir} ...")
    model.export(output_dir=str(export_dir))
    onnx_files = sorted(export_dir.glob("*.onnx"))
    if not onnx_files:
        raise SystemExit(f"rfdetr export produced no .onnx in {export_dir}")
    dst = outdir / f"rfdetr-{size}.onnx"
    shutil.copyfile(onnx_files[0], dst)
    return dst


def onnx_io_names(onnx_path: Path) -> tuple[list[str], list[str]]:
    """Read input/output tensor names so the FP16 I/O format flags cover every tensor."""
    try:
        import onnx
    except ImportError:
        return [], []
    m = onnx.load(str(onnx_path), load_external_data=False)
    inits = {i.name for i in m.graph.initializer}
    return ([i.name for i in m.graph.input if i.name not in inits],
            [o.name for o in m.graph.output])


def find_trtexec(override: str | None) -> str:
    for cand in (override, shutil.which("trtexec"), TRTEXEC_DEFAULT):
        if cand and Path(cand).exists():
            return cand
    raise SystemExit("trtexec not found - pass --trtexec (on the Jetson it's usually "
                     f"{TRTEXEC_DEFAULT}).")


def build_engine(onnx_path: Path, engine_path: Path, precision: str, trtexec: str,
                 no_tf32: bool = False, mem_pool_mb: int | None = None) -> None:
    cmd = [trtexec, f"--onnx={onnx_path}", f"--saveEngine={engine_path}"]
    if no_tf32:
        cmd.append("--noTF32")
    if mem_pool_mb:
        cmd.append(f"--memPoolSize=workspace:{mem_pool_mb}")
    if precision == "fp16":
        cmd.append("--fp16")
        inputs, outputs = onnx_io_names(onnx_path)
        # rfdetr exports 1 input and 2 outputs (boxes, class logits). Fall back
        # to that count if onnx isn't installed to read the names.
        n_in, n_out = (len(inputs), len(outputs)) if inputs else (1, 2)
        cmd.append("--inputIOFormats=" + ",".join(["fp16:chw"] * n_in))
        cmd.append("--outputIOFormats=" + ",".join(["fp16:chw"] * n_out))
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    onnx_path = Path(args.onnx) if args.onnx else export_onnx(args.size, args.weights, outdir)
    print(f"ONNX: {onnx_path}")
    if args.onnx_only:
        return

    suffix = "_notf32" if args.no_tf32 else ""
    engine_path = outdir / f"rfdetr-{args.size}_{args.precision}{suffix}.engine"
    build_engine(onnx_path, engine_path, args.precision, find_trtexec(args.trtexec),
                 no_tf32=args.no_tf32, mem_pool_mb=args.mem_pool_mb)
    print(f"wrote {engine_path}")
    config = "configs/rfdetr-large.yaml" if args.size == "large" else "configs/rfdetr.yaml"
    print(f"next: python3 scripts/infer_rfdetr_trt.py --config {config} --engine {engine_path}")


if __name__ == "__main__":
    main()
