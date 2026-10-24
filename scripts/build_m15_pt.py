#!/usr/bin/env python3
"""Rebuild the m15_fast_detector training checkpoint as an Ultralytics .pt.

The SAR training pipeline saves a raw training checkpoint (state dicts under a
"detector._inner." / "_inner." prefix), not an Ultralytics model file. The
architecture is a standard yolo11s with one change: the Detect head also takes
the P2 feature (backbone layer 2), giving strides 4/8/16/32
(models/m15/yolo11s-p2-m15.yaml). This script rebuilds that model, loads the
checkpoint's weights strict (any architecture mismatch is an error), and saves
an Ultralytics-loadable .pt for ONNX export.

Weights: --weights ema (default) loads the EMA state dict, the standard choice
for evaluation; --weights model loads the raw training weights. They differ
(the EMA is a smoothed copy), and best.pt was selected on the checkpoint score,
so the EMA is what the reported validation metrics describe.

The checkpoint carries no class names (single SAR/IR class); the model is
labelled "target" for drawing.

Usage (from the repo root):
    python3 scripts/build_m15_pt.py                       # -> models/m15/m15_fast_best_ema.pt
    python3 scripts/build_m15_pt.py --weights model      # -> models/m15/m15_fast_best_model.pt
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_YAML = ROOT / "models" / "m15" / "yolo11s-p2-m15.yaml"
DEFAULT_CKPT = Path.home() / "m15_fast_detector" / "m15_fast_detector" / "best.pt"

PREFIXES = {"ema": "detector._inner.", "model": "_inner."}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="m15 checkpoint -> Ultralytics .pt")
    p.add_argument("--checkpoint", default=str(DEFAULT_CKPT),
                   help=f"Training checkpoint (default: {DEFAULT_CKPT})")
    p.add_argument("--weights", default="ema", choices=sorted(PREFIXES),
                   help="Which state dict to load: ema (default) or model (raw)")
    p.add_argument("--yaml", default=str(DEFAULT_YAML),
                   help=f"Architecture YAML (default: {DEFAULT_YAML})")
    p.add_argument("--out", default=None,
                   help="Output .pt path (default: models/m15/m15_fast_best_<weights>.pt)")
    p.add_argument("--no-verify", action="store_true",
                   help="Skip the reload-and-predict verification")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from ultralytics.nn.tasks import DetectionModel

    print(f"Building {args.yaml} (ch=1, nc=1) ...")
    model = DetectionModel(args.yaml, ch=1, nc=1, verbose=False)
    strides = model.model[-1].stride.tolist()
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}  strides: {strides}")

    print(f"Loading {args.weights} weights from {args.checkpoint} ...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    prefix = PREFIXES[args.weights]
    sd = {k[len(prefix):]: v for k, v in ckpt[f"{args.weights}_state_dict"].items()
          if k.startswith(prefix)}
    if len(sd) != len(ckpt[f"{args.weights}_state_dict"]):
        raise SystemExit(f"Checkpoint keys don't start with {prefix!r} - not an m15 checkpoint?")
    model.load_state_dict(sd, strict=True)  # strict: the architecture must match exactly
    print(f"  strict load OK ({len(sd)} tensors)")
    model.names = {0: "target"}

    out = Path(args.out) if args.out else ROOT / "models" / "m15" / f"m15_fast_best_{args.weights}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "train_args": {"model": str(Path(args.yaml).name), "task": "detect", "imgsz": 960},
        "m15": {
            "source_checkpoint": str(Path(args.checkpoint).resolve()),
            "weights": args.weights,
            "epoch": ckpt.get("epoch"),
            "best_metric": ckpt.get("best_metric"),
            "run_id": (ckpt.get("_metadata") or {}).get("run_id"),
        },
        "date": datetime.now().isoformat(timespec="seconds"),
    }
    torch.save(payload, out)
    print(f"saved {out}")

    if args.no_verify:
        return
    print("Verifying: reload via YOLO() and predict one frame ...")
    import cv2
    from ultralytics import YOLO

    yolo = YOLO(str(out))
    cap = cv2.VideoCapture(str(ROOT / "data" / "vtest.avi"))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Could not read data/vtest.avi for the verification frame")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    r = yolo.predict(gray, imgsz=960, conf=0.25, verbose=False)[0]
    print(f"  predict OK: {len(r.boxes)} boxes, speed (ms): "
          f"{ {k: round(v, 1) for k, v in r.speed.items()} }")


if __name__ == "__main__":
    main()
