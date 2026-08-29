"""v1 object-detection demo: read a video source, run YOLO, draw boxes, display.

Each frame's work is wrapped in NVTX ranges (decode / inference / draw /
display) so an Nsight Systems timeline splits into readable phases instead of
a wall of anonymous CUDA kernels.
"""
from __future__ import annotations

import argparse
import time

import cv2
import yaml
from ultralytics import YOLO

from .profiling import nvtx_range
from .sources import open_source


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Jetson YOLO detection demo (v1)")
    p.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    p.add_argument("--source", default=None,
                   help="Override source: file path, rtsp:// URL, or webcam index")
    p.add_argument("--no-display", action="store_true", help="Run headless (no window)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Override max_frames from config (0 = run to end of source)")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    source = args.source if args.source is not None else cfg["source"]
    display = cfg.get("display", True) and not args.no_display
    conf = cfg.get("conf", 0.25)
    imgsz = cfg.get("imgsz", 640)
    max_frames = args.max_frames if args.max_frames is not None else cfg.get("max_frames", 0)

    model = YOLO(cfg["model"])  # weights auto-download on first run
    cap = open_source(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {source!r}")

    frames = 0
    t0 = time.perf_counter()
    try:
        while True:
            with nvtx_range("decode"):
                ok, frame = cap.read()
            if not ok:
                break

            with nvtx_range("inference"):
                results = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)

            with nvtx_range("draw"):
                annotated = results[0].plot()

            if display:
                with nvtx_range("display"):
                    cv2.imshow("detections", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            frames += 1
            if max_frames and frames >= max_frames:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    dt = time.perf_counter() - t0
    if frames:
        print(f"processed {frames} frames in {dt:.1f}s  ->  {frames / dt:.1f} FPS")


if __name__ == "__main__":
    main()
