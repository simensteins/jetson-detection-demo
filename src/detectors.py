"""Model backends for the baseline demo (src/detect.py).

Each backend splits a frame's work into infer() and draw() so detect.py can
keep wrapping them in separate "inference" / "draw" NVTX ranges, whichever
model is loaded.

- YOLO (Ultralytics): any .pt / .engine / .onnx path Ultralytics can load.
- RF-DETR (Roboflow, `pip install rfdetr`): selected with a model name of the
  form "rfdetr-<size>", e.g. "rfdetr-nano", "rfdetr-base". Weights
  auto-download on first run, like yolov8n.pt. Optionally "rfdetr-<size>:<path>"
  loads custom-trained weights for that size.
"""
from __future__ import annotations

import cv2
import numpy as np

# rfdetr class names per size. Not every rfdetr release has every size, so
# they're looked up lazily and a missing one gives a clear error.
RFDETR_CLASSES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "base": "RFDETRBase",
    "large": "RFDETRLarge",
}


def is_rfdetr(model: str) -> bool:
    return str(model).lower().startswith("rfdetr")


def parse_rfdetr_name(model: str) -> tuple[str, str | None]:
    """'rfdetr-nano' -> ('nano', None); 'rfdetr-base:models/x.pth' -> ('base', 'models/x.pth')."""
    name, _, weights = str(model).partition(":")
    size = name.lower()[len("rfdetr"):].lstrip("-_") or "base"
    if size not in RFDETR_CLASSES:
        raise SystemExit(f"Unknown RF-DETR size {size!r} in {model!r}; "
                         f"expected one of {sorted(RFDETR_CLASSES)}")
    return size, (weights or None)


def load_rfdetr(model: str):
    """Instantiate the rfdetr model object for a 'rfdetr-<size>[:weights]' name."""
    try:
        import rfdetr
    except ImportError as e:
        raise SystemExit("RF-DETR requested but the 'rfdetr' package is not installed. "
                         "See README (RF-DETR section) for Jetson install notes.") from e
    size, weights = parse_rfdetr_name(model)
    cls = getattr(rfdetr, RFDETR_CLASSES[size], None)
    if cls is None:
        raise SystemExit(f"This rfdetr version has no {RFDETR_CLASSES[size]}; "
                         f"upgrade rfdetr or pick another size.")
    return cls(pretrain_weights=weights) if weights else cls()


class YoloDetector:
    def __init__(self, model: str, conf: float, imgsz: int):
        from ultralytics import YOLO
        self.model = YOLO(model)  # weights auto-download on first run
        self.conf = conf
        self.imgsz = imgsz

    def infer(self, frame: np.ndarray):
        return self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)

    def draw(self, frame: np.ndarray, results) -> np.ndarray:
        return results[0].plot()


class RFDETRDetector:
    """RF-DETR through rfdetr's own predict(). The input resolution is fixed by
    the model size (e.g. 384 for nano, 560 for base) - config imgsz is ignored."""

    def __init__(self, model: str, conf: float):
        self.model = load_rfdetr(model)
        self.conf = conf
        try:
            from rfdetr.util.coco_classes import COCO_CLASSES
            self.class_names = COCO_CLASSES
        except ImportError:
            self.class_names = {}

    def infer(self, frame: np.ndarray):
        # rfdetr expects RGB; OpenCV decodes BGR.
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        return self.model.predict(rgb, threshold=self.conf)  # supervision.Detections

    def draw(self, frame: np.ndarray, dets) -> np.ndarray:
        annotated = frame.copy()
        for (x1, y1, x2, y2), conf, cls in zip(dets.xyxy.tolist(), dets.confidence.tolist(),
                                                dets.class_id.tolist()):
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            label = f"{self.class_names.get(int(cls), int(cls))} {conf:.2f}"
            cv2.rectangle(annotated, p1, p2, (56, 56, 255), 2)
            cv2.putText(annotated, label, (p1[0], max(p1[1] - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (56, 56, 255), 1)
        return annotated


def load_detector(model: str, conf: float, imgsz: int):
    if is_rfdetr(model):
        return RFDETRDetector(model, conf)
    return YoloDetector(model, conf, imgsz)
