"""GStreamer pipelines driven directly from Python (PyGObject / `gi`), with
frames handed over as NumPy arrays - no OpenCV in between.

Why: the venv's pip opencv-python is built without GStreamer, and JetPack's
system OpenCV (which has it) is built against NumPy 1.x while the venv uses
NumPy 2.x. Going through `gi` (installed system-wide, visible in the venv via
include-system-site-packages) avoids touching the measured environment and
keeps the Jetson's hardware decoder (nvv4l2decoder) in the input pipeline.

GstCapture mimics the parts of cv2.VideoCapture the scripts use
(isOpened / read / get(cv2.CAP_PROP_FPS) / release); GstWriter mimics
cv2.VideoWriter (isOpened / write / release).
"""
from __future__ import annotations

import numpy as np

try:
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
    Gst.init(None)
except (ImportError, ValueError):
    Gst = None

FPS_PROP = 5  # cv2.CAP_PROP_FPS, without importing cv2 here


def available() -> bool:
    return Gst is not None


def _bus_error(pipeline) -> str:
    msg = pipeline.get_bus().pop_filtered(Gst.MessageType.ERROR)
    if msg is None:
        return ""
    err, debug = msg.parse_error()
    return f"{err.message} ({debug})" if debug else err.message


def frame_from_bytes(data, width: int, height: int) -> np.ndarray:
    """BGR bytes (rows possibly padded to a 4-byte stride) -> owned (H, W, 3) uint8 array."""
    flat = np.frombuffer(data, dtype=np.uint8)
    stride = flat.size // height
    return flat[:stride * height].reshape(height, stride)[:, :width * 3].reshape(height, width, 3).copy()


def bytes_from_frame(frame: np.ndarray) -> bytes:
    """(H, W, 3) uint8 BGR -> bytes with rows padded to a 4-byte stride, as GStreamer expects."""
    h, w = frame.shape[:2]
    row = w * 3
    stride = (row + 3) & ~3
    if stride == row:
        return np.ascontiguousarray(frame).tobytes()
    padded = np.zeros((h, stride), dtype=np.uint8)
    padded[:, :row] = frame.reshape(h, row)
    return padded.tobytes()


class GstCapture:
    """Reads BGR frames from a pipeline ending in `appsink name=sink`."""

    def __init__(self, pipeline: str, timeout_s: float = 10.0):
        self.timeout_ns = int(timeout_s * Gst.SECOND)
        self.fps = 0.0
        self.pipeline = Gst.parse_launch(pipeline)
        self.sink = self.pipeline.get_by_name("sink")
        ok = self.pipeline.set_state(Gst.State.PLAYING) != Gst.StateChangeReturn.FAILURE
        self._opened = ok and self.sink is not None
        if not self._opened:
            print(f"GStreamer source failed: {_bus_error(self.pipeline)}\n  pipeline: {pipeline}")

    def isOpened(self) -> bool:
        return self._opened

    def read(self):
        sample = self.sink.emit("try-pull-sample", self.timeout_ns)
        if sample is None:  # end of stream, timeout or error
            err = _bus_error(self.pipeline)
            if err:
                print(f"GStreamer source stopped: {err}")
            return False, None
        s = sample.get_caps().get_structure(0)
        width, height = s.get_value("width"), s.get_value("height")
        ok, num, den = s.get_fraction("framerate")
        if ok and den:
            self.fps = num / den
        buf = sample.get_buffer()
        ok, info = buf.map(Gst.MapFlags.READ)
        if not ok:
            return False, None
        try:
            frame = frame_from_bytes(info.data, width, height)
        finally:
            buf.unmap(info)
        return True, frame

    def get(self, prop: int) -> float:
        return self.fps if prop == FPS_PROP else 0.0

    def release(self) -> None:
        self.pipeline.set_state(Gst.State.NULL)


class GstWriter:
    """Pushes BGR frames into `appsrc ! <rest>`; caps and timestamps are set here."""

    def __init__(self, rest: str, width: int, height: int, fps: float):
        caps = f"video/x-raw,format=BGR,width={width},height={height},framerate={max(1, round(fps))}/1"
        self.description = (f"appsrc name=src is-live=true format=time do-timestamp=true "
                            f"caps={caps} ! {rest}")
        self.pipeline = Gst.parse_launch(self.description)
        self.src = self.pipeline.get_by_name("src")
        self._opened = self.pipeline.set_state(Gst.State.PLAYING) != Gst.StateChangeReturn.FAILURE
        if not self._opened:
            print(f"GStreamer sink failed: {_bus_error(self.pipeline)}")

    def isOpened(self) -> bool:
        return self._opened

    def write(self, frame: np.ndarray) -> None:
        self.src.emit("push-buffer", Gst.Buffer.new_wrapped(bytes_from_frame(frame)))

    def release(self) -> None:
        self.src.emit("end-of-stream")
        self.pipeline.set_state(Gst.State.NULL)
