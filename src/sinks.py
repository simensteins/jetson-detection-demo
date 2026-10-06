"""Turn a sink string into a writer for the annotated frames - the output side
of the replica, the counterpart of src/sources.py.

In the operational system the ECP does not show video on its own screen; it
sends annotated video on to the mission console (operationally the Getac
tablet; in the replica, the host PC). This module does that over the network:

- rtp://<host>:<port>   -> H.264 over RTP/UDP to <host>:<port>
- rtsp://<host>:<port>/<path> -> H.264 published to an RTSP server (e.g. mediamtx
                         on the Jetson itself), which the mission console then
                         pulls. Useful when the receiver's firewall blocks
                         incoming connections: the receiver only connects out.
                         Needs GStreamer's rtspclientsink (gstreamer1.0-rtsp).

Optional query parameters, e.g. rtp://192.168.10.1:5000?bitrate=4000&encoder=x264:
- bitrate   target bitrate in kbit/s (default 4000)
- encoder   x264 (default): software H.264 on the CPU (GStreamer x264enc)
            nvenc: Jetson hardware encoder (nvv4l2h264enc) - only on modules
                   that have one; check with `gst-inspect-1.0 nvv4l2h264enc`

Receive on the host with scripts/rtp_receiver.sdp (see README "Replica setup").

Frames go through GStreamer directly (src/gstreamer.py, PyGObject) when it is
available - the venv's pip opencv-python has no GStreamer support - and through
OpenCV's GStreamer backend otherwise.
"""
from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import cv2

from . import gstreamer

ENCODERS = {
    # Software H.264 on the CPU, tuned for low latency.
    "x264": ("videoconvert ! video/x-raw,format=I420 ! "
             "x264enc tune=zerolatency speed-preset=ultrafast bitrate={kbps} key-int-max={gop}"),
    # Jetson hardware encoder (bitrate in bit/s), fed through NVMM memory.
    "nvenc": ("videoconvert ! video/x-raw,format=BGRx ! nvvidconv ! video/x-raw(memory:NVMM),format=I420 ! "
              "nvv4l2h264enc bitrate={bps} iframeinterval={gop} insert-sps-pps=1"),
}


def rtp_pipeline(host: str, port: int, kbps: int, gop: int, encoder: str) -> str:
    """Everything after the appsrc: encode, packetize, send."""
    if encoder not in ENCODERS:
        raise SystemExit(f"Unknown encoder {encoder!r}; expected one of {sorted(ENCODERS)}")
    enc = ENCODERS[encoder].format(kbps=kbps, bps=kbps * 1000, gop=gop)
    return (f"{enc} ! h264parse ! rtph264pay config-interval=1 pt=96 ! "
            f"udpsink host={host} port={port} sync=false")


def rtsp_pipeline(location: str, kbps: int, gop: int, encoder: str) -> str:
    """Everything after the appsrc: encode, publish to an RTSP server over TCP."""
    if encoder not in ENCODERS:
        raise SystemExit(f"Unknown encoder {encoder!r}; expected one of {sorted(ENCODERS)}")
    enc = ENCODERS[encoder].format(kbps=kbps, bps=kbps * 1000, gop=gop)
    return f"{enc} ! h264parse config-interval=-1 ! rtspclientsink location={location} protocols=tcp"


def open_sink(spec: str, width: int, height: int, fps: float):
    """Open a writer for frames of size (width, height). One keyframe per second."""
    u = urlparse(str(spec))
    if u.scheme not in ("rtp", "rtsp") or not u.hostname or not u.port:
        raise SystemExit(f"Unsupported sink {spec!r}; expected rtp://<host>:<port> "
                         "or rtsp://<host>:<port>/<path>")
    q = parse_qs(u.query)
    kbps = int(q.get("bitrate", ["4000"])[0])
    encoder = q.get("encoder", ["x264"])[0]
    fps = fps if fps and fps > 0 else 30.0
    gop = max(1, round(fps))
    if u.scheme == "rtp":
        rest = rtp_pipeline(u.hostname, u.port, kbps, gop, encoder)
    else:
        rest = rtsp_pipeline(f"rtsp://{u.hostname}:{u.port}{u.path}", kbps, gop, encoder)
    if gstreamer.available():
        writer = gstreamer.GstWriter(rest, width, height, fps)
        pipeline = writer.description
    else:
        pipeline = f"appsrc ! {rest}"
        writer = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, fps, (width, height), True)
    if not writer.isOpened():
        raise SystemExit(
            f"Could not open sink {spec!r}. Is GStreamer available (PyGObject, or OpenCV built "
            f"with GStreamer), and is the '{encoder}' encoder installed? Pipeline was:\n  {pipeline}"
        )
    print(f"sink: {spec} ({width}x{height} @ {fps:.1f} fps, {encoder}, {kbps} kbit/s)")
    return writer
