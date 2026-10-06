"""Turn a sink string into a writer for the annotated frames - the output side
of the replica, the counterpart of src/sources.py.

In the operational system the ECP does not show video on its own screen; it
sends annotated video on to the mission console (operationally the Getac
tablet; in the replica, the host PC). This module does that over the network:

- rtp://<host>:<port>   -> H.264 over RTP/UDP to <host>:<port>

Optional query parameters, e.g. rtp://192.168.10.1:5000?bitrate=4000&encoder=x264:
- bitrate   target bitrate in kbit/s (default 4000)
- encoder   x264 (default): software H.264 on the CPU (GStreamer x264enc)
            nvenc: Jetson hardware encoder (nvv4l2h264enc) - only on modules
                   that have one; check with `gst-inspect-1.0 nvv4l2h264enc`

Receive on the host with scripts/rtp_receiver.sdp (see README "Replica setup").

Needs an OpenCV build with GStreamer (JetPack's system OpenCV has it; the pip
opencv-python wheel does not) - check with
    python -c "import cv2; print(cv2.getBuildInformation())" | grep -i gstreamer
"""
from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import cv2

ENCODERS = {
    # Software H.264 on the CPU, tuned for low latency.
    "x264": ("videoconvert ! video/x-raw,format=I420 ! "
             "x264enc tune=zerolatency speed-preset=ultrafast bitrate={kbps} key-int-max={gop}"),
    # Jetson hardware encoder (bitrate in bit/s), fed through NVMM memory.
    "nvenc": ("videoconvert ! video/x-raw,format=BGRx ! nvvidconv ! video/x-raw(memory:NVMM),format=I420 ! "
              "nvv4l2h264enc bitrate={bps} iframeinterval={gop} insert-sps-pps=1"),
}


def rtp_pipeline(host: str, port: int, kbps: int, gop: int, encoder: str) -> str:
    if encoder not in ENCODERS:
        raise SystemExit(f"Unknown encoder {encoder!r}; expected one of {sorted(ENCODERS)}")
    enc = ENCODERS[encoder].format(kbps=kbps, bps=kbps * 1000, gop=gop)
    return (f"appsrc ! {enc} ! h264parse ! rtph264pay config-interval=1 pt=96 ! "
            f"udpsink host={host} port={port} sync=false")


def open_sink(spec: str, width: int, height: int, fps: float) -> cv2.VideoWriter:
    """Open a writer for frames of size (width, height). One keyframe per second."""
    u = urlparse(str(spec))
    if u.scheme != "rtp" or not u.hostname or not u.port:
        raise SystemExit(f"Unsupported sink {spec!r}; expected rtp://<host>:<port>")
    q = parse_qs(u.query)
    kbps = int(q.get("bitrate", ["4000"])[0])
    encoder = q.get("encoder", ["x264"])[0]
    fps = fps if fps and fps > 0 else 30.0
    pipeline = rtp_pipeline(u.hostname, u.port, kbps, max(1, round(fps)), encoder)
    writer = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, fps, (width, height), True)
    if not writer.isOpened():
        raise SystemExit(
            f"Could not open sink {spec!r}. Is this OpenCV built with GStreamer, and is the "
            f"'{encoder}' encoder available? Pipeline was:\n  {pipeline}"
        )
    print(f"sink: {spec} ({width}x{height} @ {fps:.1f} fps, {encoder}, {kbps} kbit/s)")
    return writer
