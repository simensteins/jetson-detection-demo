"""Turn a source string into a cv2.VideoCapture (or a compatible reader).

- digit string ("0")  -> webcam index
- rtsp:// URL         -> network stream (v1a / replica), hardware-decoded on the
                         Jetson (nvv4l2decoder). Read through GStreamer directly
                         (src/gstreamer.py) when PyGObject is available, since the
                         venv's pip OpenCV has no GStreamer support; otherwise
                         through OpenCV's GStreamer backend.
- anything else       -> file path (v1b)
"""
from __future__ import annotations

import cv2

from . import gstreamer


def open_source(source: str):
    s = str(source)
    if s.isdigit():
        return cv2.VideoCapture(int(s))
    if s.startswith("rtsp://"):
        decode = (f"rtspsrc location={s} latency=0 ! rtph264depay ! h264parse ! "
                  "nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx ! "
                  "videoconvert ! video/x-raw,format=BGR")
        if gstreamer.available():
            return gstreamer.GstCapture(f"{decode} ! appsink name=sink drop=true max-buffers=1 sync=false")
        return cv2.VideoCapture(f"{decode} ! appsink drop=1", cv2.CAP_GSTREAMER)
    return cv2.VideoCapture(s)
