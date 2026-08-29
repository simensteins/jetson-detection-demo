"""Turn a source string into a cv2.VideoCapture.

- digit string ("0")  -> webcam index
- rtsp:// URL         -> network stream (v1a)
- anything else       -> file path (v1b)
"""
from __future__ import annotations

import cv2


def open_source(source: str) -> cv2.VideoCapture:
    s = str(source)
    if s.isdigit():
        return cv2.VideoCapture(int(s))
    if s.startswith("rtsp://"):
        pipeline = (
            f"rtspsrc location={s} latency=0 ! rtph264depay ! h264parse ! "
            "nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
        )
        return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    return cv2.VideoCapture(s)
