#!/usr/bin/env bash
# v1a ONLY — run this on the MAC (not the Jetson) to stream a looping video
# file to the Jetson over the USB-C link.
#
# Prereqs on the Mac:  brew install ffmpeg mediamtx
#   1. Start the RTSP server in one terminal:   mediamtx
#   2. Run this script in another to publish a looping file to it.
#
# The Jetson then reads:  rtsp://192.168.55.100:8554/demo
# (set that as `source` in configs/default.yaml, or pass --source)
set -euo pipefail

VIDEO="${1:-sample.mp4}"   # pass a file path as the first argument

ffmpeg -re -stream_loop -1 -i "${VIDEO}" \
  -c:v libx264 -preset veryfast -tune zerolatency \
  -f rtsp rtsp://127.0.0.1:8554/demo
