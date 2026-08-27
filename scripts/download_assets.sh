#!/usr/bin/env bash
# Fetch a sample video for v1b. YOLO weights auto-download on first model load.
set -euo pipefail
mkdir -p data

# vtest.avi: short pedestrian clip from the OpenCV samples (COCO "person"
# class). Swap in traffic footage from pexels.com/videos for cars, etc.
if [ ! -f data/vtest.avi ]; then
  curl -L -o data/vtest.avi \
    https://github.com/opencv/opencv/raw/master/samples/data/vtest.avi
fi
echo "assets ready in data/"
