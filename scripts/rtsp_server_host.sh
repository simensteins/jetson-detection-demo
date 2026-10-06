#!/usr/bin/env bash
# Replica video source - run this on the HOST (the OptiPlex, Ubuntu), not the
# Jetson. It stands in for the Video Recording Device (VRD): a looping file
# published as an RTSP stream that the Jetson reads over Ethernet.
# (The Ubuntu counterpart of scripts/rtsp_server_mac.sh, which is kept as is.)
#
# Prereqs on the host:
#   sudo apt install ffmpeg
#   mediamtx: download the linux_amd64 release from
#   https://github.com/bluenviron/mediamtx/releases and put `mediamtx` on PATH
#
# Usage:  bash scripts/rtsp_server_host.sh [video file]      (default: data/vtest.avi)
# The Jetson then reads:  rtsp://<host-ip>:8554/demo
#
# The stream is re-encoded with FIXED settings, so every measurement sees the
# same input: H.264, 4 Mbit/s constant-ish bitrate, a keyframe every 30 frames,
# no B-frames (zerolatency), yuv420p. Change them here, and note it, if needed.
set -euo pipefail

VIDEO="${1:-data/vtest.avi}"
[ -f "${VIDEO}" ] || { echo "video not found: ${VIDEO}"; exit 1; }

# Start mediamtx (the RTSP server) unless it is already running.
if ! pgrep -x mediamtx >/dev/null; then
  mediamtx >/tmp/mediamtx.log 2>&1 &
  MEDIAMTX_PID=$!
  trap 'kill ${MEDIAMTX_PID} 2>/dev/null || true' EXIT
  sleep 1
fi

echo "publishing ${VIDEO} on rtsp://$(hostname -I | awk '{print $1}'):8554/demo  (Ctrl+C to stop)"
ffmpeg -hide_banner -loglevel warning -re -stream_loop -1 -i "${VIDEO}" -an \
  -c:v libx264 -preset veryfast -tune zerolatency -pix_fmt yuv420p \
  -b:v 4M -maxrate 4M -bufsize 8M -g 30 \
  -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/demo
