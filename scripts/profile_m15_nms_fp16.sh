#!/usr/bin/env bash
# Profile only M15's last two stages in the new order (same as YOLOv8n):
#   m15_nms   TF32 engine, custom NMS captured in the CUDA graph (k = 100)
#   m15_fp16  FP16 engine, same NMS
# Same conditions, checks and logs as scripts/profile_paper_set.sh (25 W,
# jetson_clocks, desktop stopped - see docs/measurement-protocol.md).
#     bash scripts/profile_m15_nms_fp16.sh [set-name]   # default: paper25w_v2
set -euo pipefail
exec bash "$(dirname "$0")/profile_paper_set.sh" "${1:-paper25w_v2}" m15_nms m15_fp16
