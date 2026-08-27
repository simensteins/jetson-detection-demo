#!/usr/bin/env bash
# Profile the detection demo on the Jetson with Nsight Systems.
# Produces reports/<timestamp>.nsys-rep — copy it to the Mac to view.
set -euo pipefail

mkdir -p reports
stamp=$(date +%Y%m%d_%H%M%S)
out="reports/v1_${stamp}"

# --duration bounds the capture so the report stays a sensible size.
# --trace=cuda,nvtx captures GPU work plus our decode/inference/draw ranges.
nsys profile \
  --trace=cuda,nvtx \
  --duration=20 \
  --force-overwrite true \
  -o "${out}" \
  python -m src.detect --config configs/default.yaml --no-display

echo "wrote ${out}.nsys-rep"
