#!/usr/bin/env bash
# Profile the detection demo on the Jetson with Nsight Systems.
# Produces reports/<timestamp>.nsys-rep — copy it to the Mac to view.
set -euo pipefail

mkdir -p reports
stamp=$(date +%Y%m%d_%H%M%S)
out="reports/v1_${stamp}"

# --duration bounds the capture so the report stays a sensible size.
# --trace=nvtx only (no cuda): tracing cuda hangs on this JetPack/nsys combo
# in GpuTicksConverter during report export. NVTX still gives the
# decode/inference/draw/display range breakdown we actually care about.
nsys profile \
  --trace=nvtx \
  --duration=20 \
  --force-overwrite true \
  -o "${out}" \
  python -m src.detect --config configs/default.yaml --no-display "$@"

echo "wrote ${out}.nsys-rep"
