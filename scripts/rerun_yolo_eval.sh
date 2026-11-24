#!/usr/bin/env bash
# Rerun the YOLOv8n accuracy evaluation for every optimization stage, on the
# person class only (the SAR task) and on all 80 COCO classes.
#
# scripts/eval_yolo_coco.py runs every stage in one go: baseline, gpupre
# (+ GPU preprocessing), customnms (+ fixed-shape NMS), fp16 (+ FP16, the
# final pipeline) and fp16_ultranms (a control: the FP16 engine with the
# standard NMS). The CUDA graph stage replays the same kernels as the
# baseline, so it has no pipeline of its own. Accuracy does not depend on the
# power mode or the clocks, so no measurement protocol is needed; the desktop
# may stay up.
#
# On the Jetson, from anywhere, venv active:
#     bash scripts/rerun_yolo_eval.sh            # first 600 COCO val2017 images, as before
#     bash scripts/rerun_yolo_eval.sh 5000       # all images (several times longer)
# Needs COCO val2017 (bash scripts/download_coco_val.sh), pycocotools, and the
# engines yolov8n.engine and yolov8n_fp16_io.engine in the repo root.
#
# Writes reports/eval/yolo_coco_person_* (person only) and reports/eval/yolo_coco_*
# (all classes): *_summary.md/.csv per stage, *_buckets.* by objects per image,
# *_<stage>_dets.json, plus reports/eval/rerun_<timestamp>.log with everything
# printed.
set -euo pipefail

IMAGES="${1:-600}"
cd "$(dirname "$0")/.."
[[ "$IMAGES" =~ ^[0-9]+$ ]] || { echo "the image count must be a number, got '$IMAGES'"; exit 1; }

# --- preconditions --------------------------------------------------------------
for f in yolov8n.engine yolov8n_fp16_io.engine data/coco/annotations/instances_val2017.json; do
    [[ -e "$f" ]] || { echo "missing: $f"; [[ "$f" == data/coco/* ]] && echo "  run: bash scripts/download_coco_val.sh"; exit 1; }
done
[[ -d data/coco/val2017 ]] || { echo "missing: data/coco/val2017 - run: bash scripts/download_coco_val.sh"; exit 1; }
python3 -c "import tensorrt, torch, ultralytics" 2>/dev/null \
    || { echo "activate the venv first: source .venv/bin/activate"; exit 1; }
python3 -c "import pycocotools" 2>/dev/null || { echo "missing pycocotools: pip install pycocotools"; exit 1; }
python3 -c "import sys; sys.path.insert(0, 'scripts'); import yolo_fixed_nms" 2>/dev/null \
    || { echo "scripts/yolo_fixed_nms.py missing - git pull first"; exit 1; }

mkdir -p reports/eval
LOG="reports/eval/rerun_$(date +%Y%m%d_%H%M%S).log"
{
    echo "=== YOLOv8n accuracy rerun, $IMAGES images, started $(date -Is)"
    echo "--- git: $(git rev-parse --short HEAD) $(git status --porcelain | wc -l) uncommitted file(s)"
} | tee "$LOG"

run() {  # run <label> <eval arguments...>
    local label="$1"; shift
    echo | tee -a "$LOG"
    echo "=== $label  $(date +%T)" | tee -a "$LOG"
    python3 scripts/eval_yolo_coco.py --max-images "$IMAGES" "$@" 2>&1 | tee -a "$LOG"
}

run "person only (the SAR task)" --person-only
run "all 80 classes"

echo | tee -a "$LOG"
echo "=== finished $(date -Is)" | tee -a "$LOG"
for f in reports/eval/yolo_coco_person_summary.md reports/eval/yolo_coco_summary.md; do
    echo; echo "--- $f"; cat "$f"
done | tee -a "$LOG"
echo
echo "Copy to the Mac (from the Mac):"
echo "  rsync -av --exclude '*_dets.json' studenta@<jetson address>:~/nsight-demo/jetson-detection-demo/reports/eval/ reports/eval/"
