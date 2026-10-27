#!/usr/bin/env bash
# Profile every optimization stage of both models (YOLOv8n and M15) in one
# session, under identical conditions, for the paper's comparison figures.
#
# Meant to run with the desktop stopped, over SSH:
#     sudo systemctl isolate multi-user.target     # stop the desktop
#     sudo nvpmodel -m <25W id> && sudo jetson_clocks
#     source .venv/bin/activate
#     bash scripts/profile_paper_set.sh [set-name] [all|yolo|m15]
#                                    # defaults: set paper25w, both models
#     sudo systemctl isolate graphical.target      # desktop back afterwards
#
# Refuses to start unless the clocks are pinned (jetson_clocks); warns if the
# desktop is running. Writes, per stage, reports/<set>_<model>_<stage>.nsys-rep,
# .sqlite and .log (the script's own output), plus <set>_session.log with the
# clock state, the over-current counters before/after, and a tegrastats log.
#
# Both models use the same stage order: baseline, + CUDA graph, + GPU
# preprocessing, + custom NMS, + FP16 (each stage keeps the earlier ones).
# The set paper25w (25.10.2026) ran M15 with FP16 before the custom NMS.
set -euo pipefail

SET="${1:-paper25w}"
MODELS="${2:-all}"
[[ "$MODELS" =~ ^(all|yolo|m15)$ ]] || { echo "models must be all, yolo or m15"; exit 1; }
SRC="data/testdata.mkv"
PAUSE=60   # seconds between runs, so each starts from the same thermal state
NSYS="nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --force-overwrite true"
YOLO_ARGS="--config configs/default.yaml --source $SRC --no-display --max-frames 600"
M15_FP32="models/m15/m15_fast_fp32-tf32.engine"
M15_FP16="models/m15/m15_fast_fp16.engine"
M15_ARGS="--source $SRC --no-display --max-frames 600 --warmup-frames 50"
SESSION="reports/${SET}_session.log"

cd "$(dirname "$0")/.."
mkdir -p reports

# --- preconditions --------------------------------------------------------
for f in "$SRC" yolov8n.engine yolov8n_fp16_io.engine "$M15_FP32" "$M15_FP16"; do
    [[ -e "$f" ]] || { echo "missing: $f"; exit 1; }
done
python3 -c "import tensorrt, torch, ultralytics" 2>/dev/null \
    || { echo "activate the venv first: source .venv/bin/activate"; exit 1; }
clocks="$(sudo jetson_clocks --show)"
gpu_line="$(grep '^GPU MinFreq' <<<"$clocks")"
gpu_min="$(sed -E 's/.*MinFreq=([0-9]+).*/\1/' <<<"$gpu_line")"
gpu_max="$(sed -E 's/.*MaxFreq=([0-9]+).*/\1/' <<<"$gpu_line")"
if [[ "$gpu_min" != "$gpu_max" ]]; then
    echo "GPU clock is not pinned ($gpu_line) - run: sudo jetson_clocks"; exit 1
fi
if systemctl is-active --quiet graphical.target; then
    echo "WARNING: the desktop is running (graphical.target). For the clean set, stop it:"
    echo "         sudo systemctl isolate multi-user.target"
    read -r -p "Continue anyway? [y/N] " ans; [[ "$ans" == "y" ]] || exit 1
fi

{
    echo "=== set $SET ($MODELS), started $(date -Is)"
    echo "--- git: $(git rev-parse --short HEAD) $(git status --porcelain | wc -l) uncommitted file(s)"
    echo "--- $(sudo nvpmodel -q | tr '\n' ' ')"
    echo "$clocks"
    echo "--- desktop: $(systemctl is-active graphical.target || true)"
    echo "--- top CPU users:"; top -b -n 1 -o %CPU | sed -n '7,14p'
    echo "--- over-current counters BEFORE:"; grep . /sys/class/hwmon/hwmon*/oc*_event_cnt || true
} | tee "$SESSION"

sudo tegrastats --interval 1000 --logfile "reports/${SET}_tegrastats.txt" &
trap 'sudo tegrastats --stop >/dev/null 2>&1 || true' EXIT

run() {  # run <name> <command...>
    local name="${SET}_$1"; shift
    echo; echo "=== $(date +%T) $name: $*" | tee -a "$SESSION"
    $NSYS -o "reports/$name" "$@" 2>&1 | tee "reports/$name.log" | grep -E "FPS|--nms fixed|Error|error" || true
    nsys export --type=sqlite --force-overwrite true -o "reports/$name.sqlite" "reports/$name.nsys-rep" >/dev/null
    local size; size=$(stat -c %s "reports/$name.nsys-rep")
    echo "    report $((size / 1024)) KiB" | tee -a "$SESSION"
    (( size > 1024 * 1024 )) || echo "    WARNING: report under 1 MiB - this run probably failed, see reports/$name.log" | tee -a "$SESSION"
    sleep "$PAUSE"
}

# --- YOLOv8n: five stages --------------------------------------------------
if [[ "$MODELS" != m15 ]]; then
run yolo_baseline python3 -m src.detect $YOLO_ARGS
run yolo_graph    python3 scripts/infer_cuda_graph.py $YOLO_ARGS
run yolo_gpupre   python3 scripts/infer_cuda_graph_gpu_resize.py $YOLO_ARGS
run yolo_nms      python3 scripts/infer_cuda_graph_custom_nms.py $YOLO_ARGS
run yolo_fp16     python3 scripts/infer_cuda_graph_fp16.py $YOLO_ARGS --engine yolov8n_fp16_io.engine
fi

# --- M15: five stages ------------------------------------------------------
if [[ "$MODELS" != yolo ]]; then
run m15_baseline  python3 scripts/infer_m15_trt.py --engine "$M15_FP32" $M15_ARGS --no-graph --preprocess cpu
run m15_graph     python3 scripts/infer_m15_trt.py --engine "$M15_FP32" $M15_ARGS --preprocess cpu
run m15_gpupre    python3 scripts/infer_m15_trt.py --engine "$M15_FP32" $M15_ARGS
run m15_nms       python3 scripts/infer_m15_trt.py --engine "$M15_FP32" $M15_ARGS --nms fixed
run m15_fp16      python3 scripts/infer_m15_trt.py --engine "$M15_FP16" $M15_ARGS --nms fixed
fi

{
    echo; echo "--- over-current counters AFTER (should equal BEFORE):"
    grep . /sys/class/hwmon/hwmon*/oc*_event_cnt || true
    echo "=== finished $(date -Is)"
} | tee -a "$SESSION"
echo
echo "Copy to the Mac:  scp 'studenta@10.0.0.2:~/nsight-demo/jetson-detection-demo/reports/${SET}_*' reports/"
