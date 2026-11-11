#!/usr/bin/env bash
# Every optimization stage of both models (YOLOv8n and M15) in every power
# mode, one run each, unattended: the data behind all figures, per mode.
#
# For each power mode the script sets it (nvpmodel), pins the clocks
# (jetson_clocks), checks them, lets the board settle, and then runs each of
# the ten stages twice:
#   fps   without a profiler: the script's own FPS (600 frames, first 50
#         excluded), the board's input power (VDD_IN) over the measured frames,
#         max junction temperature and the GPU clock range from tegrastats
#   nsys  under nsys (as profile_paper_set.sh), exported to .sqlite, for the
#         breakdown / idle-gap / dispatch / swimlane figures
# The over-current counters are read before and after every run; a run during
# which they increased is marked throttled=1 (kept, not dropped: see
# docs/measurement-protocol.md).
#
# Run it on the Jetson with the desktop stopped (it checks), over SSH or the
# text console, venv active:
#     sudo systemctl isolate multi-user.target
#     source .venv/bin/activate
#     bash scripts/run_power_modes.sh [set-name] [mode-id ...]
#                     # defaults: set "modes", every mode in /etc/nvpmodel.conf
#     sudo systemctl isolate graphical.target      # desktop back afterwards
# sudo's password is asked once at the start.
#
# Environment options:
#     STAGES="m15_fp16 yolo"  only these stages (full name, or prefix up to "_")
#     KINDS="fps"             only fps or only nsys runs (default "fps nsys")
#     FORCE=1                 rerun runs that are already done
# Resumable: a run already in runs.csv is skipped, so after an interruption or
# a reboot the same command continues where it stopped. If a mode needs a
# reboot to switch to, the script stops and says what to do.
#
# Writes reports/<set>/:
#     runs.csv                 one row per run (mode, kind, stage, FPS, power, throttled, ...)
#     session.log              per mode: clocks, over-current counters, every result
#     <tag>/<stage>_fps.log    each run's own output; <tag> = mode name, e.g. 25w
#     <tag>/<stage>_fps.tegrastats.txt
#     <tag>/<stage>.nsys-rep, .sqlite, _nsys.log, _nsys.tegrastats.txt
# About 25-35 minutes per mode. Plots (on the Mac): scripts/plot_power_modes.py.
set -euo pipefail

SET="${1:-modes}"
MODE_ARGS=("${@:2}")   # empty = every mode
read -r -a STAGE_SEL <<<"${STAGES:-}"
read -r -a KIND_SEL <<<"${KINDS:-fps nsys}"
FORCE="${FORCE:-0}"
SRC="data/testdata.mkv"
PAUSE=20     # seconds between runs
SETTLE=60    # seconds after switching mode before the first run
TS_MS=500    # tegrastats interval
NSYS="nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --force-overwrite true"
COMMON="--source $SRC --no-display --max-frames 600 --warmup-frames 50"
YOLO_ARGS="--config configs/default.yaml $COMMON"
M15_FP32="models/m15/m15_fast_fp32-tf32.engine"
M15_FP16="models/m15/m15_fast_fp16.engine"
OUT="reports/$SET"
CSV="$OUT/runs.csv"
SESSION="$OUT/session.log"

cd "$(dirname "$0")/.."

# --- stages: name and command, in pipeline order ------------------------------
NAMES=(); CMDS=()
stage() { NAMES+=("$1"); CMDS+=("$2"); }
stage yolo_baseline "python3 -m src.detect $YOLO_ARGS"
stage yolo_graph    "python3 scripts/infer_cuda_graph.py $YOLO_ARGS"
stage yolo_gpupre   "python3 scripts/infer_cuda_graph_gpu_resize.py $YOLO_ARGS"
stage yolo_nms      "python3 scripts/infer_cuda_graph_custom_nms.py $YOLO_ARGS"
stage yolo_fp16     "python3 scripts/infer_cuda_graph_fp16.py $YOLO_ARGS --engine yolov8n_fp16_io.engine"
stage m15_baseline  "python3 scripts/infer_m15_trt.py --engine $M15_FP32 $COMMON --no-graph --preprocess cpu"
stage m15_graph     "python3 scripts/infer_m15_trt.py --engine $M15_FP32 $COMMON --preprocess cpu"
stage m15_gpupre    "python3 scripts/infer_m15_trt.py --engine $M15_FP32 $COMMON"
stage m15_nms       "python3 scripts/infer_m15_trt.py --engine $M15_FP32 $COMMON --nms fixed"
stage m15_fp16      "python3 scripts/infer_m15_trt.py --engine $M15_FP16 $COMMON --nms fixed"

wanted() {  # wanted <stage>: is it selected by STAGES?
    (( ${#STAGE_SEL[@]} == 0 )) && return 0
    local s
    for s in "${STAGE_SEL[@]}"; do [[ "$1" == "$s" || "$1" == "$s"_* ]] && return 0; done
    return 1
}
SEL=()
for i in "${!NAMES[@]}"; do wanted "${NAMES[$i]}" && SEL+=("$i"); done
(( ${#SEL[@]} > 0 )) || { echo "no stage matches STAGES=${STAGES:-}"; exit 1; }
for k in "${KIND_SEL[@]}"; do
    [[ "$k" == fps || "$k" == nsys ]] || { echo "KINDS must be fps and/or nsys, got '$k'"; exit 1; }
done

# --- power modes from /etc/nvpmodel.conf ---------------------------------------
declare -A MODE_NAME
ALL_MODES=()
while read -r id name; do
    MODE_NAME[$id]="$name"; ALL_MODES+=("$id")
done < <(sed -nE 's/.*POWER_MODEL ID=([0-9]+) NAME=([^ >]+).*/\1 \2/p' /etc/nvpmodel.conf)
(( ${#ALL_MODES[@]} > 0 )) || { echo "no POWER_MODEL entries in /etc/nvpmodel.conf"; exit 1; }
if (( ${#MODE_ARGS[@]} > 0 )); then MODES=("${MODE_ARGS[@]}"); else MODES=("${ALL_MODES[@]}"); fi
for m in "${MODES[@]}"; do
    [[ -n "${MODE_NAME[$m]:-}" ]] || { echo "unknown mode id '$m'; known: $(for a in "${ALL_MODES[@]}"; do printf '%s=%s ' "$a" "${MODE_NAME[$a]}"; done)"; exit 1; }
done
tag_of() { tr '[:upper:]' '[:lower:]' <<<"${MODE_NAME[$1]}" | tr -cd 'a-z0-9'; }

# --- preconditions --------------------------------------------------------------
for f in "$SRC" yolov8n.engine yolov8n_fp16_io.engine "$M15_FP32" "$M15_FP16"; do
    [[ -e "$f" ]] || { echo "missing: $f"; exit 1; }
done
python3 -c "import tensorrt, torch, ultralytics" 2>/dev/null \
    || { echo "activate the venv first: source .venv/bin/activate"; exit 1; }
if systemctl is-active --quiet graphical.target; then
    echo "The desktop is running. Stop it first (from SSH or Ctrl+Alt+F3):"
    echo "    sudo systemctl isolate multi-user.target"
    exit 1
fi
sudo -v
while sleep 50; do sudo -n true 2>/dev/null || exit; done &   # keep sudo alive
KEEPALIVE=$!
trap 'kill $KEEPALIVE 2>/dev/null; sudo tegrastats --stop >/dev/null 2>&1 || true' EXIT

mkdir -p "$OUT"
[[ -e "$CSV" ]] || echo "mode_id,mode,kind,stage,fps,frames,seconds,oc_delta,throttled,power_mw,tj_max_c,gpu_mhz_min,gpu_mhz_max,started" > "$CSV"
oc_total() { cat /sys/class/hwmon/hwmon*/oc*_event_cnt 2>/dev/null | awk '{s += $1} END {print s + 0}'; }
oc_list() { grep . /sys/class/hwmon/hwmon*/oc*_event_cnt 2>/dev/null | tr '\n' ' ' || true; }
done_already() {  # done_already <mode-id> <kind> <stage>
    [[ "$FORCE" == 1 ]] && return 1
    awk -F, -v m="$1" -v k="$2" -v s="$3" '$1 == m && $3 == k && $4 == s && $5 != "" {f = 1} END {exit !f}' "$CSV"
}
start_mode=$(sudo nvpmodel -q | tail -1 | tr -d ' ')
{
    echo "=== power-mode set $SET, started $(date -Is)"
    echo "--- modes: $(for m in "${MODES[@]}"; do printf '%s=%s ' "$m" "${MODE_NAME[$m]}"; done)"
    echo "--- stages: $(for i in "${SEL[@]}"; do printf '%s ' "${NAMES[$i]}"; done); kinds: ${KIND_SEL[*]}"
    echo "--- git: $(git rev-parse --short HEAD) $(git status --porcelain | wc -l) uncommitted file(s)"
    echo "--- desktop: $(systemctl is-active graphical.target || true); starting mode id $start_mode"
} | tee -a "$SESSION"

# tegrastats over the measured frames: average VDD_IN, max tj, GPU clock range.
ts_stats() {  # ts_stats <tegrastats log> <measured seconds> -> "power_mw,tj_max,gpu_min,gpu_max"
    python3 - "$1" "$2" "$TS_MS" <<'EOF'
import re, sys
lines = open(sys.argv[1], errors="replace").read().splitlines()
n = max(1, round(float(sys.argv[2] or 0) * 1000 / int(sys.argv[3])))
lines = lines[-(n + 1):-1] or lines   # the last sample can be after the run ended
def vals(rx, cast=float):
    return [cast(m.group(1)) for m in (re.search(rx, l) for l in lines) if m]
p, tj, g = vals(r"VDD_IN (\d+)mW"), vals(r"tj@([\d.]+)C"), vals(r"GR3D_FREQ \d+%@\[?(\d+)", int)
print(",".join([f"{sum(p) / len(p):.0f}" if p else "", f"{max(tj):.1f}" if tj else "",
                str(min(g)) if g else "", str(max(g)) if g else ""]))
EOF
}

measure() {  # measure <mode-id> <kind> <stage-index>
    local id="$1" kind="$2" i="$3" tag; tag="$(tag_of "$1")"
    local name="${NAMES[$i]}" base="$OUT/$tag/${NAMES[$i]}"
    local log="${base}_$kind.log" ts="${base}_$kind.tegrastats.txt" started oc0 oc1
    if done_already "$id" "$kind" "$name"; then
        echo "  $kind $name: done already, skipped"; return 0
    fi
    started="$(date -Is)"; oc0="$(oc_total)"
    sudo tegrastats --interval "$TS_MS" --logfile "$ts" &
    sleep 1
    # Word splitting of the command string is intended (no quoted arguments in it).
    if [[ "$kind" == nsys ]]; then
        $NSYS -o "$base" ${CMDS[$i]} > "$log" 2>&1 || true
    else
        ${CMDS[$i]} > "$log" 2>&1 || true
    fi
    sudo tegrastats --stop >/dev/null 2>&1 || true
    oc1="$(oc_total)"
    local line fps="" frames="" secs="" stats=",,," throttled=0
    line="$(grep -E '^processed [0-9]+ frames in [0-9.]+s.*FPS' "$log" | tail -1 || true)"
    if [[ -n "$line" ]]; then
        frames="$(sed -E 's/^processed ([0-9]+) frames.*/\1/' <<<"$line")"
        secs="$(sed -E 's/^processed [0-9]+ frames in ([0-9.]+)s.*/\1/' <<<"$line")"
        fps="$(sed -E 's/.*->  *([0-9.]+) FPS.*/\1/' <<<"$line")"
        stats="$(ts_stats "$ts" "$secs" 2>/dev/null || echo ",,,")"
    fi
    if [[ "$kind" == nsys ]]; then
        if [[ -e "$base.nsys-rep" ]]; then
            nsys export --type=sqlite --force-overwrite true -o "$base.sqlite" "$base.nsys-rep" >/dev/null 2>&1 \
                || echo "  WARNING: sqlite export failed for $base.nsys-rep" | tee -a "$SESSION"
        fi
        if [[ ! -e "$base.sqlite" || $(stat -c %s "$base.nsys-rep" 2>/dev/null || echo 0) -lt $((1024 * 1024)) ]]; then
            echo "  WARNING: nsys report missing or under 1 MiB - see $log" | tee -a "$SESSION"
            fps=""   # not done: rerun next time
        fi
    fi
    (( oc1 > oc0 )) && throttled=1
    echo "$id,${MODE_NAME[$id]},$kind,$name,$fps,$frames,$secs,$((oc1 - oc0)),$throttled,$stats,$started" >> "$CSV"
    if [[ -z "$fps" ]]; then
        echo "  $kind $name: FAILED - see $log" | tee -a "$SESSION"
    else
        IFS=, read -r pw tj gmin gmax <<<"$stats"
        printf "  %-4s %-14s %6s FPS  %5s mW  tj %4s C  GPU %s-%s MHz%s\n" "$kind" "$name" "$fps" \
            "$pw" "$tj" "$gmin" "$gmax" "$( ((throttled)) && echo "  THROTTLED (+$((oc1 - oc0)) over-current events)")" \
            | tee -a "$SESSION"
    fi
    sleep "$PAUSE"
}

set_mode() {  # set_mode <mode-id>: switch, pin the clocks, check; exits if a reboot is needed
    local id="$1" out cur clocks gpu_line gmin gmax
    cur="$(sudo nvpmodel -q | tail -1 | tr -d ' ')"
    if [[ "$cur" != "$id" ]]; then
        out="$(echo no | sudo nvpmodel -m "$id" 2>&1 || true)"
        if grep -qi "reboot" <<<"$out"; then
            echo "$out" | tee -a "$SESSION"
            echo "Mode $id (${MODE_NAME[$id]}) needs a reboot. Run: sudo nvpmodel -m $id  (answer YES)," | tee -a "$SESSION"
            echo "then after the reboot stop the desktop again and rerun this same command; done runs are skipped." | tee -a "$SESSION"
            exit 2
        fi
    fi
    cur="$(sudo nvpmodel -q | tail -1 | tr -d ' ')"
    [[ "$cur" == "$id" ]] || { echo "nvpmodel is in mode $cur, not $id" | tee -a "$SESSION"; exit 1; }
    sudo jetson_clocks
    sleep 2
    clocks="$(sudo jetson_clocks --show)"
    gpu_line="$(grep '^GPU MinFreq' <<<"$clocks")"
    gmin="$(sed -E 's/.*MinFreq=([0-9]+).*/\1/' <<<"$gpu_line")"
    gmax="$(sed -E 's/.*MaxFreq=([0-9]+).*/\1/' <<<"$gpu_line")"
    {
        echo; echo "=== mode $id ${MODE_NAME[$id]}  $(date -Is)"
        echo "$clocks"
        echo "--- over-current counters: $(oc_list)"
    } | tee -a "$SESSION"
    [[ "$gmin" == "$gmax" ]] || { echo "GPU clock not pinned after jetson_clocks ($gpu_line)" | tee -a "$SESSION"; exit 1; }
}

for id in "${MODES[@]}"; do
    todo=0
    for i in "${SEL[@]}"; do for k in "${KIND_SEL[@]}"; do
        done_already "$id" "$k" "${NAMES[$i]}" || todo=1
    done; done
    if (( todo == 0 )); then echo "mode $id ${MODE_NAME[$id]}: all runs done, skipped"; continue; fi
    set_mode "$id"
    mkdir -p "$OUT/$(tag_of "$id")"
    echo "settling ${SETTLE}s"; sleep "$SETTLE"
    for i in "${SEL[@]}"; do
        for k in "${KIND_SEL[@]}"; do measure "$id" "$k" "$i"; done
    done
    echo "--- over-current counters after mode $id: $(oc_list)" | tee -a "$SESSION"
done

{
    echo; echo "=== finished $(date -Is)"
    echo "--- throttled runs:"
    awk -F, 'NR > 1 && $9 == 1 {printf "    %s %s %s (+%s events)\n", $2, $3, $4, $8}' "$CSV"
} | tee -a "$SESSION"
if [[ "$start_mode" != "$(sudo nvpmodel -q | tail -1 | tr -d ' ')" ]]; then
    echo "Restoring the starting power mode $start_mode"
    echo no | sudo nvpmodel -m "$start_mode" || true
    echo "jetson_clocks is reset by the mode change: run sudo jetson_clocks if you need pinned clocks"
fi
echo
echo "Copy to the Mac (without the large .nsys-rep files):"
echo "    rsync -av --exclude '*.nsys-rep' 'studenta@10.0.0.2:~/nsight-demo/jetson-detection-demo/$OUT' reports/"
echo "Then: python3 scripts/plot_power_modes.py --set $SET"
