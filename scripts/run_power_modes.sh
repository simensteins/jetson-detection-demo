#!/usr/bin/env bash
# FPS of every optimization stage of both models (YOLOv8n and M15) in every
# power mode, without a profiler and under nsys, unattended.
#
# For each power mode the script sets it (nvpmodel), pins the clocks
# (jetson_clocks), checks them, lets the board settle for a minute, and runs
# the ten stages one after the other, each once without a profiler (plain)
# and once under nsys (same flags as profile_paper_set.sh): 600 frames of
# data/testdata.mkv, the first 50 excluded, no display; the FPS is the
# scripts' own in both cases. With RUNS > 1 the stages are repeated in
# rounds inside each power mode (interleaved, as in bench_fps.sh). The over-current
# counters are read before and after every run; a run during which they
# increased is marked throttled (kept, not dropped: see
# docs/measurement-protocol.md).
#
# On the Jetson, desktop stopped (it checks), over SSH or the text console:
#     sudo systemctl isolate multi-user.target
#     source .venv/bin/activate
#     bash scripts/run_power_modes.sh [set-name] [mode-id ...]
#                     # defaults: set "modes", every mode in /etc/nvpmodel.conf
#     sudo systemctl isolate graphical.target      # desktop back afterwards
# sudo's password is asked once at the start.
#
# Environment options:
#     STAGES="m15_fp16 yolo"  only these stages (full name, or prefix up to "_")
#     PROFILERS="plain nsys"  plain = no profiler, nsys = under nsys (default:
#                             both; PROFILERS=plain for the old FPS-only runs)
#     RUNS=1                  rounds per power mode (default 1); the tables
#                             give the median over the rounds
#     KEEP_REPORTS=1          keep each nsys run's .nsys-rep (5-25 MB each);
#                             0 deletes it once its FPS is read
#     FORCE=1                 rerun runs that are already done
# Resumable: a run already in fps.csv is skipped, so after an interruption or
# a reboot the same command continues where it stopped. If a mode needs a
# reboot to switch to, the script stops and says what to do.
#
# Writes reports/<set>/:
#     fps.csv          one row per run (mode, stage, profiler, round, FPS,
#                      over-current events, throttled)
#     fps_table.md     FPS tables, one per profiler setting: one row per
#                      stage, one column per power mode (median over the
#                      rounds); with both settings also the profiler overhead
#                      (nsys FPS / plain FPS) per stage and power mode
#     session.log      per mode: clocks, over-current counters, every result
#     <mode>/<stage>_<profiler>_r<NN>.log  each run's own output
#     <mode>/<stage>_nsys_r<NN>.nsys-rep   each nsys run's report (export to
#                      .sqlite on the Jetson: nsys export --type=sqlite ...)
# About 10-15 minutes per power mode per round without the profiler, and
# roughly 25-35 minutes per power mode per round with both settings.
set -euo pipefail

SET="${1:-modes}"
MODE_ARGS=("${@:2}")   # empty = every mode
read -r -a STAGE_SEL <<<"${STAGES:-}"
FORCE="${FORCE:-0}"
read -r -a PROFILERS <<<"${PROFILERS:-plain nsys}"
RUNS="${RUNS:-1}"
KEEP_REPORTS="${KEEP_REPORTS:-1}"
NSYS="nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --force-overwrite true"
SRC="data/testdata.mkv"
PAUSE=15     # seconds between runs
SETTLE=60    # seconds after switching mode before the first run
COMMON="--source $SRC --no-display --max-frames 600 --warmup-frames 50"
YOLO_ARGS="--config configs/default.yaml $COMMON"
M15_FP32="models/m15/m15_fast_fp32-tf32.engine"
M15_FP16="models/m15/m15_fast_fp16.engine"
OUT="reports/$SET"
CSV="$OUT/fps.csv"
SESSION="$OUT/session.log"

cd "$(dirname "$0")/.."
(( ${#PROFILERS[@]} > 0 )) || { echo "PROFILERS is empty - use plain, nsys or both"; exit 1; }
for p in "${PROFILERS[@]}"; do
    [[ "$p" == plain || "$p" == nsys ]] || { echo "unknown profiler setting '$p' in PROFILERS - use plain and/or nsys"; exit 1; }
done
[[ "$RUNS" =~ ^[0-9]+$ ]] && (( RUNS > 0 )) || { echo "RUNS must be a positive number, got '$RUNS'"; exit 1; }
[[ "$KEEP_REPORTS" == 0 || "$KEEP_REPORTS" == 1 ]] || { echo "KEEP_REPORTS must be 0 or 1"; exit 1; }

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
if [[ " ${PROFILERS[*]} " == *" nsys "* ]]; then
    command -v nsys >/dev/null || { echo "nsys not found - install Nsight Systems or run with PROFILERS=plain"; exit 1; }
fi
if systemctl is-active --quiet graphical.target; then
    echo "The desktop is running. Stop it first (from SSH or Ctrl+Alt+F3):"
    echo "    sudo systemctl isolate multi-user.target"
    exit 1
fi
sudo -v
while sleep 50; do sudo -n true 2>/dev/null || exit; done &   # keep sudo alive
KEEPALIVE=$!
trap 'kill $KEEPALIVE 2>/dev/null || true' EXIT

mkdir -p "$OUT"
HEADER="mode_id,mode,stage,profiler,run,fps,frames,seconds,oc_delta,throttled,started"
[[ -e "$CSV" ]] || echo "$HEADER" > "$CSV"
[[ "$(head -1 "$CSV")" == "$HEADER" ]] \
    || { echo "$CSV is from an older version of this script (other columns) - pick another set name"; exit 1; }
oc_total() { cat /sys/class/hwmon/hwmon*/oc*_event_cnt 2>/dev/null | awk '{s += $1} END {print s + 0}'; }
oc_list() { grep . /sys/class/hwmon/hwmon*/oc*_event_cnt 2>/dev/null | tr '\n' ' ' || true; }
done_already() {  # done_already <mode-id> <stage> <profiler> <round>
    [[ "$FORCE" == 1 ]] && return 1
    awk -F, -v m="$1" -v s="$2" -v p="$3" -v r="$4" \
        '$1 == m && $3 == s && $4 == p && $5 == r && $6 != "" {f = 1} END {exit !f}' "$CSV"
}
start_mode=$(sudo nvpmodel -q | tail -1 | tr -d ' ')
{
    echo "=== power-mode FPS set $SET, started $(date -Is)"
    echo "--- modes: $(for m in "${MODES[@]}"; do printf '%s=%s ' "$m" "${MODE_NAME[$m]}"; done)"
    echo "--- stages: $(for i in "${SEL[@]}"; do printf '%s ' "${NAMES[$i]}"; done)"
    echo "--- profiler: ${PROFILERS[*]}; rounds per mode: $RUNS; nsys: $NSYS; keep reports: $KEEP_REPORTS"
    echo "--- git: $(git rev-parse --short HEAD) $(git status --porcelain | wc -l) uncommitted file(s)"
    echo "--- desktop: $(systemctl is-active graphical.target || true); starting mode id $start_mode"
} | tee -a "$SESSION"

measure() {  # measure <mode-id> <stage-index> <profiler> <round>
    local id="$1" i="$2" name="${NAMES[$2]}" prof="$3" r="$4" base log started oc0 oc1
    base="$OUT/$(tag_of "$1")/${name}_${prof}_r$(printf %02d "$r")"
    log="$base.log"
    if done_already "$id" "$name" "$prof" "$r"; then
        echo "  $name ($prof, round $r): done already, skipped"; return 0
    fi
    started="$(date -Is)"; oc0="$(oc_total)"
    # Word splitting of the command strings is intended (no quoted arguments in them).
    if [[ "$prof" == nsys ]]; then
        $NSYS -o "$base" ${CMDS[$i]} > "$log" 2>&1 || true
        (( KEEP_REPORTS )) || rm -f "$base.nsys-rep"
    else
        ${CMDS[$i]} > "$log" 2>&1 || true
    fi
    oc1="$(oc_total)"
    local line fps="" frames="" secs="" throttled=0
    line="$(grep -E '^processed [0-9]+ frames in [0-9.]+s.*FPS' "$log" | tail -1 || true)"
    if [[ -n "$line" ]]; then
        frames="$(sed -E 's/^processed ([0-9]+) frames.*/\1/' <<<"$line")"
        secs="$(sed -E 's/^processed [0-9]+ frames in ([0-9.]+)s.*/\1/' <<<"$line")"
        fps="$(sed -E 's/.*->  *([0-9.]+) FPS.*/\1/' <<<"$line")"
    fi
    (( oc1 > oc0 )) && throttled=1
    echo "$id,${MODE_NAME[$id]},$name,$prof,$r,$fps,$frames,$secs,$((oc1 - oc0)),$throttled,$started" >> "$CSV"
    if [[ -z "$fps" ]]; then
        echo "  $name ($prof, round $r): FAILED - see $log" | tee -a "$SESSION"
    else
        printf "  %-14s %-5s r%-2s %6s FPS%s\n" "$name" "$prof" "$r" "$fps" \
            "$( ((throttled)) && echo "  THROTTLED (+$((oc1 - oc0)) over-current events)")" | tee -a "$SESSION"
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
    for (( r = 1; r <= RUNS; r++ )); do
        for i in "${SEL[@]}"; do
            for p in "${PROFILERS[@]}"; do done_already "$id" "${NAMES[$i]}" "$p" "$r" || todo=1; done
        done
    done
    if (( todo == 0 )); then echo "mode $id ${MODE_NAME[$id]}: all runs done, skipped"; continue; fi
    set_mode "$id"
    mkdir -p "$OUT/$(tag_of "$id")"
    echo "settling ${SETTLE}s"; sleep "$SETTLE"
    for (( r = 1; r <= RUNS; r++ )); do
        (( RUNS > 1 )) && echo "--- round $r/$RUNS  $(date +%T)" | tee -a "$SESSION"
        for i in "${SEL[@]}"; do
            for p in "${PROFILERS[@]}"; do measure "$id" "$i" "$p" "$r"; done
        done
    done
    echo "--- over-current counters after mode $id: $(oc_list)" | tee -a "$SESSION"
done
echo "=== finished $(date -Is)" | tee -a "$SESSION"

# --- tables: one row per stage, one column per power mode (median of the rounds) ---
python3 - "$CSV" "$OUT/fps_table.md" <<'EOF'
import csv, statistics, sys
all_rows = list(csv.DictReader(open(sys.argv[1])))   # failed runs too: keeps the run order
modes = list(dict.fromkeys(r["mode"] for r in all_rows))
stages = list(dict.fromkeys(r["stage"] for r in all_rows))
profs = [p for p in ("plain", "nsys") if any(r["profiler"] == p for r in all_rows)]
# Per (power mode, stage, profiler, round) the last successful run counts (FORCE=1 reruns).
last = {}
for r in all_rows:
    if r["fps"]:
        last[(r["mode"], r["stage"], r["profiler"], r["run"])] = r
cell = {}
for (m, s, p, _), r in last.items():
    cell.setdefault((m, s, p), []).append(r)
med = {k: statistics.median(float(r["fps"]) for r in v) for k, v in cell.items()}

TITLES = {"plain": "FPS without a profiler", "nsys": "FPS under nsys"}
md = []
for p in profs:
    md += [f"### {TITLES[p]}", "", "| Stage | " + " | ".join(modes) + " |", "|---|" + "---|" * len(modes)]
    for s in stages:
        cells = []
        for m in modes:
            v = cell.get((m, s, p))
            if not v:
                cells.append(""); continue
            txt = f"{med[(m, s, p)]:g}"
            if len(v) > 1:
                txt += f" (n={len(v)})"
            if any(r["throttled"] == "1" for r in v):
                txt += " ⚠"
            cells.append(txt)
        md.append(f"| {s} | " + " | ".join(cells) + " |")
    md.append("")
if len(profs) == 2:
    md += ["### Profiler overhead: nsys FPS / plain FPS", "", "| Stage | " + " | ".join(modes) + " |",
           "|---|" + "---|" * len(modes)]
    for s in stages:
        cells = [f"{med[(m, s, 'nsys')] / med[(m, s, 'plain')]:.3f}"
                 if (m, s, "nsys") in med and (m, s, "plain") in med else "" for m in modes]
        md.append(f"| {s} | " + " | ".join(cells) + " |")
    md.append("")
md += ["600 frames per run, first 50 excluded; median over the rounds where n > 1. "
       "⚠ = the over-current counters increased during at least one of the runs (throttled)."]
open(sys.argv[2], "w").write("\n".join(md) + "\n")
print("\n" + "\n".join(md))
EOF

if [[ "$start_mode" != "$(sudo nvpmodel -q | tail -1 | tr -d ' ')" ]]; then
    echo; echo "Restoring the starting power mode $start_mode"
    echo no | sudo nvpmodel -m "$start_mode" || true
    echo "jetson_clocks is reset by the mode change: run sudo jetson_clocks if you need pinned clocks"
fi
echo
echo "Commit with: git add -f $OUT/fps.csv $OUT/fps_table.md $SESSION"
