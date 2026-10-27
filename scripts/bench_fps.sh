#!/usr/bin/env bash
# FPS of every optimization stage of both models (YOLOv8n and M15), WITHOUT a
# profiler, repeated N times, for the paper's speed numbers. nsys adds its own
# overhead; this measures what the scripts print on their own.
#
# Same conditions as scripts/profile_paper_set.sh (see docs/measurement-protocol.md):
#     sudo systemctl isolate multi-user.target     # stop the desktop
#     sudo nvpmodel -m <25W id> && sudo jetson_clocks
#     source .venv/bin/activate
#     bash scripts/bench_fps.sh [set-name] [runs] [stage ...]
#                                    # defaults: set fps25w, 10 runs, every stage
#     sudo systemctl isolate graphical.target      # desktop back afterwards
#
# The runs are interleaved: round 1 runs every stage once, then round 2, and so
# on. Slow drift (temperature, background activity) then spreads over all
# stages instead of landing on whichever stage happens to run last.
# Stage arguments work as in profile_paper_set.sh: a full name (m15_nms) or a
# prefix up to an underscore (m15, yolo).
#
# Every run: 600 frames of data/testdata.mkv, the first 50 excluded from the
# FPS (the scripts' own timer), no display. Writes reports/fps/:
#   <set>_runs.csv      one row per run (stage, round, FPS, frames, seconds)
#   <set>_summary.csv   per stage: median, mean, std, min, max, CV, speed-up
#   <set>_summary.md    the same as a Markdown table
#   <set>_session.log   clocks, power mode, over-current counters before,
#                       after every round and at the end
#   <set>_tegrastats.txt, <set>/<stage>_r<NN>.log (each run's own output)
# About 75-90 minutes for all ten stages x 10 runs.
set -euo pipefail

SET="${1:-fps25w}"
RUNS="${2:-10}"
STAGES=("${@:3}")   # empty = every stage
SRC="data/testdata.mkv"
PAUSE=15   # seconds between runs
COMMON="--source $SRC --no-display --max-frames 600 --warmup-frames 50"
YOLO_ARGS="--config configs/default.yaml $COMMON"
M15_FP32="models/m15/m15_fast_fp32-tf32.engine"
M15_FP16="models/m15/m15_fast_fp16.engine"
OUT="reports/fps"
SESSION="$OUT/${SET}_session.log"
CSV="$OUT/${SET}_runs.csv"

cd "$(dirname "$0")/.."
[[ "$RUNS" =~ ^[0-9]+$ ]] && (( RUNS > 0 )) || { echo "runs must be a positive number, got '$RUNS'"; exit 1; }
mkdir -p "$OUT/$SET"
[[ -e "$CSV" ]] && { echo "$CSV exists - pick another set name"; exit 1; }

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

wanted() {  # wanted <stage>: is it selected by the stage arguments?
    (( ${#STAGES[@]} == 0 )) && return 0
    local s
    for s in "${STAGES[@]}"; do [[ "$1" == "$s" || "$1" == "$s"_* ]] && return 0; done
    return 1
}
SEL=()
for i in "${!NAMES[@]}"; do wanted "${NAMES[$i]}" && SEL+=("$i"); done
(( ${#SEL[@]} > 0 )) || { echo "no stage matches: ${STAGES[*]}"; exit 1; }

# --- preconditions (as in profile_paper_set.sh) ------------------------------
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
    echo "WARNING: the desktop is running (graphical.target). For clean numbers, stop it:"
    echo "         sudo systemctl isolate multi-user.target"
    read -r -p "Continue anyway? [y/N] " ans; [[ "$ans" == "y" ]] || exit 1
fi
oc_counters() { grep . /sys/class/hwmon/hwmon*/oc*_event_cnt 2>/dev/null | tr '\n' ' ' || true; }

selected=(); for i in "${SEL[@]}"; do selected+=("${NAMES[$i]}"); done
{
    echo "=== FPS set $SET: ${#SEL[@]} stages x $RUNS runs, started $(date -Is)"
    echo "--- stages: ${selected[*]}"
    echo "--- git: $(git rev-parse --short HEAD) $(git status --porcelain | wc -l) uncommitted file(s)"
    echo "--- $(sudo nvpmodel -q | tr '\n' ' ')"
    echo "$clocks"
    echo "--- desktop: $(systemctl is-active graphical.target || true)"
    echo "--- top CPU users:"; top -b -n 1 -o %CPU | sed -n '7,14p'
    echo "--- over-current counters BEFORE: $(oc_counters)"
} | tee "$SESSION"

sudo tegrastats --interval 1000 --logfile "$OUT/${SET}_tegrastats.txt" &
trap 'sudo tegrastats --stop >/dev/null 2>&1 || true' EXIT

echo "stage,round,fps,frames,seconds,started" > "$CSV"
oc_before="$(oc_counters)"
for (( r = 1; r <= RUNS; r++ )); do
    echo; echo "=== round $r/$RUNS  $(date +%T)" | tee -a "$SESSION"
    for i in "${SEL[@]}"; do
        name="${NAMES[$i]}"
        log="$OUT/$SET/${name}_r$(printf %02d "$r").log"
        started="$(date -Is)"
        # Word splitting of the command string is intended (no quoted arguments in it).
        ${CMDS[$i]} > "$log" 2>&1 || true
        line="$(grep -E '^processed [0-9]+ frames in [0-9.]+s.*FPS' "$log" | tail -1 || true)"
        if [[ -z "$line" ]]; then
            echo "  $name: FAILED - no FPS line, see $log" | tee -a "$SESSION"
            echo "$name,$r,,,,$started" >> "$CSV"
        else
            frames="$(sed -E 's/^processed ([0-9]+) frames.*/\1/' <<<"$line")"
            secs="$(sed -E 's/^processed [0-9]+ frames in ([0-9.]+)s.*/\1/' <<<"$line")"
            fps="$(sed -E 's/.*->  *([0-9.]+) FPS.*/\1/' <<<"$line")"
            printf "  %-14s %6s FPS\n" "$name" "$fps" | tee -a "$SESSION"
            echo "$name,$r,$fps,$frames,$secs,$started" >> "$CSV"
        fi
        sleep "$PAUSE"
    done
    oc_now="$(oc_counters)"
    echo "  over-current counters: $oc_now" >> "$SESSION"
    [[ "$oc_now" == "$oc_before" ]] || echo "  WARNING: over-current counters changed - the chip throttled in round $r" | tee -a "$SESSION"
done

{
    echo; echo "--- over-current counters AFTER (should equal BEFORE): $(oc_counters)"
    echo "=== finished $(date -Is)"
} | tee -a "$SESSION"

# --- summary ---------------------------------------------------------------
python3 - "$CSV" "$OUT/${SET}_summary" <<'EOF'
import csv, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1])))
order = list(dict.fromkeys(r["stage"] for r in rows))
by = {s: [float(r["fps"]) for r in rows if r["stage"] == s and r["fps"]] for s in order}
failed = {s: sum(1 for r in rows if r["stage"] == s and not r["fps"]) for s in order}
out = []
for s in order:
    v = by[s]
    if not v:
        out.append({"stage": s, "runs": 0, "failed": failed[s]}); continue
    sd = statistics.stdev(v) if len(v) > 1 else 0.0
    model = s.split("_")[0]
    base = by.get(f"{model}_baseline")
    out.append({"stage": s, "runs": len(v), "failed": failed[s],
                "median_fps": round(statistics.median(v), 2), "mean_fps": round(statistics.mean(v), 2),
                "std_fps": round(sd, 2), "min_fps": min(v), "max_fps": max(v),
                "cv_pct": round(100 * sd / statistics.mean(v), 2),
                "median_ms": round(1000 / statistics.median(v), 2),
                "speedup_vs_baseline": round(statistics.median(v) / statistics.median(base), 2) if base else ""})
keys = ["stage", "runs", "failed", "median_fps", "mean_fps", "std_fps", "min_fps", "max_fps",
        "cv_pct", "median_ms", "speedup_vs_baseline"]
with open(sys.argv[2] + ".csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(out)
md = ["| Stage | Runs | Median FPS | Mean ± std | Min – max | CV | Median frame | Speed-up |",
      "|---|---|---|---|---|---|---|---|"]
for o in out:
    if not o["runs"]:
        md.append(f"| {o['stage']} | 0 ({o['failed']} failed) | | | | | | |"); continue
    runs = f"{o['runs']}" + (f" ({o['failed']} failed)" if o["failed"] else "")
    md.append(f"| {o['stage']} | {runs} | **{o['median_fps']}** | {o['mean_fps']} ± {o['std_fps']} "
              f"| {o['min_fps']} – {o['max_fps']} | {o['cv_pct']} % | {o['median_ms']} ms "
              f"| {str(o['speedup_vs_baseline']) + '×' if o['speedup_vs_baseline'] != '' else '–'} |")
open(sys.argv[2] + ".md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
EOF
echo
echo "Results: $OUT/${SET}_summary.md (+ .csv), all runs in $CSV, conditions in $SESSION"
echo "Commit them with: git add -f $OUT/${SET}_*.csv $OUT/${SET}_*.md $SESSION"
