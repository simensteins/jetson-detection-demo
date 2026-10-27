#!/usr/bin/env python3
"""Summarize the Nsight Compute reports written by scripts/ncu_top_kernels.py:
one row per profiled launch, a time-weighted summary per kernel, and the
warnings Nsight Compute's own rules raised.

Per launch: duration, grid, waves per SM, registers, theoretical vs achieved
occupancy, SM (compute) vs memory throughput as % of peak, what limits it,
the busiest pipeline, the top warp stall reason, L2 hit rate, and memory
reads (L2 misses x 32 B) with the bandwidth they imply.

"Limited by" follows Nsight Compute's speed-of-light guidance: a unit above
60 % of its peak is the limiter (memory or compute); if neither reaches 60 %,
the kernel is latency-bound / under-utilizing the GPU (too few warps, stalls).
Memory throughput on the Orin is the busiest of L1/L2: its DRAM counters are
not available (the GPU shares LPDDR5 with the CPU), so DRAM traffic is
estimated from L2 misses.

Durations come from Nsight Compute's replays, which flush the caches before
each pass, so they are longer than in a normal run; use them to compare
launches, and nsys for real times.

Reads the reports with Nsight Compute's Python module (ncu_report), found in
the Nsight Compute install: the macOS app, /opt/nvidia/nsight-compute/<ver>/extras/python,
or set NCU_PYTHON to its folder.

Usage:
    python3 scripts/summarize_ncu.py                 # reports/ncu/ncu_m15_fp16_k*.ncu-rep
    python3 scripts/summarize_ncu.py --tag yolo_fp16
Writes reports/ncu/ncu_<tag>_launches.csv and ncu_<tag>_summary.md.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NCU_PYTHON_DIRS = [os.environ.get("NCU_PYTHON", ""),
                   "/Applications/NVIDIA Nsight Compute.app/Contents/MacOS/python",
                   *sorted(glob.glob("/opt/nvidia/nsight-compute/*/extras/python"), reverse=True),
                   *sorted(glob.glob("/usr/local/cuda*/nsight-compute*/extras/python"), reverse=True)]
SOL_LIMIT = 60.0  # % of peak above which a unit counts as the limiter
SECTOR_BYTES = 32
STALL = re.compile(r"^smsp__average_warps_issue_stalled_(\w+)_per_issue_active\.ratio$")
PIPE = re.compile(r"^sm__inst_executed_pipe_(\w+)\.avg\.pct_of_peak_sustained_active$")


def load_ncu_report():
    for d in NCU_PYTHON_DIRS:
        if d and (Path(d) / "ncu_report.py").exists():
            sys.path.insert(0, d)
            import ncu_report
            return ncu_report
    raise SystemExit("ncu_report not found - set NCU_PYTHON to Nsight Compute's extras/python folder")


def metric(action, name: str, default: float = float("nan")) -> float:
    m = action.metric_by_name(name)
    if m is None:
        return default
    try:
        return m.as_double()
    except Exception:
        return default


def launch_row(action) -> dict:
    names = list(action.metric_names())
    dur_us = metric(action, "gpu__time_duration.sum") / 1e3
    sm = metric(action, "sm__throughput.avg.pct_of_peak_sustained_elapsed")
    mem = metric(action, "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed")
    if max(sm, mem) >= SOL_LIMIT:
        limiter = "memory" if mem >= sm else "compute"
    else:
        limiter = "latency"
    pipes = {PIPE.match(n).group(1): metric(action, n) for n in names if PIPE.match(n)}
    pipes = {k: v for k, v in pipes.items() if "fp64" not in k and v == v}
    top_pipe = max(pipes, key=pipes.get) if pipes else ""
    stalls = {STALL.match(n).group(1): metric(action, n) for n in names if STALL.match(n)}
    stalls.pop("selected", None)
    top_stall = max(stalls, key=stalls.get) if stalls else ""
    miss_mb = metric(action, "lts__t_sectors_lookup_miss.sum", 0.0) * SECTOR_BYTES / 1e6
    return {
        "duration_us": round(dur_us, 1),
        "grid": int(metric(action, "launch__grid_size")),
        "block": int(metric(action, "launch__block_size")),
        "regs": int(metric(action, "launch__registers_per_thread")),
        "waves_per_sm": round(metric(action, "launch__waves_per_multiprocessor"), 2),
        "occ_theoretical_pct": round(metric(action, "sm__maximum_warps_per_active_cycle_pct"), 1),
        "occ_achieved_pct": round(metric(action, "sm__warps_active.avg.pct_of_peak_sustained_active"), 1),
        "sm_pct": round(sm, 1),
        "mem_pct": round(mem, 1),
        "l2_pct": round(metric(action, "lts__throughput.avg.pct_of_peak_sustained_elapsed"), 1),
        "l1_pct": round(metric(action, "l1tex__throughput.avg.pct_of_peak_sustained_elapsed"), 1),
        "limited_by": limiter,
        "top_pipe": f"{top_pipe} {pipes[top_pipe]:.0f}%" if top_pipe else "",
        "tensor_pct": round(pipes.get("tensor_op_hmma", 0.0), 1),
        "top_stall": f"{top_stall} {stalls[top_stall]:.2f}" if top_stall else "",
        "l2_hit_pct": round(metric(action, "lts__t_sector_hit_rate.pct"), 1),
        "mem_read_mb": round(miss_mb, 2),
        "mem_read_gbps": round(miss_mb / 1e3 / (dur_us / 1e6), 1) if dur_us else 0.0,
    }


def warnings(action) -> list[tuple[str, str]]:
    """(rule name, message) for each warning Nsight Compute's rules raised."""
    out = []
    for r in action.rule_results_as_dicts():
        msg = r.get("rule_message") or {}
        if msg.get("type") == 2 and msg.get("message"):  # 2 = warning
            out.append((r.get("name", ""), msg["message"]))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize ncu_top_kernels.py reports")
    p.add_argument("--tag", default="m15_fp16")
    p.add_argument("--dir", default="reports/ncu")
    args = p.parse_args()
    ncu_report = load_ncu_report()
    reports = sorted(glob.glob(str(ROOT / args.dir / f"ncu_{args.tag}_k*.ncu-rep")),
                     key=lambda s: int(re.search(r"_k(\d+)\.ncu-rep$", s).group(1)))
    if not reports:
        raise SystemExit(f"no reports matching {args.dir}/ncu_{args.tag}_k*.ncu-rep")

    rows, md = [], [f"# Nsight Compute summary: {args.tag}", "",
                    "Durations are from Nsight Compute's replays (caches flushed per pass): compare "
                    "launches with them, take real times from nsys. Limited by: a unit above "
                    f"{SOL_LIMIT:.0f} % of peak, else latency (neither compute nor memory near peak).", ""]
    for path in reports:
        k = re.search(r"_(k\d+)\.ncu-rep$", path).group(1)
        rng = ncu_report.load_report(path).range_by_idx(0)
        actions = [rng.action_by_idx(i) for i in range(rng.num_actions())]
        name = actions[0].name()
        launch = [launch_row(a) for a in actions]
        for i, r in enumerate(launch):
            rows.append({"k": k, "kernel": name, "launch": i + 1, **r})
        total = sum(r["duration_us"] for r in launch)
        w = lambda key: sum(r[key] * r["duration_us"] for r in launch) / total  # noqa: E731
        by_limit = {}
        for r in launch:
            by_limit[r["limited_by"]] = by_limit.get(r["limited_by"], 0) + r["duration_us"]
        md += [f"## {k}: `{name}`", "",
               f"{len(launch)} launches, {total / 1e3:.2f} ms under ncu. Time-weighted: SM {w('sm_pct'):.0f} %, "
               f"memory {w('mem_pct'):.0f} %, tensor pipe {w('tensor_pct'):.0f} %, achieved occupancy "
               f"{w('occ_achieved_pct'):.0f} %, L2 hit {w('l2_hit_pct'):.0f} %. Time by limiter: "
               + ", ".join(f"{lim} {100 * t / total:.0f} %" for lim, t in
                           sorted(by_limit.items(), key=lambda kv: -kv[1])) + ".", "",
               "| # | µs | Grid | Waves/SM | Regs | Occ. theor./achieved | SM % | Mem % | Limited by "
               "| Top pipe | Tensor % | Top stall | L2 hit | Mem read MB (GB/s) |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for i, r in enumerate(launch, 1):
            md.append(f"| {i} | {r['duration_us']} | {r['grid']} | {r['waves_per_sm']} | {r['regs']} "
                      f"| {r['occ_theoretical_pct']} / {r['occ_achieved_pct']} % | {r['sm_pct']} | {r['mem_pct']} "
                      f"| {r['limited_by']} | {r['top_pipe']} | {r['tensor_pct']} | {r['top_stall']} "
                      f"| {r['l2_hit_pct']} % | {r['mem_read_mb']} ({r['mem_read_gbps']}) |")
        # Messages carry each launch's numbers, so group by rule and show the
        # message of the longest launch raising it as the example.
        by_rule: dict[str, dict[int, tuple]] = {}  # rule -> {launch index: (duration, message)}
        for i, (a, r) in enumerate(zip(actions, launch)):
            for rule, wmsg in warnings(a):
                by_rule.setdefault(rule, {}).setdefault(i, (r["duration_us"], wmsg))
        if by_rule:
            md += ["", "Nsight Compute warnings (launches raising each; example from the longest of them):", ""]
            for rule, hits in sorted(by_rule.items(), key=lambda kv: -len(kv[1])):
                example = max(hits.values())[1].replace("@section:", "").replace("@url:", "")
                md.append(f"- **{rule}** ({len(hits)}/{len(actions)}): {example[:500]}")
        md.append("")
        print(f"{k}: {len(launch)} launches, {total / 1e3:.2f} ms under ncu, "
              + ", ".join(f"{lim} {100 * t / total:.0f}%" for lim, t in by_limit.items()) + f"  {name[:70]}")

    out = ROOT / args.dir
    with open(out / f"ncu_{args.tag}_launches.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    (out / f"ncu_{args.tag}_summary.md").write_text("\n".join(md) + "\n")
    print(f"wrote {args.dir}/ncu_{args.tag}_launches.csv and ncu_{args.tag}_summary.md")


if __name__ == "__main__":
    main()
