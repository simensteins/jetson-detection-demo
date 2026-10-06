#!/usr/bin/env python3
"""Rank GPU kernels by time inside the 'inference' NVTX range of an nsys report.

Step 1 of the kernel-level analysis: find the heaviest kernels, to select
which ones to profile with Nsight Compute.

Uses the same frame definition as scripts/analyze_nsys.py, so the numbers
match its tables: only kernels that start AND end inside an 'inference' NVTX
range count, and the first `skip_frames` inference ranges are dropped as
warm-up. `nsys stats --report cuda_gpu_kern_sum` gives a similar list, but
over the whole recording (warm-up and work outside 'inference' included).

Kernels are grouped by name. One name is often launched several times per
frame with different sizes (e.g. a GEMM kernel reused by several layers), so
the grid sizes are listed too, and the suggested Nsight Compute filter covers
one full frame of launches rather than the first few:

    --launch-skip  = launches per frame x skip_frames   (lands after warm-up)
    --launch-count = launches per frame                  (exactly one frame)

These assume ncu is run with --nvtx --nvtx-include "inference/", so only
launches inside the 'inference' range are counted.

Input is a report exported with:
    nsys export --type=sqlite -o report.sqlite report.nsys-rep
captured with --trace=cuda,nvtx (and --cuda-graph-trace=node if the script
uses a CUDA graph, so graph kernels appear individually).

Usage:
    python3 scripts/top_kernels.py <report.sqlite> [skip_frames] [top_n]
"""
from __future__ import annotations

import bisect
import re
import sqlite3
import sys


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: top_kernels.py <report.sqlite> [skip_frames] [top_n]")
        raise SystemExit(1)
    db_path = sys.argv[1]
    skip = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 15

    cur = sqlite3.connect(db_path).cursor()
    existing = {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    missing = [t for t in ("NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_KERNEL", "StringIds") if t not in existing]
    if missing:
        raise SystemExit(f"Report is missing table(s) {missing}. "
                         "Was it captured with --trace=cuda,nvtx (not nvtx-only)?")

    frames = cur.execute(
        "SELECT start, end FROM NVTX_EVENTS WHERE text='inference' ORDER BY start").fetchall()
    frames = frames[skip:] if len(frames) > skip else frames
    if not frames:
        raise SystemExit("No 'inference' NVTX ranges found after skipping warm-up.")
    starts = [s for s, _ in frames]
    n_frames = len(frames)

    rows = cur.execute(
        "SELECT s.value, k.start, k.end, k.gridX * k.gridY * k.gridZ, "
        "k.blockX * k.blockY * k.blockZ, k.registersPerThread "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.shortName "
        "WHERE k.start >= ? AND k.end <= ?", (frames[0][0], frames[-1][1])).fetchall()

    agg: dict[str, dict] = {}
    for name, start, end, grid, block, regs in rows:
        i = bisect.bisect_right(starts, start) - 1
        if i < 0 or end > frames[i][1]:
            continue  # not fully inside an analysed 'inference' range
        a = agg.setdefault(name, {"n": 0, "ns": 0, "grids": set(), "blocks": set(), "regs": set()})
        a["n"] += 1
        a["ns"] += end - start
        a["grids"].add(grid)
        a["blocks"].add(block)
        a["regs"].add(regs)

    total_ns = sum(a["ns"] for a in agg.values())
    launches = sum(a["n"] for a in agg.values())
    print(f"=== GPU kernels inside 'inference' (skipped first {skip} ranges as warm-up, n={n_frames}) ===")
    print(f"kernel time {total_ns / n_frames / 1e6:.3f} ms/frame, {launches / n_frames:.1f} launches/frame, "
          f"{len(agg)} distinct kernel names\n")
    print(f"{'#':>2} {'share':>6} {'cum':>6} {'ms/frame':>9} {'per frame':>9} {'grid (blocks)':>14} "
          f"{'block':>6} {'regs':>8}  kernel")

    ranked = sorted(agg.items(), key=lambda kv: -kv[1]["ns"])[:top_n]
    cum = 0
    for rank, (name, a) in enumerate(ranked, 1):
        cum += a["ns"]
        grids = sorted(a["grids"])
        grid = f"{grids[0]}-{grids[-1]}" if len(grids) > 1 else str(grids[0])
        block = "/".join(map(str, sorted(a["blocks"])))
        regs = sorted(a["regs"])
        reg = f"{regs[0]}-{regs[-1]}" if len(regs) > 1 else str(regs[0])
        print(f"{rank:2d} {100 * a['ns'] / total_ns:5.1f}% {100 * cum / total_ns:5.1f}% "
              f"{a['ns'] / n_frames / 1e6:9.3f} {a['n'] / n_frames:9.1f} {grid:>14} {block:>6} {reg:>8}  {name}")

    print("\n=== Suggested Nsight Compute filters (with --nvtx --nvtx-include \"inference/\") ===")
    for rank, (name, a) in enumerate(ranked, 1):
        per_frame = a["n"] / n_frames
        count = round(per_frame)
        note = "" if abs(per_frame - count) < 0.01 else f"   (WARNING: {per_frame:.2f} launches/frame is not constant)"
        print(f"{rank:2d} --kernel-name regex:'^{re.escape(name)}$' "
              f"--launch-skip {count * skip} --launch-count {count}{note}")


if __name__ == "__main__":
    main()
