#!/usr/bin/env python3
"""Extract per-frame pipeline timing from an nsys sqlite export.

Input is a report produced by:
    nsys export --type=sqlite -o report.sqlite report.nsys-rep
captured with --trace=cuda,nvtx (kernel-level tables required, not nvtx-only).

Reports, per inference call (averaged over all captured frames after
skipping a warm-up window):
  - NVTX range durations: decode / inference / draw / display
  - What "inference" is actually made of: GPU kernel execution, CPU-side
    CUDA Runtime API dispatch overhead, synchronization, memcpy/memset,
    and time before/after the kernel sequence (Python-side pre/post work)
  - Frame-to-frame variance and what it correlates with, to explain why
    some inference calls take longer than others

Usage:
    python3 scripts/analyze_nsys.py <report.sqlite> [skip_frames]
"""
from __future__ import annotations

import sqlite3
import statistics
import sys

CUPTI_TABLES = ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_RUNTIME",
                 "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION", "CUPTI_ACTIVITY_KIND_MEMCPY",
                 "CUPTI_ACTIVITY_KIND_MEMSET")


def check_tables(cur: sqlite3.Cursor) -> None:
    existing = {r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    missing = [t for t in ("NVTX_EVENTS",) + CUPTI_TABLES if t not in existing]
    if missing:
        raise SystemExit(
            f"Report is missing required table(s): {missing}. "
            "Was it captured with --trace=cuda,nvtx (not nvtx-only)?"
        )


def sum_overlap(cur: sqlite3.Cursor, table: str, start: int, end: int) -> tuple[int, float]:
    """Count and total duration (us) of events in `table` fully inside [start, end]."""
    row = cur.execute(
        f"SELECT COUNT(*), COALESCE(SUM(end-start),0) FROM {table} "
        f"WHERE start >= ? AND end <= ?", (start, end),
    ).fetchone()
    return row[0], row[1] / 1000.0


def nvtx_range_stats(cur: sqlite3.Cursor, skip: int) -> dict[str, list[float]]:
    ranges: dict[str, list[float]] = {}
    for name in ("decode", "inference", "draw", "display"):
        rows = cur.execute(
            "SELECT start, end FROM NVTX_EVENTS WHERE text=? ORDER BY start", (name,),
        ).fetchall()
        rows = rows[skip:] if len(rows) > skip else rows
        if rows:
            ranges[name] = [(e - s) / 1000.0 for s, e in rows]
    return ranges


def per_frame_breakdown(cur: sqlite3.Cursor, skip: int) -> list[dict]:
    frames = cur.execute(
        "SELECT start, end FROM NVTX_EVENTS WHERE text='inference' ORDER BY start",
    ).fetchall()
    frames = frames[skip:] if len(frames) > skip else frames

    out = []
    for f_start, f_end in frames:
        kernels = cur.execute(
            "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL "
            "WHERE start >= ? AND end <= ? ORDER BY start", (f_start, f_end),
        ).fetchall()
        kernel_count = len(kernels)
        kernel_time = sum(e - s for s, e in kernels) / 1000.0

        _, runtime_us = sum_overlap(cur, "CUPTI_ACTIVITY_KIND_RUNTIME", f_start, f_end)
        _, sync_us = sum_overlap(cur, "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION", f_start, f_end)
        _, memcpy_us = sum_overlap(cur, "CUPTI_ACTIVITY_KIND_MEMCPY", f_start, f_end)
        _, memset_us = sum_overlap(cur, "CUPTI_ACTIVITY_KIND_MEMSET", f_start, f_end)

        total_us = (f_end - f_start) / 1000.0
        if kernels:
            k_first = min(s for s, e in kernels)
            k_last = max(e for s, e in kernels)
            kernel_span_us = (k_last - k_first) / 1000.0
            pre_kernel_us = (k_first - f_start) / 1000.0
            post_kernel_us = (f_end - k_last) / 1000.0
            idle_gap_us = kernel_span_us - kernel_time
        else:
            kernel_span_us = pre_kernel_us = post_kernel_us = idle_gap_us = 0.0

        out.append(dict(
            total_us=total_us, kernel_count=kernel_count, kernel_time_us=kernel_time,
            kernel_span_us=kernel_span_us, idle_gap_us=idle_gap_us,
            runtime_us=runtime_us, sync_us=sync_us, memcpy_us=memcpy_us, memset_us=memset_us,
            pre_kernel_us=pre_kernel_us, post_kernel_us=post_kernel_us,
        ))
    return out


def summarize(values: list[float]) -> dict:
    return dict(
        n=len(values),
        mean=statistics.mean(values),
        stdev=statistics.stdev(values) if len(values) > 1 else 0.0,
        min=min(values),
        max=max(values),
    )


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    sx, sy = statistics.pstdev(xs), statistics.pstdev(ys)
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: analyze_nsys.py <report.sqlite> [skip_frames]")
        raise SystemExit(1)
    db_path = sys.argv[1]
    skip = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    con = sqlite3.connect(db_path)
    cur = con.cursor()
    check_tables(cur)

    ranges = nvtx_range_stats(cur, skip)
    print(f"=== NVTX range summary (skipped first {skip} 'inference' events as warm-up) ===")
    for name, durations in ranges.items():
        s = summarize(durations)
        print(f"{name:10s} n={s['n']:4d}  mean={s['mean']:9.2f}us  "
              f"stdev={s['stdev']:8.2f}us  min={s['min']:9.2f}us  max={s['max']:9.2f}us")

    per_frame = per_frame_breakdown(cur, skip)
    con.close()

    if not per_frame:
        raise SystemExit("No inference events found after skipping warm-up.")

    print(f"\n=== Inference breakdown, averaged over {len(per_frame)} frames ===")
    keys = ["total_us", "kernel_count", "kernel_time_us", "idle_gap_us",
            "runtime_us", "sync_us", "memcpy_us", "memset_us",
            "pre_kernel_us", "post_kernel_us"]
    means = {}
    for k in keys:
        vals = [f[k] for f in per_frame]
        s = summarize(vals)
        means[k] = s["mean"]
        print(f"{k:16s} mean={s['mean']:10.2f}  stdev={s['stdev']:9.2f}  "
              f"min={s['min']:9.2f}  max={s['max']:9.2f}")

    total_mean = means["total_us"]
    print(f"\n=== Share of total inference time (mean total = {total_mean:.2f}us) ===")
    for k, label in [("kernel_time_us", "GPU kernel execution"),
                      ("idle_gap_us", "GPU idle gaps between kernels"),
                      ("runtime_us", "CUDA Runtime API dispatch"),
                      ("sync_us", "Synchronization"),
                      ("memcpy_us", "Memcpy"), ("memset_us", "Memset"),
                      ("pre_kernel_us", "Pre-kernel (Python/setup)"),
                      ("post_kernel_us", "Post-kernel (Python/postproc)")]:
        pct = 100.0 * means[k] / total_mean
        print(f"{label:32s} {means[k]:9.2f}us  ({pct:5.1f}%)")

    print("\n=== Variance investigation: what correlates with total inference time? ===")
    totals = [f["total_us"] for f in per_frame]
    for k in ["kernel_count", "kernel_time_us", "idle_gap_us", "runtime_us",
              "pre_kernel_us", "post_kernel_us"]:
        vals = [f[k] for f in per_frame]
        r = pearson(totals, vals)
        print(f"corr(total, {k:16s}) = {r:+.3f}")

    idx_sorted = sorted(range(len(per_frame)), key=lambda i: per_frame[i]["total_us"])
    slowest = per_frame[idx_sorted[-1]]
    fastest = per_frame[idx_sorted[0]]
    median = per_frame[idx_sorted[len(idx_sorted) // 2]]
    print("\nFastest frame:", {k: round(v, 1) for k, v in fastest.items()})
    print("Median frame: ", {k: round(v, 1) for k, v in median.items()})
    print("Slowest frame:", {k: round(v, 1) for k, v in slowest.items()})


if __name__ == "__main__":
    main()
