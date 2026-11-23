#!/usr/bin/env python3
"""Where the time goes, per stage and power mode: the per-frame breakdown of
every nsys run of a scripts/run_power_modes.sh set, in one small CSV.

Runs on the Jetson (nsys export only works on Linux). For each nsys run in
reports/<set>/fps.csv it exports <mode>/<stage>_nsys_r<NN>.nsys-rep to .sqlite
(skipped if the .sqlite exists) and computes, with the same definitions as
plot_compare_reports.py's summarize_stage, the mean per frame of: frame time,
the `inference` range, and its parts - CPU work before the first kernel, GPU
kernel execution, idle gaps between kernels, work after the last kernel.
Copy the CSV to the Mac and plot it with plot_power_modes.py. Needs only the
standard library (no numpy or matplotlib), so it runs with any python3.

Usage (on the Jetson, from the repo root, venv active):
    python3 scripts/summarize_power_modes.py --set modes_v2            # every round
    python3 scripts/summarize_power_modes.py --set modes_v2 --rounds 1 # round 1 only (faster)
    python3 scripts/summarize_power_modes.py --set modes_v2 --delete-sqlite
Writes reports/<set>/breakdown.csv. Exporting takes ~10-60 s per report.
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_nsys import check_tables, per_frame_breakdown  # noqa: E402  (standard library only)

KEYS = ["frame_time_ms", "fps", "total_inf_ms", "pre_kernel_ms", "kernel_exec_ms", "idle_gap_ms",
        "post_kernel_ms", "dispatch_ms", "kernel_count", "frames"]


def summarize(path: str, skip: int) -> dict:
    """The numbers plot_compare_reports.summarize_stage computes, without its
    matplotlib import: frame time = decode start to the next decode start; the
    inference parts from analyze_nsys.per_frame_breakdown; means per frame."""
    con = sqlite3.connect(path)
    cur = con.cursor()
    check_tables(cur)
    frames = per_frame_breakdown(cur, skip)
    dec = [s for (s,) in cur.execute("SELECT start FROM NVTX_EVENTS WHERE text='decode' ORDER BY start")]
    con.close()
    frame_times = [(b - a) / 1e6 for a, b in zip(dec[skip:], dec[skip + 1:])]
    if not frames or not frame_times:
        raise SystemExit(f"no frames after skipping {skip} in {path}")

    def mean_ms(key: str) -> float:
        return statistics.mean(f[key] for f in frames) / 1000.0

    frame_ms = statistics.mean(frame_times)
    return dict(frame_time_ms=frame_ms, fps=1000.0 / frame_ms, total_inf_ms=mean_ms("total_us"),
                pre_kernel_ms=mean_ms("pre_kernel_us"), kernel_exec_ms=mean_ms("kernel_time_us"),
                idle_gap_ms=mean_ms("idle_gap_us"), post_kernel_ms=mean_ms("post_kernel_us"),
                dispatch_ms=mean_ms("runtime_us"),
                kernel_count=statistics.mean(f["kernel_count"] for f in frames), frames=len(frames))


def tag_of(mode: str) -> str:   # the folder name run_power_modes.sh uses for a mode
    return "".join(c for c in mode.lower() if c.isalnum())


def main() -> None:
    p = argparse.ArgumentParser(description="Per-frame time breakdown of a power-mode set's nsys runs")
    p.add_argument("--set", default="modes_v2")
    p.add_argument("--dir", default=None, help="Default: reports/<set>")
    p.add_argument("--rounds", nargs="+", default=None, help="Only these rounds (default: all)")
    p.add_argument("--skip-frames", type=int, default=50)
    p.add_argument("--delete-sqlite", action="store_true", help="Remove each .sqlite after reading it")
    args = p.parse_args()
    src = Path(args.dir or f"reports/{args.set}")

    runs = []
    for r in csv.DictReader(open(src / "fps.csv")):
        if r["profiler"] == "nsys" and r["fps"] and (args.rounds is None or r["run"] in args.rounds):
            runs.append((r["mode"], r["stage"], r["run"]))
    runs = list(dict.fromkeys(runs))   # FORCE=1 reruns appear twice; the files are the latest
    if not runs:
        raise SystemExit(f"no nsys runs in {src / 'fps.csv'}")

    rows, missing = [], []
    for i, (mode, stage, rnd) in enumerate(runs, 1):
        base = src / tag_of(mode) / f"{stage}_nsys_r{int(rnd):02d}"
        rep, db = base.with_suffix(".nsys-rep"), base.with_suffix(".sqlite")
        if not db.exists():
            if not rep.exists():
                missing.append(str(rep)); continue
            print(f"[{i}/{len(runs)}] exporting {rep}", flush=True)
            subprocess.run(["nsys", "export", "--type=sqlite", "--force-overwrite=true",
                            "-o", str(db), str(rep)], check=True, stdout=subprocess.DEVNULL)
        st = summarize(str(db), args.skip_frames)
        rows.append({"mode": mode, "stage": stage, "run": rnd, **{k: round(st[k], 3) for k in KEYS}})
        print(f"  {mode:11s} {stage:14s} r{rnd}  inference {st['total_inf_ms']:6.2f} ms  "
              f"(kernels {st['kernel_exec_ms']:.2f}, gaps {st['idle_gap_ms']:.2f}, "
              f"pre {st['pre_kernel_ms']:.2f}, post {st['post_kernel_ms']:.2f})", flush=True)
        if args.delete_sqlite:
            db.unlink()

    out = src / "breakdown.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mode", "stage", "run"] + KEYS)
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} runs)")
    if missing:
        print(f"no report for {len(missing)} run(s) (KEEP_REPORTS=0?), e.g. {missing[0]}")


if __name__ == "__main__":
    main()
