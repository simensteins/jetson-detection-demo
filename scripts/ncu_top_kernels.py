#!/usr/bin/env python3
"""Profile the heaviest kernels of a pipeline stage with Nsight Compute, one command.

The generic form of ncu_rfdetr_kernels.py (on the RF-DETR branch), so no
kernel names or launch counts have to be copied by hand:

1. Rank the kernels in an nsys report of the stage with scripts/top_kernels.py
   (kernels fully inside 'inference', warm-up frames skipped, grouped by name).
2. For each selected kernel, run Nsight Compute on the same stage the report
   was captured from, restricted to the 'inference' NVTX range, after warm-up,
   covering exactly one frame of that kernel's launches (a name launched by
   several layers is profiled once per layer). --set full; --clock-control
   none, so the clocks are the jetson_clocks ones, as in the nsys runs.
3. Export each report's details page to CSV.

The default stage is the final M15 pipeline (FP16 engine, GPU preprocessing,
fixed NMS in the CUDA graph), ranked from its paper-set report. Any other
stage: pass its report with --sqlite and its command after "--" (without
--max-frames; it is appended: skip + 10 frames).

Files written per kernel k1..kN (in reports/ncu/):
    ncu_<tag>_kN.ncu-rep      the Nsight Compute report (open in the GUI)
    ncu_<tag>_kN.csv          its details page as CSV
    ncu_<tag>_kN.cmd.txt      kernel, share, launches/frame and the exact command
and once:
    ncu_<tag>_selection.txt   the ranking and which kernels were selected

Nsight Compute needs root for the GPU performance counters on the Jetson, so
each ncu run is started with sudo (password asked once); the files are given
back to the calling user afterwards.

Usage (on the Jetson, from the repo root, venv active, 25 W + sudo jetson_clocks first):
    python3 scripts/ncu_top_kernels.py --dry-run          # show the ranking and the commands
    python3 scripts/ncu_top_kernels.py                    # top 5 kernels of the M15 FP16 stage
    python3 scripts/ncu_top_kernels.py --ranks 1,2,4,7    # a specific selection
    python3 scripts/ncu_top_kernels.py --sqlite reports/paper25w_yolo_fp16.sqlite --tag yolo_fp16 -- \\
        python3 scripts/infer_cuda_graph_fp16.py --config configs/default.yaml \\
        --source data/testdata.mkv --no-display --engine yolov8n_fp16_io.engine
"""
from __future__ import annotations

import argparse
import getpass
import glob
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from top_kernels import ncu_filter, rank_kernels  # noqa: E402

DEFAULT_SQLITE = "reports/paper25w_v2_m15_fp16.sqlite"
DEFAULT_CMD = ["python3", "scripts/infer_m15_trt.py", "--engine", "models/m15/m15_fast_fp16.engine",
               "--source", "data/testdata.mkv", "--nms", "fixed", "--no-display"]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    argv = sys.argv[1:]
    cmd = DEFAULT_CMD
    if "--" in argv:
        i = argv.index("--")
        argv, cmd = argv[:i], argv[i + 1:]
    p = argparse.ArgumentParser(description="Nsight Compute on the heaviest kernels of a stage")
    p.add_argument("--sqlite", default=DEFAULT_SQLITE,
                   help="nsys report of the stage, exported to .sqlite, to rank the kernels from "
                        f"(default {DEFAULT_SQLITE}; also looked for in reports/nsys_paper_reports/)")
    p.add_argument("--skip", type=int, default=50, help="warm-up frames (as in analyze_nsys.py)")
    p.add_argument("--top", type=int, default=5, help="profile the top N kernels")
    p.add_argument("--ranks", default=None,
                   help="comma-separated ranks to profile instead of --top, e.g. 1,2,4,7")
    p.add_argument("--tag", default="m15_fp16", help="file name tag")
    p.add_argument("--ncu", default=None, help="ncu path (default: PATH, then /opt/nvidia/nsight-compute)")
    p.add_argument("--dry-run", action="store_true", help="print the selection and commands only")
    return p.parse_args(argv), cmd


def find_ncu(override: str | None) -> str:
    candidates = [override, shutil.which("ncu"), "/usr/local/cuda/bin/ncu",
                  *sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"), reverse=True)]
    for c in candidates:
        if c and Path(c).exists():
            return c
    raise SystemExit("ncu not found - pass --ncu /path/to/ncu")


def find_sqlite(path: str) -> Path:
    for cand in (ROOT / path, ROOT / "reports" / "nsys_paper_reports" / Path(path).name):
        if cand.exists():
            return cand
    raise SystemExit(f"{path} not found - export it from its .nsys-rep first: "
                     f"nsys export --type=sqlite -o {path} {path[:-7]}.nsys-rep")


def main() -> None:
    args, cmd = parse_args()
    if "--max-frames" in cmd:
        raise SystemExit("leave --max-frames out of the command; it is set to skip + 10")
    # sudo resets PATH, so run the venv's interpreter by its full path.
    if cmd and cmd[0] in ("python", "python3"):
        cmd = [sys.executable, *cmd[1:]]
    sqlite = find_sqlite(args.sqlite)

    ranked, n_frames, total_ns = rank_kernels(str(sqlite), args.skip)
    ranks = ([int(r) for r in args.ranks.split(",")] if args.ranks
             else list(range(1, args.top + 1)))
    if max(ranks) > len(ranked):
        raise SystemExit(f"Only {len(ranked)} kernels in the report; can't select rank {max(ranks)}")

    outdir = ROOT / "reports" / "ncu"
    outdir.mkdir(parents=True, exist_ok=True)
    lines = [f"Ranking from {sqlite.relative_to(ROOT)} (kernels inside 'inference', first {args.skip} "
             f"frames skipped, n={n_frames}); shares are of GPU kernel time inside 'inference' "
             f"({total_ns / n_frames / 1e6:.3f} ms/frame)", f"Stage command: {shlex.join(cmd)}", ""]
    cum = 0
    for rank, (name, a) in enumerate(ranked[:max(max(ranks), 15)], 1):
        cum += a["ns"]
        mark = "*" if rank in ranks else " "
        lines.append(f"{mark} {rank:2d} {100 * a['ns'] / total_ns:5.1f}% (cum {100 * cum / total_ns:5.1f}%) "
                     f"{a['n'] / n_frames:6.1f}/frame  {name}")
    selected_share = sum(ranked[r - 1][1]["ns"] for r in ranks) / total_ns
    lines += ["", f"Selected (*): ranks {ranks}, together {100 * selected_share:.1f}% of GPU kernel time "
                  "inside 'inference'"]
    print("\n".join(lines))

    ncu = find_ncu(args.ncu) if not args.dry_run else (args.ncu or "ncu")
    user = getpass.getuser()
    if not args.dry_run:
        (outdir / f"ncu_{args.tag}_selection.txt").write_text("\n".join(lines) + "\n")

    for k, rank in enumerate(ranks, 1):
        name, a = ranked[rank - 1]
        launch_skip, count, per_frame = ncu_filter(a, n_frames, args.skip)
        out = outdir / f"ncu_{args.tag}_k{k}"
        ncu_cmd = ["sudo", ncu, "--nvtx", "--nvtx-include", "inference/",
                   "--set", "full", "--clock-control", "none",
                   "--kernel-name", f"regex:^{re.escape(name)}$",
                   "--launch-skip", str(launch_skip), "--launch-count", str(count),
                   "-f", "-o", str(out.relative_to(ROOT)),
                   *cmd, "--max-frames", str(args.skip + 10)]
        header = (f"k{k} = rank {rank}: {name}\n"
                  f"share {100 * a['ns'] / total_ns:.1f}% of GPU kernel time inside 'inference', "
                  f"{per_frame:.2f} launches/frame -> launch-skip {launch_skip}, launch-count {count}\n")
        print(f"\n=== {header.rstrip()}\n$ {shlex.join(ncu_cmd)}")
        if abs(per_frame - count) >= 0.01:
            print(f"WARNING: launches/frame is not constant ({per_frame:.2f}); the profiled set "
                  "may not be exactly one frame")
        if args.dry_run:
            continue

        out.with_suffix(".cmd.txt").write_text(header + "command: " + shlex.join(ncu_cmd) + "\n")
        result = subprocess.run(ncu_cmd, cwd=ROOT)
        rep = out.with_suffix(".ncu-rep")
        if result.returncode != 0 or not rep.exists():
            print(f"ncu failed for k{k} (exit {result.returncode}) - see the output above; continuing")
            continue
        subprocess.run(["sudo", "chown", f"{user}:{user}", str(rep)], check=False)
        with open(out.with_suffix(".csv"), "w") as f:
            subprocess.run([ncu, "--import", str(rep), "--page", "details", "--csv"], stdout=f, check=False)
        print(f"wrote {rep.name}, {out.with_suffix('.csv').name}, {out.with_suffix('.cmd.txt').name}")


if __name__ == "__main__":
    main()
