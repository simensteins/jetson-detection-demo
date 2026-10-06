#!/usr/bin/env python3
"""Profile the heaviest RF-DETR nano kernels with Nsight Compute, one command.

Steps 2-4 of the kernel-level analysis, automated so no kernel names or
launch counts have to be copied by hand:

1. Rank the kernels in an nsys report with scripts/top_kernels.py's logic
   (kernels fully inside 'inference', warm-up frames skipped).
2. For each selected kernel, run Nsight Compute on the same pipeline the report
   was captured from - scripts/infer_rfdetr_trt.py --preprocess gpu, CUDA graph
   on - restricted to the 'inference' NVTX range, after warm-up, covering
   exactly one frame of that kernel's launches (--set full, --clock-control
   none so clocks are the jetson_clocks ones, like the nsys runs).
3. Export each report's details page to CSV.

Files written per kernel k1..kN (in reports/):
    ncu_<tag>_kN.ncu-rep      the Nsight Compute report (open in the GUI)
    ncu_<tag>_kN.csv          its details page as CSV
    ncu_<tag>_kN.cmd.txt      kernel, share, launches/frame and the exact command
and once:
    ncu_<tag>_selection.txt   the ranking and which kernels were selected

Nsight Compute needs root for the GPU performance counters on the Jetson, so
each ncu run is started with sudo (password asked once); the files are given
back to the calling user afterwards.

Usage (from the repo root, venv active, sudo jetson_clocks first):
    python scripts/ncu_rfdetr_kernels.py --dry-run          # show what would run
    python scripts/ncu_rfdetr_kernels.py                    # top 5 kernels
    python scripts/ncu_rfdetr_kernels.py --ranks 1,2,3,5,7  # a specific selection
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nsight Compute on the heaviest RF-DETR nano kernels")
    p.add_argument("--sqlite", default="reports/rfdetr_trt_graph_gpupre.sqlite",
                   help="nsys report (exported to .sqlite) the kernels are ranked from")
    p.add_argument("--engine", default="models/rfdetr-nano_fp32.engine",
                   help="the engine that report was captured with")
    p.add_argument("--skip", type=int, default=50, help="warm-up frames (as in analyze_nsys.py)")
    p.add_argument("--top", type=int, default=5, help="profile the top N kernels")
    p.add_argument("--ranks", default=None,
                   help="comma-separated ranks to profile instead of --top, e.g. 1,2,3,5,7")
    p.add_argument("--tag", default="rfdetr_nano", help="file name tag")
    p.add_argument("--ncu", default=None, help="ncu path (default: PATH, then /opt/nvidia/nsight-compute)")
    p.add_argument("--dry-run", action="store_true", help="print the selection and commands only")
    return p.parse_args()


def find_ncu(override: str | None) -> str:
    candidates = [override, shutil.which("ncu"), "/usr/local/cuda/bin/ncu",
                  *sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"), reverse=True)]
    for c in candidates:
        if c and Path(c).exists():
            return c
    raise SystemExit("ncu not found - pass --ncu /path/to/ncu")


def main() -> None:
    args = parse_args()
    sqlite = ROOT / args.sqlite
    if not sqlite.exists():
        raise SystemExit(f"{sqlite} not found - export it first: "
                         f"nsys export --type=sqlite -o {args.sqlite} {args.sqlite[:-7]}.nsys-rep")
    if not args.dry_run and not (ROOT / args.engine).exists():
        raise SystemExit(f"Engine {args.engine} not found")

    ranked, n_frames, total_ns = rank_kernels(str(sqlite), args.skip)
    ranks = ([int(r) for r in args.ranks.split(",")] if args.ranks
             else list(range(1, args.top + 1)))
    if max(ranks) > len(ranked):
        raise SystemExit(f"Only {len(ranked)} kernels in the report; can't select rank {max(ranks)}")

    reports = ROOT / "reports"
    lines = [f"Ranking from {args.sqlite} (kernels inside 'inference', first {args.skip} frames skipped, "
             f"n={n_frames}); shares are of GPU kernel time inside 'inference' "
             f"({total_ns / n_frames / 1e6:.3f} ms/frame)", ""]
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
        (reports / f"ncu_{args.tag}_selection.txt").write_text("\n".join(lines) + "\n")

    for k, rank in enumerate(ranks, 1):
        name, a = ranked[rank - 1]
        launch_skip, count, per_frame = ncu_filter(a, n_frames, args.skip)
        out = reports / f"ncu_{args.tag}_k{k}"
        cmd = ["sudo", ncu, "--nvtx", "--nvtx-include", "inference/",
               "--set", "full", "--clock-control", "none",
               "--kernel-name", f"regex:^{re.escape(name)}$",
               "--launch-skip", str(launch_skip), "--launch-count", str(count),
               "-f", "-o", str(out.relative_to(ROOT)),
               sys.executable, "scripts/infer_rfdetr_trt.py", "--engine", args.engine,
               "--preprocess", "gpu", "--no-display", "--max-frames", str(args.skip + 10)]
        header = (f"k{k} = rank {rank}: {name}\n"
                  f"share {100 * a['ns'] / total_ns:.1f}% of GPU kernel time inside 'inference', "
                  f"{per_frame:.2f} launches/frame -> launch-skip {launch_skip}, launch-count {count}\n")
        print(f"\n=== {header.rstrip()}\n$ {shlex.join(cmd)}")
        if abs(per_frame - count) >= 0.01:
            print(f"WARNING: launches/frame is not constant ({per_frame:.2f}); the profiled set "
                  "may not be exactly one frame")
        if args.dry_run:
            continue

        (out.with_suffix(".cmd.txt")).write_text(header + "command: " + shlex.join(cmd) + "\n")
        result = subprocess.run(cmd, cwd=ROOT)
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
