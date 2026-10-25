# Figures

| Folder | What | Conditions | Made with |
|---|---|---|---|
| `paper25w/` | **The paper set.** YOLOv8n and M15, all optimization stages, five charts each (`01_frame_time` … `05_cpu_gpu_swimlane`) plus `*_summary.csv` with the numbers behind them. | 25 W, `jetson_clocks`, desktop stopped, `data/testdata.mkv` (1920×1080), one session on 2026-10-25; no throttling (over-current counters 0 before and after). Conditions and per-run output: `reports/nsys_paper_reports/logfiles/`. | `scripts/profile_paper_set.sh`, then `scripts/plot_compare_reports.py --name paper25w_yolo` / `--name paper25w_m15` |
| `yolov8n_vtest_2026-09/` | The original YOLOv8n iteration figures (baseline → CUDA graph → GPU preprocessing → custom NMS) and, in `single_reports/`, single-report figures of the Iteration 3 run and the display test. | `data/vtest.avi` (768×576), September 2026; power mode and clocks not recorded. The baseline bar in `01_frame_time` comes from a different baseline run (33.45 ms) than the rest of its baseline numbers (30.6 ms). | `scripts/plot_paper_figures.py`, `scripts/plot_nsys_report.py` |
| `archive/m15_25w_2026-10-24/` | Earlier M15 figures (baseline → FP16). Superseded by `paper25w/`. | 25 W, desktop running. | `scripts/plot_compare_reports.py --name m15` |
| `archive/yolov8n_25w_desktop_2026-10-25/` | YOLOv8n set from the morning of 2026-10-25. Superseded: the desktop and a browser were loading the CPU, and the GPU preprocessing fix was not in yet. Local only (not in git). | 25 W, desktop + Firefox running. | `scripts/plot_compare_reports.py --name yolov8n_25w` |

New figures from `plot_compare_reports.py` and `plot_nsys_report.py` land in `reports/figures/` itself unless `--outdir` says otherwise; move them into a folder here (or pass `--outdir reports/figures/<set>`).
