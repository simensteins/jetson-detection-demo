# Measurement protocol

How profiling runs on the Jetson must be set up so that their numbers can be compared, and the mistakes that taught us each rule. Every number in [results-paper25w.md](results-paper25w.md) was measured this way.

## The rules

| Rule | How | Why |
|---|---|---|
| **One power mode for everything: 25 W** | `sudo nvpmodel -q` / `sudo nvpmodel -m <id>` (IDs: `grep POWER_MODEL /etc/nvpmodel.conf`) | MAXN SUPER throttles (over-current) with M15 and RF-DETR Large, so its numbers change during a run. 15 W is stable but slow. 25 W was stable for both models. |
| **Pin the clocks** | `sudo jetson_clocks`, then check `sudo jetson_clocks --show`: GPU Min = Max = Current = 918 MHz, all six CPUs 1344 MHz, EMC 3199 MHz | Without it the GPU clock floats between 306 and 918 MHz. The same code then varied 78–101 FPS between runs and climbed run after run. **`jetson_clocks` is lost after every `nvpmodel` change and every reboot**: run it last, and check. |
| **No desktop** | `sudo systemctl isolate multi-user.target`, run from the text console (Ctrl+Alt+F3) or SSH; back with `sudo systemctl isolate graphical.target` | With the desktop up, `gnome-shell` used 92 % of a core and a Firefox tab another ~44 %. YOLOv8n FP16 is CPU-bound, so that load slowed it and made frame times noisy. |
| **Same video for everything** | `data/testdata.mkv` (1920×1080), 600 frames, the first 50 excluded | Preprocessing cost depends on the input size; the old September runs used `vtest.avi` (768×576) and are not comparable. |
| **All stages in one session, one at a time** | `bash scripts/profile_paper_set.sh` | Pasting several long commands at once fed the later lines into the running process; three reports came out ~370 KB and empty. The script runs every stage in order, pauses 60 s between runs, and flags any report under 1 MiB. |
| **Check for throttling** | over-current counters before and after: `grep . /sys/class/hwmon/hwmon*/oc*_event_cnt` | If a counter increased, the chip cut its clocks during the run and the numbers are not valid. The script logs both. |

## Running a set

On the Jetson, in the repo, with the venv active (`source .venv/bin/activate`):

```bash
sudo systemctl isolate multi-user.target      # from a text console or SSH
sudo nvpmodel -m <25W id>
sudo jetson_clocks
sudo jetson_clocks --show
top -o %CPU                                   # nothing else busy (q to quit)
bash scripts/profile_paper_set.sh [set-name]  # ~25 min, default name: paper25w
sudo systemctl isolate graphical.target
```

It writes, per stage, `reports/<set>_<model>_<stage>.nsys-rep`, `.sqlite` and `.log`, plus `reports/<set>_session.log` (power mode, clocks, git commit, top CPU users, over-current counters before and after) and a `tegrastats` log.

Every profile uses `nsys profile --trace=cuda,nvtx --cuda-graph-trace=node`. Without `--cuda-graph-trace=node`, a CUDA graph shows up as one block instead of its individual kernels; that once made Iteration 1 look like it had removed 85 % of the kernels.

## Analysing

The `.sqlite` files are analysed on the Mac. The nsys command-line tool in the macOS app only runs on Linux, so export to sqlite on the Jetson: `nsys export --type=sqlite -o <name>.sqlite <name>.nsys-rep`.

- `python3 scripts/analyze_nsys.py <report.sqlite>`: per-frame breakdown of one report.
- `python3 scripts/plot_compare_reports.py --name <set> --outdir reports/figures/<set> --reports <one .sqlite per stage> --labels <one label per stage>`: the five comparison figures and a summary CSV.
- `python3 scripts/plot_nsys_report.py <report.sqlite>`: figures for a single report.

Definitions used in all figures:

| Quantity | Definition |
|---|---|
| Frame time / FPS | mean time from one `decode` NVTX start to the next, after warm-up; FPS = 1000 / that |
| Pre-kernel | time from the start of `inference` to the first GPU kernel (CPU preprocessing, uploads, launch) |
| Kernel execution | sum of GPU kernel durations inside `inference` (includes GPU preprocessing and NMS kernels) |
| GPU idle | span from first to last kernel minus kernel execution: gaps where the GPU waits for the CPU |
| Post-kernel | time from the last kernel to the end of `inference` |
| CUDA API time | CPU time in CUDA runtime calls inside `inference`, split into active dispatch and blocked waits (calls over 1 ms) |

The scripts' own printed FPS and the NVTX-derived FPS matched exactly in the paper set (e.g. 86.7 / 86.7), which is a useful cross-check.

## Pitfalls we hit

- **Run-to-run variation is not frame-to-frame variation.** At ~90 FPS, 1 ms per frame is ~8 FPS. Report a median and spread over several runs (`--warmup-frames 100 --max-frames 2000`, five runs), not a single printed FPS.
- **The report does not record the power mode.** nsys's `CpuInfo maxfreqhz` always shows 1728 MHz, whatever the mode. Write the mode down, or use the session log.
- **Content-dependent frames.** Frames with more detections take longer with Ultralytics/torchvision NMS (more kernels). This shows as slow frames in otherwise clean runs; it is not interference.
- **TensorRT engines are tied to the TensorRT version that built them.** Two YOLO FP16 engines built elsewhere failed with "Serialized Engine Version: 243 vs 240". Rebuild on the Jetson, inside the same venv. `python3 scripts/inspect_engine.py <file.engine>` shows an engine's inputs/outputs and dtypes.
- **"Using an engine plan file across different models of devices"** appears for every engine on this board, including ones built on it minutes earlier. It did not affect speed.
- **Ultralytics `half=True` keeps FP32 engine I/O.** Use `scripts/export_yolo_fp16.py` (YOLO) or `scripts/export_m15.py --precision fp16` (M15) for FP16 input/output.
- **Production FPS is a separate measurement.** The product shows detections on a monitor; `cv2.imshow` cost ~3.8 ms per frame (YOLO 74 → 52 FPS when measured). Measure the deployed configuration with display separately; the stage comparison is measured without it.
