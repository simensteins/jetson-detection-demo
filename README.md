# jetson-detection-demo

v1 of the profiling replica: a COCO-pretrained YOLO detector running on the
Jetson Orin Nano, reading a video source, drawing boxes on a monitor, and
profiled with Nsight Systems. Built so the data feed and model can later be
swapped for the real (prod) pipeline.

## Structure

```
├── configs/default.yaml     # source, model, thresholds — the one file you edit
├── src/
│   ├── detect.py            # entrypoint: source -> detect -> draw -> display
│   ├── detectors.py         # model backends: YOLO (Ultralytics) or RF-DETR
│   ├── sources.py           # file / rtsp / webcam -> cv2.VideoCapture
│   └── profiling.py         # NVTX ranges (no-op when CUDA torch absent)
├── scripts/
│   ├── download_assets.sh   # fetch the sample video
│   ├── profile.sh           # nsys wrapper -> reports/*.nsys-rep
│   └── rtsp_server_mac.sh   # v1a: stream a looping file from the Mac
├── models/  data/  reports/ # gitignored (weights, footage, profiles)
```

## Setup (on the Jetson)

1. **torch / torchvision** must match your JetPack. Check your version:
   ```
   sudo apt show nvidia-jetpack 2>/dev/null | grep Version
   ```
   Then install the matching torch wheels via the Ultralytics NVIDIA Jetson
   guide (https://docs.ultralytics.com/guides/nvidia-jetson/). Do NOT
   `pip install torch` from PyPI — it has no CUDA for Tegra and will run on CPU.

2. Rest of the deps:
   ```
   pip install -r requirements.txt
   ```

3. Confirm OpenCV is present (JetPack usually ships it):
   ```
   python3 -c "import cv2; print(cv2.__version__)"
   ```

## Run — v1b (local file, start here)

```
bash scripts/download_assets.sh          # gets data/vtest.avi
python -m src.detect                      # uses configs/default.yaml
```
A window with detection boxes should appear on the Jetson's monitor. Press `q`
to quit.

## Run — v1a (streamed from the Mac over USB-C)

On the **Mac**: `mediamtx` in one terminal, then
`bash scripts/rtsp_server_mac.sh yourfile.mp4` in another.
On the **Jetson**: set `source: rtsp://192.168.55.100:8554/demo` in the config
(or `--source`), then run `python -m src.detect`.

## RF-DETR

The same pipeline can run Roboflow's RF-DETR (a DETR-style transformer
detector) instead of YOLO, so the two can be compared under identical
conditions. `configs/rfdetr.yaml` is `default.yaml` with `model: rfdetr-nano`.

Install (on the Jetson, in the activated `.venv`): the RF-DETR packages are
pinned in `requirements.txt` to a set known to work together (newer
transformers/huggingface_hub/pyDeprecate releases break it). Check first that
pip won't replace the Jetson torch/torchvision with CPU-only PyPI builds:
```
pip install --dry-run -r requirements.txt   # torch/torchvision must not be under "Would install"
pip install -r requirements.txt
python3 -c "import rfdetr, torch; print(torch.__version__, torch.cuda.is_available())"   # must print True
```

**Baseline** (PyTorch, rfdetr's own `predict()`), same NVTX ranges as YOLO:
```
python -m src.detect --config configs/rfdetr.yaml
bash scripts/profile.sh --config configs/rfdetr.yaml   # later --config wins
```

Engines are built with `scripts/export_rfdetr.py` (ONNX export, then trtexec).
The ONNX export step can run on another machine (`--onnx-only`, then
`--onnx <file>` on the Jetson); the engine build must run on the Jetson.
Precision (`--precision`, named in the engine file so every result traces to one):
- `fp32-strict` (default): every layer in full FP32 (`trtexec --noTF32`).
- `fp32-tf32`: FP32, but TensorRT may run matmuls/convolutions in TF32
  (10-bit mantissa) - TensorRT's own default on the Orin. The engine used in
  the nano experiments, `rfdetr-nano_fp32.engine` (built before these names
  existed, with plain trtexec), is this variant.
- `fp16`: reduced precision - not used in this project (detection accuracy first).

RF-DETR has no NMS, uses a plain resize (no letterbox), and its input
resolution is fixed by the model size (nano = 384, large = 704).

### Nano experiments (`scripts/infer_rfdetr_trt.py`)

The script exactly as the nano experiments were run (commit `87dceb4`),
engine `models/rfdetr-nano_fp32.engine` (fp32-tf32), stage flags
`--no-graph` and `--preprocess {cpu,gpu}`:
```
python3 scripts/infer_rfdetr_trt.py --engine models/rfdetr-nano_fp32.engine --no-graph --preprocess cpu ...  # "baseline" as run
python3 scripts/infer_rfdetr_trt.py --engine models/rfdetr-nano_fp32.engine --preprocess cpu ...             # + CUDA graph
python3 scripts/infer_rfdetr_trt.py --engine models/rfdetr-nano_fp32.engine --preprocess gpu ...             # + GPU preprocessing
```
**Note on the baseline - kept unchanged on purpose, for traceability.** These
runs used `--no-graph --preprocess cpu` as the TensorRT baseline, to mirror
YOLO's starting point (Ultralytics letterboxes on the CPU). That turned out not
to be RF-DETR's starting point: rfdetr's own `predict()` (source-checked,
rfdetr 1.11.1) already preprocesses on the GPU - uint8 upload through pinned
memory, then float / resize (antialias=False) / ImageNet-normalize on the GPU -
and uses no CUDA graph. Consequences for reading the nano results:
- graph vs. no graph (with CPU preprocessing) is a valid single-variable A/B;
- "+ GPU preprocessing" is a gain over a CPU path rfdetr itself never uses, so
  it is a controlled experiment from YOLO's starting point, not an
  optimization of RF-DETR.

The script and its results are left as they were; the corrected setup is the
Large script below.

### Large (`scripts/infer_rfdetr_large_trt.py`)

704×704, the architecture of the project's SAR/IR detector (here with COCO
weights), strict FP32. A separate script whose baseline is rfdetr's own
default path with the model run by TensorRT: GPU preprocessing only (no
`--preprocess` option - rfdetr never preprocesses on the CPU), no CUDA graph.
The one variable is `--no-graph`. `--mem-pool-mb` caps the engine build's
memory, which the Orin Nano shares with the CPU:
```
python -m src.detect --config configs/rfdetr-large.yaml --no-display --max-frames 600 --warmup-frames 50   # PyTorch
python3 scripts/export_rfdetr.py --size large --mem-pool-mb 2048   # -> models/rfdetr-large_fp32-strict.engine
python3 scripts/infer_rfdetr_large_trt.py --engine models/rfdetr-large_fp32-strict.engine --no-graph \
    --no-display --max-frames 600 --warmup-frames 50   # TensorRT baseline
python3 scripts/infer_rfdetr_large_trt.py --engine models/rfdetr-large_fp32-strict.engine \
    --no-display --max-frames 600 --warmup-frames 50   # + CUDA graph
```

## Profile

```
bash scripts/profile.sh                   # writes reports/v1_<timestamp>.nsys-rep
```
Copy the report to the Mac (scp over USB-C, or a USB stick) and open it in the
Nsight Systems GUI. The NVTX ranges label the timeline as decode / inference /
draw so you can see where each frame's time goes.

## Roadmap

- **v1** — Mac host, USB-C, video file, view-only reports *(this repo)*
- **v2** — add Ethernet: stable transport for stream in + report out
- **v3** — add x86 Ubuntu host: remote profiling, report pulled back automatically
- **prod** — real camera feed + custom-trained model (swap source + weights)
