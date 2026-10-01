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

**Optimized** (TensorRT + CUDA graph + GPU pre/postprocessing, i.e. the RF-DETR
version of Iterations 1-4):
```
python3 scripts/export_rfdetr.py --size nano --precision fp32   # -> models/rfdetr-nano_fp32.engine
python3 scripts/export_rfdetr.py --size nano --precision fp16   # -> models/rfdetr-nano_fp16.engine
python3 scripts/infer_rfdetr_trt.py --engine models/rfdetr-nano_fp16.engine --no-display --max-frames 600
```
The ONNX export step can run on another machine (`--onnx-only`, then
`--onnx <file>` on the Jetson); the engine build must run on the Jetson.
RF-DETR has no NMS, uses a plain resize (no letterbox), and its input
resolution is fixed by the model size (nano = 384). See the docstring of
`scripts/infer_rfdetr_trt.py` for what differs from YOLO.

**Large** (704×704, the architecture of the project's SAR/IR detector, here with
COCO weights). FP32 only. `--precision fp32` lets TensorRT use TF32 on the
Orin (its default); `--no-tf32` builds a strict-FP32 engine (`_notf32` suffix).
`--mem-pool-mb` caps the build's memory, which the Orin Nano shares with the CPU:
```
python -m src.detect --config configs/rfdetr-large.yaml --no-display --max-frames 600 --warmup-frames 50   # PyTorch
python3 scripts/export_rfdetr.py --size large --precision fp32 --mem-pool-mb 2048
python3 scripts/export_rfdetr.py --size large --precision fp32 --no-tf32 --mem-pool-mb 2048 --onnx models/rfdetr-large.onnx
python3 scripts/infer_rfdetr_trt.py --config configs/rfdetr-large.yaml --engine models/rfdetr-large_fp32.engine \
    --no-graph --preprocess gpu --no-display --max-frames 600 --warmup-frames 50   # TensorRT baseline
python3 scripts/infer_rfdetr_trt.py --config configs/rfdetr-large.yaml --engine models/rfdetr-large_fp32.engine \
    --preprocess gpu --no-display --max-frames 600 --warmup-frames 50              # + CUDA graph
```
The TensorRT baseline uses `--preprocess gpu` because rfdetr's own `predict()`
already preprocesses on the GPU (uint8 upload, then resize + normalize on the
GPU) - unlike Ultralytics, which letterboxes on the CPU.

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
