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
