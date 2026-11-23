### FPS without a profiler

| Stage | 15W | 25W | MAXN_SUPER |
|---|---|---|---|
| yolo_baseline | 43.8 (n=3) | 56 (n=3) | 62.2 (n=3) |
| yolo_graph | 48.8 (n=3) | 61.5 (n=3) | 69 (n=3) |
| yolo_gpupre | 54.3 (n=3) | 74.6 (n=3) | 80.7 (n=3) |
| yolo_nms | 54.7 (n=3) | 75.8 (n=3) | 81.6 (n=3) |
| yolo_fp16 | 80.8 (n=3) | 111.9 (n=3) | 123.6 (n=3) |
| m15_baseline | 13.7 (n=3) | 19.5 (n=3) | 20.9 (n=3) ⚠ |
| m15_graph | 13.9 (n=3) | 19.8 (n=3) | 21.3 (n=3) ⚠ |
| m15_gpupre | 14.2 (n=3) | 20.5 (n=3) | 21.9 (n=3) ⚠ |
| m15_nms | 14.1 (n=3) | 20.6 (n=3) | 22 (n=3) ⚠ |
| m15_fp16 | 29.5 (n=3) | 42 (n=3) | 45.1 (n=3) ⚠ |

### FPS under nsys

| Stage | 15W | 25W | MAXN_SUPER |
|---|---|---|---|
| yolo_baseline | 40 (n=3) | 50.4 (n=3) | 56.3 (n=3) |
| yolo_graph | 45.5 (n=3) | 56.8 (n=3) | 63.9 (n=3) |
| yolo_gpupre | 51.7 (n=3) | 69.8 (n=3) | 76.1 (n=3) |
| yolo_nms | 52.7 (n=3) | 73.1 (n=3) | 79 (n=3) |
| yolo_fp16 | 77.5 (n=3) | 86 (n=3) | 100.4 (n=3) |
| m15_baseline | 13.4 (n=3) | 19 (n=3) ⚠ | 20.4 (n=3) ⚠ |
| m15_graph | 13.8 (n=3) | 19.6 (n=3) ⚠ | 21.1 (n=3) ⚠ |
| m15_gpupre | 13.9 (n=3) | 20.1 (n=3) ⚠ | 21.5 (n=3) ⚠ |
| m15_nms | 13.9 (n=3) | 20.3 (n=3) ⚠ | 21.6 (n=3) ⚠ |
| m15_fp16 | 28.9 (n=3) | 41.1 (n=3) | 44.2 (n=3) ⚠ |

### Profiler overhead: nsys FPS / plain FPS

| Stage | 15W | 25W | MAXN_SUPER |
|---|---|---|---|
| yolo_baseline | 0.913 | 0.900 | 0.905 |
| yolo_graph | 0.932 | 0.924 | 0.926 |
| yolo_gpupre | 0.952 | 0.936 | 0.943 |
| yolo_nms | 0.963 | 0.964 | 0.968 |
| yolo_fp16 | 0.959 | 0.769 | 0.812 |
| m15_baseline | 0.978 | 0.974 | 0.976 |
| m15_graph | 0.993 | 0.990 | 0.991 |
| m15_gpupre | 0.979 | 0.980 | 0.982 |
| m15_nms | 0.986 | 0.985 | 0.982 |
| m15_fp16 | 0.980 | 0.979 | 0.980 |

600 frames per run, first 50 excluded; median over the rounds where n > 1. ⚠ = the over-current counters increased during at least one of the runs (throttled).
