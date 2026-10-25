#!/usr/bin/env bash
# Download COCO val2017 (5000 images, ~780 MB) and its annotations for
# scripts/eval_yolo_coco.py. Ends up in data/coco/ (gitignored).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/coco
cd data/coco
if [[ ! -d val2017 ]]; then
    curl -L -o val2017.zip http://images.cocodataset.org/zips/val2017.zip
    unzip -q val2017.zip && rm val2017.zip
fi
if [[ ! -f annotations/instances_val2017.json ]]; then
    curl -L -o annotations.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip -q -j annotations.zip annotations/instances_val2017.json -d annotations && rm annotations.zip
fi
echo "images: $(ls val2017 | wc -l)  annotations: $(du -h annotations/instances_val2017.json | cut -f1)"
