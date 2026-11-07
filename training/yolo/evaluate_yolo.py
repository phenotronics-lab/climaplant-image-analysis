"""Compute the segmentation scores of the trained YOLO model on the validation images."""

import os

from ultralytics import YOLO

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(ROOT, "models", "yolo11l_seg_best.pt")
DATASET_YAML = os.path.join(ROOT, "data", "yolo", "dataset.yaml")

if __name__ == "__main__":
    model = YOLO(MODEL_PATH)
    results = model.val(data=DATASET_YAML, split="val", imgsz=1280, batch=4, plots=False)
    print(f"mask mAP@0.5:0.95: {results.seg.map:.3f}")
    print(f"mask mAP@0.5: {results.seg.map50:.3f}")
