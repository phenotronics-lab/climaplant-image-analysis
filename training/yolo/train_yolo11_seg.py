"""Train YOLO11l-seg on the annotated leaves."""

import os
from ultralytics import YOLO
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_YAML_PATH = os.path.join(ROOT, "data", "yolo", "dataset.yaml")
MODEL_TO_TRAIN = "yolo11l-seg.pt"
OUTPUT_DIR = os.path.join(ROOT, "results", "yolo_training")

EPOCHS = 150
IMAGE_SIZE = 1280
BATCH_SIZE = 4


def main():
    model_type = os.path.splitext(os.path.basename(MODEL_TO_TRAIN))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = f"{model_type}_run_{timestamp}"

    model = YOLO(MODEL_TO_TRAIN)

    model.train(
        data=DATASET_YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        name=run_name,
        project=OUTPUT_DIR,
        exist_ok=True,

        # Augmentations
        degrees=90.0,
        translate=0.1,
        scale=0.3,
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.02,
        hsv_s=0.3,
        hsv_v=0.3,
        mosaic=0.7,
        mixup=0.1,
        copy_paste=0.1,

        # Optimization (optimizer='auto' replaces lr0 and momentum)
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5.0,
        box=15.0,
        cls=0.3,
        patience=30,

        # Other parameters
        rect=True,
        cos_lr=True,
        seed=42,
        deterministic=True,
    )


if __name__ == "__main__":
    main()
