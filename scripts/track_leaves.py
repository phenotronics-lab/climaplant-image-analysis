"""Segment the leaves with YOLO, link them through time with LapTrack and write a video."""

import os

import cv2
import numpy as np
from laptrack import OverLapTrack
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
from tqdm import tqdm
from ultralytics import YOLO

from common import draw_label, list_images, open_video, read_image, to_8bit

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMAGE_DIR = os.path.join(ROOT, "data", "example")
MODEL_PATH = os.path.join(ROOT, "models", "yolo11l_seg_best.pt")
OUTPUT_DIR = os.path.join(ROOT, "results", "tracking")

POSITION = "POS12"

# YOLO
IMAGE_SIZE = 1280
CONFIDENCE = 0.1
IOU = 0.5

# LapTrack. Coefficients: (offset, overlap, iou, ratio_1, ratio_2)
CUTOFF = 0.8
METRIC_COEFS = (1.0, 0.0, -1.0, 0.0, 0.0)
GAP_MAX_FRAMES = 5
GAP_CUTOFF = 0.8
GAP_METRIC_COEFS = (1.0, 0.0, -0.5, -0.5, 0.0)

VIDEO_FPS = 10
MASK_OPACITY = 0.35

os.makedirs(OUTPUT_DIR, exist_ok=True)
images = list_images(IMAGE_DIR, "FLUO-NoSat", POSITION)
height, width = read_image(images[0]["path"]).shape[:2]

# Segmentation
model = YOLO(MODEL_PATH)
segmentation = np.zeros((len(images), height, width), dtype=np.uint16)
for i, entry in enumerate(tqdm(images, desc="Segmentation")):
    rgb = cv2.cvtColor(to_8bit(read_image(entry["path"])), cv2.COLOR_GRAY2RGB)
    result = model.predict(source=rgb, imgsz=IMAGE_SIZE, conf=CONFIDENCE, iou=IOU, verbose=False)[0]
    if result.masks is not None:
        for leaf_id, mask in enumerate(result.masks.data.cpu().numpy(), start=1):
            resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            segmentation[i][resized > 0.5] = leaf_id
np.save(os.path.join(OUTPUT_DIR, f"{POSITION}_segmentation.npy"), segmentation)

# Tracking
tracker = OverLapTrack(
    cutoff=CUTOFF,
    metric_coefs=METRIC_COEFS,
    gap_closing_cutoff=GAP_CUTOFF,
    gap_closing_max_frame_count=GAP_MAX_FRAMES,
    gap_closing_metric_coefs=GAP_METRIC_COEFS,
)
tracks, _, _ = tracker.predict_overlap_dataframe(segmentation)

tracking = np.zeros_like(segmentation)
for (frame, label), row in tqdm(tracks.iterrows(), total=len(tracks), desc="Tracking"):
    tracking[int(frame)][segmentation[int(frame)] == label] = int(row["tree_id"]) + 1
np.save(os.path.join(OUTPUT_DIR, f"{POSITION}_tracking.npy"), tracking)

# Video
hsv = np.zeros((int(tracking.max()) + 1, 1, 3), dtype=np.uint8)
hsv[:, 0, 0] = (np.arange(len(hsv)) * 0.618 % 1.0 * 179).astype(np.uint8)
hsv[:, 0, 1:] = (220, 255)
colors = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[:, 0, :]

video = open_video(os.path.join(OUTPUT_DIR, f"{POSITION}_tracking.mp4"), width, height, VIDEO_FPS)
for i, entry in enumerate(tqdm(images, desc="Video")):
    labels = tracking[i]
    frame = cv2.cvtColor(to_8bit(read_image(entry["path"])), cv2.COLOR_GRAY2BGR)

    leaves = labels > 0
    frame[leaves] = ((1 - MASK_OPACITY) * frame[leaves] + MASK_OPACITY * colors[labels[leaves]]).astype(np.uint8)
    edges = find_boundaries(labels, mode="inner")
    frame[edges] = colors[labels[edges]]

    for region in regionprops(labels):
        y, x = region.centroid
        draw_label(frame, str(region.label), (int(x), int(y)), scale=0.7)
    draw_label(frame, f"{POSITION} | {entry['time']} | frame {i + 1}/{len(images)}", (20, 40))
    video.write(frame)
video.release()
