"""Project the rosette masks onto the RGB and IR images and write one video per channel.

Run segment_rosettes.py first.
"""

import os

import cv2
import numpy as np
from tqdm import tqdm

from common import compute_homography, draw_label, list_images, load_points, open_video, read_image, to_display, warp_mask

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMAGE_DIR = os.path.join(ROOT, "data", "example")
CALIBRATION_DIR = os.path.join(ROOT, "data", "calibration")
OUTPUT_DIR = os.path.join(ROOT, "results", "registration")

POSITION = "POS12"
MASKS_PATH = os.path.join(ROOT, "results", "segmentation", f"{POSITION}_masks.npy")
VIDEO_FPS = 10


def make_video(image_type, name, project):
    target_images = {entry["time"]: entry["path"] for entry in list_images(IMAGE_DIR, image_type, POSITION)}
    video = None
    for i, entry in enumerate(tqdm(images, desc=name)):
        if entry["time"] not in target_images:
            continue
        frame = to_display(read_image(target_images[entry["time"]]))
        height, width = frame.shape[:2]
        if video is None:
            video = open_video(os.path.join(OUTPUT_DIR, f"{POSITION}_{name}.mp4"), width, height, VIDEO_FPS)

        contours, _ = cv2.findContours(project(masks[i], frame.shape), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0, 255, 0), max(1, width // 1000))
        scale = max(0.5, width / 1500)
        draw_label(frame, f"{POSITION} {name} | {entry['time']}", (10, int(35 * scale)), scale)
        video.write(frame)
    if video is not None:
        video.release()


os.makedirs(OUTPUT_DIR, exist_ok=True)
images = list_images(IMAGE_DIR, "FLUO-NoSat", POSITION)
masks = np.load(MASKS_PATH)

points_fluo = load_points(os.path.join(CALIBRATION_DIR, "Results_bin1.csv"))
matrix_rgb = compute_homography(points_fluo, load_points(os.path.join(CALIBRATION_DIR, "Results_RGB.csv")))
matrix_ir = compute_homography(points_fluo, load_points(os.path.join(CALIBRATION_DIR, "Results_IR.csv")))
np.save(os.path.join(OUTPUT_DIR, "H_fluo_to_rgb.npy"), matrix_rgb)
np.save(os.path.join(OUTPUT_DIR, "H_fluo_to_ir.npy"), matrix_ir)

make_video("VIS-Nat", "rgb", lambda mask, shape: warp_mask(mask, matrix_rgb, shape))
make_video("IR-Nat", "ir", lambda mask, shape: warp_mask(mask, matrix_ir, shape))
