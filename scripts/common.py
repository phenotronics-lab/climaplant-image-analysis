"""Functions shared by the analysis scripts."""

import glob
import os
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

# Homography estimation
USE_RANSAC = False
RANSAC_THRESHOLD_PX = 5.0


def list_images(folder, image_type, position):
    """Return the images of one type and one position, sorted by request time.

    Expected name: <x>-<type1>-<type2>-<ddmmyy>-<HHMMSS>-<ddmmyy>-<HHMMSS>-<POS>.tiff
    """
    images = []
    for path in glob.glob(os.path.join(folder, image_type, "*.tiff")):
        parts = os.path.basename(path).replace(".tiff", "").split("-")
        if parts[7] == position:
            time = datetime.strptime(f"{parts[3]} {parts[4]}", "%d%m%y %H%M%S")
            images.append({"path": path, "time": time})
    return sorted(images, key=lambda image: image["time"])


def read_image(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    return image


def to_8bit(image):
    """Min-max normalization of a grayscale image."""
    low, high = image.min(), image.max()
    if high == low:
        return np.zeros(image.shape, dtype=np.uint8)
    return ((image - low) / (high - low) * 255).astype(np.uint8)


def to_display(image):
    """Convert a grayscale or color image to 8-bit BGR."""
    if image.ndim > 2:
        return cv2.convertScaleAbs(image[:, :, :3])
    return cv2.cvtColor(to_8bit(image), cv2.COLOR_GRAY2BGR)


def load_points(csv_path, dx=0, dy=0):
    points = pd.read_csv(csv_path)[["X", "Y"]].to_numpy(dtype=np.float32)
    return points + np.array([dx, dy], dtype=np.float32)


def compute_homography(points_src, points_dst):
    if USE_RANSAC:
        matrix, _ = cv2.findHomography(points_src, points_dst, cv2.RANSAC, RANSAC_THRESHOLD_PX)
    else:
        matrix, _ = cv2.findHomography(points_src, points_dst)
    return matrix


def warp_mask(mask, matrix, shape):
    height, width = shape[:2]
    return cv2.warpPerspective((mask > 0).astype(np.uint8), matrix, (width, height), flags=cv2.INTER_NEAREST)


def draw_label(image, text, position, scale=0.9):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)


def open_video(path, width, height, fps):
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
