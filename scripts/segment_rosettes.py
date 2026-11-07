"""Segment the rosette on the NoSat images with the U-Net. Writes the masks and a video."""

import os

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from tqdm import tqdm

from common import draw_label, list_images, open_video, read_image, to_display

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMAGE_DIR = os.path.join(ROOT, "data", "example")
MODEL_PATH = os.path.join(ROOT, "models", "unet_best_model.pth")
OUTPUT_DIR = os.path.join(ROOT, "results", "segmentation")

POSITION = "POS12"
THRESHOLD = 0.5
MIN_AREA_PX = 100
BORDER_PX = None  # a number removes the objects that enter this margin
VIDEO_FPS = 10


def predict_mask(model, image, device):
    # The network needs image sizes divisible by 32
    height, width = image.shape
    pad_h = (height + 31) // 32 * 32 - height
    pad_w = (width + 31) // 32 * 32 - width
    top, left = pad_h // 2, pad_w // 2
    padded = np.pad(image, ((top, pad_h - top), (left, pad_w - left)), mode="reflect")

    scaled = padded.astype(np.float32) * (1.0 / 65535.0)
    mean, std = scaled.mean(), scaled.std()
    scaled = (scaled - mean) / std if std > 1e-8 else scaled - mean
    tensor = torch.from_numpy(scaled).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                output = model(tensor)
        else:
            output = model(tensor)
        proba = torch.sigmoid(output).cpu().squeeze(1).numpy()[0]

    mask = (proba > THRESHOLD).astype(np.uint8) * 255
    return mask[top:top + height, left:left + width]


def clean_mask(mask):
    height, width = mask.shape
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return np.zeros_like(mask)

    x, y = stats[1:, cv2.CC_STAT_LEFT], stats[1:, cv2.CC_STAT_TOP]
    w, h = stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]
    keep = stats[1:, cv2.CC_STAT_AREA] > MIN_AREA_PX
    if BORDER_PX is not None:
        keep &= (x > BORDER_PX) & (x + w < width - BORDER_PX) & (y > BORDER_PX) & (y + h < height - BORDER_PX)
    return np.isin(labels, np.where(keep)[0] + 1).astype(np.uint8) * 255


os.makedirs(OUTPUT_DIR, exist_ok=True)
images = list_images(IMAGE_DIR, "FLUO-NoSat", POSITION)
height, width = read_image(images[0]["path"]).shape

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=1).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False)["model_state_dict"])
model.eval()

masks = np.zeros((len(images), height, width), dtype=np.uint8)
video = open_video(os.path.join(OUTPUT_DIR, f"{POSITION}_masks.mp4"), width, height, VIDEO_FPS)

for i, entry in enumerate(tqdm(images)):
    image = read_image(entry["path"])
    masks[i] = clean_mask(predict_mask(model, image, device))

    frame = to_display(image)
    contours, _ = cv2.findContours(masks[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
    draw_label(frame, f"{POSITION} | {entry['time']} | frame {i + 1}/{len(images)}", (20, 40))
    video.write(frame)

video.release()
np.save(os.path.join(OUTPUT_DIR, f"{POSITION}_masks.npy"), masks)
print(f"Empty masks: {int(np.sum(masks.max(axis=(1, 2)) == 0))}/{len(images)}")
