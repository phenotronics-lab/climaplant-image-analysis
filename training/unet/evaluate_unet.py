"""Compute Dice and IoU of the trained U-Net on the test images, before post-processing."""

import os
import torch
from tqdm import tqdm
import segmentation_models_pytorch as smp
from dataset import ArabidopsisFullImageDataset, iou_score, dice_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(ROOT, "models", "unet_best_model.pth")
IMAGES_DIR = os.path.join(ROOT, "data", "unet", "images")
MASKS_DIR = os.path.join(ROOT, "data", "unet", "masks")
BATCH_SIZE = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=1).to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

test_dataset = ArabidopsisFullImageDataset(IMAGES_DIR, MASKS_DIR, split='test')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_dice = 0
test_iou = 0
with torch.no_grad():
    for images, masks in tqdm(test_loader, desc="Test images"):
        images = images.to(device)
        masks = masks.to(device)
        preds = model(images)

        test_dice += dice_score(preds, masks)
        test_iou += iou_score(preds, masks)

test_dice /= len(test_loader)
test_iou /= len(test_loader)

print(f"Test Dice: {test_dice:.4f}")
print(f"Test IoU: {test_iou:.4f}")
