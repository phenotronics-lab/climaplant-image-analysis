"""Train the U-Net that segments the rosette."""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from datetime import datetime
import segmentation_models_pytorch as smp
from tqdm import tqdm
import random
import numpy as np

from dataset import (
    ArabidopsisFullImageDataset,
    iou_score,
    dice_score,
    plot_metrics,
    visualize_predictions
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IMAGES_DIR = os.path.join(ROOT, "data", "unet", "images")
MASKS_DIR = os.path.join(ROOT, "data", "unet", "masks")
OUTPUT_DIR = os.path.join(ROOT, "results", "unet_training")

BATCH_SIZE = 8
EPOCHS = 60
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
ENCODER_NAME = "resnet34"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_model(
    images_dir,
    masks_dir,
    batch_size=6,
    epochs=60,
    num_workers=4,
    learning_rate=1e-4,
    encoder_name="resnet34"
):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = True if device.type == 'cuda' else False

    scaler = GradScaler('cuda', enabled=use_amp)

    run_dir = os.path.join(OUTPUT_DIR, datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    train_ds = ArabidopsisFullImageDataset(images_dir, masks_dir, split='train')
    val_ds = ArabidopsisFullImageDataset(images_dir, masks_dir, split='val')

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    model = smp.Unet(
        encoder_name=encoder_name, encoder_weights="imagenet",
        in_channels=1, classes=1
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    dice_loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)
    bce_loss_fn = nn.BCEWithLogitsLoss()
    early_stopping = EarlyStopping(patience=15)

    best_val_dice = 0.0
    history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_dice": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda', enabled=use_amp):
                preds = model(images)
                loss = 0.5 * dice_loss_fn(preds, masks) + 0.5 * bce_loss_fn(preds, masks)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix(loss=train_loss / (pbar.n + 1))

        model.eval()
        v_loss, v_iou, v_dice = 0, 0, 0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="[Val]", leave=False):
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                preds = model(images)
                loss_val = 0.5 * dice_loss_fn(preds, masks) + 0.5 * bce_loss_fn(preds, masks)

                v_loss += loss_val.item()
                v_iou += iou_score(preds, masks)
                v_dice += dice_score(preds, masks)

        metrics = {
            "train_loss": train_loss / len(train_loader),
            "val_loss": v_loss / len(val_loader),
            "val_iou": v_iou / len(val_loader),
            "val_dice": v_dice / len(val_loader)
        }

        for k, v in metrics.items(): history[k].append(v)
        scheduler.step(metrics["val_loss"])

        print(f"Epoch {epoch+1}: val loss: {metrics['val_loss']:.4f} | val Dice: {metrics['val_dice']:.4f}")

        if metrics["val_dice"] > best_val_dice:
            best_val_dice = metrics["val_dice"]
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(run_dir, "best_model.pth"))

        early_stopping(metrics["val_loss"])
        if early_stopping.early_stop:
            print("Early stopping")
            break

    plot_metrics(history["train_loss"], history["val_loss"], history["val_iou"], history["val_dice"], run_dir)
    model.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pth"))['model_state_dict'])
    visualize_predictions(model, val_ds, device, run_dir)

if __name__ == "__main__":
    train_model(
        images_dir=IMAGES_DIR,
        masks_dir=MASKS_DIR,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        num_workers=NUM_WORKERS,
        learning_rate=LEARNING_RATE,
        encoder_name=ENCODER_NAME
    )
