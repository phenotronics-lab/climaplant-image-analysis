"""Dataset, metrics and training plots for the U-Net."""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt


class ArabidopsisFullImageDataset(Dataset):
    def __init__(self, images_dir, masks_dir, split='train'):
        self.images_dir = os.path.join(images_dir, split)
        self.masks_dir = os.path.join(masks_dir, split)
        self.split = split

        if not os.path.exists(self.images_dir):
            raise ValueError(f"Folder not found: {self.images_dir}")
        if not os.path.exists(self.masks_dir):
            raise ValueError(f"Folder not found: {self.masks_dir}")

        self.image_files = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.tif')])

        if len(self.image_files) == 0:
            raise ValueError(f"No .tif image found in {self.images_dir}")

        # Augmentation for training only
        if split == 'train':
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, border_mode=0, p=0.5),
                A.Affine(translate_percent=0.05, scale=(0.95, 1.05), rotate=0, p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
                A.GaussNoise(p=0.2),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_filename)

        mask_filename = os.path.splitext(img_filename)[0] + '.png'
        mask_path = os.path.join(self.masks_dir, mask_filename)

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        image = np.array(Image.open(img_path)).astype(np.float32)
        image /= 65535.0

        # Standardize each image
        mean_img = np.mean(image)
        std_img = np.std(image)
        if std_img > 1e-8:
            image = (image - mean_img) / std_img
        else:
            image = image - mean_img

        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127).astype(np.float32)

        image = np.expand_dims(image, axis=-1)

        augmented = self.transform(image=image, mask=mask)
        image_tensor = augmented['image']
        mask_tensor = augmented['mask'].unsqueeze(0)

        return image_tensor, mask_tensor


def iou_score(outputs, targets, threshold=0.5):
    outputs = torch.sigmoid(outputs)
    outputs = (outputs > threshold).float()
    targets = targets.float()

    intersection = (outputs * targets).sum(dim=[1, 2, 3])
    union = outputs.sum(dim=[1, 2, 3]) + targets.sum(dim=[1, 2, 3]) - intersection

    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


def dice_score(outputs, targets, threshold=0.5):
    outputs = torch.sigmoid(outputs)
    outputs = (outputs > threshold).float()
    targets = targets.float()

    intersection = (outputs * targets).sum(dim=[1, 2, 3])
    dice = (2 * intersection + 1e-6) / (outputs.sum(dim=[1, 2, 3]) + targets.sum(dim=[1, 2, 3]) + 1e-6)

    return dice.mean().item()


def plot_metrics(train_losses, val_losses, val_ious, val_dices, output_dir):
    epochs = range(1, len(train_losses) + 1)

    # Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o', markersize=3)
    plt.plot(epochs, val_losses, label='Val Loss', marker='s', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss.svg"))
    plt.savefig(os.path.join(output_dir, "loss.png"), dpi=150)
    plt.close()

    # IoU
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_ious, label='Val IoU', marker='o', markersize=3, color='green')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.title("Validation IoU Score")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "iou.svg"))
    plt.savefig(os.path.join(output_dir, "iou.png"), dpi=150)
    plt.close()

    # Dice
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_dices, label='Val Dice', marker='o', markersize=3, color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.legend()
    plt.title("Validation Dice Score")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dice.svg"))
    plt.savefig(os.path.join(output_dir, "dice.png"), dpi=150)
    plt.close()


def visualize_predictions(model, dataset, device, output_dir, num_samples=4):
    model.eval()

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

    with torch.no_grad():
        for i in range(num_samples):
            idx = np.random.randint(0, len(dataset))
            image, mask = dataset[idx]

            image_input = image.unsqueeze(0).to(device)
            pred = model(image_input)
            pred = torch.sigmoid(pred).cpu().squeeze().numpy()

            image_np = image.squeeze().cpu().numpy()
            mask_np = mask.squeeze().cpu().numpy()

            axes[i, 0].imshow(image_np, cmap='gray')
            axes[i, 0].set_title('Original image')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(mask_np, cmap='gray')
            axes[i, 1].set_title('Ground truth mask')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(pred, cmap='gray')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "predictions_sample.png"), dpi=150)
    plt.close()
