"""
Dataset loader for GID Built-Up Area Segmentation.

Training images: JPG in GID-img-{1-4}/<subfolder>/
Training labels: PNG (RGB color masks) in GID-label/
    - Built-up = (255, 0, 0) → binary 1
    - Everything else → binary 0

Test images: PNG in Test/Images/
"""
import os
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

from . import config


def get_train_pairs() -> List[Tuple[Path, Path]]:
    """
    Scan all training image directories and match with labels.
    
    Images are JPG in GID-img-{1-4}/<subfolder>/
    Labels are PNG in GID-label/ with the same base name.
    
    Returns list of (image_path, label_path) tuples.
    """
    pairs = []
    label_dir = config.TRAIN_LABEL_DIR
    
    # Build a set of available label stems for fast lookup
    label_stems = {}
    for lbl_path in label_dir.glob("*.png"):
        label_stems[lbl_path.stem] = lbl_path
    
    # Scan all image directories
    for img_dir in config.TRAIN_IMG_DIRS:
        if not img_dir.exists():
            print(f"[Warning] Image dir not found: {img_dir}")
            continue
        
        # Images can be in subdirectories
        for img_path in img_dir.rglob("*.jpg"):
            stem = img_path.stem
            if stem in label_stems:
                pairs.append((img_path, label_stems[stem]))
    
    print(f"[Dataset] Found {len(pairs)} image-label pairs")
    return pairs


def get_test_images() -> List[Path]:
    """Get all test image paths."""
    test_dir = config.TEST_IMG_DIR
    images = sorted(test_dir.glob("*.png"))
    print(f"[Dataset] Found {len(images)} test images")
    return images


def color_label_to_binary(label_rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB color label to binary mask for built-up areas.
    
    Built-up = (255, 0, 0) → 1
    Everything else → 0
    
    Args:
        label_rgb: (H, W, 3) uint8 array
    
    Returns:
        (H, W) uint8 array with values 0 or 1
    """
    r, g, b = config.BUILDUP_COLOR
    mask = (
        (label_rgb[:, :, 0] == r) &
        (label_rgb[:, :, 1] == g) &
        (label_rgb[:, :, 2] == b)
    ).astype(np.uint8)
    return mask


def get_train_augmentation():
    """Training augmentation pipeline."""
    if not HAS_ALBUMENTATIONS:
        return None
    
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.1, rotate_limit=15,
            border_mode=0, p=0.3
        ),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1.0),
        ], p=0.3),
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
        ToTensorV2(),
    ])


def get_val_augmentation():
    """Validation/test augmentation (normalize only)."""
    if not HAS_ALBUMENTATIONS:
        return None
    
    return A.Compose([
        ToTensorV2(),
    ])


def _normalize_per_image(image: torch.Tensor) -> torch.Tensor:
    """Normalize a tensor image per-image (Z-score).
    
    Handles uint8 (0-255) and float32 (0-1) inputs consistently.
    Always returns a float32 tensor.
    """
    # Ensure float32
    if image.dtype != torch.float32:
        image = image.float()
    # Scale to 0-1 if still in 0-255 range
    if image.max() > 1.0:
        image = image / 255.0
    # Z-score normalization per channel
    mean = image.mean(dim=(1, 2), keepdim=True)
    std = image.std(dim=(1, 2), keepdim=True) + 1e-6
    return (image - mean) / std


class GIDSegmentationDataset(Dataset):
    """
    PyTorch Dataset for GID segmentation training/validation.
    """
    
    def __init__(
        self,
        pairs: List[Tuple[Path, Path]],
        transform=None,
    ):
        self.pairs = pairs
        self.transform = transform
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img_path, lbl_path = self.pairs[idx]
        
        # Load image (JPG, RGB)
        image = np.array(Image.open(img_path).convert("RGB"))
        
        # Load label (PNG, RGB color mask) -> binary
        label_rgb = np.array(Image.open(lbl_path).convert("RGB"))
        mask = color_label_to_binary(label_rgb)
        
        # Apply augmentation
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]        # (C, H, W) tensor (uint8 from ToTensorV2)
            mask = augmented["mask"]           # (H, W) tensor
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask)

        # Per-image normalization
        if config.USE_PER_IMAGE_NORM:
            image = _normalize_per_image(image)
        
        # Ensure mask is float with channel dim: (1, H, W)
        mask = mask.unsqueeze(0).float()
        
        return image, mask


class GIDTestDataset(Dataset):
    """
    PyTorch Dataset for GID test inference.
    """
    
    def __init__(self, image_paths: List[Path], transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        
        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Per-image normalization (same as training)
        if config.USE_PER_IMAGE_NORM:
            image = _normalize_per_image(image)
        
        return image, img_path.name
