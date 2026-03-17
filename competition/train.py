"""
Training script for Track 3: Built-Up Area Segmentation.

Usage:
    python -m competition.train
"""
import os
import sys
import time
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from . import config
from .dataset import (
    get_train_pairs,
    get_train_augmentation,
    get_val_augmentation,
    GIDSegmentationDataset,
)
from .model import build_model, BCEDiceLoss, compute_metrics


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """Train for one epoch with optional mixed precision. Returns avg loss and metrics."""
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_miou = 0.0
    total_kappa = 0.0
    num_batches = 0
    use_amp = scaler is not None
    
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)
        
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        metrics = compute_metrics(logits.detach().cpu(), masks.detach().cpu())
        
        total_loss += loss.item()
        total_iou += metrics["iou"]
        total_dice += metrics["dice"]
        total_miou += metrics["miou"]
        total_kappa += metrics["kappa"]
        num_batches += 1
    
    return {
        "loss": total_loss / max(num_batches, 1),
        "iou": total_iou / max(num_batches, 1),
        "dice": total_dice / max(num_batches, 1),
        "miou": total_miou / max(num_batches, 1),
        "kappa": total_kappa / max(num_batches, 1),
    }


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate. Returns average loss and metrics."""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_miou = 0.0
    total_kappa = 0.0
    num_batches = 0
    use_amp = device.type == 'cuda'
    
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)
        
        metrics = compute_metrics(logits.cpu(), masks.cpu())
        
        total_loss += loss.item()
        total_iou += metrics["iou"]
        total_dice += metrics["dice"]
        total_miou += metrics["miou"]
        total_kappa += metrics["kappa"]
        num_batches += 1
    
    return {
        "loss": total_loss / max(num_batches, 1),
        "iou": total_iou / max(num_batches, 1),
        "dice": total_dice / max(num_batches, 1),
        "miou": total_miou / max(num_batches, 1),
        "kappa": total_kappa / max(num_batches, 1),
    }


def main():
    print("=" * 60)
    print("Track 3: Built-Up Area Segmentation - Training")
    print("=" * 60)
    
    # Setup
    set_seed(config.SEED)
    device = config.get_device()
    
    # Create output dirs
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data pairs
    pairs = get_train_pairs()
    if len(pairs) == 0:
        print("[Error] No image-label pairs found! Check data paths in config.py")
        sys.exit(1)
    
    # Split into train/val
    num_val = int(len(pairs) * config.VAL_SPLIT)
    num_train = len(pairs) - num_val
    
    # Deterministic split
    generator = torch.Generator().manual_seed(config.SEED)
    train_pairs_idx, val_pairs_idx = random_split(
        range(len(pairs)), [num_train, num_val], generator=generator
    )
    
    train_pairs = [pairs[i] for i in train_pairs_idx]
    val_pairs = [pairs[i] for i in val_pairs_idx]
    
    print(f"[Data] Train: {len(train_pairs)}, Val: {len(val_pairs)}")
    
    # Datasets
    train_dataset = GIDSegmentationDataset(
        train_pairs, transform=get_train_augmentation()
    )
    val_dataset = GIDSegmentationDataset(
        val_pairs, transform=get_val_augmentation()
    )
    
    # DataLoaders
    use_cuda = device.type == 'cuda'
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=use_cuda,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=use_cuda,
    )
    
    # Model
    model = build_model()
    model = model.to(device)
    print(f"[Model] U-Net with {config.ENCODER_NAME} encoder loaded")
    
    # Loss & Optimizer
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.SCHEDULER_T_MAX
    )
    
    # Mixed precision scaler (CUDA only)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    # Training loop
    best_val_iou = 0.0
    best_epoch = 0
    
    amp_status = "AMP enabled" if scaler else "AMP disabled"
    print(f"\n[Training] {config.NUM_EPOCHS} epochs, batch_size={config.BATCH_SIZE}, {amp_status}")
    print("-" * 60)
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        t_start = time.time()
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Step scheduler
        scheduler.step()
        
        elapsed = time.time() - t_start
        lr = optimizer.param_groups[0]["lr"]
        
        print(
            f"Epoch {epoch:3d}/{config.NUM_EPOCHS} | "
            f"Train Loss: {train_metrics['loss']:.4f} IoU: {train_metrics['iou']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} IoU: {val_metrics['iou']:.4f} "
            f"Dice: {val_metrics['dice']:.4f} mIoU: {val_metrics['miou']:.4f} "
            f"Kappa: {val_metrics['kappa']:.4f} | "
            f"LR: {lr:.6f} | {elapsed:.1f}s"
        )
        
        # Save best model (based on mIoU)
        if val_metrics["miou"] > best_val_iou:
            best_val_iou = val_metrics["miou"]
            best_epoch = epoch
            ckpt_path = config.CHECKPOINT_DIR / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_iou": val_metrics["iou"],
                "val_dice": val_metrics["dice"],
                "val_miou": best_val_iou,
                "val_kappa": val_metrics["kappa"],
            }, ckpt_path)
            print(f"  -> Saved best model (mIoU: {best_val_iou:.4f})")
    
    print("-" * 60)
    print(f"[Done] Best Val mIoU: {best_val_iou:.4f} at epoch {best_epoch}")
    print(f"[Done] Checkpoint: {config.CHECKPOINT_DIR / 'best_model.pth'}")


if __name__ == "__main__":
    main()
