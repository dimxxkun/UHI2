"""
Inference script for Track 3: Built-Up Area Segmentation.

Loads the best trained model, runs inference on all test images,
and packages the binary masks into a submission ZIP file.

Usage:
    python -m competition.predict
"""
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from . import config
from .dataset import get_test_images, get_val_augmentation, GIDTestDataset
from .model import build_model


def main():
    print("=" * 60)
    print("Track 3: Built-Up Area Segmentation - Inference")
    print("=" * 60)
    
    device = config.get_device()
    
    # Load model
    ckpt_path = config.CHECKPOINT_DIR / "best_model.pth"
    if not ckpt_path.exists():
        print(f"[Error] No checkpoint found at {ckpt_path}")
        print("       Run training first: python -m competition.train")
        sys.exit(1)
    
    model = build_model()
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    print(f"[Model] Loaded checkpoint from epoch {checkpoint['epoch']}")
    val_iou = checkpoint.get('val_iou', 'N/A')
    val_dice = checkpoint.get('val_dice', 'N/A')
    val_miou = checkpoint.get('val_miou', 'N/A')
    val_kappa = checkpoint.get('val_kappa', 'N/A')
    print(f"        Val IoU: {val_iou if isinstance(val_iou, str) else f'{val_iou:.4f}'}, "
          f"Val Dice: {val_dice if isinstance(val_dice, str) else f'{val_dice:.4f}'}, "
          f"Val mIoU: {val_miou if isinstance(val_miou, str) else f'{val_miou:.4f}'}, "
          f"Val Kappa: {val_kappa if isinstance(val_kappa, str) else f'{val_kappa:.4f}'}")
    
    # Load test data
    test_images = get_test_images()
    if len(test_images) == 0:
        print("[Error] No test images found!")
        sys.exit(1)
    
    test_dataset = GIDTestDataset(test_images, transform=get_val_augmentation())
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(device.type == 'cuda'),
    )
    
    # Create directories
    config.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    config.VISUAL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run inference
    print(f"\n[Inference] Processing {len(test_images)} test images...")
    print(f"            Threshold: {config.PREDICT_THRESHOLD}")
    
    output_paths = []
    with torch.no_grad():
        for i, (image_tensor, filename) in enumerate(test_loader):
            # Keep original image for visualization
            orig_path = test_images[i]
            orig_img = Image.open(orig_path).convert("RGB")
            
            image_tensor = image_tensor.to(device)
            
            # Forward pass
            logits = model(image_tensor)
            
            # Binarize
            probs = torch.sigmoid(logits)
            mask_np = (probs > config.PREDICT_THRESHOLD).cpu().numpy()[0, 0].astype(np.uint8)
            
            # 1. Save submission mask (0/1 labels)
            fname = filename[0]
            out_path = config.SUBMISSION_DIR / fname
            Image.fromarray(mask_np).save(out_path)
            output_paths.append(out_path)
            
            # 2. Save visual preview (Red overlay)
            if mask_np.max() > 0:
                # Create red mask
                red_mask = np.zeros((512, 512, 3), dtype=np.uint8)
                red_mask[mask_np == 1] = [255, 0, 0]
                
                # Blend with original
                preview = Image.blend(orig_img, Image.fromarray(red_mask), alpha=0.4)
                preview.save(config.VISUAL_DIR / f"preview_{fname}")
            
            if (i + 1) % 50 == 0 or (i + 1) == len(test_images):
                print(f"  Processed {i + 1}/{len(test_images)}")
    
    # Verify outputs
    print(f"\n[Verify] Checking {len(output_paths)} output masks...")
    errors = []
    has_detections = 0
    for p in output_paths:
        img = np.array(Image.open(p))
        if img.shape != (512, 512):
            errors.append(f"  {p.name}: wrong shape {img.shape}")
        unique = np.unique(img)
        if not all(v in [0, 1] for v in unique):
            errors.append(f"  {p.name}: unexpected values {unique}")
        if 1 in unique:
            has_detections += 1
    
    if errors:
        print("[Warning] Issues found:")
        for e in errors:
            print(e)
    else:
        print(f"[Verify] All masks are 512x512 with values in {{0, 1}} [OK]")
        print(f"[Verify] Images with built-up detections: {has_detections}/{len(output_paths)}")
    
    if has_detections > 0:
        print(f"[Info] Visual previews saved to: {config.VISUAL_DIR}")
    
    # Package into ZIP
    zip_path = config.OUTPUT_DIR / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in output_paths:
            zf.write(p, p.name)
    
    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"\n[Done] Submission ZIP: {zip_path}")
    print(f"       Size: {zip_size_mb:.2f} MB")
    print(f"       Contains: {len(output_paths)} PNG masks")


if __name__ == "__main__":
    main()
