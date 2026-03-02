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
    print(f"        Val IoU: {checkpoint['val_iou']:.4f}, " 
          f"Val Dice: {checkpoint['val_dice']:.4f}")
    
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
    
    # Create submission directory
    config.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run inference
    print(f"\n[Inference] Processing {len(test_images)} test images...")
    
    output_paths = []
    with torch.no_grad():
        for i, (image, filename) in enumerate(test_loader):
            image = image.to(device)
            
            # Forward pass
            logits = model(image)
            
            # Binarize: sigmoid → threshold 0.5
            pred = (torch.sigmoid(logits) > 0.5).cpu().numpy()
            
            # Extract single image prediction: (1, 1, H, W) → (H, W)
            mask = pred[0, 0].astype(np.uint8)
            
            # Save as PNG
            fname = filename[0]  # DataLoader wraps in list
            out_path = config.SUBMISSION_DIR / fname
            Image.fromarray(mask).save(out_path)
            output_paths.append(out_path)
            
            if (i + 1) % 50 == 0 or (i + 1) == len(test_images):
                print(f"  Processed {i + 1}/{len(test_images)}")
    
    # Verify outputs
    print(f"\n[Verify] Checking {len(output_paths)} output masks...")
    errors = []
    for p in output_paths:
        img = np.array(Image.open(p))
        if img.shape != (512, 512):
            errors.append(f"  {p.name}: wrong shape {img.shape}")
        unique = np.unique(img)
        if not all(v in [0, 1] for v in unique):
            errors.append(f"  {p.name}: unexpected values {unique}")
    
    if errors:
        print("[Warning] Issues found:")
        for e in errors:
            print(e)
    else:
        print("[Verify] All masks are 512×512 with values ∈ {0, 1} ✓")
    
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
