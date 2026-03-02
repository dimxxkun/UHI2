"""
U-Net model with pretrained encoder for binary segmentation.
"""
import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except ImportError:
    HAS_SMP = False

from . import config


def build_model():
    """
    Build U-Net with pretrained ResNet34 encoder.
    
    Returns:
        nn.Module: segmentation model
    """
    if not HAS_SMP:
        raise ImportError(
            "segmentation-models-pytorch is required. "
            "Install with: pip install segmentation-models-pytorch"
        )
    
    model = smp.Unet(
        encoder_name=config.ENCODER_NAME,
        encoder_weights=config.ENCODER_WEIGHTS,
        in_channels=3,
        classes=1,                         # Binary segmentation
        activation=None,                   # We use BCEWithLogitsLoss
    )
    return model


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        # Flatten
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """Combined BCE + Dice loss."""
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    
    def forward(self, logits, targets):
        return (
            self.bce_weight * self.bce(logits, targets) +
            self.dice_weight * self.dice(logits, targets)
        )


def compute_metrics(logits, targets, threshold=0.5):
    """
    Compute IoU and Dice score.
    
    Args:
        logits: (B, 1, H, W) raw model output
        targets: (B, 1, H, W) binary ground truth
        threshold: binarization threshold
    
    Returns:
        dict with 'iou' and 'dice' values
    """
    with torch.no_grad():
        preds = (torch.sigmoid(logits) > threshold).float()
        
        # Flatten spatial dims
        preds_flat = preds.view(preds.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        
        intersection = (preds_flat * targets_flat).sum(dim=1)
        union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection
        
        # IoU
        iou = (intersection + 1e-6) / (union + 1e-6)
        
        # Dice
        dice = (2.0 * intersection + 1e-6) / (
            preds_flat.sum(dim=1) + targets_flat.sum(dim=1) + 1e-6
        )
        
        return {
            "iou": iou.mean().item(),
            "dice": dice.mean().item(),
        }
