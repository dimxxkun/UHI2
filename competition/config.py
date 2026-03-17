"""
Configuration for Track 3: Built-Up Area Segmentation
Optimized for NVIDIA CUDA GPU (RTX 5060)
"""
from pathlib import Path

# ============================================
# Paths
# ============================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "track 3 dataset - preliminary round" / "dataset-preliminary round"

TRAIN_IMG_DIRS = [
    DATA_ROOT / "Train" / f"GID-img-{i}"
    for i in range(1, 5)
]
TRAIN_LABEL_DIR = DATA_ROOT / "Train" / "GID-label"
TEST_IMG_DIR = DATA_ROOT / "Test" / "Images"

# Output
OUTPUT_DIR = PROJECT_ROOT / "competition" / "output"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
# Prediction
SUBMISSION_DIR = OUTPUT_DIR / "submission"
VISUAL_DIR = OUTPUT_DIR / "visual_preview"
PREDICT_THRESHOLD = 0.3

# ============================================
# Label Encoding (from readme.md)
# ============================================
BUILDUP_COLOR = (255, 0, 0)  # Built-up = RED → Track 3 target

# ============================================
# Training Hyperparameters
# ============================================
IMG_SIZE = 512
BATCH_SIZE = 8
NUM_WORKERS = 4
VAL_SPLIT = 0.15
SEED = 42

# Model
ENCODER_NAME = "resnet50"
ENCODER_WEIGHTS = "imagenet"

# Optimizer
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 30
USE_PER_IMAGE_NORM = True  # Enable robust normalization for satellite images

# Scheduler
SCHEDULER_T_MAX = NUM_EPOCHS

# ImageNet normalization
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ============================================
# Device Selection
# ============================================
def get_device():
    """Select best available device: CUDA > CPU"""
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Using CUDA: {torch.cuda.get_device_name(0)}")
        return device

    print("[Device] Using CPU (training will be slow)")
    return torch.device("cpu")
