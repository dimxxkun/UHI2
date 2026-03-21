"""
Configuration for Track 3: Built-Up Area Segmentation
Optimized for NVIDIA CUDA GPU (RTX 5060)
"""
from pathlib import Path
from glob import glob
from pathlib import Path
import zipfile
# ============================================
# Paths
# ============================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "track 3 dataset - preliminary round" / "dataset-preliminary round"

TRAIN_IMG_DIRS = [
    DATA_ROOT / "Train" / f"GID-img-{i}" / f"{i}"
    for i in range(1, 5)
]
TRAIN_LABEL_DIR = DATA_ROOT / "Train" / "GID-label"


def ensure_unzipped_train_dirs():
    """Unzips any missing GID-img-* directories and the GID-label folder automatically."""
    train_root = DATA_ROOT / "Train"

    # --- Unzip GID-img-1 to GID-img-4 ---
    for i in range(1, 5):
        zip_path = train_root / f"GID-img-{i}.zip"
        dest_dir = train_root / f"GID-img-{i}"

        if not dest_dir.exists() and zip_path.exists():
            print(f"[Setup] Extracting {zip_path.name} → {dest_dir}")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(dest_dir)
        elif not dest_dir.exists():
            print(f"[Warning] Neither folder nor zip found for {dest_dir.name}")

    # --- Unzip GID-label.zip ---
    label_zip = train_root / "GID-label.zip"
    label_dir = train_root / "GID-label"

    if not label_dir.exists() and label_zip.exists():
        print(f"[Setup] Extracting {label_zip.name} → {label_dir}")
        with zipfile.ZipFile(label_zip, "r") as zf:
            zf.extractall(label_dir)
    elif not label_dir.exists():
        print("[Warning] Neither folder nor zip found for GID-label")

# Run once when config loads
ensure_unzipped_train_dirs()


total_imgs = sum(len(glob(str(d / "*.tif"))) for d in TRAIN_IMG_DIRS if d.exists())
print(f"[Dataset Check] Total training images found: {total_imgs}")


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
BATCH_SIZE = 24    # inreased from 8 to 24 
NUM_WORKERS = 8
PERSISTENT_WORKERS = True          # ← new (keeps workers alive)
PIN_MEMORY = True                  # already true, but make sure

VAL_SPLIT = 0.15
SEED = 42

# Model
ENCODER_NAME = "resnet50"
ENCODER_WEIGHTS = "imagenet"

# Optimizer
LEARNING_RATE = 1.5e-4       #for 1e-4, the model converges but with some fluctuations. 1.5e-4 seems to provide a smoother convergence.
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
