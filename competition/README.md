# Track 3: Built-Up Area Segmentation

## Quick Setup (NVIDIA GPU)

### 1. Install Python dependencies
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install segmentation-models-pytorch albumentations Pillow tqdm
```

### 2. Folder structure needed
Make sure the folder looks like this:
```
UHI_LST/
├── competition/          ← the code (this folder)
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   └── requirements.txt
└── data/
    └── track 3 dataset - preliminary round/
        └── dataset-preliminary round/
            ├── Train/
            │   ├── GID-img-1/
            │   ├── GID-img-2/
            │   ├── GID-img-3/
            │   ├── GID-img-4/
            │   └── GID-label/
            └── Test/
                └── Images/
```

### 3. Train the model
```bash
cd UHI_LST
python -m competition.train
```
Training takes ~30-60 minutes on an RTX 5060. It will print progress per epoch.

### 4. Generate submission
```bash
python -m competition.predict
```
This creates `competition/output/submission.zip` — send this file back!
