# Model Weights

This directory contains the trained GazeSymCAT model weights.

## Download Model

The `best_model.pth` file (1.1GB) is **not included in the git repository** due to GitHub file size limits.

### Option 1: Download from Google Drive/Cloud
You need to download the model file separately and place it here.

```bash
# Place your downloaded best_model.pth file in this directory
# Expected path: weights/best_model.pth
# File size: ~1.1GB
```

### Option 2: Use Pre-trained Model
If you don't have the model file, the system will use random weights (for testing only).

## Verification

To verify the model file is correctly placed:

```bash
ls -lh weights/best_model.pth
# Should show: -rw-rw-r-- 1 user user 1.1G ... best_model.pth
```

## File Format
- **Format**: PyTorch checkpoint (.pth)
- **Size**: ~1.1GB
- **Contains**: Model state dict with trained weights for GazeSymCAT
- **Architecture**: d_model=512, num_blocks=2, with head pose conditioning
