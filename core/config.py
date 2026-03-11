import os

DATA_ROOT      = '/kaggle/input/metagazeversiontwo'
TRAIN_DIR      = os.path.join(DATA_ROOT, '')
VAL_DIR        = os.path.join(DATA_ROOT, 'test/test')
CHECKPOINT_DIR = '/kaggle/working/last.pth'

D_MODEL                = 512
NUM_TRANSFORMER_BLOCKS = 2
NUM_HEADS              = 8
FFN_DIM                = 2048
DROPOUT                = 0.1

USE_HEAD_POSE = True

EPOCHS      = 29
BATCH_SIZE  = 24

BACKBONE_LR = 1e-5
TRANS_LR    = 1e-4
WEIGHT_DECAY = 0.05

GRAD_CLIP = 1.0

WARMUP_EPOCHS = 5
LR_MIN        = 1e-6

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
