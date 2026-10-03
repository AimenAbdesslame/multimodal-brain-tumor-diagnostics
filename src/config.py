import os
from pathlib import Path
import torch

# ==========================================
# 1. PROJECT & PATHS MANAGEMENT
# ==========================================
PROJECT_NAME = "multimodal-brain-tumor-diagnostics"

# Automatically resolve Repository Root (assumes config.py is inside src/)
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

IS_KAGGLE = os.path.exists("/kaggle/input")

if IS_KAGGLE:
    DATA_DIR = Path(
        "/kaggle/input/datasets/awsaf49/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    )
    OUTPUT_DIR = Path("/kaggle/working/outputs")
else:
    DATA_DIR = PROJECT_ROOT / "data" / "raw"
    OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Central Checkpoint Directory
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. HARDWARE & DEVICE SELECTION
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 3. BASE MODEL & DATA DIMENSIONS
# ==========================================
IN_CHANNELS = 1
OUT_CHANNELS = 1
IMAGE_SIZE = (240, 240)


# ==========================================
# 4. GLOBAL TRAINING & REPRODUCIBILITY
# ==========================================
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 20
VAL_SPLIT = 0.15  # 15% validation split at patient level
RANDOM_SEED = 42  # For reproducibility


# ==========================================
# 5. CLASSIFICATION TASK CONFIG (Baseline B1)
# ==========================================
CLS_IN_CHANNELS = (
    512  # Bottleneck feature map channels for classification head
)
CLS_NUM_CLASSES = 1  # Binary classification (tumor present or not)
CLS_DROPOUT = 0.4  # Dropout rate for classification MLP head

# Training Hyperparameters
CLS_BATCH_SIZE = 16  # Batch size for classification training
CLS_LR = 1e-4  # Learning rate for classification head
CLS_EPOCHS = 25  # Total training epochs for baseline B1
CLS_WEIGHT_DECAY = 1e-4  # L2 regularization weight decay

# Logging & Checkpoint Settings
WANDB_PROJECT = PROJECT_NAME
WANDB_CLS_RUN_NAME = "Cls_Single_Baseline"

# Classification Model Checkpoint
CLS_CHECKPOINT_DIR = CHECKPOINT_DIR
CLS_CHECKPOINT_PATH = CLS_CHECKPOINT_DIR / "best_model_classification.pth"