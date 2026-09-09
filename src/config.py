import os 
from pathlib import Path 
import torch 



PROJECT_NAME = "multimodal-brain-tumor-diagnostics"

#1. environment & path management : 
IS_KAGGLE = os.path.exists("/kaggle/input")


if IS_KAGGLE : 
    DATA_DIR = Path("/kaggle/input/datasets/awsaf49/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData")
    OUTPUT_DIR = Path("/kaggle/working/outputs")
else : 
    DATA_DIR = Path('./data/raw')
    OUTPUT_DIR = Path('./outputs')
    
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


#2.hardware And running : 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#3. Model & Data_dimensions : 
IN_CHANNELS = 1
OUT_CHANNELS = 1
IMAGE_SIZE = (240, 240)  # Assuming all images are resized to 240x240


#4. Training Hyperparameters :
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 20
VAL_SPLIT = 0.15  # 15% validation split at patient level
RANDOM_SEED = 42  # For reproducibility
