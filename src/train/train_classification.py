from pathlib import Path
import sys
import torch
import wandb
import src.config as cfg
import glob
import random

from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.tumor_dataset import BraTS2DDataset
from src.models.classification_head import UNetClassificationWrapper
from src.models.unet_segmentation import UNet

from sklearn.metrics import roc_auc_score 


# 1. Dynamically append project root to_s python path
FILE_DIR = Path(__file__).resolve().parent
SRC_DIR = FILE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# 2. Central project imports
import src.config as cfg
from src.models.classification_head import UNetClassificationWrapper

# 3. Reproducibility & Device configuration
torch.manual_seed(cfg.RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.RANDOM_SEED)

# 4. Initialize W&B Experiment Tracking
wandb.init(
    project=cfg.WANDB_PROJECT,
    name=cfg.WANDB_CLS_RUN_NAME,
    config={
        "architecture": "UNet_Bottleneck_MLP",
        "task_type": "Classification_Single",
        "batch_size": cfg.CLS_BATCH_SIZE,
        "learning_rate": cfg.CLS_LR,
        "epochs": cfg.CLS_EPOCHS,
        "in_channels": cfg.CLS_IN_CHANNELS,
        "dropout": cfg.CLS_DROPOUT,
        "weight_decay": cfg.CLS_WEIGHT_DECAY,
        "seed": cfg.RANDOM_SEED,
        "device": str(cfg.DEVICE),
    },
)

print(
    f" Block 1 Success: W&B initialized for '{cfg.WANDB_CLS_RUN_NAME}' on {cfg.DEVICE}"
)


# Find all patient preprocessed folders inside OUTPUT_DIR
patient_dirs = sorted([p for p in cfg.OUTPUT_DIR.iterdir() if p.is_dir()])

# Deterministic Patient-Level Split (prevents slice leakage)
random.seed(cfg.RANDOM_SEED)
random.shuffle(patient_dirs)

val_count = max(1, int(len(patient_dirs) * cfg.VAL_SPLIT))
train_patient_dirs = patient_dirs[val_count:]
val_patient_dirs = patient_dirs[:val_count]

# Collect all .npz slice paths for train and validation sets
train_slice_paths = sorted(
    [str(p) for pdir in train_patient_dirs for p in pdir.glob("*.npz")]
)
val_slice_paths = sorted(
    [str(p) for pdir in val_patient_dirs for p in pdir.glob("*.npz")]
)

train_ds = BraTS2DDataset(data_source=train_slice_paths)
val_ds = BraTS2DDataset(data_source=val_slice_paths)

train_loader = DataLoader(
    train_ds, batch_size=cfg.CLS_BATCH_SIZE, shuffle=True, drop_last=False
)
val_loader = DataLoader(
    val_ds, batch_size=cfg.CLS_BATCH_SIZE, shuffle=False, drop_last=False
)


# ==========================================
# 2. MODEL WRAPPING & BOTTLENECK HOOK
# ==========================================
base_unet = UNet(in_channels=cfg.IN_CHANNELS, out_channels=cfg.OUT_CHANNELS)


# Dynamically locate bottleneck layer inside base U-Net
bottleneck_layer = getattr(
    base_unet,
    "bottleneck",
    getattr(base_unet, "center", list(base_unet.children())[2]),
)

# Wrap base U-Net with MLP classification head hook
model = UNetClassificationWrapper(
    unet_model=base_unet,
    bottleneck_layer=bottleneck_layer,
    in_channels=cfg.CLS_IN_CHANNELS,
).to(cfg.DEVICE)
# Optional: Explicitly freeze decoder parameters
for name, param in model.unet.named_parameters():
    if "up" in name or "outc" in name:  # Decoder layer naming conventions
        param.requires_grad = False

# ==========================================
# 3. LOSS & OPTIMIZER SETUP (END-TO-END)
# ==========================================
criterion = nn.BCEWithLogitsLoss()

# Pass model.parameters() to optimize BOTH Encoder and MLP Head together
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=cfg.CLS_LR,
    weight_decay=cfg.CLS_WEIGHT_DECAY,
)

trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)

print(
    f" Block 2 Success: Split {len(patient_dirs)} patients into "
    f"{len(train_patient_dirs)} train ({len(train_slice_paths)} slices) and "
    f"{len(val_patient_dirs)} val ({len(val_slice_paths)} slices)."
)
print(f" Block 3 Success: Model initialized with {trainable_params:,} trainable parameters.")
print(f"End-to-End Training Enabled: Encoder + MLP Head training together.")
print(f"   Total Trainable Parameters: {trainable_params:,}")



# 1. EPOCH STEP FUNCTIONS
# ==========================================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Executes a single training epoch over Encoder + Classification Head."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, _, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device):
    """Evaluates the full model on the validation set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_targets = []
    all_probs = []

    for images, _, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = criterion(logits, targets)

        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        correct += (preds == targets).sum().item()
        total += targets.size(0)

        all_targets.extend(targets.cpu().numpy().flatten())
        all_probs.extend(probs.cpu().numpy().flatten())

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    # Compute ROC-AUC safely
    try:
        epoch_auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_acc, epoch_auc


# ==========================================
# 2. MAIN TRAINING LOOP & CHECKPOINTING
# ==========================================
if __name__ == "__main__":
    best_val_auc = 0.0

    print(
        f"\n Starting End-to-End Classification Training ({cfg.WANDB_CLS_RUN_NAME}) for {cfg.CLS_EPOCHS} Epochs...\n"
    )

    for epoch in range(1, cfg.CLS_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, cfg.DEVICE
        )
        val_loss, val_acc, val_auc = validate_one_epoch(
            model, val_loader, criterion, cfg.DEVICE
        )

        # Log metrics to Weights & Biases
        wandb.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "val/auc": val_auc,
            }
        )

        print(
            f"Epoch [{epoch:02d}/{cfg.CLS_EPOCHS:02d}] | "
            f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - Val AUC: {val_auc:.4f}"
        )

        # Save best model checkpoint (Saves both UNet Encoder and Head weights)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "unet_state_dict": model.unet.state_dict(),
                    "mlp_head_state_dict": model.mlp_head.state_dict(),
                    "best_val_auc": best_val_auc,
                },
                cfg.CLS_CHECKPOINT_PATH,
            )
            print(
                f"   💾 Checkpoint saved -> Best Val AUC: {best_val_auc:.4f} ({cfg.CLS_CHECKPOINT_PATH.name})"
            )

    # Detach hook and close W&B logging
    model.remove_hook()
    wandb.finish()

    print(
        f"\n Classification Baseline B1 Complete! Best Validation ROC-AUC: {best_val_auc:.4f}"
    )




