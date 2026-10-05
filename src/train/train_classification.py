import glob

# ==========================================
# 1. PATH RESOLUTION & IMPORTS
# ==========================================
from pathlib import Path
import random
import sys

from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

FILE_DIR = Path(__file__).resolve().parent
SRC_DIR = FILE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.config as cfg
from src.data.tumor_dataset import BraTS2DDataset
from src.models.classification_head import UNetClassificationWrapper

try:
    from src.models.unet import UNet
except ModuleNotFoundError:
    from src.models.unet_segmentation import UNet

# Reproducibility seeds
torch.manual_seed(cfg.RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.RANDOM_SEED)


# ==========================================
# 2. EARLY STOPPING CLASS
# ==========================================
class EarlyStopping:
    """Stops training if validation score doesn't improve after a given patience."""

    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_auc: float) -> bool:
        if self.best_score is None:
            self.best_score = val_auc
        elif val_auc > self.best_score + self.min_delta:
            self.best_score = val_auc
            self.counter = 0  # Reset counter on improvement
        else:
            self.counter += 1
            print(
                f"    EarlyStopping counter: {self.counter}/{self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


# ==========================================
# 3. W&B & DATASET INITIALIZATION
# ==========================================
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

# Patient-Level Split (prevents slice leakage)
patient_dirs = sorted([p for p in cfg.OUTPUT_DIR.iterdir() if p.is_dir()])
random.seed(cfg.RANDOM_SEED)
random.shuffle(patient_dirs)

val_count = max(1, int(len(patient_dirs) * cfg.VAL_SPLIT))
train_patient_dirs = patient_dirs[val_count:]
val_patient_dirs = patient_dirs[:val_count]

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
# 4. MODEL SETUP & OPTIMIZER
# ==========================================
base_unet = UNet(in_channels=cfg.IN_CHANNELS, out_channels=cfg.OUT_CHANNELS)

bottleneck_layer = getattr(
    base_unet,
    "bottleneck",
    getattr(base_unet, "center", list(base_unet.children())[2]),
)

model = UNetClassificationWrapper(
    unet_model=base_unet,
    bottleneck_layer=bottleneck_layer,
    in_channels=cfg.CLS_IN_CHANNELS,
).to(cfg.DEVICE)

# Explicitly freeze decoder parameters
for name, param in model.unet.named_parameters():
    if "up" in name or "outc" in name:
        param.requires_grad = False

criterion = nn.BCEWithLogitsLoss()

# Optimize Encoder + MLP Head together
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
print(f"🔥 End-to-End Training Enabled: Encoder + MLP Head training together.")
print(f"   Total Trainable Parameters: {trainable_params:,}")


# ==========================================
# 5. EPOCH STEP FUNCTIONS
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

    try:
        epoch_auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_acc, epoch_auc


# ==========================================
# 6. MAIN TRAINING LOOP & CHECKPOINTING
# ==========================================
if __name__ == "__main__":
    best_val_auc = 0.0
    early_stopper = EarlyStopping(patience=5, min_delta=0.001)

    print(
        f"\n🚀 Starting End-to-End Classification Training ({cfg.WANDB_CLS_RUN_NAME}) for up to {cfg.CLS_EPOCHS} Epochs...\n"
    )

    for epoch in range(1, cfg.CLS_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, cfg.DEVICE
        )
        val_loss, val_acc, val_auc = validate_one_epoch(
            model, val_loader, criterion, cfg.DEVICE
        )

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

        # 1. Save checkpoint on improvement
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
                f"    Checkpoint saved -> Best Val AUC: {best_val_auc:.4f} ({cfg.CLS_CHECKPOINT_PATH.name})"
            )

        # 2. Early stopping evaluation
        if early_stopper(val_auc):
            print(
                f"\n Early stopping triggered at Epoch {epoch}! Validation ROC-AUC plateaued."
            )
            break

    # Detach hook and close W&B logging
    model.remove_hook()
    wandb.finish()

    print(
        f"\n Classification Baseline B1 Complete! Best Validation ROC-AUC: {best_val_auc:.4f}"
    )