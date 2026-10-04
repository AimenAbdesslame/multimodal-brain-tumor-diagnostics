import glob
import os
from pathlib import Path
import random
import sys
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

# ==========================================
# 1. PATH RESOLUTION & SETUP
# ==========================================
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

# Set reproducibility seeds
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
            self.counter = 0  # Reset counter on meaningful gain
        else:
            self.counter += 1
            print(
                f"    EarlyStopping counter: {self.counter}/{self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


# ==========================================
# 3. CHECKPOINT SEARCH & LOADING LOGIC
# ==========================================
def find_checkpoint(project_root: Path) -> Path:
    """Find the trained segmentation checkpoint in common project locations."""
    preferred_paths = [
        project_root / "outputs" / "checkpoints" / "best_model.zip",
        project_root / "outputs" / "checkpoints" / "best_model.pth",
        project_root / "models" / "best_model.pth",
        project_root / "best_model.pth",
    ]
    for candidate in preferred_paths:
        if candidate.is_file():
            return candidate

    candidates = sorted(
        path
        for pattern in ("*.pth", "*.pt", "*.zip")
        for path in project_root.rglob(pattern)
        if "data" not in path.parts
    )
    if not candidates:
        raise FileNotFoundError(
            f"No segmentation checkpoint found under {project_root}."
        )
    return candidates[0]


def load_state_dict(checkpoint_path: Path, device: torch.device):
    """Load a raw state dict or a training checkpoint containing one."""
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict) or not all(
        isinstance(key, str) for key in checkpoint
    ):
        raise TypeError(
            f"Checkpoint {checkpoint_path} does not contain a valid state dict."
        )

    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in checkpoint.items()
    }


# ==========================================
# 4. W&B & DATA INITIALIZATION
# ==========================================
wandb.init(
    project=cfg.WANDB_PROJECT,
    name="Cls_PretrainedEncoder_Baseline",
    config={
        "architecture": "UNet_Pretrained_Bottleneck_MLP",
        "task_type": "Classification_PretrainedEncoder",
        "batch_size": cfg.CLS_BATCH_SIZE,
        "learning_rate": cfg.CLS_LR,
        "epochs": cfg.CLS_EPOCHS,
        "in_channels": cfg.CLS_IN_CHANNELS,
        "dropout": cfg.CLS_DROPOUT,
        "weight_decay": cfg.CLS_WEIGHT_DECAY,
        "seed": cfg.RANDOM_SEED,
    },
)

# Patient-level splitting to prevent data leakage
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
    train_ds, batch_size=cfg.CLS_BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(val_ds, batch_size=cfg.CLS_BATCH_SIZE, shuffle=False)


# ==========================================
# 5. INSTANTIATE & LOAD PRE-TRAINED ENCODER
# ==========================================
base_unet = UNet(
    in_channels=cfg.IN_CHANNELS, out_channels=cfg.OUT_CHANNELS
).to(cfg.DEVICE)

ckpt_path = find_checkpoint(PROJECT_ROOT)
state_dict = load_state_dict(ckpt_path, cfg.DEVICE)
base_unet.load_state_dict(state_dict, strict=True)
print(
    f" Loaded pre-trained segmentation U-Net from: {ckpt_path.name} | device: {cfg.DEVICE}"
)

# Attach bottleneck hook and classification head
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

criterion = nn.BCEWithLogitsLoss()

# End-to-end optimization (Pre-trained Encoder + Classification Head)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=cfg.CLS_LR,
    weight_decay=cfg.CLS_WEIGHT_DECAY,
)


# ==========================================
# 6. EPOCH FUNCTIONS & MAIN LOOP
# ==========================================
def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, _, targets in dataloader:
        images, targets = images.to(cfg.DEVICE), targets.to(cfg.DEVICE)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits)
        correct += ((probs > 0.5).float() == targets).sum().item()
        total += targets.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_targets, all_probs = [], []
    for images, _, targets in dataloader:
        images, targets = images.to(cfg.DEVICE), targets.to(cfg.DEVICE)
        logits = model(images)
        loss = criterion(logits, targets)

        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits)
        correct += ((probs > 0.5).float() == targets).sum().item()
        total += targets.size(0)

        all_targets.extend(targets.cpu().numpy().flatten())
        all_probs.extend(probs.cpu().numpy().flatten())

    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = 0.5
    return running_loss / total, correct / total, auc


if __name__ == "__main__":
    best_val_auc = 0.0
    checkpoint_out_path = (
        cfg.CHECKPOINT_DIR / "best_model_cls_pretrained_encoder.pth"
    )
    early_stopper = EarlyStopping(patience=5, min_delta=0.001)

    print(
        f"\n Training Classification with Pre-trained Encoder (up to {cfg.CLS_EPOCHS} epochs)..."
    )

    for epoch in range(1, cfg.CLS_EPOCHS + 1):
        t_loss, t_acc = train_one_epoch(
            model, train_loader, criterion, optimizer
        )
        v_loss, v_acc, v_auc = validate_one_epoch(model, val_loader, criterion)

        wandb.log(
            {
                "epoch": epoch,
                "train/loss": t_loss,
                "train/acc": t_acc,
                "val/loss": v_loss,
                "val/acc": v_acc,
                "val/auc": v_auc,
            }
        )

        print(
            f"Epoch [{epoch:02d}/{cfg.CLS_EPOCHS:02d}] | "
            f"Train Loss: {t_loss:.4f} Acc: {t_acc:.4f} | "
            f"Val Loss: {v_loss:.4f} Acc: {v_acc:.4f} AUC: {v_auc:.4f}"
        )

        # 1. Save checkpoint on improvement
        if v_auc > best_val_auc:
            best_val_auc = v_auc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "unet_state_dict": model.unet.state_dict(),
                    "mlp_head_state_dict": model.mlp_head.state_dict(),
                    "best_val_auc": best_val_auc,
                },
                checkpoint_out_path,
            )
            print(
                f"    Saved Checkpoint -> Best Val AUC: {best_val_auc:.4f} ({checkpoint_out_path.name})"
            )

        # 2. Check Early Stopping
        if early_stopper(v_auc):
            print(
                f"\n Early stopping triggered at Epoch {epoch}! Validation ROC-AUC plateaued."
            )
            break

    model.remove_hook()
    wandb.finish()
    print(
        f"\n Completed! Best Pre-trained Encoder Val AUC: {best_val_auc:.4f}"
    )