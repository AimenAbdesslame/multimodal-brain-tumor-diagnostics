# Block 1: Imports, Path Resolution & Seed Setup
import glob
import os
from pathlib import Path
import random
import sys

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

# Dynamic path resolution to handle imports across environments
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd()
    if PROJECT_ROOT.name in ["train", "src"]:
        PROJECT_ROOT = (
            PROJECT_ROOT.parent
            if PROJECT_ROOT.name == "src"
            else PROJECT_ROOT.parent.parent
        )

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.config as cfg
from src.data.tumor_dataset import BraTS2DDataset
from src.losses.multitask_loss import MultiTaskLoss
from src.models.mtl_model import UNetMultiTaskModel
from src.models.unet_segmentation import UNet

# Set seeds for reproducibility
random.seed(cfg.RANDOM_SEED)
torch.manual_seed(cfg.RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.RANDOM_SEED)


# Block 2: Metrics, Data Helpers & Early Stopping
def compute_dice_score(preds, targets, smooth=1e-6):
    preds = (torch.sigmoid(preds) > 0.5).float()
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean()


def get_slice_paths(patient_dirs):
    slice_paths = []
    for p_dir in patient_dirs:
        p_dir_str = str(p_dir)
        if os.path.isdir(p_dir_str):
            slices = glob.glob(os.path.join(p_dir_str, "*.npz"))
        else:
            slices = glob.glob(f"{p_dir_str}*.npz")
        slice_paths.extend(slices)
    return sorted(slice_paths)


class EarlyStopping:

    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_auc):
        if self.best_score is None:
            self.best_score = val_auc
        elif val_auc > self.best_score + self.min_delta:
            self.best_score = val_auc
            self.counter = 0
        else:
            self.counter += 1
            print(f"   EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


print(f"Block 1 Loaded successfully! Target Device: {cfg.DEVICE}")


if __name__ == "__main__":
    wandb.init(
        project=cfg.PROJECT_NAME,
        config={
            "learning_rate": cfg.LEARNING_RATE,
            "batch_size": cfg.BATCH_SIZE,
            "epochs": cfg.EPOCHS,
            "device": str(cfg.DEVICE),
        },
    )

    # Patient directory splitting
    search_dir = cfg.OUTPUT_DIR
    patient_dirs = sorted(
        glob.glob(os.path.join(search_dir, "BraTS20_Training_*"))
    )

    if not patient_dirs:
        patient_dirs = sorted(
            glob.glob("/kaggle/working/**/BraTS20_Training_*", recursive=True)
        )

    if not patient_dirs:
        all_npz = glob.glob(
            os.path.join(search_dir, "**/*.npz"), recursive=True
        ) or glob.glob("/kaggle/working/**/*.npz", recursive=True)
        if not all_npz:
            raise FileNotFoundError(
                f"No .npz files found in {search_dir} or /kaggle/working/"
            )
        patient_dirs = sorted(list(set(os.path.dirname(p) for p in all_npz)))

    print(
        f"[Data Setup] Discovered {len(patient_dirs)} patient folders for splitting."
    )
    train_data, val_data = train_test_split(
        patient_dirs, test_size=cfg.VAL_SPLIT, random_state=cfg.RANDOM_SEED
    )

    train_slice_paths = get_slice_paths(train_data)
    val_slice_paths = get_slice_paths(val_data)
    print(
        f"[Data Setup] Train Slices: {len(train_slice_paths)} | Val Slices: {len(val_slice_paths)}"
    )

    train_dataset = BraTS2DDataset(train_slice_paths)
    val_dataset = BraTS2DDataset(val_slice_paths)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Block 4: Model, Loss Function and Optimizer Instantiation
    base_unet = UNet(
        in_channels=cfg.IN_CHANNELS, out_channels=cfg.OUT_CHANNELS
    ).to(cfg.DEVICE)

    if hasattr(base_unet, "bottleneck"):
        bottleneck_layer = base_unet.bottleneck
    elif hasattr(base_unet, "center"):
        bottleneck_layer = base_unet.center
    else:
        bottleneck_layer = list(base_unet.children())[2]

    model = UNetMultiTaskModel(
        unet_model=base_unet,
        bottleneck_layer=bottleneck_layer,
        in_channels=512,
        hidden_channels=128,
        dropout=0.4,
        num_classes=1,
    ).to(cfg.DEVICE)

    loss_fn = MultiTaskLoss(
        uncertainty_weighting=True, alpha=0.25, gamma=2.0
    ).to(cfg.DEVICE)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=cfg.LEARNING_RATE,
        weight_decay=getattr(cfg, "WEIGHT_DECAY", 1e-4),
    )

    # Tracking & Checkpoint setup
    best_val_auc = 0.0
    early_stopper = EarlyStopping(patience=5, min_delta=0.001)

    ckpt_dir = getattr(cfg, "CHECKPOINT_DIR", PROJECT_ROOT / "checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best_model_mtl.pth"

    # Main Epoch Training Loop
    print(f"\nStarting Multi-Task Training on {cfg.DEVICE}...")

    for epoch in range(1, cfg.EPOCHS + 1):

        # --- A. TRAINING PHASE ---
        model.train()
        loss_fn.train()

        train_total_loss = 0.0
        train_seg_loss = 0.0
        train_cls_loss = 0.0
        train_samples = 0

        for images, masks, targets in train_loader:
            images = images.to(cfg.DEVICE)
            masks = masks.to(cfg.DEVICE).float()
            targets = targets.to(cfg.DEVICE).float()

            if targets.ndim == 1:
                targets = targets.unsqueeze(1)
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)

            optimizer.zero_grad()
            seg_logits, cls_logits = model(images)

            total_loss, seg_loss, cls_loss = loss_fn(
                seg_logits, masks, cls_logits, targets
            )
            total_loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            train_total_loss += total_loss.item() * batch_size
            train_seg_loss += seg_loss.item() * batch_size
            train_cls_loss += cls_loss.item() * batch_size
            train_samples += batch_size

        avg_t_loss = train_total_loss / train_samples
        avg_t_seg_loss = train_seg_loss / train_samples
        avg_t_cls_loss = train_cls_loss / train_samples

        # --- B. VALIDATION PHASE ---
        model.eval()
        loss_fn.eval()

        val_total_loss = 0.0
        val_seg_loss = 0.0
        val_cls_loss = 0.0
        val_samples = 0

        all_val_targets = []
        all_val_probs = []
        val_dice_scores = []

        with torch.no_grad():
            for images, masks, targets in val_loader:
                images = images.to(cfg.DEVICE)
                masks = masks.to(cfg.DEVICE).float()
                targets = targets.to(cfg.DEVICE).float()

                if targets.ndim == 1:
                    targets = targets.unsqueeze(1)
                if masks.ndim == 3:
                    masks = masks.unsqueeze(1)

                seg_logits, cls_logits = model(images)
                total_loss, seg_loss, cls_loss = loss_fn(
                    seg_logits, masks, cls_logits, targets
                )

                batch_size = images.size(0)
                val_total_loss += total_loss.item() * batch_size
                val_seg_loss += seg_loss.item() * batch_size
                val_cls_loss += cls_loss.item() * batch_size
                val_samples += batch_size

                cls_probs = torch.sigmoid(cls_logits)
                all_val_targets.extend(targets.cpu().numpy().flatten())
                all_val_probs.extend(cls_probs.cpu().numpy().flatten())
                val_dice_scores.append(compute_dice_score(seg_logits, masks))

        avg_v_loss = val_total_loss / val_samples
        avg_v_seg_loss = val_seg_loss / val_samples
        avg_v_cls_loss = val_cls_loss / val_samples
        mean_v_dice = (
            sum(val_dice_scores) / len(val_dice_scores)
            if val_dice_scores
            else 0.0
        )

        try:
            v_auc = roc_auc_score(all_val_targets, all_val_probs)
        except ValueError:
            v_auc = 0.5

        if hasattr(loss_fn, "log_vars"):
            s_seg = loss_fn.log_vars[0].item()
            s_cls = loss_fn.log_vars[1].item()
            w_seg = 0.5 * torch.exp(-loss_fn.log_vars[0]).item()
            w_cls = 0.5 * torch.exp(-loss_fn.log_vars[1]).item()
        else:
            s_seg, s_cls, w_seg, w_cls = 0.0, 0.0, 1.0, 1.0

        # --- C. W&B LOGGING ---
        wandb.log(
            {
                "epoch": epoch,
                "train/total_loss": avg_t_loss,
                "train/seg_loss": avg_t_seg_loss,
                "train/cls_loss": avg_t_cls_loss,
                "val/total_loss": avg_v_loss,
                "val/seg_loss": avg_v_seg_loss,
                "val/cls_loss": avg_v_cls_loss,
                "val/auc": v_auc,
                "val/dice": mean_v_dice,
                "weights/seg_weight": w_seg,
                "weights/cls_weight": w_cls,
                "log_vars/s_seg": s_seg,
                "log_vars/s_cls": s_cls,
            }
        )

        print(
            f"Epoch [{epoch:02d}/{cfg.EPOCHS:02d}] | "
            f"Train Loss: {avg_t_loss:.4f} (Seg: {avg_t_seg_loss:.4f}, Cls: {avg_t_cls_loss:.4f}) | "
            f"Val Loss: {avg_v_loss:.4f} | Val AUC: {v_auc:.4f} | Val Dice: {mean_v_dice:.4f} | "
            f"Task Weights [Seg: {w_seg:.3f}, Cls: {w_cls:.3f}]"
        )

        # --- D. CHECKPOINTING & EARLY STOPPING ---
        if v_auc > best_val_auc:
            best_val_auc = v_auc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "loss_state_dict": loss_fn.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_auc": best_val_auc,
                    "val_dice": mean_v_dice,
                },
                ckpt_path,
            )
            print(
                f"   Checkpoint saved! New Best Val AUC: {best_val_auc:.4f} ({ckpt_path.name})"
            )

        if early_stopper(v_auc):
            print(f"\nEarly stopping triggered at Epoch {epoch}!")
            break

    model.remove_hook()
    wandb.finish()
    print(
        f"\nMulti-Task Training Completed! Peak Validation ROC-AUC: {best_val_auc:.4f}"
    )