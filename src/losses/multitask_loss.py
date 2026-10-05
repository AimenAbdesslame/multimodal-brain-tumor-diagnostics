import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. DICE LOSS
# ==========================================
class DiceLoss(nn.Module):

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        cardinality = (probs_flat**2).sum(dim=1) + (targets_flat**2).sum(
            dim=1
        )

        dice_score = (2.0 * intersection + self.smooth) / (
            cardinality + self.smooth
        )
        return (1.0 - dice_score).mean()


# ==========================================
# 2. FOCAL LOSS
# ==========================================
class FocalLoss(nn.Module):

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        p_t = torch.exp(-bce_loss)
        focal_weight = (1.0 - p_t) ** self.gamma
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)

        focal_loss = alpha_t * focal_weight * bce_loss
        return focal_loss.mean()


# ==========================================
# 3. COMBINED SEGMENTATION LOSS
# ==========================================
class SegmentationLoss(nn.Module):

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return self.dice(logits, targets) + self.focal(logits, targets)


# ==========================================
# 4. MULTI-TASK LOSS WRAPPER
# ==========================================
class MultiTaskLoss(nn.Module):

    def __init__(
        self,
        uncertainty_weighting: bool = True,
        static_weights: dict = None,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.uncertainty_weighting = uncertainty_weighting
        self.static_weights = static_weights or {
            "segmentation": 1.0,
            "classification": 1.0,
        }

        # Passes alpha and gamma to SegmentationLoss
        self.seg_loss_fn = SegmentationLoss(alpha=alpha, gamma=gamma)
        self.cls_loss_fn = nn.BCEWithLogitsLoss()

        if self.uncertainty_weighting:
            self.log_vars = nn.Parameter(torch.zeros(2))

    def forward(
        self,
        seg_logits: torch.Tensor,
        seg_targets: torch.Tensor,
        cls_logits: torch.Tensor,
        cls_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        seg_loss = self.seg_loss_fn(seg_logits, seg_targets)
        cls_loss = self.cls_loss_fn(cls_logits, cls_targets)

        if self.uncertainty_weighting:
            seg_weight = 0.5 * torch.exp(-self.log_vars[0])
            cls_weight = 0.5 * torch.exp(-self.log_vars[1])

            total_loss = (
                (seg_weight * seg_loss + 0.5 * self.log_vars[0])
                + (cls_weight * cls_loss + 0.5 * self.log_vars[1])
            )
        else:
            total_loss = (
                self.static_weights["segmentation"] * seg_loss
                + self.static_weights["classification"] * cls_loss
            )

        return total_loss, seg_loss, cls_loss