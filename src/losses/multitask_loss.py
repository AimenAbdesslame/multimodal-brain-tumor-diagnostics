import torch 
import torch.nn as nn






#Block 1.1: DiceLoss class
class DiceLoss(nn.Module):
    def __init__(self, smooth: float =1e-6) : 
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits) 
        progs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersection = (probs * targets).sum(dim=1)
        cardinality = (probs ** 2).sum(dim=1) + (targets ** 2).sum(dim=1)
        dice_score = (2.0 * intersection  + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_score.mean()


#Block 1.2: FocalLoss class

class FocalLoss(nn.Module): 
    def __init__(self, alpha: float=0.25 , gamma:float=2.0) : 
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, logits:torch.Tensor , targets:torch.Tensor) -> torch.Tensor:
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.exp(-bce_loss)
        focal_weight = (1.0 - p_t) ** self.gamma
        loss = self.alpha * focal_weight * bce_loss
        return loss.mean()

class SegmentationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Total Loss = Dice Loss + Focal Loss
        return self.dice(logits, targets) + self.focal(logits, targets)

#Block 1.3: MultiTaskLoss wrapper class (combining both static and uncertainty weighting)
class MultiTaskLoss(nn.Module):
    """Combines Segmentation Loss (Dice+Focal) and Classification Loss (BCE)

    using either Homoscedastic Uncertainty Weighting (Kendall et al.) or Static Weights.
    """

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

        # Initialize loss functions ONCE here
        self.seg_loss_fn = SegmentationLoss(alpha=alpha, gamma=gamma)
        self.cls_loss_fn = nn.BCEWithLogitsLoss()

        if self.uncertainty_weighting:
            # Learnable log-variance parameters s_seg and s_cls
            # Initialized to zero -> exp(0) = 1 (equal starting weights)
            self.log_vars = nn.Parameter(torch.zeros(2))

    def forward(
        self,
        seg_logits: torch.Tensor,
        seg_targets: torch.Tensor,
        cls_logits: torch.Tensor,
        cls_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # 1. Compute individual task losses
        seg_loss = self.seg_loss_fn(seg_logits, seg_targets)
        cls_loss = self.cls_loss_fn(cls_logits, cls_targets)

        # 2. Combine task losses
        if self.uncertainty_weighting:
            # L_total = 0.5 * exp(-s_seg) * L_seg + 0.5 * exp(-s_cls) * L_cls + 0.5 * s_seg + 0.5 * s_cls
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