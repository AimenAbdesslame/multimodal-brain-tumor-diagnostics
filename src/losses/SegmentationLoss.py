import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Step 1: Turn raw model predictions into probabilities (0.0 to 1.0)
        probs = torch.sigmoid(logits)
        
        # Step 2: Flatten 2D images into 1D vectors for easy calculation
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        
        # Step 3: Math formula for Dice
        intersection = (probs * targets).sum(dim=1)
        cardinality = (probs ** 2).sum(dim=1) + (targets ** 2).sum(dim=1)
        
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        
        # Step 4: Return 1 - average dice score
        return 1.0 - dice_score.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Step 1: Use PyTorch's built-in safe BCE calculation
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Step 2: Calculate p_t safely without getting numerical errors
        p_t = torch.exp(-bce_loss)
        
        # Step 3: Apply the focal weighting (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma
        
        # Step 4: Multiply everything together
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