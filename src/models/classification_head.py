# Input(B, 1, 240, 240) => {Encoder} => {Bottleneck Feature Map } (B, C, H', W') => Hook Captured {GAP } (B, C, 1, 1)=> {Logit } (B, 1)

import torch 
import torch.nn as nn



import torch
import torch.nn as nn


class UNetClassificationWrapper(nn.Module):
    """Wraps a U-Net backbone, intercepts bottleneck feature maps via a forward hook,

    and passes them through an MLP classification head.
    """

    def __init__(
        self,
        unet_model: nn.Module,
        bottleneck_layer: nn.Module,
        in_channels: int = 512,
        hidden_channels: int = 128,
        dropout: float = 0.4,
        num_classes: int = 1,
    ):
        super().__init__()
        self.unet = unet_model
        self.bottleneck_features = None

        # MLP Head: Global Avg Pool -> Linear -> ReLU -> Dropout -> Linear
        self.mlp_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_channels, num_classes),
        )

        # Register forward hook on specified bottleneck layer
        self.hook_handle = bottleneck_layer.register_forward_hook(
            self.hook_fn
        )

    def hook_fn(self, module, input, output):
        """Callback to store bottleneck tensor during forward pass."""
        self.bottleneck_features = output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Run forward pass through UNet to trigger bottleneck hook
        _ = self.unet(x)

        if self.bottleneck_features is None:
            raise RuntimeError(
                "Bottleneck features were not captured during UNet forward pass."
            )

        # Compute logits through MLP head
        logits = self.mlp_head(self.bottleneck_features)

        # Clear cached feature reference to prevent memory leaks across iterations
        self.bottleneck_features = None

        return logits

    def remove_hook(self):
        """Cleanly unregister the forward hook handle."""
        if hasattr(self, "hook_handle") and self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
        