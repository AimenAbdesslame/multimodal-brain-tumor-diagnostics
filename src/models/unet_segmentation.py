## the u-net model from the paper "U-Net: Convolutional Networks for Biomedical Image Segmentation" by Olaf Ronneberger, Philipp Fischer, and Thomas Brox (2015)

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Convolution 3x3 -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    """Downscaling with MaxPool2d then DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upscaling with ConvTranspose2d, concatenating skip connection, then DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle spatial dimension matching for skip connection concatenation
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Full 5-level U-Net Architecture (Ronneberger et al., 2015)
    Progression: 64 -> 128 -> 256 -> 512 -> 1024 (Bottleneck) -> 512 -> 256 -> 128 -> 64
    """
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder (Downsampling path)
        self.enc1 = DoubleConv(in_channels, 64)
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)

        # Bottleneck (Target 1: Deepest layer at 1024 channels)
        self.bottleneck = DownBlock(512, 1024)

        # Decoder (Upsampling path)
        self.up1 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        
        # Final Decoder Block (Target 2: Restored spatial resolution at 64 channels)
        self.up4 = UpBlock(128, 64)

        # Final 1x1 Convolution to map features to output mask channels
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)

        b = self.bottleneck(e4)

        d1 = self.up1(b, e4)
        d2 = self.up2(d1, e3)
        d3 = self.up3(d2, e2)
        d4 = self.up4(d3, e1)

        logits = self.outc(d4)
        return logits