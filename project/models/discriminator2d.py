"""
2D PatchGAN discriminator for PX images.
"""

import torch
import torch.nn as nn


class Discriminator2D(nn.Module):
    """
    PatchGAN discriminator for 2D panoramic X-ray images.

    Produces a grid of real/fake predictions (patch-level discrimination).
    """

    def __init__(self, in_channels: int = 1, ndf: int = 64):
        """
        Args:
            in_channels: Number of input channels (1 for grayscale).
            ndf: Base number of discriminator filters.
        """
        super().__init__()

        self.model = nn.Sequential(
            # (1, 256, 256) → (64, 128, 128)
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # (64, 128, 128) → (128, 64, 64)
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (128, 64, 64) → (256, 32, 32)
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (256, 32, 32) → (512, 31, 31)
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (512, 31, 31) → (1, 30, 30)
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) image.

        Returns:
            patch_pred: (B, 1, H', W') patch predictions.
        """
        return self.model(x)
