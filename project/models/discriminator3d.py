"""
3D CNN discriminator for CBCT volumes.
"""

import torch
import torch.nn as nn


class Discriminator3D(nn.Module):
    """
    3D PatchGAN discriminator for CBCT volumes.

    Produces a grid of real/fake predictions.
    """

    def __init__(self, in_channels: int = 1, ndf: int = 32):
        """
        Args:
            in_channels: Number of input channels.
            ndf: Base number of discriminator filters.
        """
        super().__init__()

        self.model = nn.Sequential(
            # (1, 128, 128, 128) → (32, 64, 64, 64)
            nn.Conv3d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # (32, 64, 64, 64) → (64, 32, 32, 32)
            nn.Conv3d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (64, 32, 32, 32) → (128, 16, 16, 16)
            nn.Conv3d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (128, 16, 16, 16) → (256, 15, 15, 15)
            nn.Conv3d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (256, 15, 15, 15) → (1, 14, 14, 14)
            nn.Conv3d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, D, H, W) volume.

        Returns:
            patch_pred: (B, 1, D', H', W') patch predictions.
        """
        return self.model(x)
