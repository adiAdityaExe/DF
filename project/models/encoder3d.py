"""
3D Encoder for CBCT volumes.
Custom 3D CNN that maps a volume to a latent vector Z.
"""

import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    """Conv3d → InstanceNorm → LeakyReLU block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CBCTEncoder(nn.Module):
    """
    Encodes a 3D CBCT volume into a latent vector of dimension `latent_dim`.

    Architecture:
        - Series of strided 3D convolutions
        - AdaptiveAvgPool3d to (1,1,1)
        - Linear projection to latent_dim
    """

    def __init__(self, latent_dim: int = 512, in_channels: int = 1):
        super().__init__()

        self.encoder = nn.Sequential(
            # (1, 128, 128, 128) → (32, 64, 64, 64)
            ConvBlock3D(in_channels, 32),
            # (32, 64, 64, 64) → (64, 32, 32, 32)
            ConvBlock3D(32, 64),
            # (64, 32, 32, 32) → (128, 16, 16, 16)
            ConvBlock3D(64, 128),
            # (128, 16, 16, 16) → (256, 8, 8, 8)
            ConvBlock3D(128, 256),
            # (256, 8, 8, 8) → (512, 4, 4, 4)
            ConvBlock3D(256, 512),
        )

        self.pool = nn.AdaptiveAvgPool3d(1)

        self.fc = nn.Sequential(
            nn.Linear(512, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, D, H, W) CBCT volume normalized to [0, 1].

        Returns:
            z: (B, latent_dim) latent vector.
        """
        x = self.encoder(x)
        x = self.pool(x)              # (B, 512, 1, 1, 1)
        x = x.view(x.size(0), -1)     # (B, 512)
        z = self.fc(x)
        return z
