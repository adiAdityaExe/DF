"""
Cycle consistency and latent consistency losses.
"""

import torch
import torch.nn as nn


class CycleLoss(nn.Module):
    """
    Cycle consistency loss: ensures round-trip translation recovers the original.

    L_cycle = ||x - G(G(x))|| (L1 loss)
    """

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, real: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            real: Original input (B, ...).
            reconstructed: Round-trip reconstructed input (B, ...).

        Returns:
            Scalar cycle consistency loss.
        """
        return self.l1(reconstructed, real)


class LatentConsistencyLoss(nn.Module):
    """
    Latent consistency loss: enforces that the latent codes from both
    encoders are aligned in the Unified Latent Anatomical Space.

    L_latent = ||z_px - z_cbct||_2
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1: (B, latent_dim) latent from one modality.
            z2: (B, latent_dim) latent from another modality.

        Returns:
            Scalar latent consistency loss.
        """
        return self.mse(z1, z2)
