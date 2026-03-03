"""
Structural losses: SSIM for 2D and gradient consistency for 3D.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) loss for 2D images.
    Returns 1 - SSIM (so it can be minimized).
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

        # Create 1D Gaussian kernel
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()

        # Create 2D Gaussian kernel
        kernel_2d = gauss.unsqueeze(1) * gauss.unsqueeze(0)
        self.register_buffer(
            "kernel",
            kernel_2d.unsqueeze(0).unsqueeze(0),  # (1, 1, K, K)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) predicted image.
            y: (B, 1, H, W) target image.

        Returns:
            Scalar SSIM loss (1 - SSIM).
        """
        kernel = self.kernel.to(x.device)
        pad = self.window_size // 2

        mu_x = F.conv2d(x, kernel, padding=pad)
        mu_y = F.conv2d(y, kernel, padding=pad)

        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.conv2d(x ** 2, kernel, padding=pad) - mu_x_sq
        sigma_y_sq = F.conv2d(y ** 2, kernel, padding=pad) - mu_y_sq
        sigma_xy = F.conv2d(x * y, kernel, padding=pad) - mu_xy

        ssim_map = ((2 * mu_xy + self.C1) * (2 * sigma_xy + self.C2)) / (
            (mu_x_sq + mu_y_sq + self.C1) * (sigma_x_sq + sigma_y_sq + self.C2)
        )

        return 1.0 - ssim_map.mean()


class GradientConsistencyLoss(nn.Module):
    """
    3D gradient consistency loss for CBCT volumes.
    Encourages similar spatial gradients between real and generated volumes.
    """

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def _compute_gradients(self, vol: torch.Tensor) -> tuple:
        """Compute spatial gradients along D, H, W axes."""
        grad_d = vol[:, :, 1:, :, :] - vol[:, :, :-1, :, :]
        grad_h = vol[:, :, :, 1:, :] - vol[:, :, :, :-1, :]
        grad_w = vol[:, :, :, :, 1:] - vol[:, :, :, :, :-1]
        return grad_d, grad_h, grad_w

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 1, D, H, W) predicted volume.
            target: (B, 1, D, H, W) target volume.

        Returns:
            Scalar gradient consistency loss.
        """
        pred_grads = self._compute_gradients(pred)
        target_grads = self._compute_gradients(target)

        loss = sum(
            self.l1(pg, tg) for pg, tg in zip(pred_grads, target_grads)
        )
        return loss / 3.0


class StructuralLoss(nn.Module):
    """
    Combined structural loss:
        - SSIM for 2D PX images
        - Gradient consistency for 3D CBCT volumes
    """

    def __init__(self):
        super().__init__()
        self.ssim_loss = SSIMLoss()
        self.grad_loss = GradientConsistencyLoss()

    def px_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """SSIM loss for PX images."""
        return self.ssim_loss(pred, target)

    def cbct_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Gradient consistency loss for CBCT volumes."""
        return self.grad_loss(pred, target)
