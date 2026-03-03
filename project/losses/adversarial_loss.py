"""
LSGAN adversarial loss for stable training.
"""

import torch
import torch.nn as nn


class AdversarialLoss(nn.Module):
    """
    Least-Squares GAN (LSGAN) adversarial loss.

    D loss: 0.5 * E[(D(real) - 1)^2] + 0.5 * E[D(fake)^2]
    G loss: 0.5 * E[(D(fake) - 1)^2]
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def discriminator_loss(
        self, pred_real: torch.Tensor, pred_fake: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred_real: Discriminator output on real data.
            pred_fake: Discriminator output on fake (generated) data.

        Returns:
            Scalar discriminator loss.
        """
        real_target = torch.ones_like(pred_real)
        fake_target = torch.zeros_like(pred_fake)
        loss_real = self.mse(pred_real, real_target)
        loss_fake = self.mse(pred_fake, fake_target)
        return 0.5 * (loss_real + loss_fake)

    def generator_loss(self, pred_fake: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_fake: Discriminator output on fake (generated) data.

        Returns:
            Scalar generator loss (fool the discriminator).
        """
        real_target = torch.ones_like(pred_fake)
        return 0.5 * self.mse(pred_fake, real_target)
