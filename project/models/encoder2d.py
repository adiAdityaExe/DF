"""
2D Encoder for Panoramic X-ray images.
Uses ResNet18 backbone to extract latent vector Z.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class PXEncoder(nn.Module):
    """
    Encodes a 2D panoramic X-ray into a latent vector of dimension `latent_dim`.

    Architecture:
        - ResNet18 backbone (pretrained optional)
        - Replace first conv to accept 1-channel grayscale
        - Replace final FC with projection to latent_dim
    """

    def __init__(self, latent_dim: int = 512, pretrained: bool = True):
        super().__init__()

        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )

        # Modify first conv layer: 1 channel input instead of 3
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            # Initialize from pretrained weights by averaging across channel dim
            with torch.no_grad():
                self.conv1.weight.copy_(resnet.conv1.weight.mean(dim=1, keepdim=True))

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # Projection head to latent space
        self.fc = nn.Sequential(
            nn.Linear(512, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) grayscale PX image normalized to [-1, 1].

        Returns:
            z: (B, latent_dim) latent vector.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        z = self.fc(x)
        return z
