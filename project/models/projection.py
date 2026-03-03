"""
Differentiable projection module for converting 3D CBCT volumes to 2D PX images.
Implements Beer-Lambert attenuation projection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableProjection(nn.Module):
    """
    Projects a 3D volume to a 2D image using differentiable forward projection.

    Two modes:
        - 'sum': Simple sum along depth axis (DRR-like)
        - 'beer_lambert': Exponential attenuation: I(u,v) = exp(-sum(density))

    The output is resized to the target PX image size.
    """

    def __init__(
        self,
        mode: str = "beer_lambert",
        target_size: int = 256,
        projection_axis: int = 2,
    ):
        """
        Args:
            mode: 'sum' or 'beer_lambert'.
            target_size: Output image spatial size.
            projection_axis: Depth axis to project along (0=sagittal, 1=coronal, 2=axial).
        """
        super().__init__()
        assert mode in ("sum", "beer_lambert"), f"Unknown mode: {mode}"
        self.mode = mode
        self.target_size = target_size
        self.projection_axis = projection_axis

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Args:
            volume: (B, 1, D, H, W) 3D volume with density in [0, 1].

        Returns:
            projection: (B, 1, target_size, target_size) 2D image.
                - 'sum' mode: normalized sum projection
                - 'beer_lambert' mode: values in [-1, 1] for compatibility with PX
        """
        vol = volume.squeeze(1)  # (B, D, H, W)

        if self.mode == "sum":
            # Sum along depth axis
            proj = vol.sum(dim=self.projection_axis + 0)  # project along chosen axis
            # Normalize to [0, 1]
            proj_min = proj.amin(dim=(-2, -1), keepdim=True)
            proj_max = proj.amax(dim=(-2, -1), keepdim=True)
            proj = (proj - proj_min) / (proj_max - proj_min + 1e-8)
            # Convert to [-1, 1] for PX compatibility
            proj = proj * 2.0 - 1.0

        elif self.mode == "beer_lambert":
            # Beer-Lambert: I(u,v) = exp(-sum(density along ray))
            # Scale density by a learnable or fixed attenuation factor
            attenuation = vol.sum(dim=self.projection_axis + 0)  # (B, H, W) or (B, D, W) etc.
            proj = torch.exp(-attenuation)
            # Normalize to [-1, 1]
            proj_min = proj.amin(dim=(-2, -1), keepdim=True)
            proj_max = proj.amax(dim=(-2, -1), keepdim=True)
            proj = (proj - proj_min) / (proj_max - proj_min + 1e-8)
            proj = proj * 2.0 - 1.0

        # Add channel dim and resize
        proj = proj.unsqueeze(1)  # (B, 1, H', W')
        if proj.shape[-2] != self.target_size or proj.shape[-1] != self.target_size:
            proj = F.interpolate(
                proj,
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
            )

        return proj
