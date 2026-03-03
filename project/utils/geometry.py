"""
Geometry utilities for coordinate grids, positional encoding, and ray operations.
"""

import torch
import numpy as np


def create_coordinate_grid(resolution: int = 128, device: str = "cuda") -> torch.Tensor:
    """
    Create a normalized 3D coordinate grid in [-1, 1].

    Args:
        resolution: Number of points along each axis.
        device: Target device.

    Returns:
        Tensor of shape (resolution^3, 3) with (x, y, z) coordinates.
    """
    coords = torch.linspace(-1.0, 1.0, resolution, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(coords, coords, coords, indexing="ij")
    grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # (R, R, R, 3)
    return grid.reshape(-1, 3)  # (R^3, 3)


def positional_encoding(x: torch.Tensor, num_freqs: int = 10) -> torch.Tensor:
    """
    Apply sinusoidal positional encoding to input coordinates.

    Args:
        x: Input coordinates of shape (..., D).
        num_freqs: Number of frequency bands.

    Returns:
        Encoded coordinates of shape (..., D * (2 * num_freqs + 1)).
        Original coordinates are included.
    """
    encodings = [x]
    for i in range(num_freqs):
        freq = 2.0 ** i * np.pi
        encodings.append(torch.sin(freq * x))
        encodings.append(torch.cos(freq * x))
    return torch.cat(encodings, dim=-1)


def get_positional_encoding_dim(input_dim: int = 3, num_freqs: int = 10) -> int:
    """Return the output dimension after positional encoding."""
    return input_dim * (2 * num_freqs + 1)


def create_projection_rays(
    height: int = 256,
    width: int = 256,
    depth_samples: int = 128,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Create ray sample points for a front-parallel projection (AP view).
    Rays go along the Z axis for each (u, v) pixel.

    Args:
        height: Image height.
        width: Image width.
        depth_samples: Number of depth samples along each ray.
        device: Target device.

    Returns:
        Tensor of shape (H * W, depth_samples, 3) with sample coordinates in [-1, 1].
    """
    u = torch.linspace(-1.0, 1.0, width, device=device)
    v = torch.linspace(-1.0, 1.0, height, device=device)
    z = torch.linspace(-1.0, 1.0, depth_samples, device=device)

    grid_u, grid_v = torch.meshgrid(u, v, indexing="xy")  # (H, W)
    grid_u = grid_u.reshape(-1, 1).expand(-1, depth_samples)  # (H*W, D)
    grid_v = grid_v.reshape(-1, 1).expand(-1, depth_samples)  # (H*W, D)
    grid_z = z.unsqueeze(0).expand(height * width, -1)        # (H*W, D)

    rays = torch.stack([grid_u, grid_v, grid_z], dim=-1)  # (H*W, D, 3)
    return rays
