"""
NeRF-style implicit decoder for generating 3D CBCT volumes from latent codes.
"""

import torch
import torch.nn as nn
from utils.geometry import positional_encoding, get_positional_encoding_dim


class NeRFDecoder(nn.Module):
    """
    Implicit Neural Representation (NeRF-style MLP) decoder.

    Input: 3D coordinates (x, y, z) + latent code Z
    Output: Density value at that point (1 channel)

    Uses positional encoding on coordinates for high-frequency detail.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 8,
        num_freqs: int = 10,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_freqs = num_freqs
        self.coord_dim = get_positional_encoding_dim(3, num_freqs)

        input_dim = self.coord_dim + latent_dim

        # Build MLP with skip connection at layer 4
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))

        for i in range(1, num_layers):
            if i == num_layers // 2:
                # Skip connection: concatenate input again
                layers.append(nn.Linear(hidden_dim + input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        self.mlp = nn.ModuleList(layers)
        self.skip_layer_idx = (num_layers // 2) * 2  # index in ModuleList

        # Output head: density
        self.density_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # density in [0, 1]
        )

    def forward(self, coords: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (N, 3) or (B, N, 3) 3D coordinates in [-1, 1].
            z: (B, latent_dim) latent vector.

        Returns:
            density: (B, N, 1) density values.
        """
        if coords.dim() == 2:
            # (N, 3) → (B, N, 3)
            coords = coords.unsqueeze(0).expand(z.size(0), -1, -1)

        B, N, _ = coords.shape

        # Positional encoding on coordinates
        encoded_coords = positional_encoding(coords, self.num_freqs)  # (B, N, coord_dim)

        # Expand latent to match coordinate count
        z_expanded = z.unsqueeze(1).expand(-1, N, -1)  # (B, N, latent_dim)

        # Concatenate
        x = torch.cat([encoded_coords, z_expanded], dim=-1)  # (B, N, input_dim)
        h = x

        # Forward through MLP with skip connection
        layer_idx = 0
        for i, layer in enumerate(self.mlp):
            if i == self.skip_layer_idx:
                # Skip connection
                h = torch.cat([h, x], dim=-1)
            h = layer(h)

        density = self.density_head(h)  # (B, N, 1)
        return density

    def generate_volume(
        self,
        z: torch.Tensor,
        resolution: int = 128,
        chunk_size: int = 65536,
    ) -> torch.Tensor:
        """
        Generate a full 3D volume from a latent code by querying the NeRF at
        a dense coordinate grid.

        Args:
            z: (B, latent_dim) latent vector.
            resolution: Grid resolution (output will be resolution^3).
            chunk_size: Number of points to process at once (memory control).

        Returns:
            volume: (B, 1, R, R, R) 3D volume.
        """
        from utils.geometry import create_coordinate_grid

        device = z.device
        B = z.size(0)
        coords = create_coordinate_grid(resolution, device)  # (R^3, 3)
        total_points = coords.size(0)

        densities = []
        for start in range(0, total_points, chunk_size):
            end = min(start + chunk_size, total_points)
            chunk_coords = coords[start:end]  # (C, 3)
            chunk_density = self.forward(chunk_coords, z)  # (B, C, 1)
            densities.append(chunk_density)

        density = torch.cat(densities, dim=1)  # (B, R^3, 1)
        volume = density.view(B, 1, resolution, resolution, resolution)
        return volume
