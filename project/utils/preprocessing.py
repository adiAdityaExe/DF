"""
Preprocessing utilities for CBCT volumes and PX images.
"""

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from PIL import Image


def load_nii(path: str) -> np.ndarray:
    """Load a NIfTI file and return the volume as a numpy array."""
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    return data


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Normalize a 3D volume to [0, 1]."""
    vmin = volume.min()
    vmax = volume.max()
    if vmax - vmin > 0:
        volume = (volume - vmin) / (vmax - vmin)
    else:
        volume = np.zeros_like(volume)
    return volume.astype(np.float32)


def resample_volume(volume: np.ndarray, target_shape: tuple = (128, 128, 128)) -> np.ndarray:
    """
    Resample a 3D volume to target_shape using trilinear interpolation.

    Args:
        volume: Input volume of shape (D, H, W).
        target_shape: Desired output shape.

    Returns:
        Resampled volume as numpy array.
    """
    tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    resampled = F.interpolate(
        tensor,
        size=target_shape,
        mode="trilinear",
        align_corners=False,
    )
    return resampled.squeeze(0).squeeze(0).numpy()


def preprocess_cbct(path: str, target_size: int = 128) -> torch.Tensor:
    """
    Full preprocessing pipeline for a single CBCT volume.
    Returns tensor of shape (1, S, S, S) normalized to [0, 1].
    """
    volume = load_nii(path)
    volume = resample_volume(volume, (target_size, target_size, target_size))
    volume = normalize_volume(volume)
    return torch.from_numpy(volume).float().unsqueeze(0)  # (1, D, H, W)


def preprocess_px(path: str, target_size: int = 256) -> torch.Tensor:
    """
    Full preprocessing pipeline for a single PX image.
    Returns tensor of shape (1, H, W) normalized to [-1, 1].
    """
    img = Image.open(path).convert("L")  # grayscale
    img = img.resize((target_size, target_size), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0  # [0, 1]
    arr = arr * 2.0 - 1.0                            # [-1, 1]
    return torch.from_numpy(arr).float().unsqueeze(0)  # (1, H, W)
