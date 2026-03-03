"""
Dataset class for Panoramic X-ray (PX) images.
"""

import os
import glob
import torch
from torch.utils.data import Dataset
from utils.preprocessing import preprocess_px


class PXDataset(Dataset):
    """
    Dataset for loading and preprocessing 2D panoramic dental X-ray images.

    Each item returns a tensor of shape (1, H, W) normalized to [-1, 1].
    """

    def __init__(self, data_dir: str, target_size: int = 256):
        """
        Args:
            data_dir: Path to directory containing PNG/JPG images.
            target_size: Resize images to (target_size, target_size).
        """
        self.data_dir = data_dir
        self.target_size = target_size

        self.file_paths = sorted(
            glob.glob(os.path.join(data_dir, "*.png"))
            + glob.glob(os.path.join(data_dir, "*.jpg"))
            + glob.glob(os.path.join(data_dir, "*.jpeg"))
        )

        if len(self.file_paths) == 0:
            raise RuntimeError(
                f"No PNG/JPG images found in {data_dir}. "
                "Please check your px_data_dir config."
            )

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.file_paths[idx]
        image = preprocess_px(path, self.target_size)  # (1, H, W)
        return image
