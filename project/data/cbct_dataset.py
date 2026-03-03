"""
Dataset class for CBCT volumes (.nii files).
"""

import os
import glob
import torch
from torch.utils.data import Dataset
from utils.preprocessing import preprocess_cbct


class CBCTDataset(Dataset):
    """
    Dataset for loading and preprocessing 3D CBCT volumes from NIfTI files.

    Each item returns a tensor of shape (1, D, H, W) normalized to [0, 1].
    """

    def __init__(self, data_dir: str, target_size: int = 128):
        """
        Args:
            data_dir: Path to directory containing .nii / .nii.gz files.
            target_size: Resample volumes to (target_size)^3.
        """
        self.data_dir = data_dir
        self.target_size = target_size

        self.file_paths = sorted(
            glob.glob(os.path.join(data_dir, "*.nii"))
            + glob.glob(os.path.join(data_dir, "*.nii.gz"))
        )

        if len(self.file_paths) == 0:
            raise RuntimeError(
                f"No .nii or .nii.gz files found in {data_dir}. "
                "Please check your cbct_data_dir config."
            )

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.file_paths[idx]
        volume = preprocess_cbct(path, self.target_size)  # (1, D, H, W)
        return volume
