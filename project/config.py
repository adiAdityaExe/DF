"""
Configuration for Bidirectional Unpaired Cross-Modal Translation
between Dental PX (2D X-ray) and CBCT (3D volumes).
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ---- Paths ----
    cbct_data_dir: str = os.path.join("..", "assets", "cbctImg")
    px_data_dir: str = os.path.join("..", "assets", "xRay")
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    output_dir: str = "outputs"

    # ---- Data ----
    cbct_volume_size: int = 128          # Resample CBCT to 128^3
    px_image_size: int = 256             # Resize PX to 256x256
    num_workers: int = 2

    # ---- Model ----
    latent_dim: int = 512                # Unified latent dimension Z
    nerf_hidden_dim: int = 256           # NeRF MLP hidden dimension
    nerf_num_layers: int = 8             # NeRF MLP depth
    positional_encoding_freqs: int = 10  # Positional encoding frequencies

    # ---- Training ----
    batch_size: int = 1
    learning_rate_g: float = 2e-4        # Generator LR
    learning_rate_d: float = 2e-4        # Discriminator LR
    beta1: float = 0.5                   # Adam beta1
    beta2: float = 0.999                 # Adam beta2
    num_epochs: int = 200
    use_amp: bool = True                 # Mixed precision training
    grad_clip: float = 1.0

    # ---- Loss Weights ----
    lambda_cycle: float = 10.0
    lambda_latent: float = 1.0
    lambda_structural: float = 5.0
    lambda_adv: float = 1.0

    # ---- Phased Training ----
    # Phase 1: CBCT autoencoder only
    # Phase 2: PX encoder + projection consistency
    # Phase 3: Full adversarial cycle training
    training_phase: int = 1              # 1, 2, or 3
    phase1_epochs: int = 10
    phase2_epochs: int = 15
    phase3_epochs: int = 75

    # ---- Logging ----
    use_tensorboard: bool = True
    log_interval: int = 10               # Log every N iterations
    save_interval: int = 5               # Save checkpoint every N epochs
    sample_interval: int = 50            # Save sample images every N iters

    # ---- Device ----
    device: str = "cuda"                 # "cuda" or "cpu"

    # ---- NeRF Sampling ----
    nerf_sample_points: int = 64         # Points sampled per ray during proj
    nerf_chunk_size: int = 65536         # Chunk size for NeRF eval

    # ---- Reproducibility ----
    seed: int = 42

    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
