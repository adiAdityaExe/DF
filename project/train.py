"""
Training pipeline for Bidirectional Unpaired Cross-Modal Translation
between Dental PX (2D X-ray) and CBCT (3D volumes).

Supports phased training:
    Phase 1: CBCT autoencoder (encoder3d + nerf_decoder)
    Phase 2: PX encoder + projection consistency
    Phase 3: Full adversarial + cycle training
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from config import Config
from data.cbct_dataset import CBCTDataset
from data.px_dataset import PXDataset
from models.encoder2d import PXEncoder
from models.encoder3d import CBCTEncoder
from models.nerf_decoder import NeRFDecoder
from models.projection import DifferentiableProjection
from models.discriminator2d import Discriminator2D
from models.discriminator3d import Discriminator3D
from losses.cycle_loss import CycleLoss, LatentConsistencyLoss
from losses.adversarial_loss import AdversarialLoss
from losses.structural_loss import StructuralLoss


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state: dict, path: str):
    """Save training checkpoint."""
    torch.save(state, path)
    print(f"  Checkpoint saved: {path}")


def infinite_loader(loader: DataLoader):
    """Wrap a DataLoader to yield infinitely (for unpaired training)."""
    while True:
        for batch in loader:
            yield batch


class Trainer:
    """Main trainer for the cross-modal translation pipeline."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(
            cfg.device if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")

        # ---- Build models ----
        self.encoder2d = PXEncoder(latent_dim=cfg.latent_dim).to(self.device)
        self.encoder3d = CBCTEncoder(latent_dim=cfg.latent_dim).to(self.device)
        self.nerf_decoder = NeRFDecoder(
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.nerf_hidden_dim,
            num_layers=cfg.nerf_num_layers,
            num_freqs=cfg.positional_encoding_freqs,
        ).to(self.device)
        self.projection = DifferentiableProjection(
            mode="beer_lambert",
            target_size=cfg.px_image_size,
        ).to(self.device)
        self.disc2d = Discriminator2D().to(self.device)
        self.disc3d = Discriminator3D().to(self.device)

        # ---- Losses ----
        self.cycle_loss = CycleLoss()
        self.latent_loss = LatentConsistencyLoss()
        self.adv_loss = AdversarialLoss()
        self.struct_loss = StructuralLoss().to(self.device)

        # ---- Optimizers ----
        gen_params = (
            list(self.encoder2d.parameters())
            + list(self.encoder3d.parameters())
            + list(self.nerf_decoder.parameters())
        )
        disc_params = (
            list(self.disc2d.parameters())
            + list(self.disc3d.parameters())
        )

        self.optimizer_G = torch.optim.Adam(
            gen_params, lr=cfg.learning_rate_g, betas=(cfg.beta1, cfg.beta2)
        )
        self.optimizer_D = torch.optim.Adam(
            disc_params, lr=cfg.learning_rate_d, betas=(cfg.beta1, cfg.beta2)
        )

        # ---- AMP ----
        self.scaler_G = GradScaler("cuda", enabled=cfg.use_amp)
        self.scaler_D = GradScaler("cuda", enabled=cfg.use_amp)

        # ---- Logging ----
        self.writer = None
        if cfg.use_tensorboard:
            self.writer = SummaryWriter(log_dir=cfg.log_dir)

        self.global_step = 0

    def _build_dataloaders(self):
        """Build data loaders for CBCT and PX datasets."""
        cfg = self.cfg

        cbct_dataset = CBCTDataset(
            data_dir=cfg.cbct_data_dir, target_size=cfg.cbct_volume_size
        )
        px_dataset = PXDataset(
            data_dir=cfg.px_data_dir, target_size=cfg.px_image_size
        )

        print(f"CBCT dataset: {len(cbct_dataset)} volumes")
        print(f"PX dataset:   {len(px_dataset)} images")

        use_pin = self.device.type == "cuda"

        cbct_loader = DataLoader(
            cbct_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=use_pin,
            drop_last=True,
        )
        px_loader = DataLoader(
            px_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=use_pin,
            drop_last=True,
        )

        return cbct_loader, px_loader

    def _log_scalar(self, tag: str, value: float):
        """Log a scalar to TensorBoard."""
        if self.writer is not None:
            self.writer.add_scalar(tag, value, self.global_step)

    def _log_image(self, tag: str, img: torch.Tensor):
        """Log a 2D image to TensorBoard. Expects (1, H, W) or (H, W)."""
        if self.writer is not None:
            if img.dim() == 2:
                img = img.unsqueeze(0)
            # Normalize to [0, 1] for display
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            self.writer.add_image(tag, img, self.global_step)

    # ------------------------------------------------------------------
    # Phase 1: CBCT Autoencoder
    # ------------------------------------------------------------------
    def train_phase1(self, num_epochs: int):
        """
        Phase 1: Train CBCT autoencoder (encoder3d + nerf_decoder).
        Reconstruction loss only.
        """
        print("\n" + "=" * 60)
        print("PHASE 1: CBCT Autoencoder Training")
        print("=" * 60)

        cbct_loader, _ = self._build_dataloaders()
        recon_criterion = nn.L1Loss()

        for epoch in range(1, num_epochs + 1):
            self.encoder3d.train()
            self.nerf_decoder.train()
            epoch_loss = 0.0

            for batch_idx, cbct in enumerate(cbct_loader):
                cbct = cbct.to(self.device)  # (B, 1, D, H, W)

                self.optimizer_G.zero_grad()

                with autocast(str(self.device), enabled=self.cfg.use_amp):
                    z = self.encoder3d(cbct)  # (B, latent_dim)
                    # Generate volume at lower resolution for memory efficiency
                    gen_res = min(self.cfg.cbct_volume_size, 64)
                    cbct_recon = self.nerf_decoder.generate_volume(
                        z, resolution=gen_res, chunk_size=self.cfg.nerf_chunk_size
                    )

                    # Downsample real CBCT to match generated resolution
                    if gen_res != self.cfg.cbct_volume_size:
                        cbct_down = nn.functional.interpolate(
                            cbct,
                            size=(gen_res, gen_res, gen_res),
                            mode="trilinear",
                            align_corners=False,
                        )
                    else:
                        cbct_down = cbct

                    loss = recon_criterion(cbct_recon, cbct_down)

                self.scaler_G.scale(loss).backward()
                if self.cfg.grad_clip > 0:
                    self.scaler_G.unscale_(self.optimizer_G)
                    nn.utils.clip_grad_norm_(
                        list(self.encoder3d.parameters())
                        + list(self.nerf_decoder.parameters()),
                        self.cfg.grad_clip,
                    )
                self.scaler_G.step(self.optimizer_G)
                self.scaler_G.update()

                epoch_loss += loss.item()
                self.global_step += 1

                if self.global_step % self.cfg.log_interval == 0:
                    self._log_scalar("phase1/recon_loss", loss.item())
                    print(
                        f"  [Phase1] Epoch {epoch}/{num_epochs} "
                        f"Batch {batch_idx+1} Loss: {loss.item():.6f}"
                    )

            avg_loss = epoch_loss / max(len(cbct_loader), 1)
            print(f"  [Phase1] Epoch {epoch}/{num_epochs} Avg Loss: {avg_loss:.6f}")

            if epoch % self.cfg.save_interval == 0:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "encoder3d": self.encoder3d.state_dict(),
                        "nerf_decoder": self.nerf_decoder.state_dict(),
                        "optimizer_G": self.optimizer_G.state_dict(),
                    },
                    os.path.join(self.cfg.checkpoint_dir, f"phase1_epoch{epoch}.pt"),
                )

    # ------------------------------------------------------------------
    # Phase 2: PX Encoder + Projection Consistency
    # ------------------------------------------------------------------
    def train_phase2(self, num_epochs: int):
        """
        Phase 2: Train PX encoder with projection consistency.
        PX → encode → generate CBCT → project back → compare with PX.
        """
        print("\n" + "=" * 60)
        print("PHASE 2: PX Encoder + Projection Consistency")
        print("=" * 60)

        _, px_loader = self._build_dataloaders()
        recon_criterion = nn.L1Loss()

        # Freeze encoder3d and nerf_decoder initially (use pretrained)
        for p in self.encoder3d.parameters():
            p.requires_grad = False
        self.encoder3d.eval()

        px_optimizer = torch.optim.Adam(
            self.encoder2d.parameters(),
            lr=self.cfg.learning_rate_g,
            betas=(self.cfg.beta1, self.cfg.beta2),
        )
        scaler = GradScaler("cuda", enabled=self.cfg.use_amp)

        for epoch in range(1, num_epochs + 1):
            self.encoder2d.train()
            self.nerf_decoder.train()
            epoch_loss = 0.0

            for batch_idx, px in enumerate(px_loader):
                px = px.to(self.device)  # (B, 1, H, W)

                px_optimizer.zero_grad()

                with autocast(str(self.device), enabled=self.cfg.use_amp):
                    z_px = self.encoder2d(px)  # (B, latent_dim)

                    # Generate CBCT from PX latent
                    gen_res = min(self.cfg.cbct_volume_size, 64)
                    cbct_fake = self.nerf_decoder.generate_volume(
                        z_px, resolution=gen_res, chunk_size=self.cfg.nerf_chunk_size
                    )

                    # Project generated CBCT back to 2D
                    px_recon = self.projection(cbct_fake)  # (B, 1, 256, 256)

                    # Projection consistency loss
                    loss_proj = recon_criterion(px_recon, px)

                    # SSIM structural loss
                    loss_ssim = self.struct_loss.px_loss(px_recon, px)

                    loss = loss_proj + self.cfg.lambda_structural * loss_ssim

                scaler.scale(loss).backward()
                if self.cfg.grad_clip > 0:
                    scaler.unscale_(px_optimizer)
                    nn.utils.clip_grad_norm_(
                        self.encoder2d.parameters(), self.cfg.grad_clip
                    )
                scaler.step(px_optimizer)
                scaler.update()

                epoch_loss += loss.item()
                self.global_step += 1

                if self.global_step % self.cfg.log_interval == 0:
                    self._log_scalar("phase2/proj_loss", loss_proj.item())
                    self._log_scalar("phase2/ssim_loss", loss_ssim.item())
                    self._log_scalar("phase2/total_loss", loss.item())
                    print(
                        f"  [Phase2] Epoch {epoch}/{num_epochs} "
                        f"Batch {batch_idx+1} Loss: {loss.item():.6f} "
                        f"(proj={loss_proj.item():.4f} ssim={loss_ssim.item():.4f})"
                    )

                if self.global_step % self.cfg.sample_interval == 0:
                    self._log_image("phase2/px_real", px[0])
                    self._log_image("phase2/px_recon", px_recon[0])

            avg_loss = epoch_loss / max(len(px_loader), 1)
            print(f"  [Phase2] Epoch {epoch}/{num_epochs} Avg Loss: {avg_loss:.6f}")

            if epoch % self.cfg.save_interval == 0:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "encoder2d": self.encoder2d.state_dict(),
                        "nerf_decoder": self.nerf_decoder.state_dict(),
                        "optimizer": px_optimizer.state_dict(),
                    },
                    os.path.join(self.cfg.checkpoint_dir, f"phase2_epoch{epoch}.pt"),
                )

        # Unfreeze encoder3d for phase 3
        for p in self.encoder3d.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------------
    # Phase 3: Full Adversarial Cycle Training
    # ------------------------------------------------------------------
    def train_phase3(self, num_epochs: int):
        """
        Phase 3: Full adversarial cycle training.

        Per iteration:
            1) Encode PX and CBCT
            2) PX → CBCT_fake, CBCT → PX_fake
            3) Cycle: PX → CBCT_fake → PX_cycle; CBCT → PX_fake → CBCT_cycle
            4) Compute all losses
            5) Update generators, then discriminators
        """
        print("\n" + "=" * 60)
        print("PHASE 3: Full Adversarial Cycle Training")
        print("=" * 60)

        cbct_loader, px_loader = self._build_dataloaders()
        cbct_iter = infinite_loader(cbct_loader)
        px_iter = infinite_loader(px_loader)

        iters_per_epoch = max(len(cbct_loader), len(px_loader))

        for epoch in range(1, num_epochs + 1):
            self.encoder2d.train()
            self.encoder3d.train()
            self.nerf_decoder.train()
            self.disc2d.train()
            self.disc3d.train()

            epoch_losses = {
                "G_total": 0.0,
                "D_total": 0.0,
                "adv_px": 0.0,
                "adv_cbct": 0.0,
                "cycle_px": 0.0,
                "cycle_cbct": 0.0,
                "latent": 0.0,
            }

            for batch_idx in range(iters_per_epoch):
                px = next(px_iter).to(self.device)      # (B, 1, H, W)
                cbct = next(cbct_iter).to(self.device)   # (B, 1, D, H, W)

                gen_res = min(self.cfg.cbct_volume_size, 64)

                # ============ GENERATOR UPDATE ============
                self.optimizer_G.zero_grad()

                with autocast(str(self.device), enabled=self.cfg.use_amp):
                    # --- Encode ---
                    z_px = self.encoder2d(px)      # (B, latent_dim)
                    z_cbct = self.encoder3d(cbct)   # (B, latent_dim)

                    # --- PX → CBCT_fake ---
                    cbct_fake = self.nerf_decoder.generate_volume(
                        z_px, resolution=gen_res,
                        chunk_size=self.cfg.nerf_chunk_size,
                    )

                    # --- CBCT → PX_fake ---
                    # Downsample real CBCT if needed
                    if gen_res != self.cfg.cbct_volume_size:
                        cbct_for_proj = nn.functional.interpolate(
                            cbct,
                            size=(gen_res, gen_res, gen_res),
                            mode="trilinear",
                            align_corners=False,
                        )
                    else:
                        cbct_for_proj = cbct
                    px_fake = self.projection(cbct_for_proj)  # (B, 1, 256, 256)

                    # --- Cycle PX: PX → CBCT_fake → PX_cycle ---
                    px_cycle = self.projection(cbct_fake)

                    # --- Cycle CBCT: CBCT → PX_fake → z_px_fake → CBCT_cycle ---
                    z_px_fake = self.encoder2d(px_fake)
                    cbct_cycle = self.nerf_decoder.generate_volume(
                        z_px_fake, resolution=gen_res,
                        chunk_size=self.cfg.nerf_chunk_size,
                    )

                    # Downsample real cbct for comparison
                    if gen_res != self.cfg.cbct_volume_size:
                        cbct_down = nn.functional.interpolate(
                            cbct,
                            size=(gen_res, gen_res, gen_res),
                            mode="trilinear",
                            align_corners=False,
                        )
                    else:
                        cbct_down = cbct

                    # --- Adversarial losses (generator wants disc to say "real") ---
                    loss_adv_cbct = self.adv_loss.generator_loss(
                        self.disc3d(cbct_fake)
                    )
                    loss_adv_px = self.adv_loss.generator_loss(
                        self.disc2d(px_fake)
                    )

                    # --- Cycle losses ---
                    loss_cycle_px = self.cycle_loss(px, px_cycle)
                    loss_cycle_cbct = self.cycle_loss(cbct_down, cbct_cycle)

                    # --- Latent consistency ---
                    loss_latent = self.latent_loss(z_px, z_cbct)

                    # --- Total generator loss ---
                    loss_G = (
                        self.cfg.lambda_adv * (loss_adv_px + loss_adv_cbct)
                        + self.cfg.lambda_cycle * (loss_cycle_px + loss_cycle_cbct)
                        + self.cfg.lambda_latent * loss_latent
                    )

                self.scaler_G.scale(loss_G).backward()
                if self.cfg.grad_clip > 0:
                    self.scaler_G.unscale_(self.optimizer_G)
                    gen_params = (
                        list(self.encoder2d.parameters())
                        + list(self.encoder3d.parameters())
                        + list(self.nerf_decoder.parameters())
                    )
                    nn.utils.clip_grad_norm_(gen_params, self.cfg.grad_clip)
                self.scaler_G.step(self.optimizer_G)
                self.scaler_G.update()

                # ============ DISCRIMINATOR UPDATE ============
                self.optimizer_D.zero_grad()

                with autocast(str(self.device), enabled=self.cfg.use_amp):
                    # Detach fake data
                    loss_D_px = self.adv_loss.discriminator_loss(
                        self.disc2d(px), self.disc2d(px_fake.detach())
                    )
                    loss_D_cbct = self.adv_loss.discriminator_loss(
                        self.disc3d(cbct_down), self.disc3d(cbct_fake.detach())
                    )
                    loss_D = loss_D_px + loss_D_cbct

                self.scaler_D.scale(loss_D).backward()
                if self.cfg.grad_clip > 0:
                    self.scaler_D.unscale_(self.optimizer_D)
                    disc_params = (
                        list(self.disc2d.parameters())
                        + list(self.disc3d.parameters())
                    )
                    nn.utils.clip_grad_norm_(disc_params, self.cfg.grad_clip)
                self.scaler_D.step(self.optimizer_D)
                self.scaler_D.update()

                # ---- Logging ----
                epoch_losses["G_total"] += loss_G.item()
                epoch_losses["D_total"] += loss_D.item()
                epoch_losses["adv_px"] += loss_adv_px.item()
                epoch_losses["adv_cbct"] += loss_adv_cbct.item()
                epoch_losses["cycle_px"] += loss_cycle_px.item()
                epoch_losses["cycle_cbct"] += loss_cycle_cbct.item()
                epoch_losses["latent"] += loss_latent.item()

                self.global_step += 1

                if self.global_step % self.cfg.log_interval == 0:
                    for k, v in epoch_losses.items():
                        self._log_scalar(f"phase3/{k}", v / (batch_idx + 1))
                    print(
                        f"  [Phase3] Epoch {epoch}/{num_epochs} "
                        f"Batch {batch_idx+1}/{iters_per_epoch} "
                        f"G={loss_G.item():.4f} D={loss_D.item():.4f} "
                        f"cyc_px={loss_cycle_px.item():.4f} "
                        f"cyc_cbct={loss_cycle_cbct.item():.4f} "
                        f"lat={loss_latent.item():.4f}"
                    )

                if self.global_step % self.cfg.sample_interval == 0:
                    self._log_image("phase3/px_real", px[0])
                    self._log_image("phase3/px_fake", px_fake[0])
                    self._log_image("phase3/px_cycle", px_cycle[0])

            # Epoch summary
            for k in epoch_losses:
                epoch_losses[k] /= iters_per_epoch
            print(
                f"  [Phase3] Epoch {epoch}/{num_epochs} "
                + " ".join(f"{k}={v:.4f}" for k, v in epoch_losses.items())
            )

            if epoch % self.cfg.save_interval == 0:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "encoder2d": self.encoder2d.state_dict(),
                        "encoder3d": self.encoder3d.state_dict(),
                        "nerf_decoder": self.nerf_decoder.state_dict(),
                        "disc2d": self.disc2d.state_dict(),
                        "disc3d": self.disc3d.state_dict(),
                        "optimizer_G": self.optimizer_G.state_dict(),
                        "optimizer_D": self.optimizer_D.state_dict(),
                        "global_step": self.global_step,
                    },
                    os.path.join(self.cfg.checkpoint_dir, f"phase3_epoch{epoch}.pt"),
                )

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def train(self):
        """Run the full phased training pipeline."""
        cfg = self.cfg

        if cfg.training_phase <= 1:
            self.train_phase1(cfg.phase1_epochs)

        if cfg.training_phase <= 2:
            # Load best phase1 checkpoint if starting from phase 2
            if cfg.training_phase == 2:
                self._load_latest_checkpoint("phase1")
            self.train_phase2(cfg.phase2_epochs)

        if cfg.training_phase <= 3:
            # Load best phase2 checkpoint if starting from phase 3
            if cfg.training_phase == 3:
                self._load_latest_checkpoint("phase2")
            self.train_phase3(cfg.phase3_epochs)

        print("\nTraining complete!")
        if self.writer is not None:
            self.writer.close()

    def _load_latest_checkpoint(self, phase_prefix: str):
        """Load the latest checkpoint for a given phase."""
        ckpt_dir = self.cfg.checkpoint_dir
        ckpts = sorted(
            [f for f in os.listdir(ckpt_dir) if f.startswith(phase_prefix)],
            key=lambda x: int(x.split("epoch")[1].split(".")[0]),
        )
        if ckpts:
            path = os.path.join(ckpt_dir, ckpts[-1])
            print(f"Loading checkpoint: {path}")
            state = torch.load(path, map_location=self.device, weights_only=False)
            if "encoder3d" in state:
                self.encoder3d.load_state_dict(state["encoder3d"])
            if "encoder2d" in state:
                self.encoder2d.load_state_dict(state["encoder2d"])
            if "nerf_decoder" in state:
                self.nerf_decoder.load_state_dict(state["nerf_decoder"])
            if "disc2d" in state:
                self.disc2d.load_state_dict(state["disc2d"])
            if "disc3d" in state:
                self.disc3d.load_state_dict(state["disc3d"])
        else:
            print(f"No checkpoint found for {phase_prefix}, starting fresh.")


def main():
    cfg = Config()
    set_seed(cfg.seed)
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
