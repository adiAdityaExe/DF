"""
Inference script for Bidirectional Cross-Modal Translation.

Usage:
    # PX → CBCT
    python inference.py --input path/to/xray.png --mode px2cbct --checkpoint checkpoints/phase3_epoch100.pt

    # CBCT → PX
    python inference.py --input path/to/scan.nii --mode cbct2px --checkpoint checkpoints/phase3_epoch100.pt
"""

import os
import argparse
import numpy as np
import torch
import nibabel as nib
from PIL import Image

from config import Config
from models.encoder2d import PXEncoder
from models.encoder3d import CBCTEncoder
from models.nerf_decoder import NeRFDecoder
from models.projection import DifferentiableProjection
from utils.preprocessing import preprocess_px, preprocess_cbct


def load_models(cfg: Config, checkpoint_path: str, device: torch.device):
    """Load all generator models from a checkpoint."""
    encoder2d = PXEncoder(latent_dim=cfg.latent_dim).to(device)
    encoder3d = CBCTEncoder(latent_dim=cfg.latent_dim).to(device)
    nerf_decoder = NeRFDecoder(
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.nerf_hidden_dim,
        num_layers=cfg.nerf_num_layers,
        num_freqs=cfg.positional_encoding_freqs,
    ).to(device)
    projection = DifferentiableProjection(
        mode="beer_lambert",
        target_size=cfg.px_image_size,
    ).to(device)

    # Load checkpoint
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "encoder2d" in state:
        encoder2d.load_state_dict(state["encoder2d"])
    if "encoder3d" in state:
        encoder3d.load_state_dict(state["encoder3d"])
    if "nerf_decoder" in state:
        nerf_decoder.load_state_dict(state["nerf_decoder"])

    encoder2d.eval()
    encoder3d.eval()
    nerf_decoder.eval()
    projection.eval()

    print(f"Models loaded from: {checkpoint_path}")
    return encoder2d, encoder3d, nerf_decoder, projection


@torch.no_grad()
def px_to_cbct(
    input_path: str,
    output_path: str,
    encoder2d: PXEncoder,
    nerf_decoder: NeRFDecoder,
    cfg: Config,
    device: torch.device,
):
    """
    Convert a panoramic X-ray to a CBCT volume.

    Args:
        input_path: Path to input PX image (PNG/JPG).
        output_path: Path to save output NIfTI volume.
    """
    print(f"PX → CBCT: {input_path}")

    # Preprocess input PX
    px = preprocess_px(input_path, cfg.px_image_size).unsqueeze(0).to(device)
    # (1, 1, H, W)

    # Encode
    z = encoder2d(px)  # (1, latent_dim)

    # Generate 3D volume
    volume = nerf_decoder.generate_volume(
        z,
        resolution=cfg.cbct_volume_size,
        chunk_size=cfg.nerf_chunk_size,
    )  # (1, 1, D, H, W)

    # Convert to numpy and save as NIfTI
    vol_np = volume.squeeze().cpu().numpy()  # (D, H, W)
    vol_np = (vol_np * 255).astype(np.uint8)

    nii_img = nib.Nifti1Image(vol_np, affine=np.eye(4))
    nib.save(nii_img, output_path)
    print(f"CBCT volume saved: {output_path} (shape={vol_np.shape})")


@torch.no_grad()
def cbct_to_px(
    input_path: str,
    output_path: str,
    encoder3d: CBCTEncoder,
    projection: DifferentiableProjection,
    cfg: Config,
    device: torch.device,
):
    """
    Convert a CBCT volume to a panoramic X-ray.

    Args:
        input_path: Path to input NIfTI volume.
        output_path: Path to save output PNG image.
    """
    print(f"CBCT → PX: {input_path}")

    # Preprocess input CBCT
    cbct = preprocess_cbct(input_path, cfg.cbct_volume_size).unsqueeze(0).to(device)
    # (1, 1, D, H, W)

    # Project to 2D
    px_out = projection(cbct)  # (1, 1, H, W)

    # Convert to numpy image
    px_np = px_out.squeeze().cpu().numpy()  # (H, W)
    # Convert from [-1, 1] to [0, 255]
    px_np = ((px_np + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)

    img = Image.fromarray(px_np, mode="L")
    img.save(output_path)
    print(f"PX image saved: {output_path} (shape={px_np.shape})")


@torch.no_grad()
def px_to_cbct_nerf(
    input_path: str,
    output_path: str,
    encoder2d: PXEncoder,
    nerf_decoder: NeRFDecoder,
    cfg: Config,
    device: torch.device,
):
    """
    Full NeRF-based PX → CBCT using the encoder + implicit decoder.
    Same as px_to_cbct but explicit about the NeRF pipeline.
    """
    px_to_cbct(input_path, output_path, encoder2d, nerf_decoder, cfg, device)


def main():
    parser = argparse.ArgumentParser(
        description="Inference: Bidirectional PX ↔ CBCT Translation"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input file (PNG/JPG for PX, .nii for CBCT)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["px2cbct", "cbct2px"],
        help="Translation direction",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint (.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (auto-generated if not provided)",
    )
    args = parser.parse_args()

    cfg = Config()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    encoder2d, encoder3d, nerf_decoder, projection = load_models(
        cfg, args.checkpoint, device
    )

    # Auto-generate output path if not provided
    if args.output is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        if args.mode == "px2cbct":
            args.output = os.path.join(cfg.output_dir, f"{base}_generated.nii")
        else:
            args.output = os.path.join(cfg.output_dir, f"{base}_projected.png")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if args.mode == "px2cbct":
        px_to_cbct(args.input, args.output, encoder2d, nerf_decoder, cfg, device)
    elif args.mode == "cbct2px":
        cbct_to_px(args.input, args.output, encoder3d, projection, cfg, device)

    print("Inference complete!")


if __name__ == "__main__":
    main()
