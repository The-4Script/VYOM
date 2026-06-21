"""
denoiser/test.py

Single-star inference for the trained Noise2Noise denoiser.
Accepts a TIC ID or a direct .fits file path.
Returns denoised flux array + uncertainty estimate via MC Dropout.

Usage:
  python -m denoiser.test --tic 12345678
  python -m denoiser.test --fits data/raw/tess/sector_01/tess_lc.fits
  python -m denoiser.test --fits data/raw/tess/sector_01/tess_lc.fits --no-plot
  python -m denoiser.test --tic 12345678 --mc-passes 100
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .config import CFG
from .model import NoiseToNoiseUNet
from .dataset import _read_fits, _chunk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

plt.rcParams.update({
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

def load_model(
    weights_path: Path = CFG.best_weights,
    device:       torch.device = None,
) -> tuple[NoiseToNoiseUNet, torch.device]:
    """
    Load trained denoiser from checkpoint.

    Args:
        weights_path : path to .pth file
        device       : torch device (auto-detected if None)

    Returns:
        (model, device)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights_path}\n"
            "Train first: python -m denoiser.train"
        )

    model = NoiseToNoiseUNet().to(device)
    ckpt  = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    trained_epoch = ckpt.get("epoch", "?")
    logger.info(f"Loaded weights from {weights_path}  (epoch {trained_epoch})")

    return model, device


# ─────────────────────────────────────────────────────────────────────────────
# MC Dropout inference
# ─────────────────────────────────────────────────────────────────────────────

def mc_dropout_predict(
    model:    NoiseToNoiseUNet,
    x:        torch.Tensor,
    n_passes: int = CFG.mc_dropout_passes,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Monte Carlo Dropout uncertainty estimation.

    Keeps dropout ACTIVE at inference time by calling model.train().
    Runs n_passes forward passes on the same input.
    Mean of outputs = best prediction.
    Std  of outputs = uncertainty estimate.

    Why this works:
      Dropout randomly zeros channels each forward pass.
      Each pass = slightly different model = slightly different output.
      The spread across passes estimates how confident the model is.
      High std → model uncertain → treat output cautiously.

    Args:
        model    : trained NoiseToNoiseUNet
        x        : [1, 1, T] input tensor (single chunk)
        n_passes : number of stochastic forward passes (default 50)

    Returns:
        mean : [1, 1, T] — best denoised estimate
        std  : [1, 1, T] — per-timestep uncertainty
    """
    # Keep dropout active — do NOT call model.eval()
    model.train()

    preds = []
    with torch.no_grad():
        for _ in range(n_passes):
            preds.append(model(x))

    stacked = torch.stack(preds, dim=0)    # [n_passes, 1, 1, T]
    mean    = stacked.mean(dim=0)          # [1, 1, T]
    std     = stacked.std(dim=0)           # [1, 1, T]

    return mean, std


# ─────────────────────────────────────────────────────────────────────────────
# FITS → denoised flux
# ─────────────────────────────────────────────────────────────────────────────

def denoise_fits(
    fits_path:    Path,
    model:        NoiseToNoiseUNet,
    device:       torch.device,
    mc_passes:    int = CFG.mc_dropout_passes,
) -> dict:
    """
    Run full denoising pipeline on a single FITS file.

    Steps:
      1. Read + preprocess FITS → normalised flux array
      2. Chunk into T=1000 windows (stride=500, overlapping)
      3. Denoise each chunk with MC Dropout
      4. Stitch chunks back → full-length denoised array
         Overlapping regions averaged for smooth stitching

    Args:
        fits_path : path to TESS FITS file
        model     : loaded NoiseToNoiseUNet
        device    : torch device
        mc_passes : MC Dropout forward passes per chunk

    Returns:
        dict with keys:
          noisy      : [N] float32 — preprocessed input flux
          denoised   : [N] float32 — denoised flux
          uncertainty: [N] float32 — per-timestep std from MC Dropout
          fits_path  : str
    """
    # Step 1: read FITS
    flux = _read_fits(fits_path)
    if flux is None:
        raise ValueError(f"Could not read or too few points in {fits_path}")

    N = len(flux)
    logger.info(f"Loaded {fits_path.name} — {N} time steps")

    # Step 2: chunk
    chunks = _chunk(flux, CFG.chunk_length, CFG.chunk_stride)
    logger.info(f"Split into {len(chunks)} overlapping chunks")

    # Step 3: denoise each chunk
    denoised_sum = np.zeros(N, dtype=np.float32)
    uncertainty_sum = np.zeros(N, dtype=np.float32)
    count_map    = np.zeros(N, dtype=np.float32)   # tracks how many chunks cover each point

    for idx, chunk in enumerate(chunks):
        x = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0).to(device)
        # [T] → [1, T] → [1, 1, T]

        mean, std = mc_dropout_predict(model, x, n_passes=mc_passes)

        denoised_chunk    = mean.squeeze().cpu().numpy()   # [T]
        uncertainty_chunk = std.squeeze().cpu().numpy()    # [T]

        # Map chunk back to original positions
        start = idx * CFG.chunk_stride
        end   = start + CFG.chunk_length

        denoised_sum[start:end]    += denoised_chunk
        uncertainty_sum[start:end] += uncertainty_chunk
        count_map[start:end]       += 1.0

    # Step 4: average overlapping regions
    count_map   = np.maximum(count_map, 1.0)          # avoid divide by zero
    denoised    = denoised_sum    / count_map
    uncertainty = uncertainty_sum / count_map

    return {
        "noisy":       flux,
        "denoised":    denoised,
        "uncertainty": uncertainty,
        "fits_path":   str(fits_path),
        "n_chunks":    len(chunks),
        "n_timesteps": N,
    }


def denoise_tic(
    tic_id:    str,
    model:     NoiseToNoiseUNet,
    device:    torch.device,
    mc_passes: int = CFG.mc_dropout_passes,
) -> dict:
    """
    Find the FITS file for a given TIC ID in data/raw/tess/ and denoise it.

    Uses the first matching file found across all sectors.
    """
    fits_files = list(CFG.data_raw_dir.rglob(f"*{tic_id.zfill(16)}*.fits"))

    if not fits_files:
        # Try without zero-padding
        fits_files = list(CFG.data_raw_dir.rglob(f"*{tic_id}*.fits"))

    if not fits_files:
        raise FileNotFoundError(
            f"No FITS file found for TIC {tic_id} in {CFG.data_raw_dir}\n"
            "Download data first: see docs/setup/data_download.md"
        )

    fits_path = fits_files[0]
    logger.info(f"TIC {tic_id} → {fits_path.name}")

    return denoise_fits(fits_path, model, device, mc_passes)


# ─────────────────────────────────────────────────────────────────────────────
# Plot result
# ─────────────────────────────────────────────────────────────────────────────

def plot_result(result: dict, save_path: Path = None) -> None:
    """
    Three-panel plot:
      Panel 1 : noisy input
      Panel 2 : denoised output with ±1σ uncertainty band
      Panel 3 : uncertainty (std) over time
    """
    noisy       = result["noisy"]
    denoised    = result["denoised"]
    uncertainty = result["uncertainty"]
    t           = np.arange(len(noisy))

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(3, 1, hspace=0.5)

    # Panel 1: noisy
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(t, noisy, color="#64748B", linewidth=0.6, alpha=0.85)
    ax0.set_ylabel("Norm. Flux")
    ax0.set_title("Input — Noisy TESS light curve (preprocessed)")
    ax0.set_xlim(0, len(noisy))

    # Panel 2: denoised + uncertainty band
    ax1 = fig.add_subplot(gs[1])
    ax1.plot(t, denoised, color="#2563EB", linewidth=1.0, label="Denoised")
    ax1.fill_between(
        t,
        denoised - uncertainty,
        denoised + uncertainty,
        alpha=0.25,
        color="#2563EB",
        label="±1σ uncertainty",
    )
    ax1.set_ylabel("Norm. Flux")
    ax1.set_title("Output — Noise2Noise U-Net denoised (MC Dropout ±1σ)")
    ax1.set_xlim(0, len(denoised))
    ax1.legend(loc="upper right", fontsize=8)

    # Panel 3: uncertainty
    ax2 = fig.add_subplot(gs[2])
    ax2.plot(t, uncertainty, color="#DC2626", linewidth=0.7)
    ax2.set_ylabel("Std (σ)")
    ax2.set_xlabel("Time step")
    ax2.set_title("Per-timestep uncertainty (MC Dropout std over 50 passes)")
    ax2.set_xlim(0, len(uncertainty))

    fig.suptitle(
        f"Vyom Denoiser — {Path(result['fits_path']).name}",
        fontsize=12, fontweight="bold", y=1.01,
    )

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Plot saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Public API — used by pipeline/run_pipeline.py
# ─────────────────────────────────────────────────────────────────────────────

def run_denoiser(
    fits_path:    Path,
    weights_path: Path = CFG.best_weights,
    mc_passes:    int  = CFG.mc_dropout_passes,
) -> dict:
    """
    Minimal public API for pipeline integration.

    Called by pipeline/run_pipeline.py — returns result dict.
    Does NOT plot (pipeline handles visualisation separately).

    Args:
        fits_path    : path to preprocessed TESS FITS file
        weights_path : trained model weights
        mc_passes    : MC Dropout passes for uncertainty

    Returns:
        dict with noisy, denoised, uncertainty arrays
    """
    model, device = load_model(weights_path)
    return denoise_fits(fits_path, model, device, mc_passes)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Vyom denoiser on a single star")

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--tic",  type=str, help="TIC ID of the star")
    group.add_argument("--fits", type=str, help="Direct path to FITS file")

    p.add_argument("--weights",    type=str, default=str(CFG.best_weights),
                   help="Path to trained weights (.pth)")
    p.add_argument("--mc-passes", type=int, default=CFG.mc_dropout_passes,
                   help="MC Dropout forward passes (default 50)")
    p.add_argument("--no-plot",   action="store_true",
                   help="Skip plot — just print summary stats")
    p.add_argument("--save-plot", type=str, default=None,
                   help="Save plot to this path instead of showing")

    return p.parse_args()


if __name__ == "__main__":
    args  = _parse_args()
    model, device = load_model(Path(args.weights))

    if args.tic:
        result = denoise_tic(args.tic, model, device, mc_passes=args.mc_passes)
    else:
        result = denoise_fits(Path(args.fits), model, device, mc_passes=args.mc_passes)

    # Summary
    logger.info("=" * 45)
    logger.info(f"  Time steps   : {result['n_timesteps']}")
    logger.info(f"  Chunks       : {result['n_chunks']}")
    logger.info(f"  Noisy std    : {result['noisy'].std():.4f}")
    logger.info(f"  Denoised std : {result['denoised'].std():.4f}")
    logger.info(f"  Mean uncertainty : {result['uncertainty'].mean():.4f}")
    logger.info("=" * 45)

    if not args.no_plot:
        save_path = Path(args.save_plot) if args.save_plot else None
        plot_result(result, save_path=save_path)
