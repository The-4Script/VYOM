"""
denoiser/evaluate.py

Evaluation script for the trained Noise2Noise denoiser.

Metrics computed:
  - MSE, RMSE, MAE          (reconstruction error — lower is better)
  - PSNR in dB              (standard denoising metric — higher is better)
  - SNR improvement ratio   (SNR_denoised / SNR_noisy — target > 2.0x)
  - Noise reduction ratio   (std_noisy / std_denoised — how much noise removed)

Plots saved to results/denoiser/:
  - loss_curve.png          train vs val loss over epochs
  - psnr_curve.png          val PSNR over epochs
  - examples/               5 side-by-side noisy vs denoised light curve plots

Usage:
  python -m denoiser.evaluate
  python -m denoiser.evaluate --weights weights/denoiser_best.pth
  python -m denoiser.evaluate --n-examples 10
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

from .config import CFG
from .model import NoiseToNoiseUNet
from .dataset import get_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Plot style — clean, publication-ready ─────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":      150,
    "font.family":     "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid":       True,
    "grid.alpha":      0.3,
    "grid.linestyle":  "--",
})


# ─────────────────────────────────────────────────────────────────────────────
# Metric functions
# ─────────────────────────────────────────────────────────────────────────────

def compute_snr(signal: np.ndarray) -> float:
    """
    Signal-to-Noise Ratio of a 1D light curve.
    SNR = mean(|signal|) / std(signal)
    Higher = cleaner signal.
    """
    std = signal.std()
    if std == 0:
        return float("inf")
    return float(np.abs(signal).mean() / std)


def compute_metrics_batch(
    pred:   torch.Tensor,
    target: torch.Tensor,
) -> dict:
    """
    Compute all scalar metrics for one batch.

    Args:
        pred   : [B, 1, T] denoiser output (on CPU)
        target : [B, 1, T] noisy sector B

    Returns:
        dict with mse, rmse, mae, psnr, snr_improvement, noise_reduction
    """
    pred_np   = pred.squeeze(1).numpy()    # [B, T]
    target_np = target.squeeze(1).numpy()  # [B, T]

    mse  = float(np.mean((pred_np - target_np) ** 2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(pred_np - target_np)))
    psnr = float(10.0 * np.log10(1.0 / mse)) if mse > 0 else float("inf")

    # SNR improvement — compare denoised vs noisy (target as noisy reference)
    snr_noisy    = np.nanmean([compute_snr(target_np[i]) for i in range(len(target_np))])
    snr_denoised = np.nanmean([compute_snr(pred_np[i])   for i in range(len(pred_np))])
    if snr_noisy > 0 and np.isfinite(snr_noisy) and np.isfinite(snr_denoised):
        snr_ratio = float(snr_denoised / snr_noisy)
    else:
        snr_ratio = 0.0

    # Noise reduction — std of residual noisy vs std of residual denoised
    # residual = signal - smooth_trend (approximated by mean here)
    noise_noisy    = float(np.mean([target_np[i].std() for i in range(len(target_np))]))
    noise_denoised = float(np.mean([pred_np[i].std()   for i in range(len(pred_np))]))
    noise_reduction = float(noise_noisy / noise_denoised) if noise_denoised > 0 else 0.0

    return {
        "mse":             mse,
        "rmse":            rmse,
        "mae":             mae,
        "psnr":            psnr,
        "snr_improvement": snr_ratio,
        "noise_reduction": noise_reduction,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full test set evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_test_set(
    model:   nn.Module,
    device:  torch.device,
) -> dict:
    """
    Run model on entire test set and return averaged metrics.

    Returns:
        dict with all metrics averaged over test set
    """
    _, _, test_loader = get_dataloaders(batch_size=CFG.batch_size, num_workers=0)

    model.eval()

    all_mse  = []
    all_rmse = []
    all_mae  = []
    all_psnr = []
    all_snr  = []
    all_nr   = []

    pbar = tqdm(test_loader, desc="Evaluating test set")

    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)

        # Move to CPU for numpy metrics
        metrics = compute_metrics_batch(pred.cpu(), y.cpu())

        all_mse.append(metrics["mse"])
        all_rmse.append(metrics["rmse"])
        all_mae.append(metrics["mae"])
        all_psnr.append(metrics["psnr"])
        all_snr.append(metrics["snr_improvement"])
        all_nr.append(metrics["noise_reduction"])

        pbar.set_postfix({"psnr": f"{metrics['psnr']:.2f} dB"})

    results = {
        "mse":             float(np.mean(all_mse)),
        "rmse":            float(np.mean(all_rmse)),
        "mae":             float(np.mean(all_mae)),
        "psnr_db":         float(np.mean(all_psnr)),
        "snr_improvement": float(np.mean(all_snr)),
        "noise_reduction": float(np.mean(all_nr)),
        "n_batches":       len(test_loader),
    }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(log_csv: Path, out_dir: Path) -> None:
    """
    Read train_log.csv and plot:
      - loss_curve.png  : train_loss vs val_loss over epochs
      - psnr_curve.png  : val_psnr over epochs
    """
    import csv

    if not log_csv.exists():
        logger.warning(f"Log file not found: {log_csv} — skipping training curves")
        return

    epochs, train_loss, val_loss, val_psnr = [], [], [], []

    with open(log_csv) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            val_psnr.append(float(row["val_psnr"]))

    if len(epochs) == 0:
        logger.warning("train_log.csv is empty — skipping training curves")
        return

    # Loss curve
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_loss, label="Train loss", color="#2563EB", linewidth=2)
    ax.plot(epochs, val_loss,   label="Val loss",   color="#DC2626", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("VyomDenoiseLoss")
    ax.set_title("Denoiser — Training vs Validation Loss")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "loss_curve.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved: {path}")

    # PSNR curve
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, val_psnr, color="#059669", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Denoiser — Validation PSNR over Training")
    fig.tight_layout()
    path = out_dir / "psnr_curve.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved: {path}")


@torch.no_grad()
def plot_examples(
    model:      nn.Module,
    device:     torch.device,
    n_examples: int = 5,
    out_dir:    Path = None,
) -> None:
    """
    Plot n_examples side-by-side comparisons:
      top panel    : noisy input (sector A)
      bottom panel : denoised output

    Saves each as examples/example_NN_noisy_vs_denoised.png
    """
    if out_dir is None:
        out_dir = CFG.results_dir / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_loader = get_dataloaders(batch_size=1, num_workers=0)

    model.eval()
    count = 0

    for x, y in test_loader:
        if count >= n_examples:
            break

        x_dev  = x.to(device)
        pred   = model(x_dev).cpu()

        noisy    = x.squeeze().numpy()     # [T]
        denoised = pred.squeeze().numpy()  # [T]
        t        = np.arange(len(noisy))

        fig = plt.figure(figsize=(14, 6))
        gs  = gridspec.GridSpec(2, 1, hspace=0.4)

        # Noisy
        ax0 = fig.add_subplot(gs[0])
        ax0.plot(t, noisy, color="#64748B", linewidth=0.7, alpha=0.9)
        ax0.set_ylabel("Normalised Flux")
        ax0.set_title(f"Example {count+1:02d} — Noisy input (TESS sector A)")
        ax0.set_xlim(0, len(noisy))

        # Denoised
        ax1 = fig.add_subplot(gs[1])
        ax1.plot(t, denoised, color="#2563EB", linewidth=0.9)
        ax1.set_ylabel("Normalised Flux")
        ax1.set_xlabel("Time step")
        ax1.set_title(f"Example {count+1:02d} — Denoised output (Noise2Noise U-Net)")
        ax1.set_xlim(0, len(denoised))

        path = out_dir / f"example_{count+1:02d}_noisy_vs_denoised.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {path}")

        count += 1


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluate function
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    weights_path: Path = CFG.best_weights,
    n_examples:   int  = 5,
) -> dict:
    """
    Full evaluation pipeline:
      1. Load model from weights
      2. Run on test set → compute all metrics
      3. Save metrics to results/denoiser/metrics.json
      4. Plot training curves (loss + psnr)
      5. Plot n_examples noisy vs denoised comparisons

    Args:
        weights_path : path to .pth file (default: denoiser_best.pth)
        n_examples   : number of example plots to generate

    Returns:
        metrics dict
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights_path}\n"
            "Train first: python -m denoiser.train"
        )

    model = NoiseToNoiseUNet().to(device)
    ckpt  = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    logger.info(f"Loaded weights from {weights_path}  (epoch {ckpt.get('epoch', '?')})")

    # ── Metrics ───────────────────────────────────────────────────────────
    logger.info("Computing metrics on test set...")
    metrics = evaluate_test_set(model, device)

    logger.info("=" * 50)
    logger.info("  DENOISER EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"  MSE              : {metrics['mse']:.6f}")
    logger.info(f"  RMSE             : {metrics['rmse']:.6f}")
    logger.info(f"  MAE              : {metrics['mae']:.6f}")
    logger.info(f"  PSNR             : {metrics['psnr_db']:.2f} dB")
    logger.info(f"  SNR improvement  : {metrics['snr_improvement']:.2f}x")
    logger.info(f"  Noise reduction  : {metrics['noise_reduction']:.2f}x")
    logger.info("=" * 50)

    # Check SNR improvement target
    if metrics["snr_improvement"] >= 2.0:
        logger.info("  ✅ SNR improvement target met (> 2.0x)")
    else:
        logger.warning(f"  ⚠️  SNR improvement {metrics['snr_improvement']:.2f}x < 2.0x target")

    # Save metrics JSON
    metrics_path = CFG.results_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved: {metrics_path}")

    # ── Training curves ───────────────────────────────────────────────────
    log_csv = CFG.results_dir / "train_log.csv"
    plot_training_curves(log_csv, CFG.results_dir)

    # ── Example plots ─────────────────────────────────────────────────────
    logger.info(f"Generating {n_examples} example plots...")
    plot_examples(model, device, n_examples=n_examples)

    logger.info("Evaluation complete.")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Vyom Noise2Noise denoiser")
    p.add_argument("--weights",    type=str, default=str(CFG.best_weights),
                   help="Path to .pth weights file")
    p.add_argument("--n-examples", type=int, default=5,
                   help="Number of example plots to generate")
    return p.parse_args()


if __name__ == "__main__":
    args  = _parse_args()
    evaluate(
        weights_path=Path(args.weights),
        n_examples=args.n_examples,
    )
