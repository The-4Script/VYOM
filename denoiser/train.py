"""
denoiser/train.py

Training loop for the Noise2Noise 1D U-Net denoiser.

Features:
  - AdamW optimiser + CosineAnnealingWarmRestarts scheduler
  - Gradient clipping (max_norm=1.0)
  - Early stopping (patience=20 epochs)
  - Saves denoiser_best.pth  on every val loss improvement
  - Saves denoiser_last.pth  every epoch (crash recovery)
  - tqdm progress bars per batch
  - CSV log: epoch, train_loss, val_loss, val_psnr, lr
  - Resume from checkpoint support

Usage:
  python -m denoiser.train
  python -m denoiser.train --resume weights/denoiser_last.pth
  python -m denoiser.train --epochs 50 --batch-size 16
"""

import argparse
import csv
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from .config import CFG
from .model import NoiseToNoiseUNet
from .losses import VyomDenoiseLoss
from .dataset import get_dataloaders

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PSNR utility
# ─────────────────────────────────────────────────────────────────────────────

def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Peak Signal-to-Noise Ratio in dB — standard denoising quality metric.

    PSNR = 10 * log10(MAX² / MSE)
    Here MAX = 1.0 because light curves are normalised to unit MAD.
    Higher is better. Typical improvement: +2 to +5 dB is significant.

    Args:
        pred   : [B, 1, T] denoiser output
        target : [B, 1, T] clean reference (sector B, used as proxy)
    Returns:
        PSNR in dB (float)
    """
    with torch.no_grad():
        mse = nn.functional.mse_loss(pred, target)
        if mse == 0:
            return float("inf")
        psnr = 10.0 * torch.log10(torch.tensor(1.0) / mse)
    return psnr.item()


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    path:          Path,
    model:         nn.Module,
    optimiser:     torch.optim.Optimizer,
    scheduler:     torch.optim.lr_scheduler._LRScheduler,
    epoch:         int,
    best_val_loss: float,
    cfg_dict:      dict,
) -> None:
    """Save full training state — model, optimiser, scheduler, metadata."""
    torch.save(
        {
            "epoch":          epoch,
            "model_state":    model.state_dict(),
            "optim_state":    optimiser.state_dict(),
            "sched_state":    scheduler.state_dict(),
            "best_val_loss":  best_val_loss,
            "cfg":            cfg_dict,
        },
        path,
    )


def load_checkpoint(
    path:      Path,
    model:     nn.Module,
    optimiser: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device:    torch.device,
) -> tuple[int, float]:
    """
    Load training state from checkpoint.

    Returns:
        (start_epoch, best_val_loss)
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimiser.load_state_dict(ckpt["optim_state"])
    scheduler.load_state_dict(ckpt["sched_state"])
    start_epoch    = ckpt["epoch"] + 1
    best_val_loss  = ckpt["best_val_loss"]
    logger.info(f"Resumed from {path}  (epoch {ckpt['epoch']}, "
                f"best_val_loss={best_val_loss:.6f})")
    return start_epoch, best_val_loss


# ─────────────────────────────────────────────────────────────────────────────
# CSV logger
# ─────────────────────────────────────────────────────────────────────────────

class CSVLogger:
    """Appends one row per epoch to a CSV file."""

    HEADERS = ["epoch", "train_loss", "val_loss", "val_psnr",
               "combined_loss", "transit_pres_loss", "lr", "epoch_time_s"]

    def __init__(self, path: Path):
        self.path = path
        # Write header if file doesn't exist OR is empty (e.g. after a crash)
        if not path.exists() or path.stat().st_size == 0:
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(self.HEADERS)

    def log(self, row: dict) -> None:
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.HEADERS)
            writer.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# One epoch — train
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: VyomDenoiseLoss,
    optimiser: torch.optim.Optimizer,
    device:    torch.device,
    epoch:     int,
) -> dict:
    """
    Run one full training epoch.

    Returns:
        dict with avg train_loss, combined_loss, transit_pres_loss
    """
    model.train()

    total_loss       = 0.0
    total_combined   = 0.0
    total_transit    = 0.0
    n_batches        = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [train]", leave=False)

    for x, y in pbar:
        x = x.to(device, non_blocking=True)   # [B, 1, T] — sector A (input)
        y = y.to(device, non_blocking=True)   # [B, 1, T] — sector B (target)

        optimiser.zero_grad(set_to_none=True)  # faster than zero_grad()

        pred = model(x)                        # [B, 1, T]
        loss, components = criterion(pred, y)

        loss.backward()

        # Gradient clipping — prevents exploding gradients in deep U-Net
        nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip_norm)

        optimiser.step()

        total_loss     += components["total"]
        total_combined += components["combined"]
        total_transit  += components["transit_pres"]
        n_batches      += 1

        pbar.set_postfix({
            "loss": f"{components['total']:.4f}",
            "tp":   f"{components['transit_pres']:.4f}",
        })

    return {
        "train_loss":        total_loss     / n_batches,
        "combined_loss":     total_combined / n_batches,
        "transit_pres_loss": total_transit  / n_batches,
    }


# ─────────────────────────────────────────────────────────────────────────────
# One epoch — validate
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: VyomDenoiseLoss,
    device:    torch.device,
    epoch:     int,
) -> dict:
    """
    Run validation — no gradients, no dropout.

    Returns:
        dict with avg val_loss, val_psnr
    """
    model.eval()

    total_loss = 0.0
    total_psnr = 0.0
    n_batches  = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [val]  ", leave=False)

    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)
        _, components = criterion(pred, y)
        psnr = compute_psnr(pred, y)

        total_loss += components["total"]
        total_psnr += psnr
        n_batches  += 1

        pbar.set_postfix({
            "val_loss": f"{components['total']:.4f}",
            "psnr":     f"{psnr:.2f} dB",
        })

    return {
        "val_loss": total_loss / n_batches,
        "val_psnr": total_psnr / n_batches,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Early stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stop training when val_loss stops improving.

    Args:
        patience  : epochs to wait after last improvement (default 20)
        min_delta : minimum improvement to count as improvement (default 1e-6)
    """

    def __init__(self, patience: int = CFG.early_stop_patience, min_delta: float = 1e-6):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_loss = float("inf")

    def step(self, val_loss: float) -> bool:
        """
        Returns True if training should stop.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            return False  # improved — continue
        else:
            self.counter += 1
            logger.info(f"EarlyStopping: no improvement for {self.counter}/{self.patience} epochs")
            return self.counter >= self.patience  # stop if patience exhausted


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(
    epochs:        int   = CFG.epochs,
    batch_size:    int   = CFG.batch_size,
    resume:        Path  = None,
    num_workers:   int   = 0,
    force_rebuild: bool  = False,
) -> NoiseToNoiseUNet:
    """
    Full training run for the Noise2Noise denoiser.

    Args:
        epochs        : max training epochs
        batch_size    : samples per batch
        resume        : path to checkpoint to resume from (optional)
        num_workers   : DataLoader workers (0 = safe on Windows)
        force_rebuild : rebuild dataset cache from FITS files

    Returns:
        Trained model (best weights loaded)
    """
    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Config summary ────────────────────────────────────────────────────
    CFG.summary()

    # ── Model ─────────────────────────────────────────────────────────────
    model = NoiseToNoiseUNet(
        base_channels=CFG.base_channels,
        se_reduction=CFG.se_reduction,
        bottleneck_dropout=CFG.bottleneck_dropout,
    ).to(device)

    logger.info(f"Model parameters: {model.count_parameters():,}")

    # ── Loss ──────────────────────────────────────────────────────────────
    criterion = VyomDenoiseLoss(
        alpha=CFG.loss_alpha,
        mse_weight=CFG.mse_weight,
        dip_weight=CFG.transit_dip_weight,
    )

    # ── Optimiser ─────────────────────────────────────────────────────────
    optimiser = AdamW(
        model.parameters(),
        lr=CFG.lr,
        weight_decay=CFG.weight_decay,
    )

    # ── Scheduler ─────────────────────────────────────────────────────────
    scheduler = CosineAnnealingWarmRestarts(
        optimiser,
        T_0=CFG.T_0,
        T_mult=CFG.T_mult,
        eta_min=CFG.eta_min,
    )

    # ── Data ──────────────────────────────────────────────────────────────
    logger.info("Loading datasets...")
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        force_rebuild=force_rebuild,
    )
    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch    = 0
    best_val_loss  = float("inf")

    if resume is not None:
        resume = Path(resume)
        if resume.exists():
            start_epoch, best_val_loss = load_checkpoint(
                resume, model, optimiser, scheduler, device
            )
        else:
            logger.warning(f"Resume path not found: {resume} — starting fresh")

    # ── Logging ───────────────────────────────────────────────────────────
    log_path = CFG.results_dir / "train_log.csv"
    csv_logger = CSVLogger(log_path)
    logger.info(f"Logging to: {log_path}")

    # ── Early stopping ────────────────────────────────────────────────────
    early_stop = EarlyStopping(patience=CFG.early_stop_patience)
    # Sync early stopping with resumed best loss
    early_stop.best_loss = best_val_loss

    # ── Training loop ─────────────────────────────────────────────────────
    logger.info(f"Starting training: epochs {start_epoch}–{epochs}")

    for epoch in range(start_epoch, epochs):
        t_start = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimiser, device, epoch
        )

        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device, epoch
        )

        # Scheduler step — after each epoch
        scheduler.step(epoch)
        current_lr = optimiser.param_groups[0]["lr"]

        # Timing
        epoch_time = time.time() - t_start

        # ── Log ───────────────────────────────────────────────────────────
        log_row = {
            "epoch":             epoch,
            "train_loss":        round(train_metrics["train_loss"],        6),
            "val_loss":          round(val_metrics["val_loss"],            6),
            "val_psnr":          round(val_metrics["val_psnr"],            4),
            "combined_loss":     round(train_metrics["combined_loss"],     6),
            "transit_pres_loss": round(train_metrics["transit_pres_loss"], 6),
            "lr":                round(current_lr,                         8),
            "epoch_time_s":      round(epoch_time,                         2),
        }
        csv_logger.log(log_row)

        logger.info(
            f"Epoch {epoch:03d} | "
            f"train={train_metrics['train_loss']:.4f} | "
            f"val={val_metrics['val_loss']:.4f} | "
            f"psnr={val_metrics['val_psnr']:.2f} dB | "
            f"lr={current_lr:.2e} | "
            f"{epoch_time:.1f}s"
        )

        # ── Save last checkpoint every epoch ──────────────────────────────
        save_checkpoint(
            CFG.last_weights,
            model, optimiser, scheduler,
            epoch, best_val_loss,
            cfg_dict=CFG.__dict__,
        )

        # ── Save best checkpoint on improvement ───────────────────────────
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            save_checkpoint(
                CFG.best_weights,
                model, optimiser, scheduler,
                epoch, best_val_loss,
                cfg_dict=CFG.__dict__,
            )
            logger.info(f"  ✅ Best model saved  (val_loss={best_val_loss:.6f})")

        # ── Early stopping check ───────────────────────────────────────────
        if early_stop.step(val_metrics["val_loss"]):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

    # ── Load best weights before returning ────────────────────────────────
    if CFG.best_weights.exists():
        ckpt = torch.load(CFG.best_weights, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        logger.info(f"Best weights loaded from {CFG.best_weights}")

    logger.info("Training complete.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Vyom Noise2Noise denoiser")
    p.add_argument("--epochs",         type=int,  default=CFG.epochs,     help="Max epochs")
    p.add_argument("--batch-size",     type=int,  default=CFG.batch_size, help="Batch size")
    p.add_argument("--resume",         type=str,  default=None,           help="Path to checkpoint")
    p.add_argument("--num-workers",    type=int,  default=0,              help="DataLoader workers")
    p.add_argument("--force-rebuild",  action="store_true",               help="Rebuild dataset cache")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        resume=Path(args.resume) if args.resume else None,
        num_workers=args.num_workers,
        force_rebuild=args.force_rebuild,
    )
