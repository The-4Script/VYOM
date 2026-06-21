"""
classifier/train.py

Two-phase training for the Vyom six-class transit classifier.

Phase 1 — Kepler pretraining:
  Train all layers from scratch on Kepler KOI labeled data.
  Kepler has ~4× more confirmed planets than TESS TOI (larger, cleaner catalog).
  Same transit physics → feature-transferable. Save as classifier_kepler.pth.

Phase 2 — TESS fine-tuning:
  Load Phase 1 weights. Freeze CNN blocks (learned transit features transfer).
  Fine-tune BiLSTM + attention + classification head on TESS TOI data.
  Lower lr (1e-4 vs 5e-4). Save best as classifier_best.pth.

Hyderabad (handled separately in run_pipeline or ad-hoc):
  Load classifier_best.pth. Fine-tune all or partial layers. 10 epochs, lr=1e-5.

Features:
  - ReduceLROnPlateau scheduler (plateau-aware, better than cosine for classifier)
  - Early stopping (patience=15 Phase 1, patience=10 Phase 2)
  - Gradient clipping max_norm=1.0
  - tqdm progress bars per batch
  - CSV log per phase: epoch, train_loss, val_loss, val_f1_macro, lr
  - Checkpoint resume support
  - CNN freeze / unfreeze utilities

Usage:
  # Full two-phase run:
  python -m classifier.train

  # Phase 1 only (Kepler):
  python -m classifier.train --phase 1

  # Phase 2 only (resume from kepler weights):
  python -m classifier.train --phase 2

  # Resume Phase 1 from checkpoint:
  python -m classifier.train --phase 1 --resume weights/classifier_kepler.pth
"""

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from sklearn.metrics import f1_score

from .config import CFG
from .losses import WeightedCrossEntropyLoss, compute_class_weights
from .dataset import get_dataloaders

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CNN freeze / unfreeze — works on whatever model Durvesh builds
# Convention: CNN blocks are named starting with "cnn" in TransitClassifier
# ─────────────────────────────────────────────────────────────────────────────

def freeze_cnn_blocks(model: nn.Module) -> int:
    """
    Freeze all parameters in CNN blocks (names starting with 'cnn').
    Returns the number of parameters frozen.

    Why freeze CNN and not LSTM:
      CNN learns local morphological features — ingress/egress shape, dip width.
      These features are physically identical between Kepler and TESS transits.
      LSTM learns sequential context — how the dip fits in the broader light curve.
      This is more mission-specific (noise patterns, cadence, sector systematics).
      Fine-tuning only the LSTM+head lets the model adapt the context-reading
      without forgetting the low-level transit shape features.
    """
    n_frozen = 0
    for name, param in model.named_parameters():
        if name.startswith("cnn"):
            param.requires_grad = False
            n_frozen += param.numel()

    frozen_layers = [n for n, p in model.named_parameters()
                     if not p.requires_grad]
    logger.info(
        f"CNN frozen — {n_frozen:,} parameters locked "
        f"({len(frozen_layers)} parameter tensors)"
    )
    return n_frozen


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all parameters — used before Hyderabad fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    logger.info("All parameters unfrozen")


def count_trainable(model: nn.Module) -> int:
    """Return number of trainable (requires_grad=True) parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint save / load
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    path:        Path,
    model:       nn.Module,
    optimiser:   torch.optim.Optimizer,
    scheduler:   ReduceLROnPlateau,
    epoch:       int,
    best_val_f1: float,
    phase:       int,
) -> None:
    """Save full training state."""
    torch.save(
        {
            "epoch":       epoch,
            "phase":       phase,
            "model_state": model.state_dict(),
            "optim_state": optimiser.state_dict(),
            "sched_state": scheduler.state_dict(),
            "best_val_f1": best_val_f1,
        },
        path,
    )


def load_checkpoint(
    path:      Path,
    model:     nn.Module,
    optimiser: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    device:    torch.device,
) -> tuple[int, float, int]:
    """
    Load checkpoint. Returns (start_epoch, best_val_f1, phase).
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimiser.load_state_dict(ckpt["optim_state"])
    scheduler.load_state_dict(ckpt["sched_state"])

    start_epoch  = ckpt["epoch"] + 1
    best_val_f1  = ckpt.get("best_val_f1", 0.0)
    phase        = ckpt.get("phase", 1)

    logger.info(
        f"Resumed from {path}  "
        f"(epoch {ckpt['epoch']}, phase {phase}, best_f1={best_val_f1:.4f})"
    )
    return start_epoch, best_val_f1, phase


# ─────────────────────────────────────────────────────────────────────────────
# CSV logger — one shared file for both phases (phase column distinguishes them)
# ─────────────────────────────────────────────────────────────────────────────

class CSVLogger:
    HEADERS = [
        "phase", "epoch", "train_loss", "val_loss",
        "val_f1_macro", "val_acc", "lr", "epoch_time_s",
    ]

    def __init__(self, path: Path):
        self.path = path
        if not path.exists() or path.stat().st_size == 0:
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(self.HEADERS)

    def log(self, row: dict) -> None:
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.HEADERS).writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# Early stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stop when val_f1_macro stops improving.
    Tracks MAX (unlike denoiser which tracks MIN loss).
    """

    def __init__(self, patience: int = CFG.phase1_early_stop, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_f1   = 0.0

    def step(self, val_f1: float) -> bool:
        """Returns True if training should stop."""
        if val_f1 > self.best_f1 + self.min_delta:
            self.best_f1 = val_f1
            self.counter  = 0
            return False
        else:
            self.counter += 1
            logger.info(
                f"EarlyStopping: no F1 improvement for "
                f"{self.counter}/{self.patience} epochs "
                f"(best={self.best_f1:.4f})"
            )
            return self.counter >= self.patience


# ─────────────────────────────────────────────────────────────────────────────
# One epoch — train
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: WeightedCrossEntropyLoss,
    optimiser: torch.optim.Optimizer,
    device:    torch.device,
    epoch:     int,
    phase:     int,
) -> dict:
    """
    Run one training epoch.

    Returns:
        dict with avg train_loss
    """
    model.train()

    total_loss = 0.0
    n_batches  = 0

    pbar = tqdm(loader, desc=f"Ph{phase} Epoch {epoch:03d} [train]", leave=False)

    for x, y in pbar:
        x = x.to(device, non_blocking=True)   # [B, 1, 200]
        y = y.to(device, non_blocking=True)   # [B]

        optimiser.zero_grad(set_to_none=True)

        logits = model(x)                      # [B, 6]
        loss   = criterion(logits, y)

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip_norm)

        optimiser.step()

        total_loss += loss.item()
        n_batches  += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return {"train_loss": total_loss / max(n_batches, 1)}


# ─────────────────────────────────────────────────────────────────────────────
# One epoch — validate
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: WeightedCrossEntropyLoss,
    device:    torch.device,
    epoch:     int,
    phase:     int,
) -> dict:
    """
    Run validation epoch.

    Returns:
        dict with val_loss, val_f1_macro, val_acc
    """
    model.eval()

    total_loss = 0.0
    n_batches  = 0
    all_preds  = []
    all_labels = []

    pbar = tqdm(loader, desc=f"Ph{phase} Epoch {epoch:03d} [val]  ", leave=False)

    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)                      # [B, 6]
        loss   = criterion(logits, y)

        preds = logits.argmax(dim=1)           # [B]

        total_loss  += loss.item()
        n_batches   += 1
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(y.cpu().numpy().tolist())

        pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

    val_loss = total_loss / max(n_batches, 1)

    # Macro F1 — treats all 6 classes equally regardless of frequency
    # This is the primary metric for imbalanced classification
    f1_macro = float(f1_score(
        all_labels, all_preds,
        average="macro",
        zero_division=0,
    ))

    # Accuracy — secondary metric
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    acc     = correct / max(len(all_labels), 1)

    return {
        "val_loss":     val_loss,
        "val_f1_macro": f1_macro,
        "val_acc":      acc,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single-phase training loop
# ─────────────────────────────────────────────────────────────────────────────

def _run_phase(
    phase:         int,
    model:         nn.Module,
    train_loader:  torch.utils.data.DataLoader,
    val_loader:    torch.utils.data.DataLoader,
    criterion:     WeightedCrossEntropyLoss,
    device:        torch.device,
    save_path:     Path,
    csv_logger:    CSVLogger,
    epochs:        int,
    lr:            float,
    patience:      int,
    resume:        Optional[Path] = None,
) -> nn.Module:
    """
    Run one complete training phase.

    Args:
        phase        : 1 (Kepler) or 2 (TESS fine-tune)
        model        : model to train (may have CNN frozen for phase 2)
        train_loader : DataLoader for this phase's training data
        val_loader   : DataLoader for this phase's validation data
        criterion    : loss function (weights already set for this phase)
        device       : torch device
        save_path    : where to save best checkpoint (.pth)
        csv_logger   : shared CSV logger
        epochs       : max epochs for this phase
        lr           : initial learning rate
        patience     : early stopping patience
        resume       : checkpoint path to resume from (optional)

    Returns:
        model with best weights loaded
    """
    logger.info(f"\n{'='*55}")
    logger.info(f"  PHASE {phase} — {'Kepler pretraining' if phase == 1 else 'TESS fine-tuning'}")
    logger.info(f"  Trainable parameters: {count_trainable(model):,}")
    logger.info(f"  Max epochs: {epochs}  |  LR: {lr}  |  Patience: {patience}")
    logger.info(f"{'='*55}")

    optimiser = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=CFG.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimiser,
        mode      = "max",             # we maximise F1
        factor    = CFG.lr_scheduler_factor,
        patience  = CFG.lr_scheduler_patience,
        min_lr    = CFG.lr_scheduler_min_lr,
    )

    early_stop   = EarlyStopping(patience=patience)
    best_val_f1  = 0.0
    start_epoch  = 0

    # ── Resume ────────────────────────────────────────────────────────────
    if resume is not None and resume.exists():
        start_epoch, best_val_f1, _ = load_checkpoint(
            resume, model, optimiser, scheduler, device
        )
        early_stop.best_f1 = best_val_f1

    # ── Loop ──────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        t_start = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimiser, device, epoch, phase
        )
        val_metrics = validate(
            model, val_loader, criterion, device, epoch, phase
        )

        current_lr  = optimiser.param_groups[0]["lr"]
        epoch_time  = time.time() - t_start

        # Scheduler step — maximise val F1
        scheduler.step(val_metrics["val_f1_macro"])

        # ── Log ───────────────────────────────────────────────────────────
        log_row = {
            "phase":         phase,
            "epoch":         epoch,
            "train_loss":    round(train_metrics["train_loss"],    6),
            "val_loss":      round(val_metrics["val_loss"],        6),
            "val_f1_macro":  round(val_metrics["val_f1_macro"],    4),
            "val_acc":       round(val_metrics["val_acc"],         4),
            "lr":            round(current_lr,                     8),
            "epoch_time_s":  round(epoch_time,                     2),
        }
        csv_logger.log(log_row)

        logger.info(
            f"Ph{phase} Epoch {epoch:03d} | "
            f"train={train_metrics['train_loss']:.4f} | "
            f"val={val_metrics['val_loss']:.4f} | "
            f"f1={val_metrics['val_f1_macro']:.4f} | "
            f"acc={val_metrics['val_acc']:.3f} | "
            f"lr={current_lr:.2e} | "
            f"{epoch_time:.1f}s"
        )

        # ── Save last every epoch (crash recovery) ─────────────────────
        # Uses same path convention as denoiser: overwrite every epoch
        last_path = CFG.weights_dir / f"classifier_phase{phase}_last.pth"
        save_checkpoint(
            last_path, model, optimiser, scheduler,
            epoch, best_val_f1, phase
        )

        # ── Save best on F1 improvement ────────────────────────────────
        if val_metrics["val_f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["val_f1_macro"]
            save_checkpoint(
                save_path, model, optimiser, scheduler,
                epoch, best_val_f1, phase
            )
            logger.info(
                f"  ✅ Best model saved  "
                f"(val_f1={best_val_f1:.4f})"
            )

        # ── Early stopping ─────────────────────────────────────────────
        if early_stop.step(val_metrics["val_f1_macro"]):
            logger.info(f"Early stopping at epoch {epoch} (phase {phase})")
            break

    # ── Load best weights ─────────────────────────────────────────────────
    if save_path.exists():
        ckpt = torch.load(save_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        logger.info(
            f"Phase {phase} complete. "
            f"Best val_f1={best_val_f1:.4f}. "
            f"Weights: {save_path}"
        )

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Main training function — public API
# ─────────────────────────────────────────────────────────────────────────────

def train(
    phases:        list[int]      = [1, 2],
    resume_phase1: Optional[Path] = None,
    resume_phase2: Optional[Path] = None,
    num_workers:   int            = 0,
    force_rebuild: bool           = False,
) -> nn.Module:
    """
    Full two-phase classifier training.

    Args:
        phases        : which phases to run, e.g. [1], [2], or [1, 2]
        resume_phase1 : checkpoint to resume Phase 1 from (optional)
        resume_phase2 : checkpoint to resume Phase 2 from (optional)
                        if None and phase 2 is running, loads classifier_kepler.pth
        num_workers   : DataLoader workers
        force_rebuild : rebuild dataset cache from FITS files

    Returns:
        Trained model with best weights (from last phase that ran)
    """
    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    CFG.summary()

    # ── Import model (built by Durvesh — must exist before running train) ─
    try:
        from .model import TransitClassifier
    except ImportError as e:
        raise ImportError(
            "classifier/model.py not found or TransitClassifier not defined.\n"
            "Durvesh needs to build model.py first.\n"
            f"Original error: {e}"
        )

    # ── Model ─────────────────────────────────────────────────────────────
    model = TransitClassifier().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters (total): {total_params:,}")

    # ── Shared CSV logger ─────────────────────────────────────────────────
    log_path   = CFG.results_dir / "train_log.csv"
    csv_logger = CSVLogger(log_path)
    logger.info(f"Logging to: {log_path}")

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 1 — Kepler pretraining
    # ─────────────────────────────────────────────────────────────────────
    if 1 in phases:
        logger.info("\nLoading Kepler KOI data for Phase 1...")

        kepler_train, kepler_val, _ = get_dataloaders(
            source        = "kepler",
            batch_size    = CFG.phase1_batch_size,
            num_workers   = num_workers,
            force_rebuild = force_rebuild,
        )

        # Class weights from Kepler catalog
        kepler_weights = compute_class_weights(
            toi_csv_path    = CFG.kepler_koi_path,
            disposition_col = "koi_disposition",
            source          = "kepler",
            device          = device,
        )
        criterion_phase1 = WeightedCrossEntropyLoss(class_weights=kepler_weights)

        # All layers trainable for Phase 1
        unfreeze_all(model)

        model = _run_phase(
            phase        = 1,
            model        = model,
            train_loader = kepler_train,
            val_loader   = kepler_val,
            criterion    = criterion_phase1,
            device       = device,
            save_path    = CFG.kepler_weights,
            csv_logger   = csv_logger,
            epochs       = CFG.phase1_epochs,
            lr           = CFG.phase1_lr,
            patience     = CFG.phase1_early_stop,
            resume       = resume_phase1,
        )

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 2 — TESS fine-tuning
    # ─────────────────────────────────────────────────────────────────────
    if 2 in phases:
        logger.info("\nLoading TESS TOI data for Phase 2...")

        # Load Phase 1 weights if we didn't just train Phase 1
        if 1 not in phases:
            weights_to_load = resume_phase2 or CFG.kepler_weights
            if not weights_to_load.exists():
                raise FileNotFoundError(
                    f"Phase 2 needs Phase 1 weights at {weights_to_load}\n"
                    "Run Phase 1 first: python -m classifier.train --phase 1"
                )
            ckpt = torch.load(weights_to_load, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            logger.info(f"Loaded Phase 1 weights from {weights_to_load}")

        # Freeze CNN blocks — BiLSTM + attention + head remain trainable
        if CFG.phase2_freeze_cnn:
            freeze_cnn_blocks(model)

        logger.info(f"Phase 2 trainable parameters: {count_trainable(model):,}")

        tess_train, tess_val, _ = get_dataloaders(
            source        = "tess",
            batch_size    = CFG.phase2_batch_size,
            num_workers   = num_workers,
            force_rebuild = False,   # never force rebuild for Phase 2
        )

        # Class weights from TESS TOI catalog
        tess_weights = compute_class_weights(
            toi_csv_path    = CFG.toi_catalog_path,
            disposition_col = "TFOPWG Disp",
            source          = "toi",
            device          = device,
        )
        criterion_phase2 = WeightedCrossEntropyLoss(class_weights=tess_weights)

        model = _run_phase(
            phase        = 2,
            model        = model,
            train_loader = tess_train,
            val_loader   = tess_val,
            criterion    = criterion_phase2,
            device       = device,
            save_path    = CFG.best_weights,
            csv_logger   = csv_logger,
            epochs       = CFG.phase2_epochs,
            lr           = CFG.phase2_lr,
            patience     = CFG.phase2_early_stop,
            resume       = resume_phase2,
        )

    logger.info("\nTraining complete.")
    logger.info(f"Kepler weights : {CFG.kepler_weights}")
    logger.info(f"Best weights   : {CFG.best_weights}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Hyderabad fine-tuning helper — called at venue, not in main training flow
# ─────────────────────────────────────────────────────────────────────────────

def finetune_isro(
    isro_fits_dir:    Path,
    isro_labels_csv:  Path,
    freeze_cnn:       bool = CFG.hyderabad_freeze_cnn,
    num_workers:      int  = 0,
) -> nn.Module:
    """
    Fine-tune classifier on ISRO dataset at Hyderabad.
    Called in first 2 hours after reading ISRO data format.

    Decision tree:
      - ISRO data similar to TESS → freeze_cnn=False, fine-tune all layers
      - ISRO data very different  → freeze_cnn=True, retrain LSTM + head only

    Args:
        isro_fits_dir   : directory containing ISRO FITS files
        isro_labels_csv : CSV with columns [star_id, label] mapping to 0–5
        freeze_cnn      : whether to freeze CNN blocks (True = more conservative)
        num_workers     : DataLoader workers

    Returns:
        Fine-tuned model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Hyderabad fine-tuning — device: {device}")
    logger.info(f"ISRO FITS dir   : {isro_fits_dir}")
    logger.info(f"ISRO labels CSV : {isro_labels_csv}")
    logger.info(f"Freeze CNN      : {freeze_cnn}")

    from .model import TransitClassifier

    # Load best weights from Phase 2
    if not CFG.best_weights.exists():
        raise FileNotFoundError(
            f"Phase 2 weights not found at {CFG.best_weights}.\n"
            "Train Phase 2 first before Hyderabad fine-tuning."
        )

    model = TransitClassifier().to(device)
    ckpt  = torch.load(CFG.best_weights, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    logger.info(f"Loaded Phase 2 weights from {CFG.best_weights}")

    if freeze_cnn:
        freeze_cnn_blocks(model)
    else:
        unfreeze_all(model)

    # For ISRO, we build a minimal DataLoader directly from the provided labels CSV
    # This avoids changing the TOI/KOI catalog pipeline
    from .dataset import TOICatalogDataset, get_dataloaders
    logger.info(
        "ISRO DataLoader: using same TOICatalogDataset pipeline.\n"
        "Place ISRO FITS in data/raw/tess/ and ISRO labels in data/catalogs/toi_catalog.csv\n"
        "OR adapt data loading in first 30 minutes after reading ISRO format."
    )

    # Fine-tuning loop reuses _run_phase with Hyderabad hyperparameters
    # Use TESS dataloaders as proxy if ISRO data format matches
    tess_train, tess_val, _ = get_dataloaders(
        source="tess", batch_size=CFG.phase2_batch_size, num_workers=num_workers
    )

    tess_weights = compute_class_weights(device=device)
    criterion    = WeightedCrossEntropyLoss(class_weights=tess_weights)
    csv_logger   = CSVLogger(CFG.results_dir / "hyderabad_log.csv")

    isro_save_path = CFG.weights_dir / "classifier_isro.pth"

    model = _run_phase(
        phase        = 3,   # Hyderabad = phase 3 in CSV log
        model        = model,
        train_loader = tess_train,
        val_loader   = tess_val,
        criterion    = criterion,
        device       = device,
        save_path    = isro_save_path,
        csv_logger   = csv_logger,
        epochs       = CFG.hyderabad_epochs,
        lr           = CFG.hyderabad_lr,
        patience     = 5,   # aggressive early stopping at Hyderabad
    )

    logger.info(f"Hyderabad fine-tuning complete. Saved: {isro_save_path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Vyom six-class transit classifier")
    p.add_argument(
        "--phase",
        type=int,
        nargs="+",
        default=[1, 2],
        choices=[1, 2],
        help="Which phase(s) to run: 1, 2, or '1 2' for both (default: both)",
    )
    p.add_argument(
        "--resume-phase1",
        type=str,
        default=None,
        help="Checkpoint to resume Phase 1 from",
    )
    p.add_argument(
        "--resume-phase2",
        type=str,
        default=None,
        help="Checkpoint to resume Phase 2 from (default: uses classifier_kepler.pth)",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (0 = safe on Windows)",
    )
    p.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild dataset cache from FITS files",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        phases        = args.phase,
        resume_phase1 = Path(args.resume_phase1) if args.resume_phase1 else None,
        resume_phase2 = Path(args.resume_phase2) if args.resume_phase2 else None,
        num_workers   = args.num_workers,
        force_rebuild = args.force_rebuild,
    )
