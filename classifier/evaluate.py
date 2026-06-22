"""
classifier/evaluate.py

Evaluation script for the trained six-class transit classifier.

Metrics computed (all saved to results/classifier/metrics.json):
  Per-class : Precision, Recall, F1-score, ROC-AUC, PR-AUC
  Overall   : Macro-F1 (primary), Weighted-F1, Accuracy, Mean-ROC-AUC, Mean-PR-AUC

Plots saved to results/classifier/:
  - confusion_matrix.png    ← most important — 6×6 normalised + raw count annotation
  - roc_curves.png          ← 6 ROC curves (one vs rest), AUC in legend
  - pr_curves.png           ← 6 PR curves — critical for imbalanced class evaluation
  - loss_curve.png          ← train_loss vs val_loss from train_log.csv (both phases)
  - f1_curve.png            ← val macro-F1 over epochs from train_log.csv

Why PR curves matter here:
  Planet Transits (class 0) are rare in any catalog. ROC-AUC is optimistic for
  imbalanced classes because it counts true negatives, which are always large
  when the positive class is rare. PR-AUC does not use TN — it directly shows
  how well the model retrieves rare positives. A high PR-AUC on PT means the
  model actually finds real planets without drowning in FPs.

Usage:
  python -m classifier.evaluate
  python -m classifier.evaluate --weights weights/classifier_best.pth
  python -m classifier.evaluate --source kepler
  python -m classifier.evaluate --phase 1   # evaluate Kepler pretrained weights
"""

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.preprocessing import label_binarize

from .config import CFG
from .model import TransitClassifier
from .dataset import get_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Plot style — publication-ready ────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
})

# Consistent colour per class — used in all plots
CLASS_COLORS = [
    "#2563EB",  # 0 PT  — blue
    "#DC2626",  # 1 EB  — red
    "#D97706",  # 2 BEB — amber
    "#7C3AED",  # 3 HEB — purple
    "#059669",  # 4 SV  — emerald
    "#374151",  # 5 IA  — slate
]


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(
    weights_path: Path,
    device:       torch.device,
) -> TransitClassifier:
    """
    Load TransitClassifier from a .pth checkpoint.

    Args:
        weights_path : path to trained weights file
        device       : torch device

    Returns:
        model with loaded weights in eval mode
    """
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights_path}\n"
            "Train first:\n"
            "  python -m classifier.train              # both phases\n"
            "  python -m classifier.train --phase 1   # Kepler only\n"
            "  python -m classifier.train --phase 2   # TESS only (needs phase 1 first)"
        )

    model = TransitClassifier().to(device)
    ckpt  = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    logger.info(
        f"Loaded: {weights_path.name}  "
        f"(epoch={ckpt.get('epoch', '?')}, "
        f"phase={ckpt.get('phase', '?')}, "
        f"best_val_f1={ckpt.get('best_val_f1', '?')})"
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Prediction collection
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(
    model:  TransitClassifier,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model on an entire DataLoader in eval mode (no dropout, no gradients).

    Returns:
        all_probs  : [N, num_classes] float32 — per-class softmax probabilities
        all_preds  : [N]             int32    — argmax predicted class indices
        all_labels : [N]             int32    — true class labels
    """
    model.eval()

    all_probs:  list = []
    all_preds:  list = []
    all_labels: list = []

    pbar = tqdm(loader, desc="Evaluating test set", unit="batch")

    for x, y in pbar:
        x = x.to(device, non_blocking=True)   # [B, 1, 200]

        logits = model(x)                           # [B, num_classes]
        probs  = F.softmax(logits, dim=-1).cpu()   # [B, num_classes]
        preds  = probs.argmax(dim=1)               # [B]

        all_probs.extend(probs.numpy())
        all_preds.extend(preds.numpy())
        all_labels.extend(y.numpy())

        pbar.set_postfix({"n": len(all_labels)})

    all_probs_arr  = np.array(all_probs,  dtype=np.float32)  # [N, C]
    all_preds_arr  = np.array(all_preds,  dtype=np.int32)    # [N]
    all_labels_arr = np.array(all_labels, dtype=np.int32)    # [N]

    # Guard: filter out ignored labels (-1) — shouldn't occur after dataset filtering
    # but defensive check keeps metrics code clean
    valid = all_labels_arr >= 0
    if not valid.all():
        n_bad = int((~valid).sum())
        logger.warning(f"Filtering {n_bad} samples with label=-1 (ignore_index)")
        all_probs_arr  = all_probs_arr[valid]
        all_preds_arr  = all_preds_arr[valid]
        all_labels_arr = all_labels_arr[valid]

    return all_probs_arr, all_preds_arr, all_labels_arr


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    all_probs:  np.ndarray,   # [N, num_classes]
    all_preds:  np.ndarray,   # [N]
    all_labels: np.ndarray,   # [N]
) -> dict:
    """
    Compute the full suite of classification metrics.

    Args:
        all_probs  : softmax class probabilities from model
        all_preds  : predicted class indices (argmax of probs)
        all_labels : ground truth class indices

    Returns:
        metrics dict with keys:
          per_class    : list of dicts (one per class)
          macro_f1     : macro-averaged F1 (primary metric)
          weighted_f1  : frequency-weighted F1
          accuracy     : fraction of correct predictions
          mean_roc_auc : mean ROC-AUC across all classes
          mean_pr_auc  : mean PR-AUC across all classes
          n_samples    : total test set size
          class_counts : {class_name: count} distribution
    """
    class_labels = list(range(CFG.num_classes))

    # ── Per-class precision / recall / F1 / support ───────────────────────
    prec, rec, f1, support = precision_recall_fscore_support(
        all_labels, all_preds,
        labels      = class_labels,
        average     = None,
        zero_division = 0,
    )

    # ── Overall summary metrics ───────────────────────────────────────────
    macro_f1    = float(np.mean(f1))
    total_sup   = float(support.sum())
    weighted_f1 = float(np.dot(f1, support) / total_sup) if total_sup > 0 else 0.0
    accuracy    = float(accuracy_score(all_labels, all_preds))

    # ── ROC-AUC and PR-AUC per class (one vs rest) ────────────────────────
    labels_bin = label_binarize(all_labels, classes=class_labels)   # [N, num_classes]

    roc_aucs: list[float] = []
    pr_aucs:  list[float] = []

    for c in class_labels:
        y_true_c  = labels_bin[:, c]
        y_score_c = all_probs[:, c]

        n_pos = int(y_true_c.sum())
        n_neg = int((~y_true_c.astype(bool)).sum())

        if n_pos == 0 or n_neg == 0:
            logger.warning(
                f"Class {c} ({CFG.get_class_name(c)}): "
                f"pos={n_pos}, neg={n_neg} in test set — "
                "ROC-AUC and PR-AUC set to 0.0 (undefined)"
            )
            roc_aucs.append(0.0)
            pr_aucs.append(0.0)
        else:
            roc_aucs.append(float(roc_auc_score(y_true_c, y_score_c)))
            pr_aucs.append(float(average_precision_score(y_true_c, y_score_c)))

    # ── Assemble per-class list ───────────────────────────────────────────
    per_class = [
        {
            "class_idx":       c,
            "class_name":      CFG.get_class_name(c),
            "class_full_name": CFG.get_class_name(c, full=True),
            "precision":       round(float(prec[c]),    4),
            "recall":          round(float(rec[c]),     4),
            "f1":              round(float(f1[c]),      4),
            "roc_auc":         round(roc_aucs[c],       4),
            "pr_auc":          round(pr_aucs[c],        4),
            "support":         int(support[c]),
        }
        for c in class_labels
    ]

    return {
        "per_class":    per_class,
        "macro_f1":     round(macro_f1,             4),
        "weighted_f1":  round(weighted_f1,           4),
        "accuracy":     round(accuracy,              4),
        "mean_roc_auc": round(float(np.mean(roc_aucs)), 4),
        "mean_pr_auc":  round(float(np.mean(pr_aucs)),  4),
        "n_samples":    int(len(all_labels)),
        "class_counts": {
            CFG.get_class_name(c): int(support[c]) for c in class_labels
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Print table
# ─────────────────────────────────────────────────────────────────────────────

def print_metrics_table(metrics: dict) -> None:
    """Print a clean per-class metrics table to stdout + quality gate messages."""
    print("\n" + "=" * 75)
    print("  VYOM CLASSIFIER — EVALUATION RESULTS")
    print("=" * 75)
    print(f"  Test samples  : {metrics['n_samples']}")
    print(f"  Accuracy      : {metrics['accuracy']:.4f}")
    print(f"  Macro F1      : {metrics['macro_f1']:.4f}   ← primary metric")
    print(f"  Weighted F1   : {metrics['weighted_f1']:.4f}")
    print(f"  Mean ROC-AUC  : {metrics['mean_roc_auc']:.4f}")
    print(f"  Mean PR-AUC   : {metrics['mean_pr_auc']:.4f}")
    print()

    hdr = (
        f"  {'Idx':<4} {'Name':<32} "
        f"{'Prec':>6} {'Rec':>6} {'F1':>6} "
        f"{'ROC-AUC':>8} {'PR-AUC':>7} {'N':>6}"
    )
    print(hdr)
    print("  " + "-" * 73)

    for pc in metrics["per_class"]:
        print(
            f"  [{pc['class_idx']}]  "
            f"{pc['class_full_name']:<32} "
            f"{pc['precision']:>6.4f} "
            f"{pc['recall']:>6.4f} "
            f"{pc['f1']:>6.4f} "
            f"{pc['roc_auc']:>8.4f} "
            f"{pc['pr_auc']:>7.4f} "
            f"{pc['support']:>6}"
        )

    print("=" * 75)

    # Quality gates
    mf1 = metrics["macro_f1"]
    if mf1 >= 0.80:
        print(f"  ✅ Macro-F1 = {mf1:.4f} — target met (≥ 0.80)")
    elif mf1 >= 0.70:
        print(f"  ⚠️  Macro-F1 = {mf1:.4f} — acceptable but below 0.80 target")
    else:
        print(f"  ❌ Macro-F1 = {mf1:.4f} — below 0.70, retrain recommended")

    pt_f1 = metrics["per_class"][0]["f1"]
    pt_pr  = metrics["per_class"][0]["pr_auc"]
    if pt_f1 >= 0.75:
        print(f"  ✅ PT F1 = {pt_f1:.4f} — planet detection target met")
    else:
        print(
            f"  ⚠️  PT F1 = {pt_f1:.4f}, PR-AUC = {pt_pr:.4f} — "
            "low planet recall. Check class weights and PT sample count."
        )
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Plot — Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    all_labels: np.ndarray,
    all_preds:  np.ndarray,
    out_dir:    Path,
) -> None:
    """
    6×6 confusion matrix.

    Color = row-normalised fraction (shows per-class recall independent of support).
    Text = raw count (top) + normalised value (bottom) — both are informative.
    """
    class_labels = list(range(CFG.num_classes))
    cm      = confusion_matrix(all_labels, all_preds, labels=class_labels)
    row_sum = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm.astype(float) / row_sum

    short_names = [CFG.get_class_name(i) for i in class_labels]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.grid(False)

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall (normalised by true class)", fontsize=9)

    # Cell annotations: raw count + normalised fraction
    for i in range(CFG.num_classes):
        for j in range(CFG.num_classes):
            val  = cm_norm[i, j]
            txt_color = "white" if val > 0.55 else "black"
            ax.text(
                j, i,
                f"{cm[i, j]}\n{val:.2f}",
                ha="center", va="center",
                color=txt_color,
                fontsize=8,
                fontweight="bold" if i == j else "normal",
            )

    ax.set_xticks(class_labels)
    ax.set_yticks(class_labels)
    ax.set_xticklabels(short_names, fontsize=10)
    ax.set_yticklabels(short_names, fontsize=10)
    ax.set_xlabel("Predicted class", fontsize=11, labelpad=8)
    ax.set_ylabel("True class",      fontsize=11, labelpad=8)
    ax.set_title(
        "Vyom Classifier — Confusion Matrix\n"
        "(colour = row-normalised recall,  text = count / fraction)",
        fontsize=10, pad=10,
    )

    fig.tight_layout()
    path = out_dir / "confusion_matrix.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot — ROC Curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(
    all_labels: np.ndarray,
    all_probs:  np.ndarray,
    out_dir:    Path,
) -> None:
    """
    Six ROC curves (one vs rest) on one figure.
    Diagonal dashed line = random classifier baseline.
    AUC shown in legend for each class.
    """
    labels_bin = label_binarize(all_labels, classes=list(range(CFG.num_classes)))

    fig, ax = plt.subplots(figsize=(8, 7))

    for c in range(CFG.num_classes):
        y_true  = labels_bin[:, c]
        y_score = all_probs[:, c]

        if len(np.unique(y_true)) < 2:
            logger.debug(f"Class {c} has only one unique label in test set — skipping ROC")
            continue

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_val      = float(roc_auc_score(y_true, y_score))

        ax.plot(
            fpr, tpr,
            color     = CLASS_COLORS[c],
            linewidth = 2.2,
            label     = f"{CFG.get_class_name(c):<4}  AUC = {auc_val:.3f}",
        )

    # Chance-level baseline
    ax.plot(
        [0, 1], [0, 1],
        color="gray", linestyle="--", linewidth=1.2, alpha=0.6,
        label="Random  AUC = 0.500",
    )

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.set_title(
        "Vyom Classifier — ROC Curves (one vs rest, per class)",
        fontsize=10, pad=8,
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    path = out_dir / "roc_curves.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot — PR Curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_pr_curves(
    all_labels: np.ndarray,
    all_probs:  np.ndarray,
    out_dir:    Path,
) -> None:
    """
    Six Precision-Recall curves (one vs rest) on one figure.

    Each curve's legend entry also shows the class baseline (random AP =
    class prevalence in test set). A curve that stays well above its baseline
    indicates genuine discriminative power on that class.
    """
    labels_bin = label_binarize(all_labels, classes=list(range(CFG.num_classes)))

    fig, ax = plt.subplots(figsize=(8, 7))

    for c in range(CFG.num_classes):
        y_true  = labels_bin[:, c]
        y_score = all_probs[:, c]

        if len(np.unique(y_true)) < 2:
            logger.debug(f"Class {c} has only one unique label in test set — skipping PR")
            continue

        prec, rec, _ = precision_recall_curve(y_true, y_score)
        ap_val        = float(average_precision_score(y_true, y_score))
        prevalence    = float(y_true.mean())

        ax.plot(
            rec, prec,
            color     = CLASS_COLORS[c],
            linewidth = 2.2,
            label     = (
                f"{CFG.get_class_name(c):<4}  "
                f"AP = {ap_val:.3f}  "
                f"(base {prevalence:.3f})"
            ),
        )

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel("Recall",    fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(
        "Vyom Classifier — Precision-Recall Curves (one vs rest, per class)\n"
        "base = class prevalence in test set (random classifier AP)",
        fontsize=10, pad=8,
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    path = out_dir / "pr_curves.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot — Training Curves (loss + F1, both phases)
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(
    log_csv: Path,
    out_dir: Path,
) -> None:
    """
    Read results/classifier/train_log.csv and produce:
      loss_curve.png  — train vs val loss per row, with phase boundary
      f1_curve.png    — val macro-F1 per row, with phase boundary
    """
    if not log_csv.exists():
        logger.warning(f"train_log.csv not found: {log_csv} — skipping training curves")
        return

    phases:     list[int]   = []
    train_loss: list[float] = []
    val_loss:   list[float] = []
    val_f1:     list[float] = []

    with open(log_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                phases.append(int(row["phase"]))
                train_loss.append(float(row["train_loss"]))
                val_loss.append(float(row["val_loss"]))
                val_f1.append(float(row["val_f1_macro"]))
            except (KeyError, ValueError):
                continue

    if not phases:
        logger.warning("train_log.csv is empty or unreadable — skipping training curves")
        return

    x = list(range(len(phases)))

    # Locate phase 1→2 transition (first row where phase changes)
    phase_boundary: int | None = None
    for i in range(1, len(phases)):
        if phases[i] != phases[i - 1]:
            phase_boundary = i
            break

    def _add_boundary(ax: plt.Axes) -> None:
        if phase_boundary is not None:
            ax.axvline(
                phase_boundary - 0.5,
                color="gray", linestyle=":", linewidth=1.2,
                label="Phase 1 → Phase 2",
            )

    # ── Loss curve ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, train_loss, label="Train loss", color="#2563EB", linewidth=2.0)
    ax.plot(x, val_loss,   label="Val loss",   color="#DC2626", linewidth=2.0)
    _add_boundary(ax)
    ax.set_xlabel("Training step (log row)")
    ax.set_ylabel("Weighted CrossEntropy Loss")
    ax.set_title("Vyom Classifier — Training vs Validation Loss (both phases)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = out_dir / "loss_curve.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")

    # ── F1 curve ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, val_f1, color="#059669", linewidth=2.0, label="Val macro-F1")
    _add_boundary(ax)
    ax.set_xlabel("Training step (log row)")
    ax.set_ylabel("Macro-averaged F1")
    ax.set_title("Vyom Classifier — Validation Macro-F1 over Training")
    ax.set_ylim([-0.02, 1.05])
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = out_dir / "f1_curve.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluate function — public API
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    weights_path: Path = CFG.best_weights,
    source:       str  = "tess",
    batch_size:   int  = CFG.phase2_batch_size,
    out_dir:      Path = None,
) -> dict:
    """
    Full evaluation pipeline:
      1. Load model from weights
      2. Run model on test set, collect predictions
      3. Compute per-class and overall metrics
      4. Print formatted metrics table to stdout
      5. Save metrics.json
      6. Plot confusion matrix, ROC curves, PR curves
      7. Plot training curves from train_log.csv (if present)

    Args:
        weights_path : .pth checkpoint to evaluate  (default: classifier_best.pth)
        source       : "tess" (TESS TOI test set) or "kepler" (KOI test set)
        batch_size   : DataLoader batch size for inference
        out_dir      : directory to save plots and JSON
                       (default: CFG.results_dir = results/classifier/)

    Returns:
        full metrics dict (same as saved to metrics.json)

    Raises:
        FileNotFoundError : weights_path does not exist
        RuntimeError      : test set is empty
    """
    if out_dir is None:
        out_dir = CFG.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device      : {device}")
    logger.info(f"Weights     : {weights_path}")
    logger.info(f"Source      : {source}")
    logger.info(f"Output dir  : {out_dir}")

    # ── Load model ────────────────────────────────────────────────────────
    model = load_model(weights_path, device)
    logger.info(f"Total parameters : {model.count_all_parameters():,}")

    # ── Test data ─────────────────────────────────────────────────────────
    logger.info(f"Loading {source} test set...")
    _, _, test_loader = get_dataloaders(
        source              = source,
        batch_size          = batch_size,
        num_workers         = 0,
        force_rebuild       = False,
        use_weighted_sampler= False,   # evaluation: no resampling
    )

    n_test = len(test_loader.dataset)
    if n_test == 0:
        raise RuntimeError(
            f"Test set is empty for source='{source}'.\n"
            "Download and cache the dataset first:\n"
            "  python -m classifier.train --force-rebuild"
        )
    logger.info(f"Test samples : {n_test}")

    # ── Predictions ───────────────────────────────────────────────────────
    logger.info("Running inference on test set...")
    all_probs, all_preds, all_labels = collect_predictions(model, test_loader, device)

    # ── Metrics ───────────────────────────────────────────────────────────
    logger.info("Computing metrics...")
    metrics = compute_all_metrics(all_probs, all_preds, all_labels)

    # ── Print table ───────────────────────────────────────────────────────
    print_metrics_table(metrics)

    # ── Save JSON ─────────────────────────────────────────────────────────
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved : {metrics_path}")

    # ── Plots ─────────────────────────────────────────────────────────────
    logger.info("Generating plots...")
    plot_confusion_matrix(all_labels, all_preds, out_dir)
    plot_roc_curves(all_labels, all_probs, out_dir)
    plot_pr_curves(all_labels, all_probs, out_dir)

    log_csv = out_dir / "train_log.csv"
    plot_training_curves(log_csv, out_dir)

    logger.info(f"All outputs saved to: {out_dir}")
    logger.info("Evaluation complete.")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate Vyom six-class transit classifier on the held-out test set"
    )
    p.add_argument(
        "--weights", type=str, default=str(CFG.best_weights),
        help="Path to trained weights .pth  (default: weights/classifier_best.pth)",
    )
    p.add_argument(
        "--source", type=str, default="tess", choices=["tess", "kepler"],
        help="Which test set to use  (default: tess)",
    )
    p.add_argument(
        "--phase", type=int, default=None, choices=[1, 2],
        help=(
            "Shortcut: --phase 1 evaluates Kepler weights on Kepler test set; "
            "--phase 2 evaluates best weights on TESS test set."
        ),
    )
    p.add_argument(
        "--batch-size", type=int, default=CFG.phase2_batch_size,
        help="DataLoader batch size  (default: from CFG)",
    )
    p.add_argument(
        "--out-dir", type=str, default=None,
        help="Directory to save plots and metrics.json  (default: results/classifier/)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Phase shortcut resolves weights + source automatically
    if args.phase == 1:
        weights = CFG.kepler_weights
        source  = "kepler"
    elif args.phase == 2:
        weights = CFG.best_weights
        source  = "tess"
    else:
        weights = Path(args.weights)
        source  = args.source

    out_dir = Path(args.out_dir) if args.out_dir else CFG.results_dir

    evaluate(
        weights_path = weights,
        source       = source,
        batch_size   = args.batch_size,
        out_dir      = out_dir,
    )
