"""
classifier/losses.py

Loss function for the six-class transit classifier.

Classes:
  WeightedCrossEntropyLoss  — CrossEntropy with per-class weights
  compute_class_weights()   — derives weights from TOI catalog class frequencies

Why class weighting matters:
  Planet Transits (class 0) are rare in the TOI catalog — roughly 10–15% of
  all dispositions. Without correction, the model learns to rarely predict PT
  because predicting EB/FP gives lower average loss. Class weighting flips
  the gradient signal: rare classes get a larger per-sample penalty, forcing
  the model to take them seriously. This is critical — missing a real planet
  is far worse than a false positive for our use case.

Weight formula:
  w_i = N_total / (num_classes * N_i)
  where N_i = number of samples in class i, N_total = total samples.
  This is the standard sklearn balanced class weight formula.
  Rarer classes get higher weights; common classes get lower weights.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .config import CFG

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Class weight computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(
    toi_csv_path: Path = CFG.toi_catalog_path,
    disposition_col: str = "TFOPWG Disp",
    source: str = "toi",
    device: torch.device = None,
) -> torch.Tensor:
    """
    Compute per-class weights from a TOI or KOI catalog CSV.

    Reads the disposition column, maps each disposition to a class index
    using CFG, counts class frequencies, and returns balanced weights.

    Args:
        toi_csv_path    : path to toi_catalog.csv or kepler_koi.csv
        disposition_col : column name containing disposition strings
                          TOI: "TFOPWG Disp"  |  KOI: "koi_disposition"
        source          : "toi" or "kepler" — determines disposition mapping
        device          : torch device for the returned tensor

    Returns:
        [num_classes] float32 tensor of class weights
        Falls back to uniform weights if CSV not found or column missing.

    Example:
        weights = compute_class_weights()
        # weights[0] (PT) will be high — planet transits are rare
        # weights[1] (EB) will be lower — EBs are more common in TOI
    """
    if device is None:
        device = torch.device("cpu")

    # ── Fallback: uniform weights ─────────────────────────────────────────
    uniform = torch.ones(CFG.num_classes, dtype=torch.float32, device=device)

    if not toi_csv_path.exists():
        logger.warning(
            f"Catalog not found: {toi_csv_path}\n"
            "Using uniform class weights. Download TOI catalog first."
        )
        return uniform

    try:
        import csv

        class_counts = {i: 0 for i in range(CFG.num_classes)}
        n_unmapped   = 0

        with open(toi_csv_path, newline="", encoding="utf-8") as f:
            # Skip comment lines (TOI catalog has # header lines)
            lines = [line for line in f if not line.startswith("#")]

        reader = csv.DictReader(lines)

        # Check the column exists
        first_row = None
        rows = list(reader)
        if len(rows) == 0:
            logger.warning("Catalog CSV is empty — using uniform weights")
            return uniform

        # Try to find disposition column (case-insensitive search)
        available_cols = list(rows[0].keys())
        matched_col = None
        for col in available_cols:
            if col.strip().lower() == disposition_col.strip().lower():
                matched_col = col
                break

        if matched_col is None:
            # Try common alternatives
            for alt in ["TFOPWG Disp", "tfopwg_disp", "koi_disposition",
                        "Disposition", "disposition"]:
                if alt in available_cols:
                    matched_col = alt
                    break

        if matched_col is None:
            logger.warning(
                f"Column '{disposition_col}' not found in {toi_csv_path.name}.\n"
                f"Available: {available_cols[:10]}\n"
                "Using uniform class weights."
            )
            return uniform

        for row in rows:
            disp = row.get(matched_col, "").strip()
            if not disp:
                continue
            idx = CFG.get_class_index(disp, source=source)
            if idx == -1:
                n_unmapped += 1
                continue
            class_counts[idx] += 1

        total = sum(class_counts.values())
        if total == 0:
            logger.warning("No valid dispositions found — using uniform weights")
            return uniform

        if n_unmapped > 0:
            logger.warning(f"{n_unmapped} rows had unmappable dispositions — skipped")

        # Log class distribution
        logger.info("Class distribution from catalog:")
        for i in range(CFG.num_classes):
            pct = 100.0 * class_counts[i] / total if total > 0 else 0
            logger.info(f"  [{i}] {CFG.get_class_name(i):<5}  "
                        f"{class_counts[i]:>6} samples  ({pct:.1f}%)")

        # Balanced weight formula: w_i = N / (C * N_i)
        weights = []
        for i in range(CFG.num_classes):
            n_i = class_counts[i]
            if n_i == 0:
                logger.warning(
                    f"Class {i} ({CFG.get_class_name(i)}) has 0 samples — "
                    "assigning weight 1.0. Check catalog mappings."
                )
                weights.append(1.0)
            else:
                weights.append(total / (CFG.num_classes * n_i))

        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

        logger.info("Class weights (higher = rarer class):")
        for i, w in enumerate(weights):
            logger.info(f"  [{i}] {CFG.get_class_name(i):<5}  weight={w:.4f}")

        return weight_tensor

    except Exception as e:
        logger.warning(f"Failed to compute class weights: {e}\nUsing uniform weights.")
        return uniform


# ─────────────────────────────────────────────────────────────────────────────
# Weighted CrossEntropy Loss
# ─────────────────────────────────────────────────────────────────────────────

class WeightedCrossEntropyLoss(nn.Module):
    """
    CrossEntropyLoss with per-class weighting for imbalanced six-class problem.

    Wraps nn.CrossEntropyLoss with:
      - class_weights : [num_classes] tensor, computed from catalog frequencies
      - label_smoothing : optional smoothing (reduces overconfidence)
      - ignore_index : skip samples with label -1 (unmapped dispositions)

    Usage:
      criterion = WeightedCrossEntropyLoss()
      loss = criterion(logits, labels)
      loss.backward()

    Or with pre-computed weights:
      weights = compute_class_weights(toi_csv_path)
      criterion = WeightedCrossEntropyLoss(class_weights=weights)
    """

    def __init__(
        self,
        class_weights:    Optional[torch.Tensor] = None,
        label_smoothing:  float = CFG.label_smoothing,
        ignore_index:     int   = -1,
        toi_csv_path:     Path  = CFG.toi_catalog_path,
        auto_compute:     bool  = True,
    ):
        """
        Args:
            class_weights   : pre-computed [num_classes] weight tensor.
                              If None and auto_compute=True, computed from catalog.
            label_smoothing : smoothing factor (0.0 = no smoothing)
            ignore_index    : label value to ignore (default -1 = unmapped)
            toi_csv_path    : catalog path for auto weight computation
            auto_compute    : if True and class_weights is None, compute from catalog
        """
        super().__init__()

        self.label_smoothing = label_smoothing
        self.ignore_index    = ignore_index

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        elif auto_compute and CFG.use_class_weights:
            weights = compute_class_weights(toi_csv_path)
            self.register_buffer("class_weights", weights)
        else:
            self.register_buffer(
                "class_weights",
                torch.ones(CFG.num_classes, dtype=torch.float32)
            )

        self._build_criterion()

    def _build_criterion(self):
        """Build the underlying nn.CrossEntropyLoss with current weights."""
        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
            ignore_index=self.ignore_index,
            reduction="mean",
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits : [B, num_classes] — raw model output (before softmax)
                     NOTE: CrossEntropyLoss applies log-softmax internally.
                           Do NOT pass softmax outputs here.
            labels : [B] — integer class indices (0–5)
                           -1 values are ignored (ignore_index)

        Returns:
            scalar loss value
        """
        return self.criterion(logits, labels)

    def update_weights(self, new_weights: torch.Tensor) -> None:
        """
        Replace class weights mid-training (e.g. switching from Kepler to TESS data).

        Args:
            new_weights : [num_classes] float32 tensor
        """
        assert new_weights.shape == self.class_weights.shape, (
            f"Weight shape mismatch: got {new_weights.shape}, "
            f"expected {self.class_weights.shape}"
        )
        self.class_weights.copy_(new_weights)
        self._build_criterion()
        logger.info("Class weights updated.")

    def extra_repr(self) -> str:
        weights_str = ", ".join(
            f"{CFG.get_class_name(i)}={self.class_weights[i].item():.3f}"
            for i in range(CFG.num_classes)
        )
        return (
            f"num_classes={CFG.num_classes}, "
            f"label_smoothing={self.label_smoothing}, "
            f"weights=[{weights_str}]"
        )
