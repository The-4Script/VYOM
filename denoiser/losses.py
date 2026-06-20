"""
denoiser/losses.py

Three loss classes:
  1. CombinedLoss          — 0.8 * MSE + 0.2 * MAE
  2. TransitPreservationLoss — MSE with 5x extra weight on transit dip regions
  3. VyomDenoiseLoss        — 0.7 * Combined + 0.3 * TransitPreservation

Import only VyomDenoiseLoss in train.py — it wraps everything.
"""

import torch
import torch.nn as nn
from .config import CFG


# ─────────────────────────────────────────────────────────────────────────────
class CombinedLoss(nn.Module):
    """
    Weighted combination of MSE and MAE.

    MSE penalises large errors heavily (good for catching big noise spikes).
    MAE is more robust to outliers (good for rare cosmic ray artifacts).
    Together they balance sensitivity and robustness.

    Loss = mse_weight * MSE(pred, target) + (1 - mse_weight) * MAE(pred, target)
    Default: 0.8 * MSE + 0.2 * MAE
    """

    def __init__(self, mse_weight: float = CFG.mse_weight):
        super().__init__()
        self.mse_weight = mse_weight
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred   : [B, 1, T] — denoiser output
            target : [B, 1, T] — noisy sector B (Noise2Noise target)
        Returns:
            scalar loss value
        """
        return self.mse_weight * self.mse(pred, target) + \
               (1.0 - self.mse_weight) * self.mae(pred, target)


# ─────────────────────────────────────────────────────────────────────────────
class TransitPreservationLoss(nn.Module):
    """
    MSE loss with extra penalty on transit dip regions.

    Problem this solves:
      A naive denoiser treats a transit dip the same as noise — both are
      deviations from the mean. It will smooth out real transits while trying
      to remove noise. This is catastrophic for our pipeline.

    Solution:
      Identify dip regions (flux < mean flux) and apply dip_weight x higher
      penalty there. The model learns: "smoothing noise is okay, but smoothing
      dips is very costly."

    Weight map:
      - flux >= mean  →  weight = 1.0   (normal region)
      - flux <  mean  →  weight = dip_weight (default 5.0)

    Note: we use TARGET flux to identify dip regions, not pred.
    Using pred would create a feedback loop during training.
    """

    def __init__(self, dip_weight: float = CFG.transit_dip_weight):
        super().__init__()
        self.dip_weight = dip_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred   : [B, 1, T]
            target : [B, 1, T]
        Returns:
            scalar weighted MSE loss
        """
        # Mean flux per light curve — shape [B, 1, 1] for broadcasting
        mean_flux = target.mean(dim=2, keepdim=True)

        # Weight map: 1.0 everywhere, dip_weight where flux < mean
        # shape: [B, 1, T]
        weight_map = torch.ones_like(target)
        weight_map[target < mean_flux] = self.dip_weight

        # Weighted MSE — manually computed so we can apply per-element weights
        squared_error = (pred - target) ** 2
        weighted_loss = (weight_map * squared_error).mean()

        return weighted_loss


# ─────────────────────────────────────────────────────────────────────────────
class VyomDenoiseLoss(nn.Module):
    """
    Final training loss for the Vyom Noise2Noise denoiser.

    Loss = alpha * CombinedLoss + (1 - alpha) * TransitPreservationLoss
    Default: 0.7 * Combined + 0.3 * TransitPreservation

    Why this split:
      CombinedLoss drives overall signal reconstruction quality.
      TransitPreservationLoss ensures transit dips are never smoothed away.
      0.7/0.3 gives reconstruction quality priority while still enforcing
      transit preservation as a hard constraint through training.

    Usage:
      criterion = VyomDenoiseLoss()
      loss = criterion(pred, target)
      loss.backward()
    """

    def __init__(
        self,
        alpha: float      = CFG.loss_alpha,
        mse_weight: float = CFG.mse_weight,
        dip_weight: float = CFG.transit_dip_weight,
    ):
        super().__init__()
        self.alpha = alpha
        self.combined    = CombinedLoss(mse_weight=mse_weight)
        self.transit_pres = TransitPreservationLoss(dip_weight=dip_weight)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            pred   : [B, 1, T] — denoiser output
            target : [B, 1, T] — Noise2Noise target (noisy sector B)

        Returns:
            total_loss : scalar tensor (call .backward() on this)
            components : dict with individual loss values for logging
                         keys: 'combined', 'transit_pres', 'total'
        """
        l_combined    = self.combined(pred, target)
        l_transit_pres = self.transit_pres(pred, target)

        total = self.alpha * l_combined + (1.0 - self.alpha) * l_transit_pres

        components = {
            "combined":     l_combined.item(),
            "transit_pres": l_transit_pres.item(),
            "total":        total.item(),
        }

        return total, components
