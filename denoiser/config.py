"""
denoiser/config.py
Single source of truth for all Noise2Noise denoiser hyperparameters.
Nothing is hardcoded anywhere else — always import CFG from here.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DenoiserConfig:

    # ── Model architecture ────────────────────────────────────────────────
    input_length: int         = 1000   # T — fixed chunk size (time steps)
    base_channels: int        = 64     # encoder block 1 output channels
                                       # doubles each level: 64→128→256→512→1024
    se_reduction: int         = 16     # Squeeze-and-Excitation reduction ratio
    bottleneck_dropout: float = 0.1    # dropout only in bottleneck

    # ── Loss weights ──────────────────────────────────────────────────────
    # VyomDenoiseLoss = alpha * CombinedLoss + (1-alpha) * TransitPreservationLoss
    loss_alpha: float         = 0.7

    # CombinedLoss = mse_weight * MSE + (1 - mse_weight) * MAE
    mse_weight: float         = 0.8

    # Extra weight on dip regions in TransitPreservationLoss
    transit_dip_weight: float = 5.0

    # ── Training ──────────────────────────────────────────────────────────
    batch_size: int           = 32
    epochs: int               = 150
    early_stop_patience: int  = 20

    lr: float                 = 1e-3
    weight_decay: float       = 1e-4
    grad_clip_norm: float     = 1.0    # max norm for gradient clipping

    # CosineAnnealingWarmRestarts
    T_0: int                  = 10
    T_mult: int               = 2
    eta_min: float            = 1e-6

    # ── Data ──────────────────────────────────────────────────────────────
    tess_sectors: list        = field(default_factory=lambda: [1, 2, 3, 4, 5])
    train_frac: float         = 0.70
    val_frac: float           = 0.15
    # test_frac                = 0.15 (implicit)

    chunk_length: int         = 1000   # length of each chunk from a light curve
    chunk_stride: int         = 500    # 50% overlap = more training samples

    # ── Paths ─────────────────────────────────────────────────────────────
    data_raw_dir: Path        = Path("data/raw/tess")
    samples_dir: Path         = Path("data/samples/denoiser")
    weights_dir: Path         = Path("weights")
    results_dir: Path         = Path("results/denoiser")

    best_weights: Path        = Path("weights/denoiser_best.pth")
    last_weights: Path        = Path("weights/denoiser_last.pth")

    # ── Inference ─────────────────────────────────────────────────────────
    mc_dropout_passes: int    = 50     # Monte Carlo Dropout forward passes

    def __post_init__(self):
        # Convert strings to Path if loaded from JSON/YAML
        for attr in ["data_raw_dir", "samples_dir", "weights_dir",
                     "results_dir", "best_weights", "last_weights"]:
            setattr(self, attr, Path(getattr(self, attr)))

        # Auto-create directories on first import
        for d in [self.data_raw_dir, self.samples_dir,
                  self.weights_dir, self.results_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def summary(self):
        """Print config as a clean table — call at start of every training run."""
        print("\n" + "="*50)
        print("  VYOM DENOISER CONFIG")
        print("="*50)
        for k, v in self.__dict__.items():
            print(f"  {k:<25} {v}")
        print("="*50 + "\n")


# Singleton — import this everywhere, never instantiate DenoiserConfig again
CFG = DenoiserConfig()
