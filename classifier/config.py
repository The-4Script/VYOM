"""
classifier/config.py
Single source of truth for all CNN-LSTM classifier hyperparameters.
Nothing is hardcoded anywhere else — always import CFG from here.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ClassifierConfig:

    # ── Class definitions ─────────────────────────────────────────────────
    # Six-class false positive taxonomy
    # Indices match model output softmax order — never reorder these
    num_classes: int = 6

    class_names: dict = field(default_factory=lambda: {
        0: "PT",   # Planet Transit          — real exoplanet
        1: "EB",   # Eclipsing Binary        — two stars, deep symmetric dips
        2: "BEB",  # Background EB           — background star in aperture
        3: "HEB",  # Hierarchical EB         — EB physically bound to target
        4: "SV",   # Stellar Variability     — rotation, pulsation, starspots
        5: "IA",   # Instrumental Artifact   — momentum dump, systematics
    })

    class_full_names: dict = field(default_factory=lambda: {
        0: "Planet Transit",
        1: "Eclipsing Binary",
        2: "Background Eclipsing Binary",
        3: "Hierarchical Eclipsing Binary",
        4: "Stellar Variability",
        5: "Instrumental Artifact",
    })

    # TOI disposition → class index mapping
    # TFOPWG_Disp column values from NASA Exoplanet Archive TOI catalog
    toi_disposition_map: dict = field(default_factory=lambda: {
        "PC":  0,   # Planet Candidate  → Planet Transit
        "KP":  0,   # Known Planet      → Planet Transit
        "CP":  0,   # Confirmed Planet  → Planet Transit
        "EB":  1,   # Eclipsing Binary  → Eclipsing Binary
        "BEB": 2,   # Background EB     → Background EB
        "HEB": 3,   # Hierarchical EB   → Hierarchical EB
        "SV":  4,   # Stellar Variability → Stellar Variability
        "FP":  1,   # Generic FP → EB (most common FP type; overridden if subtype known)
        "IA":  5,   # Instrumental Artifact → IA
    })

    # Kepler KOI disposition → class index mapping
    # koi_disposition column from NASA Exoplanet Archive KOI table
    kepler_disposition_map: dict = field(default_factory=lambda: {
        "CONFIRMED":        0,   # Confirmed planet
        "CANDIDATE":        0,   # Planet candidate
        "FALSE POSITIVE":   1,   # Default FP → EB (most common)
    })

    # ── Model architecture ────────────────────────────────────────────────
    input_length: int    = 200    # T — phase-folded segment length (time steps)
    input_channels: int  = 1      # single flux channel

    # CNN blocks — kernel sizes decrease as receptive field builds
    cnn_channels: list   = field(default_factory=lambda: [32, 64, 128])
    cnn_kernels:  list   = field(default_factory=lambda: [7,  5,  3])
    cnn_dropout:  float  = 0.1    # dropout after first conv in each CNN block

    # BiLSTM layers
    # Output dim = hidden_size * 2 (bidirectional)
    lstm_hidden_sizes: list = field(default_factory=lambda: [128, 64])
    lstm_dropout: float     = 0.3   # dropout between LSTM layers

    # Classification head
    head_hidden_sizes: list = field(default_factory=lambda: [64, 32])
    head_dropouts: list     = field(default_factory=lambda: [0.5, 0.3])

    # ── Loss ─────────────────────────────────────────────────────────────
    # Class weights are computed dynamically from dataset frequencies
    # in losses.py — these are fallback uniform weights if catalog unavailable
    use_class_weights: bool  = True
    label_smoothing:   float = 0.0   # set > 0 (e.g. 0.1) if model overconfident

    # ── Training — Phase 1 (Kepler pretraining) ───────────────────────────
    phase1_epochs:        int   = 100
    phase1_lr:            float = 5e-4
    phase1_batch_size:    int   = 64
    phase1_early_stop:    int   = 15

    # ── Training — Phase 2 (TESS fine-tuning) ─────────────────────────────
    phase2_epochs:        int   = 50
    phase2_lr:            float = 1e-4
    phase2_batch_size:    int   = 64
    phase2_early_stop:    int   = 10
    phase2_freeze_cnn:    bool  = True   # freeze CNN blocks, retrain LSTM+head only

    # ── Training — Hyderabad fine-tune (ISRO data) ────────────────────────
    hyderabad_epochs:     int   = 10
    hyderabad_lr:         float = 1e-5
    hyderabad_freeze_cnn: bool  = False  # fine-tune all layers (if ISRO ~ TESS)

    # ── Shared training settings ──────────────────────────────────────────
    weight_decay:         float = 1e-4
    grad_clip_norm:       float = 1.0

    # ReduceLROnPlateau (replaces cosine schedule — plateau-based for classifier)
    lr_scheduler_factor:    float = 0.5
    lr_scheduler_patience:  int   = 10
    lr_scheduler_min_lr:    float = 1e-7

    # ── Data ─────────────────────────────────────────────────────────────
    train_frac: float = 0.70
    val_frac:   float = 0.15
    # test_frac  = 0.15 (implicit)

    # Phase folding output — number of points in folded segment
    fold_points:  int   = 200
    fold_pad:     float = 0.3   # fraction of period to include around transit centre

    # Augmentation (applied only during training, never val/test)
    aug_phase_shift:  bool  = True    # random phase shift of folded curve
    aug_noise_std:    float = 0.01    # Gaussian noise std (relative to flux std)
    aug_flux_scale:   float = 0.05    # ±5% random flux scaling

    # ── Paths ─────────────────────────────────────────────────────────────
    data_raw_tess_dir:    Path = Path("data/raw/tess")
    data_raw_kepler_dir:  Path = Path("data/raw/kepler")
    toi_catalog_path:     Path = Path("data/catalogs/toi_catalog.csv")
    kepler_koi_path:      Path = Path("data/catalogs/kepler_koi.csv")
    tic_stellar_path:     Path = Path("data/catalogs/tic_stellar.csv")

    samples_dir:          Path = Path("data/samples/classifier")
    weights_dir:          Path = Path("weights")
    results_dir:          Path = Path("results/classifier")

    kepler_weights:       Path = Path("weights/classifier_kepler.pth")
    best_weights:         Path = Path("weights/classifier_best.pth")

    # ── Inference ─────────────────────────────────────────────────────────
    mc_dropout_passes:    int  = 50    # Monte Carlo Dropout forward passes
    confidence_threshold: float = 0.5  # minimum softmax prob to accept prediction

    def __post_init__(self):
        # Convert strings to Path if loaded from JSON/YAML
        path_attrs = [
            "data_raw_tess_dir", "data_raw_kepler_dir",
            "toi_catalog_path", "kepler_koi_path", "tic_stellar_path",
            "samples_dir", "weights_dir", "results_dir",
            "kepler_weights", "best_weights",
        ]
        for attr in path_attrs:
            setattr(self, attr, Path(getattr(self, attr)))

        # Auto-create directories on first import
        for d in [self.samples_dir, self.weights_dir, self.results_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Validate class count
        assert len(self.class_names) == self.num_classes, (
            f"class_names has {len(self.class_names)} entries but num_classes={self.num_classes}"
        )

    def summary(self):
        """Print config as a clean table — call at start of every training run."""
        print("\n" + "=" * 55)
        print("  VYOM CLASSIFIER CONFIG")
        print("=" * 55)
        for k, v in self.__dict__.items():
            # Skip large dicts for readability
            if isinstance(v, dict) and len(v) > 4:
                print(f"  {k:<30} {{...{len(v)} entries}}")
            elif isinstance(v, list) and len(v) > 6:
                print(f"  {k:<30} [{len(v)} items]")
            else:
                print(f"  {k:<30} {v}")
        print("=" * 55 + "\n")

    def get_class_name(self, idx: int, full: bool = False) -> str:
        """Return short or full class name for a given class index."""
        if full:
            return self.class_full_names.get(idx, f"Unknown({idx})")
        return self.class_names.get(idx, f"UNK{idx}")

    def get_class_index(self, disposition: str, source: str = "toi") -> int:
        """
        Map a catalog disposition string to a class index.

        Args:
            disposition : raw string from TOI or KOI catalog
            source      : "toi" or "kepler"

        Returns:
            class index (0–5), or -1 if unmappable
        """
        disp = disposition.strip().upper()
        if source == "kepler":
            return self.kepler_disposition_map.get(disp, -1)
        return self.toi_disposition_map.get(disp, -1)


# Singleton — import this everywhere, never instantiate ClassifierConfig again
CFG = ClassifierConfig()
