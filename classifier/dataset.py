"""
classifier/dataset.py

TOICatalogDataset — PyTorch Dataset for the six-class transit classifier.

Two catalog sources:
  1. TESS TOI catalog  (data/catalogs/toi_catalog.csv)   — Phase 2 fine-tuning
  2. Kepler KOI table  (data/catalogs/kepler_koi.csv)    — Phase 1 pretraining

Flow for each source:
  1. Read catalog CSV → extract TIC/KIC IDs, periods, dispositions
  2. Map disposition string → class index 0–5 via CFG
  3. Split IDs into train/val/test by TIC/KIC ID (NEVER by row)
  4. For each star: read FITS → preprocess → phase-fold on known period → 200-pt segment
  5. Cache as .npy + labels.csv in data/samples/classifier/{split}/

Augmentation (train split ONLY, never val/test):
  - Random phase shift of folded segment (wrap-around safe)
  - Gaussian noise injection (std = CFG.aug_noise_std × flux_std)
  - Random flux scaling ±CFG.aug_flux_scale

Phase folding:
  Given flux[T], time[T], period P, epoch t0:
    phase = ((time - t0) % P) / P       → [0, 1)
    Sort by phase, bin into CFG.fold_points=200 bins, average each bin
    Centre transit at phase 0.5 (standard convention)

Usage:
  from classifier.dataset import get_dataloaders
  train_loader, val_loader, test_loader = get_dataloaders(source="tess")
  train_loader, val_loader, test_loader = get_dataloaders(source="kepler")
"""

import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .config import CFG

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Catalog row — internal dataclass, one row per usable star
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CatalogRow:
    """One entry from TOI or KOI catalog that we can train on."""
    star_id:    str    # TIC ID (TESS) or KIC ID (Kepler) — always string
    period:     float  # orbital period in days
    t0:         float  # reference transit epoch (BJD or BTJD)
    depth_ppm:  float  # transit depth in ppm (parts per million)
    duration_h: float  # transit duration in hours
    label:      int    # class index 0–5
    source:     str    # "tess" or "kepler"
    fits_path:  Optional[Path] = None  # filled after FITS scan


# ─────────────────────────────────────────────────────────────────────────────
# Catalog readers
# ─────────────────────────────────────────────────────────────────────────────

def _read_toi_catalog(csv_path: Path) -> list[CatalogRow]:
    """
    Read NASA Exoplanet Archive TOI catalog CSV.

    Key columns (case-insensitive, we search flexibly):
      TIC ID        — 'TIC'  or 'TIC ID' or 'tic_id'
      Period        — 'Period (days)' or 'pl_orbper'
      Epoch (t0)    — 'Epoch (BJD)' or 'pl_tranmid'
      Depth         — 'Depth (ppm)' or 'pl_trandep'
      Duration      — 'Duration (hours)' or 'pl_trandurh'
      Disposition   — 'TFOPWG Disp' (the label column)

    TOI CSV has comment lines starting with '#' — we skip them.
    Returns only rows with valid, mappable dispositions (label != -1).
    """
    if not csv_path.exists():
        logger.warning(f"TOI catalog not found: {csv_path}")
        return []

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        lines = [line for line in f if not line.startswith("#")]

    if not lines:
        logger.warning(f"TOI catalog empty or all comment lines: {csv_path}")
        return []

    reader = csv.DictReader(lines)
    raw_cols = list(reader.fieldnames or [])

    # ── Column name resolution (flexible, case-insensitive) ───────────────
    def _find_col(candidates: list[str]) -> Optional[str]:
        """Return the first matching column name (case-insensitive)."""
        lower_map = {c.strip().lower(): c for c in raw_cols}
        for cand in candidates:
            hit = lower_map.get(cand.strip().lower())
            if hit is not None:
                return hit
        return None

    col_tic      = _find_col(["TIC", "TIC ID", "tic_id", "toi_id"])
    col_period   = _find_col(["Period (days)", "pl_orbper", "period"])
    col_t0       = _find_col(["Epoch (BJD)", "Epoch (BTJD)", "pl_tranmid", "epoch"])
    col_depth    = _find_col(["Depth (ppm)", "pl_trandep", "depth_ppm", "depth"])
    col_duration = _find_col(["Duration (hours)", "pl_trandurh", "duration_hours", "duration"])
    col_disp     = _find_col(["TFOPWG Disp", "tfopwg_disp", "Disposition", "disposition"])

    missing = [name for name, col in [
        ("TIC", col_tic), ("Period", col_period),
        ("Disposition", col_disp),
    ] if col is None]

    if missing:
        logger.error(
            f"TOI catalog missing required columns: {missing}\n"
            f"Available columns: {raw_cols[:15]}"
        )
        return []

    n_skipped = 0
    for row in reader:
        try:
            tic_raw = row.get(col_tic, "").strip()
            if not tic_raw:
                continue

            # TIC IDs sometimes come as "TIC 12345" — strip prefix
            tic_id = re.sub(r"[^\d]", "", tic_raw)
            if not tic_id:
                continue

            period_str = row.get(col_period, "").strip() if col_period else ""
            t0_str     = row.get(col_t0,     "").strip() if col_t0     else ""
            depth_str  = row.get(col_depth,  "").strip() if col_depth  else ""
            dur_str    = row.get(col_duration,"").strip() if col_duration else ""
            disp       = row.get(col_disp,   "").strip()

            # Need at least period and disposition
            if not period_str or not disp:
                n_skipped += 1
                continue

            try:
                period = float(period_str)
            except ValueError:
                n_skipped += 1
                continue

            if period <= 0:
                n_skipped += 1
                continue

            label = CFG.get_class_index(disp, source="toi")
            if label == -1:
                n_skipped += 1
                continue

            t0        = float(t0_str)    if t0_str    else 0.0
            depth_ppm = float(depth_str) if depth_str else 0.0
            duration  = float(dur_str)   if dur_str   else 0.0

            rows.append(CatalogRow(
                star_id    = tic_id,
                period     = period,
                t0         = t0,
                depth_ppm  = depth_ppm,
                duration_h = duration,
                label      = label,
                source     = "tess",
            ))

        except (ValueError, KeyError):
            n_skipped += 1
            continue

    logger.info(
        f"TOI catalog: {len(rows)} usable rows, {n_skipped} skipped "
        f"(bad values / unmapped dispositions)"
    )
    return rows


def _read_kepler_catalog(csv_path: Path) -> list[CatalogRow]:
    """
    Read NASA Exoplanet Archive Kepler KOI cumulative table.

    Key columns:
      kepid           — Kepler Input Catalog ID
      koi_period      — orbital period in days
      koi_time0bk     — transit epoch (BKJD)
      koi_depth       — transit depth in ppm
      koi_duration    — transit duration in hours
      koi_disposition — 'CONFIRMED' / 'CANDIDATE' / 'FALSE POSITIVE'

    Returns only rows with valid, mappable dispositions.
    """
    if not csv_path.exists():
        logger.warning(f"Kepler KOI catalog not found: {csv_path}")
        return []

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        lines = [line for line in f if not line.startswith("#")]

    if not lines:
        logger.warning(f"Kepler catalog empty or all comment lines: {csv_path}")
        return []

    reader = csv.DictReader(lines)
    raw_cols = list(reader.fieldnames or [])

    def _find_col(candidates: list[str]) -> Optional[str]:
        lower_map = {c.strip().lower(): c for c in raw_cols}
        for cand in candidates:
            hit = lower_map.get(cand.strip().lower())
            if hit is not None:
                return hit
        return None

    col_kic      = _find_col(["kepid", "kic_id", "KIC"])
    col_period   = _find_col(["koi_period", "period"])
    col_t0       = _find_col(["koi_time0bk", "koi_time0", "epoch"])
    col_depth    = _find_col(["koi_depth", "depth_ppm", "depth"])
    col_duration = _find_col(["koi_duration", "duration"])
    col_disp     = _find_col(["koi_disposition", "disposition"])

    missing = [name for name, col in [
        ("kepid", col_kic), ("koi_period", col_period),
        ("koi_disposition", col_disp),
    ] if col is None]

    if missing:
        logger.error(
            f"Kepler catalog missing required columns: {missing}\n"
            f"Available: {raw_cols[:15]}"
        )
        return []

    n_skipped = 0
    for row in reader:
        try:
            kic_raw = row.get(col_kic, "").strip()
            if not kic_raw:
                continue
            kic_id = re.sub(r"[^\d]", "", kic_raw)
            if not kic_id:
                continue

            period_str = row.get(col_period, "").strip() if col_period else ""
            t0_str     = row.get(col_t0,     "").strip() if col_t0     else ""
            depth_str  = row.get(col_depth,  "").strip() if col_depth  else ""
            dur_str    = row.get(col_duration,"").strip() if col_duration else ""
            disp       = row.get(col_disp,   "").strip()

            if not period_str or not disp:
                n_skipped += 1
                continue

            try:
                period = float(period_str)
            except ValueError:
                n_skipped += 1
                continue

            if period <= 0:
                n_skipped += 1
                continue

            label = CFG.get_class_index(disp, source="kepler")
            if label == -1:
                n_skipped += 1
                continue

            t0        = float(t0_str)    if t0_str    else 0.0
            depth_ppm = float(depth_str) if depth_str else 0.0
            duration  = float(dur_str)   if dur_str   else 0.0

            rows.append(CatalogRow(
                star_id    = kic_id,
                period     = period,
                t0         = t0,
                depth_ppm  = depth_ppm,
                duration_h = duration,
                label      = label,
                source     = "kepler",
            ))

        except (ValueError, KeyError):
            n_skipped += 1
            continue

    logger.info(
        f"Kepler catalog: {len(rows)} usable rows, {n_skipped} skipped"
    )
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# FITS reading (minimal — shares logic with denoiser but is independent)
# ─────────────────────────────────────────────────────────────────────────────

def _read_fits_lc(
    fits_path: Path,
    source: str = "tess",
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """
    Read a TESS or Kepler FITS file and return (time, flux) arrays.

    TESS:   extension 1, columns TIME + PDCSAP_FLUX + QUALITY
    Kepler: extension 1, columns TIME + PDCSAP_FLUX + SAP_QUALITY

    Returns:
        (time, flux) both float32 numpy arrays, NaN/bad-quality removed
        None if file is unreadable or has too few valid points.
    """
    try:
        from astropy.io import fits as astrofits
    except ImportError:
        raise ImportError("astropy not installed. Run: pip install astropy")

    MIN_POINTS = CFG.fold_points * 3   # need at least 3× fold points

    try:
        with astrofits.open(fits_path) as hdul:
            data = hdul[1].data

            time  = data["TIME"].astype(np.float64)
            flux  = data["PDCSAP_FLUX"].astype(np.float32)

            # Quality column — name differs slightly between missions
            quality = None
            for q_col in ["QUALITY", "SAP_QUALITY"]:
                if q_col in data.names:
                    quality = data[q_col].astype(np.int32)
                    break

    except Exception as e:
        logger.debug(f"Could not read {fits_path.name}: {e}")
        return None

    # Apply quality mask if available
    if quality is not None:
        BAD_BITS = 1 | 2 | 4 | 8 | 32 | 64 | 512
        good = (quality & BAD_BITS) == 0
        time  = time[good]
        flux  = flux[good]

    # Remove NaN in either time or flux
    finite_mask = np.isfinite(time) & np.isfinite(flux)
    time = time[finite_mask]
    flux = flux[finite_mask]

    if len(flux) < MIN_POINTS:
        logger.debug(f"Too few points ({len(flux)}) in {fits_path.name}")
        return None

    # 5-sigma outlier removal on flux
    median = np.median(flux)
    mad    = np.median(np.abs(flux - median))
    mad    = mad if mad > 0 else 1e-8
    sigma  = 1.4826 * mad
    good   = np.abs(flux - median) < 5.0 * sigma
    time   = time[good]
    flux   = flux[good]

    if len(flux) < MIN_POINTS:
        return None

    # Normalize: (flux - median) / MAD
    median = np.median(flux)
    mad    = np.median(np.abs(flux - median))
    mad    = mad if mad > 0 else 1e-8
    flux   = ((flux - median) / (1.4826 * mad)).astype(np.float32)

    return time.astype(np.float32), flux


# ─────────────────────────────────────────────────────────────────────────────
# Phase folding
# ─────────────────────────────────────────────────────────────────────────────

def phase_fold(
    time:   np.ndarray,
    flux:   np.ndarray,
    period: float,
    t0:     float,
    n_bins: int = CFG.fold_points,
) -> Optional[np.ndarray]:
    """
    Phase-fold a light curve and bin into n_bins evenly-spaced phase bins.

    Steps:
      1. Compute phase: phi = ((time - t0) % period) / period  → [0, 1)
      2. Sort by phase
      3. Bin into n_bins bins, average flux in each bin
      4. Bins with no points → interpolated from neighbours
         (gaps happen if coverage is sparse)
      5. Pad/truncate to exactly n_bins points

    Why bin and not just sort-and-crop:
      Different sectors have different cadences. Binning gives a fixed-length
      representation regardless of how many transits stacked. It's the standard
      approach in ExoFOP, NASA Exoplanet Archive phase plots.

    Args:
        time   : [T] float32 — time axis (BJD or BTJD)
        flux   : [T] float32 — normalized flux
        period : orbital period in days
        t0     : reference transit epoch
        n_bins : output array length (default CFG.fold_points = 200)

    Returns:
        [n_bins] float32 array — folded, binned light curve
        None if folding fails (zero valid bins, etc.)
    """
    if period <= 0 or len(time) < n_bins:
        return None

    # Phase in [0, 1)
    phase = ((time - t0) % period) / period

    # Sort by phase
    sort_idx = np.argsort(phase)
    phase_sorted = phase[sort_idx]
    flux_sorted  = flux[sort_idx]

    # Bin edges: 0.0, 1/n_bins, 2/n_bins, ..., 1.0
    bin_edges   = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    folded = np.full(n_bins, np.nan, dtype=np.float32)

    # Assign each point to a bin and average
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (phase_sorted >= lo) & (phase_sorted < hi)
        if mask.any():
            folded[i] = float(np.mean(flux_sorted[mask]))

    # Fill NaN bins by interpolation from neighbouring valid bins
    nan_mask = np.isnan(folded)
    if nan_mask.all():
        return None   # nothing survived — bad period or too few points

    if nan_mask.any():
        valid_idx = np.where(~nan_mask)[0]
        for i in np.where(nan_mask)[0]:
            # Use nearest valid bin value (simple nearest-neighbour fill)
            nearest = valid_idx[np.argmin(np.abs(valid_idx - i))]
            folded[i] = folded[nearest]

    return folded


# ─────────────────────────────────────────────────────────────────────────────
# FITS file scanner
# ─────────────────────────────────────────────────────────────────────────────

def _scan_fits(data_dir: Path, source: str) -> dict[str, Path]:
    """
    Recursively scan data_dir for FITS files and map star_id → fits_path.

    For TESS: extracts 16-digit TIC ID from filename
    For Kepler: extracts KIC ID from filename

    Returns: {star_id_str: Path}
    """
    star_to_fits: dict[str, Path] = {}

    patterns = ["*.fits", "*.fit"]
    fits_files = []
    for pat in patterns:
        fits_files.extend(data_dir.rglob(pat))

    logger.info(f"Scanning {data_dir} — found {len(fits_files)} FITS files")

    for path in fits_files:
        name = path.stem

        if source == "tess":
            # Standard pattern: tess...-s0001-0000000012345678-...
            m = re.search(r"-(\d{16})-", name)
            if m:
                star_id = str(int(m.group(1)))   # strip leading zeros
            else:
                m = re.search(r"(\d{8,})", name)
                star_id = str(int(m.group(1))) if m else None

        else:  # kepler
            # Pattern: kplr012345678_...
            m = re.search(r"kplr(\d+)", name, re.IGNORECASE)
            if m:
                star_id = str(int(m.group(1)))
            else:
                m = re.search(r"(\d{8,})", name)
                star_id = str(int(m.group(1))) if m else None

        if star_id is None:
            continue

        # Keep first occurrence — multiple sectors: use any one for classifier
        if star_id not in star_to_fits:
            star_to_fits[star_id] = path

    logger.info(f"Mapped {len(star_to_fits)} unique star IDs to FITS files")
    return star_to_fits


# ─────────────────────────────────────────────────────────────────────────────
# TIC ID split — same pattern as denoiser, by star ID not by row
# ─────────────────────────────────────────────────────────────────────────────

def _split_star_ids(
    star_ids:   list[str],
    train_frac: float = CFG.train_frac,
    val_frac:   float = CFG.val_frac,
    seed:       int   = 42,
) -> tuple[list[str], list[str], list[str]]:
    """
    Split star IDs into train / val / test — NEVER by row index.

    Critical: same star must never appear in train AND test.
    Random split by sample index would leak correlated stellar noise patterns.

    Returns: (train_ids, val_ids, test_ids)
    """
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(star_ids))   # sort first for determinism across runs
    rng.shuffle(ids)

    n       = len(ids)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train_ids = ids[:n_train].tolist()
    val_ids   = ids[n_train : n_train + n_val].tolist()
    test_ids  = ids[n_train + n_val :].tolist()

    logger.info(
        f"Star ID split — train: {len(train_ids)}, "
        f"val: {len(val_ids)}, test: {len(test_ids)}"
    )
    return train_ids, val_ids, test_ids


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────────────────────────────────────

def _augment(folded: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Apply training augmentation to a 200-point folded light curve.

    Three transforms, all applied independently with their own RNG draw:
      1. Phase shift   — roll array by random offset (wrap-around safe)
      2. Noise inject  — add N(0, aug_noise_std × std(folded)) Gaussian noise
      3. Flux scale    — multiply by U(1 - aug_flux_scale, 1 + aug_flux_scale)

    Physical justification:
      Phase shift: different t0 choices produce different transit centering.
        The classifier must be invariant to where in the 200-pt window the
        dip sits — shift augmentation teaches this.
      Noise inject: real light curves have varying noise floors across stars.
        Injecting noise prevents overfitting to the specific noise of training stars.
      Flux scale: limb darkening and stellar radius uncertainties shift depth.
        Scale augmentation prevents the classifier from memorising depth values.

    Args:
        folded : [200] float32 array
        rng    : numpy random generator (passed in for reproducibility)

    Returns:
        [200] float32 augmented array
    """
    x = folded.copy()
    n = len(x)

    # 1. Phase shift — roll by random integer offset
    if CFG.aug_phase_shift:
        shift = int(rng.integers(0, n))
        x = np.roll(x, shift)

    # 2. Gaussian noise
    if CFG.aug_noise_std > 0:
        flux_std = x.std()
        noise    = rng.normal(0.0, CFG.aug_noise_std * (flux_std + 1e-8), size=n)
        x        = x + noise.astype(np.float32)

    # 3. Flux scaling
    if CFG.aug_flux_scale > 0:
        scale = rng.uniform(1.0 - CFG.aug_flux_scale, 1.0 + CFG.aug_flux_scale)
        x     = x * float(scale)

    return x.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Caching
# ─────────────────────────────────────────────────────────────────────────────

def _cache_sample_path(star_id: str, label: int, split: str, idx: int) -> Path:
    """Return the .npy path for a cached folded segment."""
    folder = CFG.samples_dir / split
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{star_id}_cls{label}_{idx:04d}.npy"


def _write_labels_csv(
    split: str,
    entries: list[tuple[str, int]],  # [(filename_stem, label), ...]
) -> None:
    """Write labels.csv mapping .npy filename → class label."""
    path = CFG.samples_dir / split / "labels.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        for stem, label in entries:
            writer.writerow([stem, label])


# ─────────────────────────────────────────────────────────────────────────────
# Main Dataset class
# ─────────────────────────────────────────────────────────────────────────────

class TOICatalogDataset(Dataset):
    """
    PyTorch Dataset for the six-class transit classifier.

    Each sample: (folded_segment, label)
      folded_segment : [1, 200] float32 tensor — phase-folded light curve
      label          : int scalar tensor (0–5)

    Augmentation is applied on-the-fly (not cached) for train split only.
    Raw folded arrays ARE cached as .npy to skip FITS re-reading on next run.

    Args:
        split         : "train", "val", or "test"
        source        : "tess" (TOI catalog) or "kepler" (KOI catalog)
        force_rebuild : if True, ignore cache and reprocess from FITS

    Example:
        dataset = TOICatalogDataset("train", source="tess")
        x, y = dataset[0]   # x: [1, 200], y: scalar tensor
    """

    def __init__(
        self,
        split:         str  = "train",
        source:        str  = "tess",
        force_rebuild: bool = False,
    ):
        assert split  in ("train", "val", "test"), \
            f"split must be train/val/test, got '{split}'"
        assert source in ("tess", "kepler"), \
            f"source must be 'tess' or 'kepler', got '{source}'"

        self.split     = split
        self.source    = source
        self.is_train  = (split == "train")
        self.rng       = np.random.default_rng(seed=0)

        # Samples: list of (folded_array [200], label_int)
        self.samples: list[tuple[np.ndarray, int]] = []

        cache_labels = CFG.samples_dir / split / "labels.csv"

        if cache_labels.exists() and not force_rebuild:
            logger.info(f"[{split}/{source}] Loading from cache")
            self._load_from_cache(split)
        else:
            logger.info(f"[{split}/{source}] Building from catalog + FITS")
            self._build(split, source)

        logger.info(f"[{split}/{source}] Dataset ready — {len(self.samples)} samples")
        self._log_class_distribution()

    # ── Build from catalog ────────────────────────────────────────────────

    def _build(self, split: str, source: str) -> None:
        """Full build: read catalog → scan FITS → phase fold → cache."""

        # 1. Read catalog
        if source == "tess":
            catalog_rows = _read_toi_catalog(CFG.toi_catalog_path)
            data_dir     = CFG.data_raw_tess_dir
        else:
            catalog_rows = _read_kepler_catalog(CFG.kepler_koi_path)
            data_dir     = CFG.data_raw_kepler_dir

        if not catalog_rows:
            raise RuntimeError(
                f"No usable rows from {source} catalog. "
                "Check catalog path in CFG and download catalogs first."
            )

        # 2. Scan available FITS files
        star_to_fits = _scan_fits(data_dir, source=source)

        # 3. Match catalog rows to FITS files
        matched: list[CatalogRow] = []
        for row in catalog_rows:
            if row.star_id in star_to_fits:
                row.fits_path = star_to_fits[row.star_id]
                matched.append(row)

        if not matched:
            raise RuntimeError(
                f"No catalog rows matched any FITS files.\n"
                f"Catalog has {len(catalog_rows)} rows, "
                f"found {len(star_to_fits)} FITS files in {data_dir}.\n"
                "Check that FITS files are downloaded and TIC IDs match."
            )

        logger.info(f"{len(matched)} catalog rows matched to FITS files")

        # 4. Split by star ID
        all_star_ids = list({row.star_id for row in matched})
        train_ids, val_ids, test_ids = _split_star_ids(all_star_ids)
        split_map = {"train": set(train_ids), "val": set(val_ids), "test": set(test_ids)}
        split_set = split_map[split]

        split_rows = [r for r in matched if r.star_id in split_set]
        logger.info(f"[{split}] {len(split_rows)} rows after TIC ID split")

        # 5. Phase fold + cache
        label_entries: list[tuple[str, int]] = []
        n_failed = 0

        for row in split_rows:
            # Read FITS
            result = _read_fits_lc(row.fits_path, source=source)
            if result is None:
                n_failed += 1
                continue

            time_arr, flux_arr = result

            # Phase fold
            folded = phase_fold(
                time_arr, flux_arr,
                period=row.period,
                t0=row.t0,
                n_bins=CFG.fold_points,
            )
            if folded is None:
                n_failed += 1
                continue

            # Cache
            idx      = len(self.samples)
            npy_path = _cache_sample_path(row.star_id, row.label, split, idx)
            np.save(npy_path, folded)

            label_entries.append((npy_path.stem, row.label))
            self.samples.append((folded, row.label))

        _write_labels_csv(split, label_entries)

        logger.info(
            f"[{split}] Built {len(self.samples)} samples, "
            f"{n_failed} failed (bad FITS / folding)"
        )

    # ── Load from cache ───────────────────────────────────────────────────

    def _load_from_cache(self, split: str) -> None:
        """Load .npy files and labels.csv from cache directory."""
        cache_dir    = CFG.samples_dir / split
        labels_csv   = cache_dir / "labels.csv"

        if not labels_csv.exists():
            raise RuntimeError(
                f"labels.csv not found in {cache_dir}. "
                "Run with force_rebuild=True."
            )

        with open(labels_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row["filename"]
                label    = int(row["label"])
                npy_path = cache_dir / f"{filename}.npy"

                if not npy_path.exists():
                    logger.warning(f"Cache file missing: {npy_path} — skipping")
                    continue

                folded = np.load(npy_path).astype(np.float32)
                self.samples.append((folded, label))

    # ── Class distribution logging ────────────────────────────────────────

    def _log_class_distribution(self) -> None:
        """Log how many samples per class — useful for catching label imbalance."""
        counts = {}
        for _, label in self.samples:
            counts[label] = counts.get(label, 0) + 1

        logger.info(f"[{self.split}] Class distribution:")
        for i in range(CFG.num_classes):
            n   = counts.get(i, 0)
            pct = 100.0 * n / max(len(self.samples), 1)
            logger.info(f"  [{i}] {CFG.get_class_name(i):<5}  {n:>5} samples  ({pct:.1f}%)")

    # ── PyTorch Dataset interface ─────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x : [1, fold_points] float32 — phase-folded segment (channel-first)
            y : scalar int64 tensor      — class label 0–5
        """
        folded, label = self.samples[idx]

        # Augmentation — train only, on-the-fly (never cached)
        if self.is_train:
            folded = _augment(folded, self.rng)

        x = torch.from_numpy(folded).unsqueeze(0)          # [200] → [1, 200]
        y = torch.tensor(label, dtype=torch.long)

        return x, y

    def get_labels(self) -> list[int]:
        """Return all labels as a list — needed for WeightedRandomSampler."""
        return [label for _, label in self.samples]


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory — single entry point for train.py
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    source:        str  = "tess",
    batch_size:    int  = CFG.phase1_batch_size,
    num_workers:   int  = 0,
    force_rebuild: bool = False,
    use_weighted_sampler: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build and return train, val, test DataLoaders.

    Args:
        source       : "tess" (TOI catalog) or "kepler" (KOI catalog)
        batch_size   : samples per batch
        num_workers  : parallel workers (0 = safe on Windows)
        force_rebuild: reprocess all FITS even if cache exists
        use_weighted_sampler: if True, oversample rare classes in training
                              (alternative to class weights in loss — can use both)

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_ds = TOICatalogDataset("train", source=source, force_rebuild=force_rebuild)
    val_ds   = TOICatalogDataset("val",   source=source, force_rebuild=False)
    test_ds  = TOICatalogDataset("test",  source=source, force_rebuild=False)

    # WeightedRandomSampler — oversamples rare classes at the batch level
    # Works ALONGSIDE class weights in loss (double correction = better for extreme imbalance)
    train_sampler = None
    if use_weighted_sampler and len(train_ds) > 0:
        from torch.utils.data import WeightedRandomSampler

        labels  = train_ds.get_labels()
        counts  = np.bincount(labels, minlength=CFG.num_classes).astype(float)
        counts  = np.maximum(counts, 1.0)   # avoid divide by zero for missing classes
        class_w = 1.0 / counts              # inverse frequency weight per class
        sample_w = [class_w[lbl] for lbl in labels]

        train_sampler = WeightedRandomSampler(
            weights     = torch.tensor(sample_w, dtype=torch.float32),
            num_samples = len(train_ds),
            replacement = True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size   = batch_size,
        sampler      = train_sampler,           # overrides shuffle when set
        shuffle      = (train_sampler is None), # shuffle only if no sampler
        num_workers  = num_workers,
        pin_memory   = torch.cuda.is_available(),
        drop_last    = True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = torch.cuda.is_available(),
        drop_last   = False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = torch.cuda.is_available(),
        drop_last   = False,
    )

    logger.info(
        f"DataLoaders ready [{source}] — "
        f"train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}"
    )

    return train_loader, val_loader, test_loader
