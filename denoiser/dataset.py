"""
denoiser/dataset.py

TESSSectorPairDataset — PyTorch Dataset for Noise2Noise denoiser training.

Core idea:
  TESS observes the same star across multiple sectors (27-day observation windows).
  Each sector is an INDEPENDENT noisy observation of the same underlying stellar signal.
  So (sector_1_flux, sector_2_flux) of the same star = perfect Noise2Noise training pair.
  No clean ground truth ever needed.

Flow:
  1. Scan data/raw/tess/ for all FITS files
  2. Group by TIC ID — find stars observed in 2+ sectors
  3. Split TIC IDs into train/val/test (never split by file — same star must stay together)
  4. For each star pair: preprocess both sectors, chunk into T=1000 windows
  5. Cache chunks as .npy to data/samples/denoiser/ — skip re-processing on next run

Usage:
  from denoiser.dataset import get_dataloaders
  train_loader, val_loader, test_loader = get_dataloaders()
"""

import os
import re
import csv
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .config import CFG

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# FITS reading + preprocessing (minimal, self-contained)
# Full preprocessing lives in pipeline/preprocess.py
# This version is lightweight — just enough for denoiser training
# ─────────────────────────────────────────────────────────────────────────────

def _read_fits(fits_path: Path) -> Optional[np.ndarray]:
    """
    Read a TESS FITS file and return a cleaned, normalized flux array.
    Returns None if file is unreadable or has too few valid points.

    Steps:
      1. Extract PDCSAP_FLUX + QUALITY columns
      2. Apply TESS quality bitmask — remove bad cadences
      3. Remove NaN values
      4. 5-sigma outlier clipping
      5. Normalize: (flux - median) / MAD
    """
    try:
        from astropy.io import fits as astrofits
    except ImportError:
        raise ImportError("astropy not installed. Run: pip install astropy")

    try:
        with astrofits.open(fits_path) as hdul:
            # TESS light curve data is in extension 1
            data = hdul[1].data
            flux    = data['PDCSAP_FLUX'].astype(np.float32)
            quality = data['QUALITY'].astype(np.int32)
    except Exception as e:
        logger.warning(f"Could not read {fits_path}: {e}")
        return None

    # TESS quality bitmask — remove flagged cadences
    # Bits 1,2,4,8,32,64,512 indicate bad data
    BAD_BITS = 1 | 2 | 4 | 8 | 32 | 64 | 512
    good_mask = (quality & BAD_BITS) == 0
    flux = flux[good_mask]

    # Remove NaN
    nan_mask = np.isfinite(flux)
    flux = flux[nan_mask]

    # Need minimum points to be useful
    if len(flux) < CFG.chunk_length:
        logger.warning(f"Too few points ({len(flux)}) in {fits_path.name} — skipping")
        return None

    # 5-sigma outlier clipping
    median = np.median(flux)
    mad    = np.median(np.abs(flux - median))
    mad    = mad if mad > 0 else 1e-8          # guard against zero MAD
    sigma  = 1.4826 * mad                      # MAD → std conversion factor
    clip_mask = np.abs(flux - median) < 5.0 * sigma
    flux = flux[clip_mask]

    if len(flux) < CFG.chunk_length:
        return None

    # Normalize: (flux - median) / MAD
    median = np.median(flux)
    mad    = np.median(np.abs(flux - median))
    mad    = mad if mad > 0 else 1e-8
    flux   = (flux - median) / (1.4826 * mad)

    return flux.astype(np.float32)


def _chunk(flux: np.ndarray, length: int, stride: int) -> list[np.ndarray]:
    """
    Split a 1D flux array into overlapping fixed-length chunks.

    Args:
        flux   : 1D numpy array
        length : chunk size (CFG.chunk_length = 1000)
        stride : step between chunk starts (CFG.chunk_stride = 500)
                 stride < length means overlapping chunks — more training samples

    Returns:
        List of 1D arrays each of shape [length]
    """
    chunks = []
    start  = 0
    while start + length <= len(flux):
        chunks.append(flux[start : start + length])
        start += stride
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# TIC ID utilities
# ─────────────────────────────────────────────────────────────────────────────

def _extract_tic_id(fits_path: Path) -> Optional[str]:
    """
    Extract TIC ID from a TESS FITS filename.

    TESS filenames follow this pattern:
      tess2018206045859-s0001-0000000012345678-0120-s_lc.fits
                                  ^^^^^^^^^^^^^^^^^^
                                  TIC ID is here (zero-padded to 16 digits)

    Returns TIC ID as string (leading zeros stripped), or None if not parseable.
    """
    name = fits_path.stem  # filename without extension

    # Try standard TESS filename pattern
    match = re.search(r'-(\d{16})-', name)
    if match:
        return str(int(match.group(1)))  # strip leading zeros

    # Fallback: try any 8+ digit number in the filename
    match = re.search(r'(\d{8,})', name)
    if match:
        return str(int(match.group(1)))

    logger.warning(f"Could not extract TIC ID from: {fits_path.name}")
    return None


def _extract_sector(fits_path: Path) -> Optional[int]:
    """
    Extract sector number from a TESS FITS filename.

    Pattern: tess2018206045859-s0001-...
                                ^^^^^ sector 1
    """
    name = fits_path.stem
    match = re.search(r'-s(\d{4})-', name)
    if match:
        return int(match.group(1))

    # Fallback: look for sector in parent directory name
    parent = fits_path.parent.name
    match = re.search(r'sector[_\-]?(\d+)', parent, re.IGNORECASE)
    if match:
        return int(match.group(1))

    return None


def _scan_fits_files(data_dir: Path) -> dict[str, dict[int, Path]]:
    """
    Recursively scan data_dir for TESS FITS files.

    Returns:
        {
          "12345678": {1: Path("sector_01/tess...fits"), 2: Path("sector_02/tess...fits")},
          "87654321": {1: Path(...), 3: Path(...)},
          ...
        }

    Only TIC IDs with 2+ sectors are useful for Noise2Noise — caller filters this.
    """
    tic_to_sectors: dict[str, dict[int, Path]] = {}

    fits_files = list(data_dir.rglob("*.fits"))
    logger.info(f"Found {len(fits_files)} FITS files in {data_dir}")

    for path in fits_files:
        tic_id = _extract_tic_id(path)
        sector  = _extract_sector(path)

        if tic_id is None or sector is None:
            continue

        if tic_id not in tic_to_sectors:
            tic_to_sectors[tic_id] = {}

        tic_to_sectors[tic_id][sector] = path

    return tic_to_sectors


# ─────────────────────────────────────────────────────────────────────────────
# Train / Val / Test split — ALWAYS by TIC ID
# ─────────────────────────────────────────────────────────────────────────────

def _split_tic_ids(
    tic_ids:    list[str],
    train_frac: float = CFG.train_frac,
    val_frac:   float = CFG.val_frac,
    seed:       int   = 42,
) -> tuple[list[str], list[str], list[str]]:
    """
    Split TIC IDs into train / val / test sets.

    CRITICAL: Split is by TIC ID, never by sample index.
    Same star's light curves from different sectors must ALL go to the same split.
    Random split by sample would leak correlated stellar patterns from train to test
    and artificially inflate reported metrics.

    Args:
        tic_ids    : list of all TIC ID strings with 2+ sectors
        train_frac : fraction for training (default 0.70)
        val_frac   : fraction for validation (default 0.15)
        seed       : random seed for reproducibility

    Returns:
        (train_ids, val_ids, test_ids) — three disjoint lists
    """
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(tic_ids))    # sort first for determinism
    rng.shuffle(ids)

    n       = len(ids)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train_ids = ids[:n_train].tolist()
    val_ids   = ids[n_train : n_train + n_val].tolist()
    test_ids  = ids[n_train + n_val :].tolist()

    logger.info(f"TIC ID split — train: {len(train_ids)}, "
                f"val: {len(val_ids)}, test: {len(test_ids)}")
    return train_ids, val_ids, test_ids


# ─────────────────────────────────────────────────────────────────────────────
# Caching — save processed chunks as .npy so we don't re-read FITS every run
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path(tic_id: str, sector_a: int, sector_b: int,
                split: str, idx: int) -> Path:
    """Return the .npy path for a specific chunk."""
    folder = CFG.samples_dir / split
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"tic{tic_id}_s{sector_a:02d}s{sector_b:02d}_chunk{idx:04d}.npy"


def _build_pairs_index(
    tic_to_sectors: dict[str, dict[int, Path]],
    split_ids:      list[str],
    split:          str,
) -> list[tuple[Path, Path]]:
    """
    For each TIC ID in split_ids, pick TWO sectors and return their FITS paths.
    Also writes a pairs_index.csv for auditing.

    Returns:
        List of (sector_A_fits_path, sector_B_fits_path) tuples
    """
    pairs   = []
    csv_rows = [["tic_id", "sector_a", "sector_b", "path_a", "path_b"]]

    for tic_id in split_ids:
        sector_map = tic_to_sectors.get(tic_id, {})
        sectors    = sorted(sector_map.keys())

        if len(sectors) < 2:
            continue  # need at least 2 sectors for a Noise2Noise pair

        # Always use first two available sectors as the pair
        # If star has 3+ sectors, we only use 2 per pair (can extend later)
        sec_a, sec_b = sectors[0], sectors[1]
        path_a = sector_map[sec_a]
        path_b = sector_map[sec_b]

        pairs.append((path_a, path_b))
        csv_rows.append([tic_id, sec_a, sec_b, str(path_a), str(path_b)])

    # Write audit CSV
    csv_path = CFG.samples_dir / split / "pairs_index.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    logger.info(f"[{split}] {len(pairs)} valid pairs from {len(split_ids)} TIC IDs")
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────────────────────────────────────

class TESSSectorPairDataset(Dataset):
    """
    PyTorch Dataset for Noise2Noise denoiser training.

    Each sample is a tuple: (chunk_from_sector_A, chunk_from_sector_B)
    Both are [1, T] tensors from the same star, different sectors.
    The model takes sector_A as input and tries to output sector_B.
    Because noise in A and B are statistically independent, the model
    learns to output the underlying clean signal.

    Args:
        split       : "train", "val", or "test"
        data_dir    : root directory containing TESS FITS files
        force_rebuild: if True, ignore cache and rebuild .npy files from scratch

    Example:
        dataset = TESSSectorPairDataset("train")
        x, y = dataset[0]   # x: [1, 1000], y: [1, 1000]
    """

    def __init__(
        self,
        split:         str  = "train",
        data_dir:      Path = CFG.data_raw_dir,
        force_rebuild: bool = False,
    ):
        assert split in ("train", "val", "test"), \
            f"split must be 'train', 'val', or 'test' — got '{split}'"

        self.split = split
        self.chunks: list[tuple[np.ndarray, np.ndarray]] = []

        cache_index = CFG.samples_dir / split / "pairs_index.csv"

        if cache_index.exists() and not force_rebuild:
            logger.info(f"[{split}] Loading from cache: {CFG.samples_dir / split}")
            self._load_from_cache(split)
        else:
            logger.info(f"[{split}] Cache not found — building from FITS files")
            self._build_from_fits(split, data_dir)

        logger.info(f"[{split}] Dataset ready — {len(self.chunks)} chunk pairs")

    # ── Build from raw FITS ────────────────────────────────────────────────

    def _build_from_fits(self, split: str, data_dir: Path):
        """Scan FITS files, split by TIC ID, preprocess, chunk, cache."""
        tic_to_sectors = _scan_fits_files(data_dir)

        # Keep only stars with 2+ sectors
        multi_sector = {
            tic: secs for tic, secs in tic_to_sectors.items()
            if len(secs) >= 2
        }

        if len(multi_sector) == 0:
            raise RuntimeError(
                f"No multi-sector TESS FITS files found in {data_dir}.\n"
                f"Download TESS data first: see data/setup/data_download.md"
            )

        all_tic_ids = list(multi_sector.keys())
        train_ids, val_ids, test_ids = _split_tic_ids(all_tic_ids)

        split_map = {"train": train_ids, "val": val_ids, "test": test_ids}
        split_ids = split_map[split]

        pairs = _build_pairs_index(multi_sector, split_ids, split)

        for path_a, path_b in pairs:
            flux_a = _read_fits(path_a)
            flux_b = _read_fits(path_b)

            if flux_a is None or flux_b is None:
                continue

            chunks_a = _chunk(flux_a, CFG.chunk_length, CFG.chunk_stride)
            chunks_b = _chunk(flux_b, CFG.chunk_length, CFG.chunk_stride)

            # Pair chunks positionally — same time window from both sectors
            n_pairs = min(len(chunks_a), len(chunks_b))

            for idx in range(n_pairs):
                ca = chunks_a[idx]
                cb = chunks_b[idx]

                # Cache to disk
                tic_id  = _extract_tic_id(path_a)
                sec_a   = _extract_sector(path_a)
                sec_b   = _extract_sector(path_b)
                np_path = _cache_path(tic_id, sec_a, sec_b, split, idx)

                # Save as (2, T) array — row 0 = sector A, row 1 = sector B
                np.save(np_path, np.stack([ca, cb], axis=0))

                self.chunks.append((ca, cb))

    # ── Load from cache ────────────────────────────────────────────────────

    def _load_from_cache(self, split: str):
        """Load all .npy chunk files from cache directory."""
        cache_dir = CFG.samples_dir / split
        npy_files = sorted(cache_dir.glob("*.npy"))

        if len(npy_files) == 0:
            raise RuntimeError(
                f"Cache directory {cache_dir} exists but has no .npy files.\n"
                "Run with force_rebuild=True to rebuild cache."
            )

        for npy_path in npy_files:
            arr = np.load(npy_path)    # shape: [2, T]
            self.chunks.append((arr[0], arr[1]))

    # ── PyTorch Dataset interface ──────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x : [1, T] float32 tensor — noisy sector A (model input)
            y : [1, T] float32 tensor — noisy sector B (Noise2Noise target)
        """
        chunk_a, chunk_b = self.chunks[idx]

        x = torch.from_numpy(chunk_a).unsqueeze(0)   # [T] → [1, T]
        y = torch.from_numpy(chunk_b).unsqueeze(0)   # [T] → [1, T]

        return x, y


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory — single entry point for train.py
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    data_dir:      Path = CFG.data_raw_dir,
    batch_size:    int  = CFG.batch_size,
    num_workers:   int  = 0,
    force_rebuild: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build and return train, val, test DataLoaders.

    Args:
        data_dir      : path to TESS FITS files
        batch_size    : samples per batch (default from CFG)
        num_workers   : parallel data loading workers
                        0 = main process only (safe on Windows)
                        4 = recommended on Linux with fast storage
        force_rebuild : if True, reprocess all FITS files even if cache exists

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_ds = TESSSectorPairDataset("train", data_dir, force_rebuild)
    val_ds   = TESSSectorPairDataset("val",   data_dir, force_rebuild)
    test_ds  = TESSSectorPairDataset("test",  data_dir, force_rebuild)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,               # shuffle training data every epoch
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,             # drop incomplete last batch for stable training
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,              # never shuffle val/test
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
