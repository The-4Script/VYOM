"""
denoiser/dataset.py  — PATCHED for VYOM folder structure

Changes from original:
  1. _scan_fits_files()  — rewrote to match TIC_{id}/mastDownload/TESS/{sector_folder}/*.fits
  2. _chunk()            — gap-aware: never chunks across Earth occultation gaps
  3. _build_pairs_index()— uses closest sector pair, not just first two
  4. _read_fits()        — unchanged (was correct)
  5. Everything else     — unchanged

Folder structure expected:
  data/raw/tess/
    TIC_384549882/
      mastDownload/
        TESS/
          tess2021039152502-s0035-0000000384549882-0205-s/
            tess2021039152502-s0035-0000000384549882-0205-s_lc.fits
"""

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
# FITS reading + preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def _read_fits(fits_path: Path) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """
    Read a TESS FITS file and return (time, flux) after cleaning.
    Returns None if file is unreadable or has too few valid points.

    Steps:
      1. Extract TIME + PDCSAP_FLUX + QUALITY
      2. Apply TESS quality bitmask
      3. Remove NaN
      4. 5-sigma outlier clipping
      5. Normalize: (flux - median) / MAD

    Returns:
        (time, flux) both float32 arrays — time needed for gap detection
    """
    try:
        from astropy.io import fits as astrofits
    except ImportError:
        raise ImportError("astropy not installed. Run: pip install astropy")

    try:
        with astrofits.open(fits_path) as hdul:
            data    = hdul[1].data
            time    = data['TIME'].astype(np.float64)
            flux    = data['PDCSAP_FLUX'].astype(np.float32)
            quality = data['QUALITY'].astype(np.int32)
    except Exception as e:
        logger.warning(f"Could not read {fits_path}: {e}")
        return None

    # Quality bitmask
    BAD_BITS  = 1 | 2 | 4 | 8 | 32 | 64 | 512
    good_mask = ((quality & BAD_BITS) == 0) & np.isfinite(flux) & np.isfinite(time)
    time = time[good_mask]
    flux = flux[good_mask]

    if len(flux) < CFG.chunk_length:
        logger.warning(f"Too few points ({len(flux)}) in {fits_path.name} — skipping")
        return None

    # 5-sigma outlier clipping
    median    = np.median(flux)
    mad       = np.median(np.abs(flux - median))
    mad       = mad if mad > 0 else 1e-8
    sigma     = 1.4826 * mad
    clip_mask = np.abs(flux - median) < 5.0 * sigma
    time      = time[clip_mask]
    flux      = flux[clip_mask]

    if len(flux) < CFG.chunk_length:
        return None

    # Normalize
    median = np.median(flux)
    mad    = np.median(np.abs(flux - median))
    mad    = mad if mad > 0 else 1e-8
    flux   = (flux - median) / (1.4826 * mad)

    return time.astype(np.float32), flux.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Gap-aware chunking  ← PATCHED
# ─────────────────────────────────────────────────────────────────────────────

def _chunk(
    flux:   np.ndarray,
    time:   np.ndarray,
    length: int,
    stride: int,
) -> list[np.ndarray]:
    """
    Split flux into fixed-length chunks, NEVER crossing Earth occultation gaps.

    A gap is detected when the time difference between consecutive cadences
    is > 10× the median cadence. Each continuous segment is chunked independently.

    Args:
        flux   : 1D normalized flux array
        time   : 1D time array (same length as flux)
        length : chunk size (CFG.chunk_length = 1000)
        stride : overlap stride (CFG.chunk_stride = 500)

    Returns:
        List of 1D arrays each of shape [length]
    """
    if len(time) < 2:
        return []

    # Find gap positions
    dt         = np.diff(time)
    median_dt  = np.median(dt)
    gap_mask   = dt > (10.0 * median_dt)
    gap_idx    = np.where(gap_mask)[0] + 1   # index of first point AFTER each gap

    # Segment boundaries: [0, gap1, gap2, ..., end]
    boundaries = [0] + gap_idx.tolist() + [len(flux)]

    chunks = []
    for i in range(len(boundaries) - 1):
        seg_start = boundaries[i]
        seg_end   = boundaries[i + 1]
        segment   = flux[seg_start:seg_end]

        # Chunk this segment independently
        start = 0
        while start + length <= len(segment):
            chunks.append(segment[start : start + length])
            start += stride

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# TIC ID and sector extraction from filename
# ─────────────────────────────────────────────────────────────────────────────

def _extract_tic_id(fits_path: Path) -> Optional[str]:
    """Extract TIC ID from TESS FITS filename (16-digit zero-padded)."""
    name  = fits_path.stem
    match = re.search(r'-(\d{16})-', name)
    if match:
        return str(int(match.group(1)))
    match = re.search(r'(\d{8,})', name)
    if match:
        return str(int(match.group(1)))
    logger.warning(f"Could not extract TIC ID from: {fits_path.name}")
    return None


def _extract_sector(fits_path: Path) -> Optional[int]:
    """Extract sector number from TESS FITS filename."""
    name  = fits_path.stem
    match = re.search(r'-s(\d{4})-', name)
    if match:
        return int(match.group(1))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# FITS scanner  ← PATCHED for your folder structure
# ─────────────────────────────────────────────────────────────────────────────

def _scan_fits_files(data_dir: Path) -> dict[str, dict[int, Path]]:
    """
    Scan data_dir for TESS FITS files matching VYOM folder structure:

      data_dir/
        TIC_{tic_id}/
          mastDownload/
            TESS/
              tess...-s{sector:04d}-{tic:016d}-.../ 
                *.fits

    Returns:
        { "tic_id_str": { sector_int: fits_path, ... }, ... }
    """
    tic_to_sectors: dict[str, dict[int, Path]] = {}

    # Iterate TIC folders directly — much faster than rglob("*.fits") on deep trees
    for tic_folder in data_dir.iterdir():
        if not tic_folder.is_dir():
            continue

        # Expect folder named TIC_{number}
        m = re.match(r'TIC_(\d+)$', tic_folder.name)
        if m is None:
            continue

        tic_id = str(int(m.group(1)))   # strip leading zeros

        mast_tess = tic_folder / "mastDownload" / "TESS"
        if not mast_tess.exists():
            continue

        for sector_folder in mast_tess.iterdir():
            if not sector_folder.is_dir():
                continue

            # Sector number from folder name  e.g. tess...-s0035-...
            sm = re.search(r'-s(\d{4})-', sector_folder.name)
            if sm is None:
                continue
            sector = int(sm.group(1))

            fits_files = list(sector_folder.glob("*.fits"))
            if not fits_files:
                continue

            if tic_id not in tic_to_sectors:
                tic_to_sectors[tic_id] = {}

            tic_to_sectors[tic_id][sector] = fits_files[0]

    n_total  = sum(len(v) for v in tic_to_sectors.values())
    n_multi  = sum(1 for v in tic_to_sectors.values() if len(v) >= 2)
    logger.info(
        f"Scanned {data_dir} — "
        f"{len(tic_to_sectors)} TIC IDs, "
        f"{n_total} FITS files, "
        f"{n_multi} stars with 2+ sectors"
    )
    return tic_to_sectors


# ─────────────────────────────────────────────────────────────────────────────
# Train / Val / Test split — always by TIC ID
# ─────────────────────────────────────────────────────────────────────────────

def _split_tic_ids(
    tic_ids:    list[str],
    train_frac: float = CFG.train_frac,
    val_frac:   float = CFG.val_frac,
    seed:       int   = 42,
) -> tuple[list[str], list[str], list[str]]:
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(tic_ids))
    rng.shuffle(ids)

    n       = len(ids)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train_ids = ids[:n_train].tolist()
    val_ids   = ids[n_train : n_train + n_val].tolist()
    test_ids  = ids[n_train + n_val :].tolist()

    logger.info(
        f"TIC ID split — train: {len(train_ids)}, "
        f"val: {len(val_ids)}, test: {len(test_ids)}"
    )
    return train_ids, val_ids, test_ids


# ─────────────────────────────────────────────────────────────────────────────
# Pairs index  ← PATCHED: picks closest sector pair, not just first two
# ─────────────────────────────────────────────────────────────────────────────

def _build_pairs_index(
    tic_to_sectors: dict[str, dict[int, Path]],
    split_ids:      list[str],
    split:          str,
) -> list[tuple[Path, Path]]:
    """
    For each TIC ID pick the CLOSEST sector pair (minimum gap between sectors).
    Writes pairs_index.csv for auditing.

    Returns:
        List of (sector_A_path, sector_B_path)
    """
    pairs    = []
    csv_rows = [["tic_id", "sector_a", "sector_b", "sector_gap", "path_a", "path_b"]]

    for tic_id in split_ids:
        sector_map = tic_to_sectors.get(tic_id, {})
        sectors    = sorted(sector_map.keys())

        if len(sectors) < 2:
            continue

        # Find the consecutive pair with smallest gap
        best_gap  = 999
        best_a    = sectors[0]
        best_b    = sectors[1]

        for i in range(len(sectors) - 1):
            gap = sectors[i + 1] - sectors[i]
            if gap < best_gap:
                best_gap = gap
                best_a   = sectors[i]
                best_b   = sectors[i + 1]

        path_a = sector_map[best_a]
        path_b = sector_map[best_b]

        pairs.append((path_a, path_b))
        csv_rows.append([tic_id, best_a, best_b, best_gap, str(path_a), str(path_b)])

    # Write audit CSV
    csv_path = CFG.samples_dir / split / "pairs_index.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    logger.info(f"[{split}] {len(pairs)} pairs built")
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Cache path
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path(
    tic_id: str, sector_a: int, sector_b: int,
    split: str, idx: int,
) -> Path:
    folder = CFG.samples_dir / split
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"tic{tic_id}_s{sector_a:02d}s{sector_b:02d}_chunk{idx:04d}.npy"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TESSSectorPairDataset(Dataset):
    """
    PyTorch Dataset for Noise2Noise denoiser training.

    Each sample: (chunk_A, chunk_B) — both [1, T] tensors.
    Model input = chunk_A, N2N target = chunk_B.

    Args:
        split        : "train", "val", or "test"
        data_dir     : root dir containing TIC_* folders
        force_rebuild: ignore cache and rebuild from FITS
    """

    def __init__(
        self,
        split:         str  = "train",
        data_dir:      Path = CFG.data_raw_dir,
        force_rebuild: bool = False,
    ):
        assert split in ("train", "val", "test"), \
            f"split must be train/val/test — got '{split}'"

        self.split  = split
        self.chunks: list[tuple[np.ndarray, np.ndarray]] = []

        cache_index = CFG.samples_dir / split / "pairs_index.csv"

        if cache_index.exists() and not force_rebuild:
            logger.info(f"[{split}] Loading from cache")
            self._load_from_cache(split)
        else:
            logger.info(f"[{split}] Building from FITS")
            self._build_from_fits(split, data_dir)

        logger.info(f"[{split}] Ready — {len(self.chunks)} chunk pairs")

    def _build_from_fits(self, split: str, data_dir: Path):
        tic_to_sectors = _scan_fits_files(data_dir)

        multi_sector = {
            tic: secs for tic, secs in tic_to_sectors.items()
            if len(secs) >= 2
        }

        if not multi_sector:
            raise RuntimeError(
                f"No multi-sector TESS data found in {data_dir}\n"
                "Check that TIC_* folders exist with mastDownload/TESS/ inside."
            )

        all_tic_ids              = list(multi_sector.keys())
        train_ids, val_ids, test_ids = _split_tic_ids(all_tic_ids)
        split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}[split]

        pairs = _build_pairs_index(multi_sector, split_ids, split)

        for path_a, path_b in pairs:
            result_a = _read_fits(path_a)
            result_b = _read_fits(path_b)

            if result_a is None or result_b is None:
                continue

            time_a, flux_a = result_a
            time_b, flux_b = result_b

            # Gap-aware chunking — pass time arrays
            chunks_a = _chunk(flux_a, time_a, CFG.chunk_length, CFG.chunk_stride)
            chunks_b = _chunk(flux_b, time_b, CFG.chunk_length, CFG.chunk_stride)

            n_pairs = min(len(chunks_a), len(chunks_b))
            if n_pairs == 0:
                continue

            tic_id = _extract_tic_id(path_a)
            sec_a  = _extract_sector(path_a)
            sec_b  = _extract_sector(path_b)

            for idx in range(n_pairs):
                ca = chunks_a[idx]
                cb = chunks_b[idx]

                np_path = _cache_path(tic_id, sec_a, sec_b, split, idx)
                np.save(np_path, np.stack([ca, cb], axis=0))   # [2, T]

                self.chunks.append((ca, cb))

    def _load_from_cache(self, split: str):
        cache_dir = CFG.samples_dir / split
        npy_files = sorted(cache_dir.glob("*.npy"))

        if not npy_files:
            raise RuntimeError(
                f"Cache dir {cache_dir} has no .npy files. "
                "Run with force_rebuild=True."
            )

        for npy_path in npy_files:
            arr = np.load(npy_path)        # [2, T]
            self.chunks.append((arr[0], arr[1]))

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ca, cb = self.chunks[idx]
        x = torch.from_numpy(ca).unsqueeze(0)   # [1, T]
        y = torch.from_numpy(cb).unsqueeze(0)   # [1, T]
        return x, y


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    data_dir:      Path = CFG.data_raw_dir,
    batch_size:    int  = CFG.batch_size,
    num_workers:   int  = 0,
    force_rebuild: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader).

    num_workers=0 is safe on Windows.
    Set force_rebuild=True first time to build cache from FITS.
    """
    train_ds = TESSSectorPairDataset("train", data_dir, force_rebuild)
    val_ds   = TESSSectorPairDataset("val",   data_dir, force_rebuild)
    test_ds  = TESSSectorPairDataset("test",  data_dir, force_rebuild)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader
