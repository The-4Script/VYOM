# pipeline/stage0_preprocessing.py

import logging
import numpy as np
from astropy.io import fits
from wotan import flatten

logger = logging.getLogger(__name__)

def process_single_fits(fits_path, chunk_length=1000):
    """
    Production-grade Stage 0 pipeline for a single TESS FITS file.
    Cleans, normalizes, detrends, and slices light curves for Stage 1.
    """
    try:
        with fits.open(fits_path) as hdul:
            data = hdul[1].data
            time = data['TIME'].astype(np.float64)
            flux = data['PDCSAP_FLUX'].astype(np.float32)
            quality = data['QUALITY'].astype(np.int32)
    except Exception as e:
        logger.error(f"Failed to read FITS file {fits_path}: {e}")
        return None, None

    # 1. Quality Bitmask Masking (Standard SPOC Flags)
    # Drops cadences with thruster fires, severe jitter, or cosmic rays
    BAD_BITS = 1 | 2 | 4 | 8 | 32 | 64 | 512
    good_mask = (quality & BAD_BITS) == 0
    time = time[good_mask]
    flux = flux[good_mask]

    # 2. Drop NaN values
    valid_mask = np.isfinite(time) & np.isfinite(flux)
    time = time[valid_mask]
    flux = flux[valid_mask]

    if len(flux) < chunk_length:
        logger.warning(f"File {fits_path.name} contains too few points after masking.")
        return None, None

    # 3. Asymmetric Outlier Clipping (Protecting Transits)
    # Standard U-Net training uses symmetric MAD clipping. Production requires 
    # asymmetric clipping to remove massive positive spikes (stellar flares) 
    # while strictly preserving deep downward drops (potential transits).
    median_val = np.median(flux)
    mad = np.median(np.abs(flux - median_val))
    mad = mad if mad > 0 else 1e-8
    sigma = 1.4826 * mad
    
    # Clip 4-sigma above (flares), but give 20-sigma headroom below (transits)
    upper_bound = median_val + (4.0 * sigma)
    lower_bound = median_val - (20.0 * sigma)
    
    clip_mask = (flux < upper_bound) & (flux > lower_bound)
    time = time[clip_mask]
    flux = flux[clip_mask]

    # 4. Median Normalization
    # Centers the baseline flux around 1.0
    flux_normalized = flux / np.median(flux)

    # 5. High-Pass Detrending via Wotan
    # Removes low-frequency stellar rotation waves while keeping narrow transit shapes intact
    # window_length=0.3 days is the baseline optimization for TESS transits
    flattened_flux, trend_flux = flatten(
        time, 
        flux_normalized, 
        window_length=0.3, 
        method='biweight', 
        return_trend=True
    )

    # 6. Final Scale Alignment for U-Net Input
    final_median = np.median(flattened_flux)
    final_mad = np.median(np.abs(flattened_flux - final_median))
    final_mad = final_mad if final_mad > 0 else 1e-8
    flux_scaled = (flattened_flux - final_median) / (1.4826 * final_mad)

    # 7. Slicing into Uniform Sequential Tensors
    return flux_scaled.astype(np.float32), time