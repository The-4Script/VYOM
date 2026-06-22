# pipeline/run_pipeline.py

import os
import csv
from pathlib import Path
import numpy as np
from astropy.io import fits
from pipeline.stage0_preprocessing import process_single_fits
from denoiser.config import CFG # Dynamically pulls your team's configuration rules

RAW_DIR = Path("data/raw")
PREPROCESSED_DIR = Path("data/preprocessed")
METADATA_CSV = PREPROCESSED_DIR / "stellar_registry.csv"

def extract_header_metadata(fits_path):
    """Parses structural metadata from primary FITS header for downstream Parameter Estimation."""
    try:
        with fits.open(fits_path) as hdul:
            header = hdul[0].header
            return {
                "tic_id": str(header.get("OBJECT", "")).replace("TIC", "").strip(),
                "teff": header.get("TEFF", "UNKNOWN"),
                "radius": header.get("RADIUS", "UNKNOWN"),
                "mass": header.get("MASS", "UNKNOWN")
            }
    except Exception:
        return None

def run_stage_0():
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    fits_files = list(RAW_DIR.glob("*.fits"))
    
    if not fits_files:
        print(f"[!] No FITS files found in {RAW_DIR}. Please place testing data there.")
        return

    print(f"[*] Beginning Stage 0 Processing on {len(fits_files)} files...")
    metadata_rows = []

    for path in fits_files:
        # Extract star traits before cleaning
        meta = extract_header_metadata(path)
        if not meta:
            continue
            
        # Process arrays to match CFG.chunk_length (1000)
        chunks, _ = process_single_fits(path, chunk_length=CFG.chunk_length)
        
        if chunks is not None and len(chunks) > 0:
            # Save preprocessed matrices as immediate local binary arrays
            output_filename = PREPROCESSED_DIR / f"TIC_{meta['tic_id']}_input.npy"
            np.save(output_filename, chunks)
            metadata_rows.append(meta)
            print(f"[✓] Processed TIC {meta['tic_id']} -> Generated {len(chunks)} tensors.")

    # Write registry mapping file
    with open(METADATA_CSV, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tic_id", "teff", "radius", "mass"])
        writer.writeheader()
        writer.writerows(metadata_rows)
        
    print(f"[+✓+] Stage 0 Complete. Stellar metadata registry built at {METADATA_CSV}")

if __name__ == "__main__":
    run_stage_0()