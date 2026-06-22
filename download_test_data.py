# download_test_data.py
import os
import shutil
from pathlib import Path

try:
    import lightkurve as lk
except ImportError:
    print("Installing lightkurve...")
    os.system("pip install lightkurve")
    import lightkurve as lk

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

import pandas as pd
from pathlib import Path

# 1. Define paths
CSV_PATH = Path("TOI_2026.06.21_04.30.50.csv")
RAW_DIR = Path("data/raw")

# 2. Read the TOI CSV and extract TIC IDs dynamically
df = pd.read_csv(CSV_PATH)
# Convert unique IDs to the "TIC <number>" format lightkurve expects
# Extract only the first 3 unique targets to verify it works safely
test_targets = [f"TIC {int(tid)}" for tid in df['tid'].unique()[:3]]

print(f"[✓] Successfully loaded {len(test_targets)} targets from TOI catalog!")
print(f"Sample targets to process: {test_targets[:3]}")

# ... now your existing download loop runs completely dynamically over test_targets!

for target in test_targets:
    print(f"Searching for {target}...")
    
    tic_digits = target.replace("TIC", "").strip()
    existing_files = list(RAW_DIR.glob(f"*{tic_digits}*_lc.fits"))
    
    # Check if file exists AND has actual data inside it (> 10 KB)
    if existing_files and existing_files[0].stat().st_size > 10240:
        print(f" [->] {target} already exists locally and is valid. Skipping download!\n")
        continue
    elif existing_files:
        print(f" [!] Found corrupted/empty file for {target}. Deleting and re-downloading...")
        existing_files[0].unlink()

    # Proceed with clean download
    try:
        search = lk.search_lightcurve(target, mission="TESS", author="SPOC")
        if len(search) > 0:
            sector_num = search.table['sequence_number'][0]
            print(f" -> Found Sector {sector_num}. Downloading clean file...")
            
            # force extraction of path
            lc_object = search[0].download(download_dir=str(RAW_DIR))
            temp_path = Path(lc_object.filename)
            
            final_path = RAW_DIR / temp_path.name
            shutil.move(str(temp_path), str(final_path))
            print(f" [✓] Saved directly to: {final_path}\n")
        else:
            print(f" [X] No SPOC files found for {target}\n")
    except Exception as e:
        print(f" [X] Error handling {target}: {e}\n")

# 3. Clean up the empty nested folder left behind by MAST
if (RAW_DIR / "mastDownload").exists():
    shutil.rmtree(RAW_DIR / "mastDownload")

print("[✓] Ready! Check your data/raw/ folder—the files are placed perfectly.")