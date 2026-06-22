# test_real_handshake.py
import numpy as np
import torch
from denoiser.dataset import _chunk

# Load the file Stage 0 just processed for you
try:
    processed_flux = np.load("data/preprocessed/TIC_50365310_input.npy")
    print(f"[*] Loaded continuous array from Stage 0. Total points: {len(processed_flux)}")
    
    # Pass it through Durvesh's chunker with a 500 stride
    chunks = _chunk(processed_flux, length=1000, stride=500)
    print(f"✅ Handshake Success! Durvesh's chunker generated {len(chunks)} overlapping tensors for the U-Net.")
    
except Exception as e:
    print(f"❌ Handshake failed: {e}")