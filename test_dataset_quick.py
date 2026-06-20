# test_dataset_quick.py
import torch
import numpy as np
import sys
sys.path.insert(0, ".")

from denoiser.dataset import _split_tic_ids, _chunk, TESSSectorPairDataset

# Test 1: TIC ID split — no overlap between sets
fake_ids = [str(i) for i in range(100)]
train_ids, val_ids, test_ids = _split_tic_ids(fake_ids)

assert len(set(train_ids) & set(val_ids))  == 0, "❌ train/val overlap"
assert len(set(train_ids) & set(test_ids)) == 0, "❌ train/test overlap"
assert len(set(val_ids)   & set(test_ids)) == 0, "❌ val/test overlap"
assert len(train_ids) + len(val_ids) + len(test_ids) == 100
print(f"✅ TIC ID split — train:{len(train_ids)} val:{len(val_ids)} test:{len(test_ids)}")

# Test 2: chunking logic
flux = np.random.randn(3000).astype(np.float32)
chunks = _chunk(flux, length=1000, stride=500)
assert all(len(c) == 1000 for c in chunks), "❌ chunk size wrong"
assert len(chunks) == 5   # (3000 - 1000) / 500 + 1 = 5
print(f"✅ Chunking — {len(chunks)} chunks from 3000 points")

print("\n✅ dataset.py verified — sab theek hai")