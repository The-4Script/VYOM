import torch
import sys
sys.path.insert(0, ".")

from denoiser.losses import VyomDenoiseLoss

criterion = VyomDenoiseLoss()

pred   = torch.randn(4, 1, 1000, requires_grad=True)  # requires_grad=True
target = torch.randn(4, 1, 1000)
target[:, :, 400:420] -= 0.5  # fake transit dip

loss, components = criterion(pred, target)

print(f"Total loss    : {components['total']:.4f}")
print(f"Combined      : {components['combined']:.4f}")
print(f"Transit pres  : {components['transit_pres']:.4f}")

assert loss.requires_grad == True, "❌ requires_grad False — backprop nahi chalega"
assert components['transit_pres'] > components['combined'], "❌ dip weight kaam nahi kar raha"
assert components['total'] > 0, "❌ loss zero ya negative hai"

print("✅ losses.py verified — sab theek hai")