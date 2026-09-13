"""Quick 2D timing test: 200 steps, M=5000."""
import sys, time
sys.path.insert(0, r'H:\2026科研\Chebyshev-NUFFT-PINN\code')
import pinn_experiments as pe
import torch
torch.set_default_dtype(torch.float64)

for method in ['hybrid', 'ad']:
    t0 = time.time()
    r = pe.train_pinn_2d_poisson(method, dict(seed=42, M=5000, steps=200, warmup=50))
    dt = time.time() - t0
    print('{}: 200 steps, M=5000: {:.1f}s total, {:.3f}s/step'.format(
        method, dt, dt/201), flush=True)
