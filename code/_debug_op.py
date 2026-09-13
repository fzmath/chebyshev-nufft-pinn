"""Check error distribution and quick training test."""
import sys
sys.path.insert(0, r'H:\2026科研\Chebyshev-NUFFT-PINN\code')
import numpy as np
import torch
import diff_hybrid as dh
import pinn_experiments as pe

torch.set_default_dtype(torch.float64)

# Check error distribution
x_np = np.linspace(0.001, 0.999, 2000)
u_np = np.sin(np.pi * x_np) + 0.5 * np.sin(3 * np.pi * x_np)
u_pp_exact = -(np.pi**2) * np.sin(np.pi * x_np) \
             - 4.5 * (np.pi**2) * np.sin(3 * np.pi * x_np)

params = dict(domain=(0.0, 1.0), delta=0.15, overlap=0.05,
              N_cheb=32, N_fourier=16, eps_nufft=1e-6,
              order=2, interior_method='nufft')
op = dh.HybridOperator1D(x_np, params)
u_t = torch.tensor(u_np, dtype=torch.float64)
u_pp = (op.A @ u_t).numpy()
err = np.abs(u_pp - u_pp_exact)

print("Error distribution along x:")
for lo, hi in [(0.0, 0.15), (0.15, 0.25), (0.25, 0.75), (0.75, 0.85), (0.85, 1.0)]:
    mask = (x_np >= lo) & (x_np < hi)
    print(f"  x in [{lo:.2f}, {hi:.2f}]: max_err={err[mask].max():.4e}, "
          f"mean_ratio={np.mean(np.abs(u_pp[mask]))/np.mean(np.abs(u_pp_exact[mask])):.4f}")

# Quick training test: 500 steps
print("\nQuick training test (hybrid, 500 steps, M=500):")
r = pe.train_pinn_1d_poisson('hybrid', dict(seed=42, M=500, steps=500, warmup=100))
print(f"  Final L2 = {r['l2']:.4e}, time = {r['time']:.1f}s")
