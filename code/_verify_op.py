"""Verify hybrid operator on a known function."""
import sys
sys.path.insert(0, r'H:\2026科研\Chebyshev-NUFFT-PINN\code')
import numpy as np
import torch
import diff_hybrid as dh

torch.set_default_dtype(torch.float64)

# Test function: u(x) = sin(pi x) + 0.5 sin(3 pi x)
# u''(x) = -pi^2 sin(pi x) - 4.5 pi^2 sin(3 pi x)
x_np = np.linspace(0.001, 0.999, 2000)
u_np = np.sin(np.pi * x_np) + 0.5 * np.sin(3 * np.pi * x_np)
u_pp_exact = -(np.pi**2) * np.sin(np.pi * x_np) \
             - 4.5 * (np.pi**2) * np.sin(3 * np.pi * x_np)

params = dict(domain=(0.0, 1.0), delta=0.15, overlap=0.05,
              N_cheb=32, N_fourier=16, eps_nufft=1e-6,
              order=2, interior_method='nufft')
op = dh.HybridOperator1D(x_np, params)

u_t = torch.tensor(u_np, dtype=torch.float64)
u_pp_computed = (op.A @ u_t).numpy()

print("Hybrid operator (interior_method='nufft'):")
print("  max |computed - exact| =", np.max(np.abs(u_pp_computed - u_pp_exact)))
print("  max |computed| =", np.max(np.abs(u_pp_computed)))
print("  max |exact|    =", np.max(np.abs(u_pp_exact)))
print("  ratio (mean)   =", np.mean(np.abs(u_pp_computed)) / np.mean(np.abs(u_pp_exact)))

# Also try interior_method='cheb'
params2 = dict(domain=(0.0, 1.0), delta=0.15, overlap=0.05,
               N_cheb=32, N_fourier=16, eps_nufft=1e-6,
               order=2, interior_method='cheb')
op2 = dh.HybridOperator1D(x_np, params2)
u_pp_cheb = (op2.A @ u_t).numpy()
print("\nHybrid operator (interior_method='cheb'):")
print("  max |computed - exact| =", np.max(np.abs(u_pp_cheb - u_pp_exact)))
print("  max |computed| =", np.max(np.abs(u_pp_cheb)))

# Check: what if we manually scale the interior?
# The interior NUFFT block computes d2/dtheta2, needs factor (2pi/L_I)^2
a, b = 0.0, 1.0
delta = 0.15
L_I = b - a - 2 * delta
scale_I = (2 * np.pi / L_I) ** 2
print(f"\nInterior physical scale (2pi/L_I)^2 = {scale_I:.4f}")
print(f"Expected ratio ~ 1/{scale_I:.4f} = {1.0/scale_I:.6f}")
