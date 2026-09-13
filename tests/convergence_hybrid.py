"""Convergence study for the differentiable Chebyshev-NUFFT hybrid layer.

Reference function: u(x) = exp(sin(pi x)) on [0, 1].
Vary N_cheb in {8, 16, 32, 64} and N_fourier in {8, 16, 32}; measure the
relative L2 error of the first hybrid derivative.  Produces a heatmap and a
convergence curve, saved to figures/hybrid_convergence.png.

Run:
    H:\\Python3.11\\python.exe tests\\convergence_hybrid.py
"""

import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.join(_HERE, "..", "code")
sys.path.insert(0, os.path.abspath(_CODE))

import diff_hybrid as dh  # noqa: E402

torch.set_default_dtype(torch.float64)
np.random.seed(0)

# ---- problem ---------------------------------------------------------------
M = 1000
x = torch.rand(M, dtype=torch.float64)
u = torch.exp(torch.sin(np.pi * x))
# exact derivative: u'(x) = exp(sin(pi x)) * pi cos(pi x)
du_ref = u * (np.pi * torch.cos(np.pi * x))

base = {
    "domain": (0.0, 1.0),
    "delta": 0.15,
    "overlap": 0.05,
    "eps_nufft": 1e-6,
    "order": 1,
}

N_cheb_list = [8, 16, 32, 64]
N_fourier_list = [8, 16, 32]

err = np.zeros((len(N_cheb_list), len(N_fourier_list)))
for i, nc in enumerate(N_cheb_list):
    for j, nf in enumerate(N_fourier_list):
        p = dict(base)
        p["N_cheb"] = nc
        p["N_fourier"] = nf
        op = dh.HybridOperator1D(x, p)
        du = op.apply(u)
        err[i, j] = float(torch.norm(du - du_ref) / torch.norm(du_ref))
        print(f"N_cheb={nc:3d}  N_fourier={nf:3d}  rel L2 = {err[i,j]:.3e}")

# ---- figure ----------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

# heatmap
im = ax1.imshow(err, aspect="auto", cmap="viridis_r",
                norm=matplotlib.colors.LogNorm(vmin=err.min(), vmax=err.max()))
ax1.set_xticks(range(len(N_fourier_list)), labels=N_fourier_list)
ax1.set_yticks(range(len(N_cheb_list)), labels=N_cheb_list)
ax1.set_xlabel(r"$N_{fourier}$ (NUFFT modes)")
ax1.set_ylabel(r"$N_{cheb}$ (Chebyshev modes)")
ax1.set_title("Relative $L^2$ error of $u'$, $u=e^{\\sin(\\pi x)}$")
for i in range(len(N_cheb_list)):
    for j in range(len(N_fourier_list)):
        ax1.text(j, i, f"{err[i,j]:.1e}", ha="center", va="center",
                 color="white", fontsize=8)
fig.colorbar(im, ax=ax1, label="rel L2 error")

# convergence curve vs N_cheb (N_fourier = 16)
j0 = N_fourier_list.index(16)
ax2.semilogy(N_cheb_list, err[:, j0], "o-", color="C0", lw=2,
             label=r"$N_{fourier}=16$")
ax2.semilogy(N_cheb_list, err[:, 0], "s--", color="C1", lw=1.5,
             label=r"$N_{fourier}=8$")
ax2.semilogy(N_cheb_list, err[:, -1], "^--", color="C2", lw=1.5,
             label=r"$N_{fourier}=32$")
ax2.set_xlabel(r"$N_{cheb}$")
ax2.set_ylabel("relative $L^2$ error")
ax2.set_title("Spectral convergence of the hybrid derivative")
ax2.grid(True, which="both", alpha=0.3)
ax2.legend()

fig.tight_layout()
out = os.path.join(_HERE, "..", "figures", "hybrid_convergence.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150)
print("saved:", os.path.abspath(out))
