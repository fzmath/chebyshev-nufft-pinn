"""Spectral-convergence study for the differentiable Chebyshev layer.

Computes the first derivative of  u(x) = exp(sin(pi x))  on [-1, 1] with the
differentiation-matrix layer for N = 8, 12, ..., 128 and plots the discrete L2
error against N on a log scale.  For an analytic, periodic-in-the-transformed-
variable function the Chebyshev error decays exponentially, so the plot is a
straight line (on a log-linear axes) until double-precision roundoff (~1e-14)
is reached.

Run:
    H:\\Python3.11\\python.exe code\\convergence_chebyshev.py
Output:
    figures/chebyshev_convergence.png
"""

import os
import sys
import math

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(_HERE))
import diff_chebyshev as dc  # noqa: E402

torch.set_default_dtype(torch.float64)

DOMAIN = (-1.0, 1.0)
OUT = os.path.join(_HERE, "..", "figures", "chebyshev_convergence.png")


def reference_derivative(x):
    """Analytic d/dx exp(sin(pi x)) = pi cos(pi x) exp(sin(pi x))."""
    return math.pi * torch.cos(math.pi * x) * torch.exp(torch.sin(math.pi * x))


def main():
    Ns = list(range(8, 129, 4))          # 8, 12, ..., 128
    errs = []
    for N in Ns:
        D, x = dc.chebyshev_diff_matrix(N)
        u = torch.exp(torch.sin(math.pi * x))
        du_num = dc.chebyshev_derivative(u, N, order=1, domain=DOMAIN)
        du_ref = reference_derivative(x)
        err = float(torch.norm(du_num - du_ref) / (torch.norm(du_ref) + 1e-300))
        errs.append(err)
        print(f"N = {N:4d}   rel L2 error = {err:.3e}")

    errs = np.array(errs)
    Ns = np.array(Ns)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=130)
    ax.semilogy(Ns, errs, "o-", color="#1f4e79", lw=1.8, ms=5,
                label="Chebyshev (matrix layer)")
    ax.axhline(1e-14, color="gray", ls="--", lw=1.0, label="machine roundoff (~1e-14)")

    # Annotate the exponential (spectral) slope.
    ax.set_xlabel(r"number of intervals $N$  ($N{+}1$ Lobatto nodes)", fontsize=12)
    ax.set_ylabel(r"relative $L^2$ error  $\|u'_\mathrm{num}-u'_\mathrm{exact}\|_2 / \|u'_\mathrm{exact}\|_2$",
                  fontsize=11)
    ax.set_title(r"Spectral convergence: $u(x)=e^{\sin(\pi x)}$ on $[-1,1]$",
                 fontsize=13)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10, loc="lower left")
    ax.set_xlim(0, 135)

    fig.tight_layout()
    out_path = os.path.abspath(OUT)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    print(f"\nSaved convergence figure to: {out_path}")


if __name__ == "__main__":
    main()
