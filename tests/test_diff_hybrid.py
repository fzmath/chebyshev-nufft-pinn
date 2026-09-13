"""Unit tests for the differentiable Chebyshev-NUFFT hybrid layer (diff_hybrid).

Run with:
    H:\\Python3.11\\python.exe tests\\test_diff_hybrid.py

Eight tests, all required to pass:
  1. torch.autograd.gradcheck on DiffHybridDerivative (double precision).
  2. 1st-derivative accuracy: u(x)=sin(pi x) on [0,1], M=500 random points.
  3. 2nd-derivative accuracy: u(x)=exp(-x^2) on [-1,1], M=500.
  4. Adjoint identity <D u, v> = <u, D^T v>.
  5. 2D Laplacian: u=sin(pi x)sin(pi y) on [0,1]^2, M=1000.
  6. Boundary layer derivative: u(x)=x^2 on [0,1].
  7. Pure Chebyshev limit: delta=0.5.
  8. Pure NUFFT limit: delta=0.01.
"""

import os
import sys
import math

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.join(_HERE, "..", "code")
sys.path.insert(0, os.path.abspath(_CODE))
sys.path.insert(0, r"H:\2026科研\JCP-Journal of Computational Physics\code")

import diff_hybrid as dh          # noqa: E402
import diff_finufft as df        # noqa: E402

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
np.random.seed(0)

PASS = []


def report(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    PASS.append(bool(ok))
    line = f"[{status}] {name}"
    if detail:
        line += f"  -- {detail}"
    print(line)


def rel_l2(err, ref):
    return float(torch.norm(err) / (torch.norm(ref) + 1e-300))


def base_params(domain):
    return {
        "domain": (float(domain[0]), float(domain[1])),
        "delta": 0.15 * (domain[1] - domain[0]),
        "overlap": 0.05 * (domain[1] - domain[0]),
        "N_cheb": 32,
        "N_fourier": 16,
        "eps_nufft": 1e-6,
        "order": 1,
    }


# -------------------------------------------------------------------------------------
def test_gradcheck():
    """1. gradcheck on the custom Function (small problem)."""
    M, Nc, Nf = 20, 8, 8
    x = torch.sort(torch.rand(M, dtype=torch.float64))[0]
    params = base_params((0.0, 1.0))
    params["N_cheb"] = Nc
    params["N_fourier"] = Nf
    op = dh.HybridOperator1D(x, params)
    A = op.A

    u = torch.randn(M, dtype=torch.float64, requires_grad=True)

    def fn(z):
        return dh.DiffHybridDerivative.apply(z, A)

    ok = torch.autograd.gradcheck(fn, u, atol=1e-4, rtol=1e-3,
                                  raise_exception=False)
    report("gradcheck DiffHybridDerivative", bool(ok))


def test_first_derivative_sin():
    """2. u(x)=sin(pi x) on [0,1], Dirichlet at endpoints, M=500 random."""
    M = 500
    x = torch.rand(M, dtype=torch.float64)
    params = base_params((0.0, 1.0))
    u = torch.sin(math.pi * x)
    op = dh.HybridOperator1D(x, params)
    du = op.apply(u)
    du_ref = math.pi * torch.cos(math.pi * x)
    err = rel_l2(du - du_ref, du_ref)
    report("1st deriv sin(pi x) [0,1] M=500", err < 1e-6, f"rel L2 = {err:.2e}")


def test_second_derivative_gaussian():
    """3. u(x)=exp(-x^2) on [-1,1], 2nd derivative, M=500."""
    M = 500
    x = torch.rand(M, dtype=torch.float64) * 2.0 - 1.0
    params = base_params((-1.0, 1.0))
    u = torch.exp(-x ** 2)
    p2 = dict(params)
    p2["order"] = 2
    op = dh.HybridOperator1D(x, p2)
    d2u = op.apply(u)
    d2u_ref = (4.0 * x ** 2 - 2.0) * torch.exp(-x ** 2)
    err = rel_l2(d2u - d2u_ref, d2u_ref)
    report("2nd deriv exp(-x^2) [-1,1] M=500", err < 1e-4, f"rel L2 = {err:.2e}")


def test_adjoint_identity():
    """4. <D u, v> = <u, D^T v> to machine precision."""
    M = 30
    x = torch.sort(torch.rand(M, dtype=torch.float64))[0]
    params = base_params((0.0, 1.0))
    params["N_cheb"] = 12
    params["N_fourier"] = 8
    op = dh.HybridOperator1D(x, params)
    A = op.A
    u = torch.randn(M, dtype=torch.float64)
    v = torch.randn(M, dtype=torch.float64)
    lhs = torch.dot(A @ u, v)
    rhs = torch.dot(u, A.T @ v)
    rel = float(abs(lhs - rhs) / (abs(lhs) + 1e-300))
    report("adjoint identity <Au,v>=<u,A^T v>", rel < 1e-8, f"rel diff = {rel:.2e}")


def test_laplacian_2d():
    """5. u=sin(pi x)sin(pi y) on [0,1]^2, M=1000, Laplacian."""
    M = 1000
    xy = torch.rand(M, 2, dtype=torch.float64)
    params = base_params((0.0, 1.0))
    u = torch.sin(math.pi * xy[:, 0]) * torch.sin(math.pi * xy[:, 1])
    lap = dh.hybrid_laplacian_2d(u, xy, params)
    lap_ref = -2.0 * math.pi ** 2 * u
    err = rel_l2(lap - lap_ref, lap_ref)
    report("2D Laplacian sin(pi x)sin(pi y) M=1000", err < 1e-4,
           f"rel L2 = {err:.2e}")


def test_boundary_derivative():
    """6. u(x)=x^2 on [0,1]; derivative 2x at the boundary layer."""
    M = 300
    x = torch.sort(torch.rand(M, dtype=torch.float64))[0]
    params = base_params((0.0, 1.0))
    u = x ** 2
    op = dh.HybridOperator1D(x, params)
    du = op.apply(u)
    du_ref = 2.0 * x
    # check the near-boundary points (left layer) specifically
    near = x < 0.15
    err_boundary = rel_l2(du[near] - du_ref[near], du_ref[near])
    err_all = rel_l2(du - du_ref, du_ref)
    ok = err_boundary < 1e-3 and err_all < 1e-3
    report("boundary derivative u=x^2 on [0,1]", ok,
           f"boundary rel L2 = {err_boundary:.2e}, all = {err_all:.2e}")


def test_pure_chebyshev_limit():
    """7. delta=0.5 covers the whole interval -> pure Chebyshev, accurate."""
    M = 500
    x = torch.rand(M, dtype=torch.float64)
    params = base_params((0.0, 1.0))
    params["delta"] = 0.5
    params["overlap"] = 0.02
    u = torch.sin(math.pi * x)
    op = dh.HybridOperator1D(x, params)
    du = op.apply(u)
    du_ref = math.pi * torch.cos(math.pi * x)
    err = rel_l2(du - du_ref, du_ref)
    report("pure Chebyshev limit delta=0.5", err < 1e-5, f"rel L2 = {err:.2e}")


def test_pure_nufft_limit():
    """8. delta=0.01: interior dominates; compare interior to finufft."""
    M = 500
    x = torch.sort(torch.rand(M, dtype=torch.float64))[0]
    params = base_params((0.0, 1.0))
    params["delta"] = 0.01
    u = torch.sin(math.pi * x)
    op = dh.HybridOperator1D(x, params)
    du = op.apply(u)
    du_ref = math.pi * torch.cos(math.pi * x)
    # deep interior where w_I ~ 1
    deep = (x > 0.2) & (x < 0.8)
    err_deep = rel_l2(du[deep] - du_ref[deep], du_ref[deep])
    report("pure NUFFT limit delta=0.01", err_deep < 5e-2,
           f"deep-interior rel L2 = {err_deep:.2e}")


if __name__ == "__main__":
    print("=" * 72)
    print("Differentiable Chebyshev-NUFFT Hybrid Layer -- unit tests")
    print(f"torch {torch.__version__} | numpy {np.__version__} | dtype float64")
    print("=" * 72)

    test_gradcheck()
    test_first_derivative_sin()
    test_second_derivative_gaussian()
    test_adjoint_identity()
    test_laplacian_2d()
    test_boundary_derivative()
    test_pure_chebyshev_limit()
    test_pure_nufft_limit()

    print("-" * 72)
    n_pass = int(sum(PASS))
    n_total = len(PASS)
    print(f"Result: {n_pass}/{n_total} tests passed")
    print("=" * 72)
    sys.exit(0 if n_pass == n_total else 1)
