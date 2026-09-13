"""Unit tests for the differentiable Chebyshev spectral layer (diff_chebyshev).

Run with:
    H:\\Python3.11\\python.exe tests\\test_diff_chebyshev.py

Seven tests, all required to pass:
  1. torch.autograd.gradcheck on DiffChebyshevDerivative (double precision).
  2. First-derivative spectral accuracy: u(x)=sin(x) on [-pi,pi], N=32.
  3. Second-derivative spectral accuracy: u(x)=exp(-x^2) on [-1,1], N=64.
  4. 2D Laplacian accuracy: u=sin(x)sin(y) on [-pi,pi]^2, N=32.
  5. Adjoint identity <D u, v> = <u, D^T v>.
  6. DCT method (B) vs differentiation-matrix method (A) consistency.
  7. Dirichlet boundary enforcement: u(x)=x^2 on [0,1].
"""

import os
import sys
import math

import numpy as np
import torch

# Make the package importable when the tests are run from the project root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.join(_HERE, "..", "code")
sys.path.insert(0, os.path.abspath(_CODE))

import diff_chebyshev as dc  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
np.random.seed(0)

PASS = []


def report(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    PASS.append(ok)
    line = f"[{status}] {name}"
    if detail:
        line += f"  -- {detail}"
    print(line)


def rel_l2(err, ref):
    """Relative discrete L2 norm ||err||_2 / ||ref||_2."""
    return float(torch.norm(err) / (torch.norm(ref) + 1e-300))


# -------------------------------------------------------------------------------------
def test_gradcheck():
    """1. autograd.gradcheck on the custom Function (the differentiability core)."""
    N = 8
    D, _ = dc.chebyshev_diff_matrix(N)
    scale = torch.as_tensor(1.0, dtype=torch.float64)
    u = torch.randn(N + 1, dtype=torch.float64, requires_grad=True)

    def fn(x):
        return dc.DiffChebyshevDerivative.apply(x, D, scale)

    ok = torch.autograd.gradcheck(
        fn, u, atol=1e-4, rtol=1e-3, raise_exception=False
    )
    report("gradcheck DiffChebyshevDerivative", bool(ok))


def test_first_derivative_sin():
    """2. u(x)=sin(x) on [-pi,pi], N=32; spectral accuracy."""
    N = 32
    domain = (-math.pi, math.pi)
    _, x = dc.chebyshev_diff_matrix(N)
    # Map reference nodes to the physical interval: x_phys = pi * x (see module doc).
    x_phys = domain[0] + (x + 1.0) / 2.0 * (domain[1] - domain[0])
    u = torch.sin(x_phys)
    du_num = dc.chebyshev_derivative(u, N, order=1, domain=domain)
    du_ref = torch.cos(x_phys)
    err = rel_l2(du_num - du_ref, du_ref)
    report("1st derivative sin(x) [-pi,pi] N=32", err < 1e-10, f"rel L2 = {err:.2e}")


def test_second_derivative_gaussian():
    """3. u(x)=exp(-x^2) on [-1,1], N=64; second derivative."""
    N = 64
    domain = (-1.0, 1.0)
    _, x = dc.chebyshev_diff_matrix(N)
    u = torch.exp(-x ** 2)
    d2u_num = dc.chebyshev_derivative(u, N, order=2, domain=domain)
    # d2/dx2 exp(-x^2) = (4x^2 - 2) exp(-x^2)
    d2u_ref = (4.0 * x ** 2 - 2.0) * torch.exp(-x ** 2)
    err = rel_l2(d2u_num - d2u_ref, d2u_ref)
    report("2nd derivative exp(-x^2) [-1,1] N=64", err < 1e-8, f"rel L2 = {err:.2e}")


def test_laplacian_2d():
    """4. u=sin(x)sin(y) on [-pi,pi]^2, N=32; Laplacian = -2 sin(x)sin(y)."""
    N = 32
    domain = ((-math.pi, math.pi), (-math.pi, math.pi))
    _, x = dc.chebyshev_diff_matrix(N)
    _, y = dc.chebyshev_diff_matrix(N)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    # physical coords = pi * (reference nodes)
    Xp = math.pi * X
    Yp = math.pi * Y
    u = torch.sin(Xp) * torch.sin(Yp)
    lap_num = dc.chebyshev_laplacian_2d(u, N, N, domain=domain)
    lap_ref = -2.0 * torch.sin(Xp) * torch.sin(Yp)
    err = rel_l2(lap_num - lap_ref, lap_ref)
    report("2D Laplacian sin(x)sin(y) [-pi,pi]^2 N=32", err < 1e-10,
           f"rel L2 = {err:.2e}")


def test_adjoint_identity():
    """5. <D u, v> = <u, D^T v> to machine precision."""
    N = 24
    D, _ = dc.chebyshev_diff_matrix(N)
    u = torch.randn(N + 1, dtype=torch.float64)
    v = torch.randn(N + 1, dtype=torch.float64)

    lhs = torch.dot(D @ u, v)          # <D u, v>
    rhs = torch.dot(u, D.T @ v)       # <u, D^T v>
    rel = float(abs(lhs - rhs) / (abs(lhs) + 1e-300))
    report("adjoint identity <Du,v>=<u,D^T v>", rel < 1e-12, f"rel diff = {rel:.2e}")


def test_dct_vs_matrix():
    """6. Method B (DCT) and Method A (matrix) agree to spectral precision."""
    N = 40
    domain = (-1.0, 1.0)
    _, x = dc.chebyshev_diff_matrix(N)
    # A smooth non-polynomial test function.
    u = torch.sin(math.pi * x) * torch.exp(-0.5 * x ** 2)

    du_matrix = dc.chebyshev_derivative(u, N, order=1, domain=domain)
    du_dct = dc.diff_chebyshev_dct(u, order=1, domain=domain)
    diff = float(torch.norm(du_matrix - du_dct))
    report("DCT vs matrix (1st deriv)", diff < 1e-10, f"||diff||_2 = {diff:.2e}")

    d2_matrix = dc.chebyshev_derivative(u, N, order=2, domain=domain)
    d2_dct = dc.diff_chebyshev_dct(u, order=2, domain=domain)
    diff2 = float(torch.norm(d2_matrix - d2_dct))
    report("DCT vs matrix (2nd deriv)", diff2 < 1e-10, f"||diff||_2 = {diff2:.2e}")


def test_dirichlet():
    """7. u(x)=x^2 on [0,1], u(0)=0, u(1)=1; derivative = 2x."""
    N = 16
    domain = (0.0, 1.0)
    D, x = dc.chebyshev_diff_matrix(N)
    x_phys = (x + 1.0) / 2.0                 # map [-1,1] -> [0,1]
    u = x_phys ** 2
    du_ref = 2.0 * x_phys

    du_cheb, u_corr = dc.apply_dirichlet_bc(D, u, bc_values=(0.0, 1.0))
    scale = 2.0 / (domain[1] - domain[0])    # = 2
    du_num = scale * du_cheb

    # Boundary values imposed exactly.  Lobatto nodes are decreasing:
    # index 0 -> x_phys = b = 1 (u = 1), index N -> x_phys = a = 0 (u = 0).
    bc_ok = bool(abs(float(u_corr[0]) - 1.0) < 1e-14 and
                 abs(float(u_corr[-1]) - 0.0) < 1e-14)
    err = rel_l2(du_num - du_ref, du_ref)
    ok = bc_ok and (err < 1e-12)
    report("Dirichlet BC u=x^2 on [0,1]", ok,
           f"rel L2 = {err:.2e}, bc_imposed = {bc_ok}")


if __name__ == "__main__":
    print("=" * 72)
    print("Differentiable Chebyshev Spectral Layer -- unit tests")
    print(f"torch {torch.__version__} | numpy {np.__version__} | dtype float64")
    print("=" * 72)

    test_gradcheck()
    test_first_derivative_sin()
    test_second_derivative_gaussian()
    test_laplacian_2d()
    test_adjoint_identity()
    test_dct_vs_matrix()
    test_dirichlet()

    print("-" * 72)
    n_pass = int(sum(PASS))
    n_total = len(PASS)
    print(f"Result: {n_pass}/{n_total} tests passed")
    print("=" * 72)
    sys.exit(0 if n_pass == n_total else 1)
