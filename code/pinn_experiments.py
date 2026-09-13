"""
Unified PINN experiment framework for the Chebyshev-NUFFT hybrid layer paper (JSC).

Experiments
-----------
  1. 1D Poisson:  -u'' = f on (0,1), Dirichlet BC u(0)=u(1)=0
  2. 2D Poisson:  -Delta u = f on (0,1)^2, Dirichlet BC
  3. Ablation: boundary-layer width delta
  4. Ablation: collocation count M

Methods compared:
  'hybrid'  - Chebyshev-NUFFT differentiable hybrid layer (this paper)
  'ad'      - standard autograd PINN (baseline)
  'cheb'    - pure Chebyshev spectral derivative on scattered points
  'finufft' - pure periodic NUFFT derivative (Gibbs ringing expected)
  'rbf_fd'  - local RBF finite-difference (baseline)
"""
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import json
import time
import math

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import qmc

sys.path.insert(0, r'H:\2026科研\Chebyshev-NUFFT-PINN\code')
sys.path.insert(0, r'H:\2026科研\JCP-Journal of Computational Physics\code')

import diff_hybrid as dh          # noqa: E402
import diff_chebyshev as dc       # noqa: E402
import diff_finufft as df         # noqa: E402

torch.set_default_dtype(torch.float64)

# ------------------------------------------------------------------ paths
ROOT = r'H:\2026科研\Chebyshev-NUFFT-PINN'
RESULTS_DIR = os.path.join(ROOT, 'results')
FIGS_DIR = os.path.join(ROOT, 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

SEEDS = [42, 123, 456]


# ==================================================================
# Network
# ==================================================================
class MLP(nn.Module):
    """Input -> 3 hidden layers (64, Tanh) -> output.  Double precision."""

    def __init__(self, dim_in=1, hidden=64, layers=3):
        super().__init__()
        m = [nn.Linear(dim_in, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            m += [nn.Linear(hidden, hidden), nn.Tanh()]
        m += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*m)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ==================================================================
# LR schedule
# ==================================================================
def get_lr(step, warmup=500, total=5000, base_lr=1e-3):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + np.cos(np.pi * progress))


def set_lr(opt, lr):
    for pg in opt.param_groups:
        pg['lr'] = lr


# ==================================================================
# PDE definitions
# ==================================================================
def exact_u_1d(x):
    return torch.sin(np.pi * x) + 0.5 * torch.sin(3.0 * np.pi * x)


def f_1d(x):
    return (np.pi ** 2) * torch.sin(np.pi * x) + \
           4.5 * (np.pi ** 2) * torch.sin(3.0 * np.pi * x)


def exact_u_2d(x, y):
    return (torch.sin(np.pi * x) * torch.sin(np.pi * y)
            + 0.3 * torch.sin(2 * np.pi * x) * torch.sin(2 * np.pi * y))


def f_2d(x, y):
    return (2.0 * np.pi ** 2) * torch.sin(np.pi * x) * torch.sin(np.pi * y) \
           + (2.4 * np.pi ** 2) * torch.sin(2 * np.pi * x) * torch.sin(2 * np.pi * y)


# ==================================================================
# Operator builders (constant matrices, precomputed once)
# ==================================================================
def build_cheb_1d_scattered(x_np, Nc, domain, order=2):
    """Pure-Chebyshev second-derivative matrix on scattered points.

    Fits a Chebyshev expansion directly from scattered points via least
    squares and evaluates the order-th derivative at every point.
    Returns a torch (M, M) matrix.
    """
    a, b = domain
    L = b - a
    xi = 2.0 * (x_np.ravel() - a) / L - 1.0
    T = (2.0 / L) ** order * dh._cheb_scattered_operator(Nc, order, xi, xi)
    return torch.from_numpy(T)


def build_rbf_interp_2d(sources, targets, epsilon):
    """2D Gaussian-RBF interpolation matrix P (n_tgt, n_src).

    v_tgt = P @ v_src,  where the RBF interpolant is built from sources.
    """
    sources = np.asarray(sources, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    d_ts = targets[:, None, :] - sources[None, :, :]
    Phi_ts = np.exp(-(epsilon ** 2) * np.sum(d_ts ** 2, axis=-1))
    d_ss = sources[:, None, :] - sources[None, :, :]
    Phi_ss = np.exp(-(epsilon ** 2) * np.sum(d_ss ** 2, axis=-1))
    Phi_ss = Phi_ss + 1e-10 * np.eye(Phi_ss.shape[0])
    X = np.linalg.solve(Phi_ss, Phi_ts.T)
    return X.T


def build_rbf_fd_1d(x_np, order=2, k=7, epsilon=None):
    """Local RBF-FD second-derivative matrix (sparse -> dense).

    For each point, select k nearest neighbours, solve the local RBF system
    for the Laplacian weights, and assemble the global (M, M) matrix.
    """
    x_np = np.asarray(x_np, dtype=np.float64).ravel()
    M = x_np.size
    if epsilon is None:
        h = np.sort(np.abs(x_np[:, None] - x_np[None, :]), axis=1)[:, 1:k+1].mean()
        epsilon = 1.0 / max(h, 1e-6)

    L = np.zeros((M, M), dtype=np.float64)
    for i in range(M):
        dists = np.abs(x_np - x_np[i])
        neighbors = np.argsort(dists)[:k]
        xn = x_np[neighbors]
        # RBF system: for each stencil point x_m, phi_jm = exp(-eps^2 (x_m - x_j)^2)
        r_jm = xn[None, :] - xn[:, None]           # (k, k)
        A = np.exp(-(epsilon ** 2) * r_jm ** 2) + 1e-10 * np.eye(k)
        # RHS: L phi evaluated at x_i
        r_im = xn - x_np[i]
        if order == 2:
            rhs = (4.0 * epsilon ** 4 * r_im ** 2 - 4.0 * epsilon ** 2) * \
                  np.exp(-(epsilon ** 2) * r_im ** 2)
        else:
            raise NotImplementedError
        w = np.linalg.solve(A, rhs)
        L[i, neighbors] = w
    return torch.from_numpy(L)


# ==================================================================
# 1D Poisson training
# ==================================================================
def train_pinn_1d_poisson(method, params=None):
    """Train a 1D Poisson PINN with the requested derivative method.

    Returns dict(l2, time, history, net).
    """
    p = dict(params or {})
    seed = p.get('seed', 42)
    M = p.get('M', 2000)
    steps = p.get('steps', 5000)
    lr = p.get('lr', 1e-3)
    delta = p.get('delta', 0.15)
    overlap = p.get('overlap', 0.05)
    N_cheb = p.get('N_cheb', 32)
    N_fourier = p.get('N_fourier', 16)
    warmup = p.get('warmup', 500)
    interior_method = p.get('interior_method', 'cheb')

    torch.manual_seed(seed)
    np.random.seed(seed)

    domain = (0.0, 1.0)

    # Halton collocation points in (0, 1)
    sampler = qmc.Halton(d=1, scramble=True, seed=seed)
    x_np = sampler.random(M).ravel()
    x = torch.tensor(x_np, dtype=torch.float64)

    f_rhs = f_1d(x)
    u_true = exact_u_1d(x)

    # Test points (uniform interior)
    x_test = torch.linspace(0.001, 0.999, 1000, dtype=torch.float64)
    u_test_exact = exact_u_1d(x_test)

    net = MLP(dim_in=1, hidden=64, layers=3)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    # ---- precompute constant operators -----------------------------
    op_A = None       # (M, M) second-derivative matrix
    theta_finufft = None
    N_finufft = None

    if method == 'hybrid':
        hparams = dict(domain=domain, delta=delta, overlap=overlap,
                       N_cheb=N_cheb, N_fourier=N_fourier,
                       eps_nufft=1e-6, order=2,
                       interior_method=interior_method)
        op = dh.HybridOperator1D(x_np, hparams)
        op_A = op.A

    elif method == 'cheb':
        op_A = build_cheb_1d_scattered(x_np, N_cheb, domain, order=2)

    elif method == 'rbf_fd':
        op_A = build_rbf_fd_1d(x_np, order=2, k=7)

    elif method == 'finufft':
        # map x in [0,1] -> theta in [-pi, pi)
        theta_finufft = 2.0 * np.pi * (x_np - 0.5)
        theta_finufft = torch.tensor(theta_finufft, dtype=torch.float64).reshape(-1, 1)
        N_finufft = max(2 * N_fourier, 16)   # comparable spectral resolution

    # ---- training loop ---------------------------------------------
    history = []
    t0 = time.time()

    for step in range(steps + 1):
        cur_lr = get_lr(step, warmup=warmup, total=steps, base_lr=lr)
        set_lr(opt, cur_lr)
        opt.zero_grad()

        if method == 'ad':
            x_ad = x.detach().reshape(-1, 1).requires_grad_(True)
            u_raw = net(x_ad)
            u_pred = u_raw * x_ad.squeeze(-1) * (1.0 - x_ad.squeeze(-1))
            g = torch.autograd.grad(u_pred, x_ad, torch.ones_like(u_pred),
                                    create_graph=True)[0].squeeze(-1)
            u_pp = torch.autograd.grad(g, x_ad, torch.ones_like(g),
                                       create_graph=True)[0].squeeze(-1)
        else:
            u_raw = net(x.reshape(-1, 1))
            u_pred = u_raw * x * (1.0 - x)
            if method in ('hybrid', 'cheb', 'rbf_fd'):
                u_pp = op_A @ u_pred
            elif method == 'finufft':
                u_pp = df.finufft_derivative(u_pred, theta_finufft,
                                             N_finufft, 2, 0, 1e-6)
            else:
                raise ValueError(f'Unknown method: {method}')

        loss_pde = torch.mean((-u_pp - f_rhs) ** 2)
        loss_data = torch.mean((u_pred - u_true) ** 2)
        loss = loss_pde + 0.1 * loss_data
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        if step % 500 == 0:
            with torch.no_grad():
                u_t = net(x_test.reshape(-1, 1)).squeeze(-1) * x_test * (1.0 - x_test)
                l2 = torch.norm(u_t - u_test_exact) / torch.norm(u_test_exact)
            history.append({'step': step, 'l2': l2.item(),
                            'loss': loss.item()})
            if step % 1000 == 0:
                print(f'    [{method}] seed={seed} step={step:5d}: '
                      f'l2={l2.item():.4e}  loss={loss.item():.4e}')

    elapsed = time.time() - t0
    with torch.no_grad():
        u_t = net(x_test.reshape(-1, 1)).squeeze(-1) * x_test * (1.0 - x_test)
        l2_final = (torch.norm(u_t - u_test_exact) /
                    torch.norm(u_test_exact)).item()

    return {'l2': l2_final, 'time': elapsed, 'history': history, 'net': net}


# ==================================================================
# 2D Poisson training
# ==================================================================
def train_pinn_2d_poisson(method, params=None):
    """Train a 2D Poisson PINN with the requested derivative method."""
    p = dict(params or {})
    seed = p.get('seed', 42)
    M = p.get('M', 5000)
    steps = p.get('steps', 8000)
    lr = p.get('lr', 1e-3)
    delta = p.get('delta', 0.15)
    overlap = p.get('overlap', 0.05)
    N_cheb = p.get('N_cheb', 32)
    N_fourier = p.get('N_fourier', 16)
    warmup = p.get('warmup', 500)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Halton collocation points in (0,1)^2
    sampler = qmc.Halton(d=2, scramble=True, seed=seed)
    samp = sampler.random(M)
    x_np = samp[:, 0]
    y_np = samp[:, 1]
    xy_np = np.stack([x_np, y_np], axis=1)
    x = torch.tensor(x_np, dtype=torch.float64)
    y = torch.tensor(y_np, dtype=torch.float64)
    xy = torch.tensor(xy_np, dtype=torch.float64)

    f_rhs = f_2d(x, y)
    u_true = exact_u_2d(x, y)

    # Test grid 50 x 50
    gx = torch.linspace(0.001, 0.999, 50, dtype=torch.float64)
    gy = torch.linspace(0.001, 0.999, 50, dtype=torch.float64)
    Xg, Yg = torch.meshgrid(gx, gy, indexing='ij')
    xy_test = torch.stack([Xg.ravel(), Yg.ravel()], dim=1)
    u_test_exact = exact_u_2d(Xg, Yg)

    net = MLP(dim_in=2, hidden=64, layers=3)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    # ---- precompute operators --------------------------------------
    hybrid_L = None           # (M, M) global RBF Laplacian
    cheb_parts = None         # (P_fwd, L_cheb, P_bwd)
    theta_finufft = None
    N_finufft_2d = None

    if method == 'hybrid':
        # global Gaussian-RBF Laplacian (fully differentiable constant matrix)
        eps_rbf = 0.03 * math.sqrt(M)
        L_np = dh._gaussian_rbf2d_laplacian_matrix(xy_np, eps_rbf)
        hybrid_L = torch.from_numpy(L_np)

    elif method == 'cheb':
        # Chebyshev tensor-product grid + RBF interpolation
        Nc = N_cheb
        x_lob = dc.chebyshev_points(Nc).numpy()       # [-1,1] decreasing
        y_lob = dc.chebyshev_points(Nc).numpy()
        x_phys = (x_lob + 1.0) / 2.0                  # [0,1]
        y_phys = (y_lob + 1.0) / 2.0
        Xg2, Yg2 = np.meshgrid(x_phys, y_phys, indexing='ij')
        grid_pts = np.stack([Xg2.ravel(), Yg2.ravel()], axis=1)
        n_grid = (Nc + 1) ** 2

        # RBF interp matrices
        rbf_eps = 0.15
        P_fwd_np = build_rbf_interp_2d(xy_np, grid_pts, rbf_eps)   # (g, M)
        P_bwd_np = build_rbf_interp_2d(grid_pts, xy_np, rbf_eps)   # (M, g)

        # Kronecker Chebyshev Laplacian on the grid
        Dx, _ = dc.chebyshev_diff_matrix(Nc)
        Dy, _ = dc.chebyshev_diff_matrix(Nc)
        Dx2 = Dx @ Dx
        Dy2 = Dy @ Dy
        nx = ny = Nc + 1
        Ix = torch.eye(nx, dtype=torch.float64)
        Iy = torch.eye(ny, dtype=torch.float64)
        sx = sy = 2.0
        L_cheb = (sx ** 2) * torch.kron(Dx2, Iy) + \
                 (sy ** 2) * torch.kron(Ix, Dy2)

        cheb_parts = (torch.from_numpy(P_fwd_np), L_cheb,
                      torch.from_numpy(P_bwd_np))

    elif method == 'finufft':
        theta = 2.0 * np.pi * (xy_np - 0.5)
        theta_finufft = torch.tensor(theta, dtype=torch.float64)
        N_finufft_2d = max(2 * N_fourier, 16)

    # ---- training loop ---------------------------------------------
    history = []
    t0 = time.time()

    for step in range(steps + 1):
        cur_lr = get_lr(step, warmup=warmup, total=steps, base_lr=lr)
        set_lr(opt, cur_lr)
        opt.zero_grad()

        if method == 'ad':
            xy_ad = xy.detach().requires_grad_(True)
            u_raw = net(xy_ad)
            u_pred = u_raw * xy_ad[:, 0] * (1.0 - xy_ad[:, 0]) \
                           * xy_ad[:, 1] * (1.0 - xy_ad[:, 1])
            g = torch.autograd.grad(u_pred, xy_ad, torch.ones_like(u_pred),
                                    create_graph=True)[0]
            u_xx = torch.autograd.grad(g[:, 0], xy_ad,
                                       torch.ones_like(g[:, 0]),
                                       create_graph=True)[0][:, 0]
            u_yy = torch.autograd.grad(g[:, 1], xy_ad,
                                       torch.ones_like(g[:, 1]),
                                       create_graph=True)[0][:, 1]
            lap_u = u_xx + u_yy
        else:
            u_raw = net(xy)
            bc_factor = x * (1.0 - x) * y * (1.0 - y)
            u_pred = u_raw * bc_factor

            if method == 'hybrid':
                lap_u = hybrid_L @ u_pred

            elif method == 'cheb':
                P_fwd, L_cheb, P_bwd = cheb_parts
                u_grid = P_fwd @ u_pred               # (g,)
                lap_grid = L_cheb @ u_grid            # (g,)
                lap_u = P_bwd @ lap_grid              # (M,)

            elif method == 'finufft':
                lap_u = df.finufft_laplacian(u_pred, theta_finufft,
                                             N_finufft_2d, eps=1e-6)
            else:
                raise ValueError(f'Unknown method: {method}')

        loss_pde = torch.mean((-lap_u - f_rhs) ** 2)
        loss_data = torch.mean((u_pred - u_true) ** 2)
        loss = loss_pde + 0.1 * loss_data
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        if step % 500 == 0:
            with torch.no_grad():
                u_t = net(xy_test) * xy_test[:, 0] * (1.0 - xy_test[:, 0]) \
                                   * xy_test[:, 1] * (1.0 - xy_test[:, 1])
                u_t = u_t.reshape(50, 50)
                l2 = torch.norm(u_t - u_test_exact) / torch.norm(u_test_exact)
            history.append({'step': step, 'l2': l2.item(),
                            'loss': loss.item()})
            if step % 1000 == 0:
                print(f'    [{method}] seed={seed} step={step:5d}: '
                      f'l2={l2.item():.4e}  loss={loss.item():.4e}')

    elapsed = time.time() - t0
    with torch.no_grad():
        u_t = net(xy_test) * xy_test[:, 0] * (1.0 - xy_test[:, 0]) \
                           * xy_test[:, 1] * (1.0 - xy_test[:, 1])
        u_t = u_t.reshape(50, 50)
        l2_final = (torch.norm(u_t - u_test_exact) /
                    torch.norm(u_test_exact)).item()

    return {'l2': l2_final, 'time': elapsed, 'history': history,
            'net': net, 'Xg': Xg, 'Yg': Yg, 'u_pred_grid': u_t,
            'u_exact_grid': u_test_exact}


# ==================================================================
# 1D Poisson with soft Neumann / Robin BC (no hard-constraint transform)
# ==================================================================
def train_pinn_1d_poisson_bc(method, params=None):
    """1D Poisson with SOFT Neumann or Robin boundary conditions.

    PDE: -u'' = f on (0,1).  The network directly outputs u(x) (no
    hard-constraint factor).  Loss = PDE residual + lambda_bc * BC residual.

    bc_type='neumann':  u'(0)=0, u'(1)=0,  exact u=cos(pi x), f=pi^2 cos(pi x)
    bc_type='robin' :  u'(0)+u(0)=2, u'(1)+u(1)=2e, exact u=exp(x), f=-exp(x)

    For the hybrid layer, order-2 and order-1 operators are assembled on the
    union of the Halton interior points and the two endpoints (0,1), so the
    boundary derivative is evaluated by the boundary Chebyshev blocks.
    """
    p = dict(params or {})
    seed = p.get('seed', 42)
    M = p.get('M', 2000)
    steps = p.get('steps', 5000)
    lr = p.get('lr', 1e-3)
    delta = p.get('delta', 0.15)
    overlap = p.get('overlap', 0.05)
    N_cheb = p.get('N_cheb', 32)
    N_fourier = p.get('N_fourier', 16)
    warmup = p.get('warmup', 500)
    bc_type = p.get('bc_type', 'neumann')
    lambda_bc = p.get('lambda_bc', 10.0)

    torch.manual_seed(seed)
    np.random.seed(seed)
    domain = (0.0, 1.0)

    if bc_type == 'neumann':
        exact_u = lambda x: torch.cos(np.pi * x)
        f_rhs = lambda x: (np.pi ** 2) * torch.cos(np.pi * x)
        bc0_target, bc1_target = 0.0, 0.0
    else:  # robin
        exact_u = lambda x: torch.exp(x)
        f_rhs = lambda x: -torch.exp(x)
        bc0_target, bc1_target = 2.0, 2.0 * np.e

    sampler = qmc.Halton(d=1, scramble=True, seed=seed)
    x_int_np = sampler.random(M).ravel()
    x_bc_np = np.array([0.0, 1.0], dtype=np.float64)
    x_int = torch.tensor(x_int_np, dtype=torch.float64)
    x_bc = torch.tensor(x_bc_np, dtype=torch.float64)

    x_test = torch.linspace(0.0, 1.0, 1000, dtype=torch.float64)
    u_test_exact = exact_u(x_test)

    net = MLP(dim_in=1, hidden=64, layers=3)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    op_A2 = None
    if method == 'hybrid':
        x_all_np = np.concatenate([x_int_np, x_bc_np])
        hp = dict(domain=domain, delta=delta, overlap=overlap,
                  N_cheb=N_cheb, N_fourier=N_fourier,
                  eps_nufft=1e-6, order=2, interior_method='cheb')
        hp1 = dict(hp, order=1)
        op_A2 = dh.HybridOperator1D(x_all_np, hp).A
        op_A1 = dh.HybridOperator1D(x_all_np, hp1).A

    history = []
    t0 = time.time()
    for step in range(steps + 1):
        cur_lr = get_lr(step, warmup=warmup, total=steps, base_lr=lr)
        set_lr(opt, cur_lr)
        opt.zero_grad()

        if method == 'ad':
            x_ad = x_int.reshape(-1, 1).requires_grad_(True)
            u_int = net(x_ad).squeeze(-1)
            g = torch.autograd.grad(u_int, x_ad, torch.ones_like(u_int),
                                    create_graph=True)[0].squeeze(-1)
            u_pp_int = torch.autograd.grad(g, x_ad, torch.ones_like(g),
                                          create_graph=True)[0].squeeze(-1)
            xb_ad = x_bc.reshape(-1, 1).requires_grad_(True)
            u_bc = net(xb_ad).squeeze(-1)
            g_bc = torch.autograd.grad(u_bc, xb_ad, torch.ones_like(u_bc),
                                       create_graph=True)[0].squeeze(-1)
            u0, u1 = u_bc[0], u_bc[1]
            up0, up1 = g_bc[0], g_bc[1]
        else:  # hybrid
            x_all = torch.tensor(np.concatenate([x_int_np, x_bc_np]),
                                 dtype=torch.float64)
            u_all = net(x_all.reshape(-1, 1)).squeeze(-1)
            u_pp_all = op_A2 @ u_all
            u_p_all = op_A1 @ u_all
            u_pp_int = u_pp_all[:M]
            u0, u1 = u_all[M], u_all[M + 1]
            up0, up1 = u_p_all[M], u_p_all[M + 1]

        loss_pde = torch.mean((-u_pp_int - f_rhs(x_int)) ** 2)
        if bc_type == 'neumann':
            loss_bc = (up0 - bc0_target) ** 2 + (up1 - bc1_target) ** 2
        else:
            loss_bc = ((up0 + u0) - bc0_target) ** 2 + \
                      ((up1 + u1) - bc1_target) ** 2
        loss = loss_pde + lambda_bc * loss_bc
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        if step % 500 == 0:
            with torch.no_grad():
                u_t = net(x_test.reshape(-1, 1)).squeeze(-1)
                if bc_type == 'neumann':
                    # Neumann problem has constant nullspace: compare centered solutions
                    u_t_c = u_t - u_t.mean()
                    u_ex_c = u_test_exact - u_test_exact.mean()
                    l2 = torch.norm(u_t_c - u_ex_c) / torch.norm(u_ex_c)
                else:
                    l2 = torch.norm(u_t - u_test_exact) / torch.norm(u_test_exact)
            history.append({'step': step, 'l2': l2.item(), 'loss': loss.item()})
            if step % 1000 == 0:
                print(f'    [{method}/{bc_type}] seed={seed} step={step:5d}: '
                      f'l2={l2.item():.4e}  loss={loss.item():.4e}')

    elapsed = time.time() - t0
    with torch.no_grad():
        u_t = net(x_test.reshape(-1, 1)).squeeze(-1)
        if bc_type == 'neumann':
            u_t_c = u_t - u_t.mean()
            u_ex_c = u_test_exact - u_test_exact.mean()
            l2_final = (torch.norm(u_t_c - u_ex_c) / torch.norm(u_ex_c)).item()
        else:
            l2_final = (torch.norm(u_t - u_test_exact) /
                        torch.norm(u_test_exact)).item()
    return {'l2': l2_final, 'time': elapsed, 'history': history, 'net': net}


# ==================================================================
# 1D heat equation (parabolic PDE), space hybrid + time autograd
# ==================================================================
def train_pinn_heat_1d(method, params=None):
    """1D heat equation:  u_t = nu * u_xx, nu=0.1, x in (0,1), t in (0,1].

    Dirichlet BC u(0,t)=u(1,t)=0, IC u(x,0)=sin(pi x).
    Exact: u(x,t) = exp(-nu*pi^2*t) * sin(pi x).

    Hard constraint satisfying BOTH BC and IC:
        u_theta(x,t) = x(1-x) * t * MLP(x,t) + (1-t) * sin(pi x)

    Spatial second derivative uses the hybrid layer (matrix A on the M
    Halton points, applied per time level); time derivative uses autograd.
    For 'ad' both derivatives use autograd.
    """
    p = dict(params or {})
    seed = p.get('seed', 42)
    M = p.get('M', 1000)
    N_t = p.get('N_t', 100)
    steps = p.get('steps', 3000)
    lr = p.get('lr', 1e-3)
    delta = p.get('delta', 0.15)
    overlap = p.get('overlap', 0.05)
    N_cheb = p.get('N_cheb', 32)
    N_fourier = p.get('N_fourier', 16)
    warmup = p.get('warmup', 300)
    nu = 0.1
    T = 1.0

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'    [heat] device={device}')

    sampler = qmc.Halton(d=1, scramble=True, seed=seed)
    x_np = sampler.random(M).ravel()
    t_np = np.linspace(0.0, T, N_t)

    X, Tg = np.meshgrid(x_np, t_np, indexing='ij')     # (M, N_t)
    x_flat = X.reshape(-1)
    t_flat = Tg.reshape(-1)

    x_t = torch.tensor(x_flat, dtype=torch.float64).reshape(-1, 1).to(device)
    t_t = torch.tensor(t_flat, dtype=torch.float64).reshape(-1, 1).to(device)

    # test grid
    x_test = torch.linspace(0.0, 1.0, 100, dtype=torch.float64)
    t_test = torch.linspace(0.0, T, 50, dtype=torch.float64)
    Xg, Tt = torch.meshgrid(x_test, t_test, indexing='ij')
    u_exact_test = torch.exp(-nu * np.pi ** 2 * Tt) * torch.sin(np.pi * Xg)

    net = MLP(dim_in=2, hidden=64, layers=3).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    op_A = None
    if method == 'hybrid':
        hp = dict(domain=(0.0, 1.0), delta=delta, overlap=overlap,
                  N_cheb=N_cheb, N_fourier=N_fourier,
                  eps_nufft=1e-6, order=2, interior_method='cheb')
        op_A = dh.HybridOperator1D(x_np, hp).A.to(device)

    history = []
    t0 = time.time()
    for step in range(steps + 1):
        cur_lr = get_lr(step, warmup=warmup, total=steps, base_lr=lr)
        set_lr(opt, cur_lr)
        opt.zero_grad()

        xt = x_t.clone()
        tt = t_t.clone().requires_grad_(True)
        inp = torch.cat([xt, tt], dim=1)
        u_raw = net(inp)
        # hard constraint: BC + IC
        u = xt.squeeze(-1) * (1.0 - xt.squeeze(-1)) * tt.squeeze(-1) \
            * u_raw.squeeze(-1) \
            + (1.0 - tt.squeeze(-1)) * torch.sin(np.pi * xt.squeeze(-1))

        # time derivative via autograd
        u_t = torch.autograd.grad(u, tt, torch.ones_like(u),
                                  create_graph=True)[0].squeeze(-1)

        if method == 'hybrid':
            u_grid = u.reshape(M, N_t)
            u_xx_grid = op_A @ u_grid                # (M, N_t)
            resid = u_t.reshape(M, N_t) - nu * u_xx_grid
        else:  # ad
            xt2 = x_t.clone().requires_grad_(True)
            inp2 = torch.cat([xt2, tt], dim=1)
            u_raw2 = net(inp2)
            xbc = xt2.squeeze(-1)
            tbc = tt.squeeze(-1)
            u2 = xbc * (1.0 - xbc) * tbc * u_raw2.squeeze(-1) \
                 + (1.0 - tbc) * torch.sin(np.pi * xbc)
            gx = torch.autograd.grad(u2, xt2, torch.ones_like(u2),
                                     create_graph=True)[0].squeeze(-1)
            u_xx = torch.autograd.grad(gx, xt2, torch.ones_like(gx),
                                       create_graph=True)[0].squeeze(-1)
            # use u_t already computed (recompute to share graph on xt2)
            u_t2 = torch.autograd.grad(u2, tt, torch.ones_like(u2),
                                       create_graph=True)[0].squeeze(-1)
            resid = (u_t2 - nu * u_xx).reshape(M, N_t)

        loss = torch.mean(resid ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        if step % 200 == 0:
            with torch.no_grad():
                xe = Xg.reshape(-1).to(device)
                te = Tt.reshape(-1).to(device)
                inp_te = torch.stack([xe, te], dim=1)
                ur = net(inp_te)
                u_te = xe * (1.0 - xe) * te * ur \
                       + (1.0 - te) * torch.sin(np.pi * xe)
                u_te = u_te.reshape(Xg.shape)
                l2 = torch.norm(u_te - u_exact_test.to(device)) / \
                     torch.norm(u_exact_test.to(device))
            history.append({'step': step, 'l2': l2.item(), 'loss': loss.item()})
            if step % 500 == 0:
                print(f'    [{method}/heat] seed={seed} step={step:5d}: '
                      f'l2={l2.item():.4e}  loss={loss.item():.4e}')

    elapsed = time.time() - t0
    with torch.no_grad():
        xe = Xg.reshape(-1).to(device)
        te = Tt.reshape(-1).to(device)
        inp_te = torch.stack([xe, te], dim=1)
        ur = net(inp_te)
        u_te = xe * (1.0 - xe) * te * ur \
               + (1.0 - te) * torch.sin(np.pi * xe)
        u_te = u_te.reshape(Xg.shape)
        l2_final = (torch.norm(u_te - u_exact_test.to(device)) /
                    torch.norm(u_exact_test.to(device))).item()
    return {'l2': l2_final, 'time': elapsed, 'history': history,
            'net': net, 'Xg': Xg, 'Tt': Tt,
            'u_pred_grid': u_te.cpu(), 'u_exact_grid': u_exact_test}


# ==================================================================
# Experiment runners
# ==================================================================
def run_1d_experiment(methods, seeds=SEEDS, **overrides):
    """Run 1D Poisson across methods and seeds.  Returns dict."""
    results = {}
    for method in methods:
        runs = []
        for seed in seeds:
            print(f'  >> 1D [{method}] seed={seed}')
            r = train_pinn_1d_poisson(method, dict(seed=seed, **overrides))
            runs.append(r)
        l2s = [r['l2'] for r in runs]
        times = [r['time'] for r in runs]
        results[method] = {
            'l2_mean': float(np.mean(l2s)),
            'l2_std': float(np.std(l2s)),
            'time_mean': float(np.mean(times)),
            'time_std': float(np.std(times)),
            'l2_per_seed': [float(v) for v in l2s],
            'time_per_seed': [float(v) for v in times],
            'history': runs[0]['history'],   # representative curve
        }
        print(f'  << 1D [{method}] L2={results[method]["l2_mean"]:.4e} '
              f'+/- {results[method]["l2_std"]:.4e}  '
              f'time={results[method]["time_mean"]:.1f}s')
    return results


def run_2d_experiment(methods, seeds=SEEDS, **overrides):
    """Run 2D Poisson across methods and seeds."""
    results = {}
    for method in methods:
        runs = []
        for seed in seeds:
            print(f'  >> 2D [{method}] seed={seed}')
            r = train_pinn_2d_poisson(method, dict(seed=seed, **overrides))
            runs.append(r)
        l2s = [r['l2'] for r in runs]
        times = [r['time'] for r in runs]
        results[method] = {
            'l2_mean': float(np.mean(l2s)),
            'l2_std': float(np.std(l2s)),
            'time_mean': float(np.mean(times)),
            'time_std': float(np.std(times)),
            'l2_per_seed': [float(v) for v in l2s],
            'time_per_seed': [float(v) for v in times],
            'history': runs[0]['history'],
            'last_run': runs[-1],
        }
        print(f'  << 2D [{method}] L2={results[method]["l2_mean"]:.4e} '
              f'+/- {results[method]["l2_std"]:.4e}  '
              f'time={results[method]["time_mean"]:.1f}s')
    return results


def run_methods_seeds(train_fn, methods, seeds=SEEDS, tag='', **overrides):
    """Generic method x seed aggregation.  train_fn(method, params) -> dict."""
    results = {}
    for method in methods:
        runs = []
        for seed in seeds:
            print(f'  >> {tag} [{method}] seed={seed}')
            r = train_fn(method, dict(seed=seed, **overrides))
            runs.append(r)
        l2s = [r['l2'] for r in runs]
        times = [r['time'] for r in runs]
        results[method] = {
            'l2_mean': float(np.mean(l2s)),
            'l2_std': float(np.std(l2s)),
            'time_mean': float(np.mean(times)),
            'time_std': float(np.std(times)),
            'l2_per_seed': [float(v) for v in l2s],
            'time_per_seed': [float(v) for v in times],
            'history': runs[0]['history'],
        }
        print(f'  << {tag} [{method}] L2={results[method]["l2_mean"]:.4e} '
              f'+/- {results[method]["l2_std"]:.4e}  '
              f'time={results[method]["time_mean"]:.1f}s')
    return results


def run_ablation_param(train_fn, param_name, param_values, seeds=SEEDS,
                       tag='', **overrides):
    """Ablation over a scalar param (hybrid only), method x seed.

    Returns dict(param_values, l2_mean[], l2_std[], time_mean[],
                 time_std[], per_value[...]).
    """
    out = {'param_values': [float(v) for v in param_values],
           'l2_mean': [], 'l2_std': [],
           'time_mean': [], 'time_std': [], 'per_value': {}}
    for val in param_values:
        runs = [train_fn('hybrid', dict(seed=s, **{param_name: val}, **overrides))
                for s in seeds]
        l2s = [r['l2'] for r in runs]
        times = [r['time'] for r in runs]
        out['l2_mean'].append(float(np.mean(l2s)))
        out['l2_std'].append(float(np.std(l2s)))
        out['time_mean'].append(float(np.mean(times)))
        out['time_std'].append(float(np.std(times)))
        out['per_value'][float(val)] = {
            'l2_per_seed': [float(v) for v in l2s],
            'time_per_seed': [float(v) for v in times]}
        print(f'  << {tag} {param_name}={val}: '
              f'L2={np.mean(l2s):.4e}+/-{np.std(l2s):.4e}  '
              f'time={np.mean(times):.1f}s')
    return results_plot_shell(out)


def results_plot_shell(out):
    return out


# ==================================================================
# Plotting
# ==================================================================
PLOT_COLORS = {
    'hybrid': '#d62728',
    'ad': '#1f77b4',
    'cheb': '#2ca02c',
    'finufft': '#ff7f0e',
    'rbf_fd': '#9467bd',
}
PLOT_LABELS = {
    'hybrid': 'Hybrid (Cheb-NUFFT)',
    'ad': 'Autograd',
    'cheb': 'Pure Chebyshev',
    'finufft': 'Pure NUFFT',
    'rbf_fd': 'RBF-FD',
}


def plot_convergence_1d(results, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for method, res in results.items():
        h = res['history']
        steps = [d['step'] for d in h]
        l2s = [d['l2'] for d in h]
        ax.semilogy(steps, l2s, '-o', ms=3, lw=1.5,
                    color=PLOT_COLORS.get(method, None),
                    label=PLOT_LABELS.get(method, method))
    ax.set_xlabel('Training step')
    ax.set_ylabel(r'Relative $L_2$ error')
    ax.set_title('1D Poisson — convergence')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_solution_1d(net, path):
    """Plot exact vs predicted solution (uses last trained hybrid net)."""
    with torch.no_grad():
        x_dense = torch.linspace(0, 1, 500, dtype=torch.float64)
        u_pred = net(x_dense.reshape(-1, 1)).squeeze(-1) * x_dense * (1.0 - x_dense)
        u_exact = exact_u_1d(x_dense)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6),
                                   gridspec_kw={'height_ratios': [3, 1]},
                                   sharex=True)
    ax1.plot(x_dense.numpy(), u_exact.numpy(), 'k-', lw=2, label='Exact')
    ax1.plot(x_dense.numpy(), u_pred.numpy(), 'r--', lw=1.5,
             label='Hybrid prediction')
    ax1.set_ylabel('u(x)')
    ax1.legend()
    ax1.set_title('1D Poisson — solution')
    ax1.grid(True, alpha=0.3)
    ax2.semilogy(x_dense.numpy(),
                 np.abs((u_pred - u_exact).numpy()) + 1e-16,
                 'b-', lw=1)
    ax2.set_xlabel('x')
    ax2.set_ylabel('|error|')
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_convergence_2d(results, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for method, res in results.items():
        h = res['history']
        steps = [d['step'] for d in h]
        l2s = [d['l2'] for d in h]
        ax.semilogy(steps, l2s, '-o', ms=3, lw=1.5,
                    color=PLOT_COLORS.get(method, None),
                    label=PLOT_LABELS.get(method, method))
    ax.set_xlabel('Training step')
    ax.set_ylabel(r'Relative $L_2$ error')
    ax.set_title('2D Poisson — convergence')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_solution_2d(last_run, path):
    Xg = last_run['Xg'].numpy()
    Yg = last_run['Yg'].numpy()
    u_pred = last_run['u_pred_grid'].numpy()
    u_exact = last_run['u_exact_grid'].numpy()
    err = np.abs(u_pred - u_exact)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    vmin = min(u_exact.min(), u_pred.min())
    vmax = max(u_exact.max(), u_pred.max())

    im0 = axes[0].pcolormesh(Xg, Yg, u_exact, cmap='viridis',
                              vmin=vmin, vmax=vmax, shading='auto')
    axes[0].set_title('Exact solution')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(Xg, Yg, u_pred, cmap='viridis',
                              vmin=vmin, vmax=vmax, shading='auto')
    axes[1].set_title('Hybrid prediction')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('y')
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].pcolormesh(Xg, Yg, err, cmap='hot_r', shading='auto')
    axes[2].set_title('|error|')
    axes[2].set_xlabel('x'); axes[2].set_ylabel('y')
    fig.colorbar(im2, ax=axes[2])

    fig.suptitle('2D Poisson — solution comparison', y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_ablation(out, xlabel, title, path):
    """Param-vs-error line plot with std band (log y)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = out['param_values']
    lm = np.array(out['l2_mean'])
    ls = np.array(out['l2_std'])
    ax.semilogy(xs, lm, 'ro-', lw=1.5, ms=8, label='Hybrid L2')
    ax.fill_between(xs, np.maximum(lm - ls, 1e-16), lm + ls,
                    color='red', alpha=0.15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'Relative $L_2$ error')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ablation_dual(out_map, x_values, xlabel, title, path):
    """Two-method (hybrid vs ad) param-vs-error log-log plot."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for method, res in out_map.items():
        ax.loglog(x_values, res['l2_mean'], 'o-', lw=1.5, ms=8,
                  color=PLOT_COLORS.get(method, None),
                  label=PLOT_LABELS.get(method, method))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'Relative $L_2$ error')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_bc_solution(net, bc_type, path, device='cpu'):
    """Exact vs predicted u(x) for Neumann/Robin Poisson."""
    with torch.no_grad():
        x_dense = torch.linspace(0, 1, 500, dtype=torch.float64)
        u_pred = net(x_dense.reshape(-1, 1).to(device)).squeeze(-1).cpu()
        if bc_type == 'neumann':
            u_exact = torch.cos(np.pi * x_dense)
        else:
            u_exact = torch.exp(x_dense)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6),
                                   gridspec_kw={'height_ratios': [3, 1]},
                                   sharex=True)
    ax1.plot(x_dense.numpy(), u_exact.numpy(), 'k-', lw=2, label='Exact')
    ax1.plot(x_dense.numpy(), u_pred.numpy(), 'r--', lw=1.5,
             label='Hybrid prediction')
    ax1.set_ylabel('u(x)')
    ax1.legend()
    ax1.set_title(f'1D Poisson ({bc_type} BC) — solution')
    ax1.grid(True, alpha=0.3)
    ax2.semilogy(x_dense.numpy(),
                 np.abs((u_pred - u_exact).numpy()) + 1e-16, 'b-', lw=1)
    ax2.set_xlabel('x')
    ax2.set_ylabel('|error|')
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_heat_solution(last_run, path):
    Xg = last_run['Xg'].numpy()
    Tt = last_run['Tt'].numpy()
    u_pred = last_run['u_pred_grid'].numpy()
    u_exact = last_run['u_exact_grid'].numpy()
    err = np.abs(u_pred - u_exact)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    vmin = min(u_exact.min(), u_pred.min())
    vmax = max(u_exact.max(), u_pred.max())
    im0 = axes[0].pcolormesh(Xg, Tt, u_exact, cmap='viridis',
                              vmin=vmin, vmax=vmax, shading='auto')
    axes[0].set_title('Exact u(x,t)')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('t')
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].pcolormesh(Xg, Tt, u_pred, cmap='viridis',
                              vmin=vmin, vmax=vmax, shading='auto')
    axes[1].set_title('Hybrid prediction')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('t')
    fig.colorbar(im1, ax=axes[1])
    im2 = axes[2].pcolormesh(Xg, Tt, err, cmap='hot_r', shading='auto')
    axes[2].set_title('|error|')
    axes[2].set_xlabel('x'); axes[2].set_ylabel('t')
    fig.colorbar(im2, ax=axes[2])
    fig.suptitle('1D heat equation — solution comparison', y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ==================================================================
# Main
# ==================================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='all',
                        choices=['all', '1d', '2d', 'ablation_delta',
                                 'ablation_epsilon', 'ablation_ncheb',
                                 'ablation_M', 'ablation_interior',
                                 'poisson_1d_neumann', 'poisson_1d_robin',
                                 'heat_1d'])
    parser.add_argument('--steps-1d', type=int, default=5000)
    parser.add_argument('--steps-2d', type=int, default=8000)
    parser.add_argument('--m-1d', type=int, default=2000)
    parser.add_argument('--m-2d', type=int, default=5000)
    parser.add_argument('--seeds', type=int, default=3,
                        help='number of seeds (3 -> [42,123,456], 1 -> [42])')
    args = parser.parse_args()

    seeds_use = SEEDS[:args.seeds]
    common_1d = dict(M=args.m_1d, steps=args.steps_1d)
    common_2d = dict(M=args.m_2d, steps=args.steps_2d)

    # ---------- Experiment 1: 1D Poisson ----------
    if args.exp in ('all', '1d'):
        print('=' * 60)
        print('Experiment 1: 1D Poisson')
        print('=' * 60)
        methods_1d = ['hybrid', 'ad', 'cheb', 'finufft']
        res_1d = run_1d_experiment(methods_1d, seeds=SEEDS, **common_1d)

        # save JSON
        save_1d = {k: {kk: vv for kk, vv in v.items() if kk != 'history'}
                   for k, v in res_1d.items()}
        save_1d['_histories'] = {k: v['history'] for k, v in res_1d.items()}
        with open(os.path.join(RESULTS_DIR, 'poisson_1d_results.json'),
                  'w') as f:
            json.dump(save_1d, f, indent=2)

        # plots
        plot_convergence_1d(res_1d,
                            os.path.join(FIGS_DIR, 'poisson_1d_convergence.png'))
        # use the hybrid net from seed=42 for the solution plot
        r_hyb = train_pinn_1d_poisson('hybrid',
                                      dict(seed=42, **common_1d))
        plot_solution_1d(r_hyb['net'],
                         os.path.join(FIGS_DIR, 'poisson_1d_solution.png'))
        print('  1D results saved.')

    # ---------- Experiment 2: 2D Poisson ----------
    if args.exp in ('all', '2d'):
        print('=' * 60)
        print('Experiment 2: 2D Poisson')
        print('=' * 60)
        methods_2d = ['hybrid', 'ad', 'cheb', 'finufft']
        res_2d = run_2d_experiment(methods_2d, seeds=SEEDS, **common_2d)

        save_2d = {k: {kk: vv for kk, vv in v.items()
                       if kk not in ('history', 'last_run')}
                   for k, v in res_2d.items()}
        save_2d['_histories'] = {k: v['history'] for k, v in res_2d.items()}
        with open(os.path.join(RESULTS_DIR, 'poisson_2d_results.json'),
                  'w') as f:
            json.dump(save_2d, f, indent=2)

        plot_convergence_2d(res_2d,
                             os.path.join(FIGS_DIR, 'poisson_2d_convergence.png'))
        if 'hybrid' in res_2d:
            plot_solution_2d(res_2d['hybrid']['last_run'],
                             os.path.join(FIGS_DIR, 'poisson_2d_solution.png'))
        print('  2D results saved.')

    # ---------- Experiment 3: ablation delta (5 values x seeds) ----------
    if args.exp in ('all', 'ablation_delta'):
        print('=' * 60)
        print('Experiment A1: ablation over delta')
        print('=' * 60)
        deltas = [0.05, 0.10, 0.15, 0.20, 0.30]
        abl_d = run_ablation_param(train_pinn_1d_poisson, 'delta', deltas,
                                   seeds=seeds_use, tag='delta',
                                   M=2000, steps=5000, overlap=0.05)
        plot_ablation(abl_d, r'Boundary-layer width $\delta$',
                      'Ablation: boundary-layer width (1D Poisson, hybrid)',
                      os.path.join(FIGS_DIR, 'ablation_delta.png'))
        with open(os.path.join(RESULTS_DIR, 'ablation_delta.json'), 'w') as f:
            json.dump(abl_d, f, indent=2)
        print('  Ablation delta saved.')

    # ---------- Experiment A2: ablation epsilon (overlap) ----------
    if args.exp in ('all', 'ablation_epsilon'):
        print('=' * 60)
        print('Experiment A2: ablation over overlap epsilon')
        print('=' * 60)
        epsilons = [0.0, 0.025, 0.05, 0.1]
        abl_e = run_ablation_param(train_pinn_1d_poisson, 'overlap', epsilons,
                                   seeds=seeds_use, tag='epsilon',
                                   M=2000, steps=5000, delta=0.15)
        plot_ablation(abl_e, r'Overlap width $\epsilon$',
                      'Ablation: overlap width (1D Poisson, hybrid)',
                      os.path.join(FIGS_DIR, 'ablation_epsilon.png'))
        with open(os.path.join(RESULTS_DIR, 'ablation_epsilon.json'),
                  'w') as f:
            json.dump(abl_e, f, indent=2)
        print('  Ablation epsilon saved.')

    # ---------- Experiment A3: ablation N_c (Chebyshev modes) ----------
    if args.exp in ('all', 'ablation_ncheb'):
        print('=' * 60)
        print('Experiment A3: ablation over N_c')
        print('=' * 60)
        nchebs = [8, 16, 32, 64]
        abl_n = run_ablation_param(train_pinn_1d_poisson, 'N_cheb', nchebs,
                                   seeds=seeds_use, tag='Nc',
                                   M=2000, steps=5000)
        plot_ablation(abl_n, r'Chebyshev modes $N_c$',
                      'Ablation: Chebyshev modes (1D Poisson, hybrid)',
                      os.path.join(FIGS_DIR, 'ablation_ncheb.png'))
        with open(os.path.join(RESULTS_DIR, 'ablation_ncheb.json'),
                  'w') as f:
            json.dump(abl_n, f, indent=2)
        print('  Ablation N_c saved.')

    # ---------- Experiment A4: ablation M (collocation count) ----------
    if args.exp in ('all', 'ablation_M'):
        print('=' * 60)
        print('Experiment A4: ablation over M')
        print('=' * 60)
        Ms = [500, 1000, 2000, 5000]
        abl_M = {}
        for method in ['hybrid', 'ad']:
            runs_by_M = []
            for Mval in Ms:
                runs = [train_pinn_1d_poisson(method,
                        dict(seed=s, M=Mval, steps=5000)) for s in seeds_use]
                l2s = [r['l2'] for r in runs]
                times = [r['time'] for r in runs]
                runs_by_M.append({
                    'l2_mean': float(np.mean(l2s)),
                    'l2_std': float(np.std(l2s)),
                    'time_mean': float(np.mean(times)),
                    'time_std': float(np.std(times)),
                    'l2_per_seed': [float(v) for v in l2s],
                    'time_per_seed': [float(v) for v in times]})
                print(f'  << M={Mval} {method}: '
                      f'L2={np.mean(l2s):.4e} time={np.mean(times):.1f}s')
            abl_M[method] = {'l2_mean': [d['l2_mean'] for d in runs_by_M],
                             'l2_std': [d['l2_std'] for d in runs_by_M],
                             'time_mean': [d['time_mean'] for d in runs_by_M],
                             'per_M': runs_by_M}
        plot_ablation_dual(abl_M, Ms, 'Collocation points M',
                           'Ablation: collocation count M (1D Poisson)',
                           os.path.join(FIGS_DIR, 'ablation_M.png'))
        with open(os.path.join(RESULTS_DIR, 'ablation_M.json'), 'w') as f:
            json.dump({'M_values': Ms, 'per_method': abl_M}, f, indent=2)
        print('  Ablation M saved.')

    # ---------- Experiment A5: interior method cheb vs nufft ----------
    if args.exp in ('all', 'ablation_interior'):
        print('=' * 60)
        print('Experiment A5: interior method cheb vs nufft')
        print('=' * 60)
        res_int = {}
        for im in ['cheb', 'nufft']:
            runs = [train_pinn_1d_poisson('hybrid',
                    dict(seed=s, M=2000, steps=5000, interior_method=im))
                    for s in seeds_use]
            l2s = [r['l2'] for r in runs]
            times = [r['time'] for r in runs]
            res_int[im] = {
                'l2_mean': float(np.mean(l2s)),
                'l2_std': float(np.std(l2s)),
                'time_mean': float(np.mean(times)),
                'time_std': float(np.std(times)),
                'l2_per_seed': [float(v) for v in l2s],
                'time_per_seed': [float(v) for v in times],
                'history': runs[0]['history']}
            print(f'  << interior={im}: L2={np.mean(l2s):.4e} '
                  f'time={np.mean(times):.1f}s')
        PLOT_LABELS['cheb'] = 'Interior: Chebyshev'
        PLOT_LABELS['nufft'] = 'Interior: NUFFT'
        PLOT_COLORS['nufft'] = '#ff7f0e'
        plot_convergence_1d(res_int,
                            os.path.join(FIGS_DIR,
                                         'ablation_interior_convergence.png'))
        save_int = {k: {kk: vv for kk, vv in v.items() if kk != 'history'}
                    for k, v in res_int.items()}
        save_int['_histories'] = {k: v['history'] for k, v in res_int.items()}
        with open(os.path.join(RESULTS_DIR, 'ablation_interior.json'),
                  'w') as f:
            json.dump(save_int, f, indent=2)
        print('  Ablation interior saved.')

    # ---------- Experiment A6: 1D Poisson Neumann BC ----------
    if args.exp in ('all', 'poisson_1d_neumann'):
        print('=' * 60)
        print('Experiment A6: 1D Poisson Neumann BC')
        print('=' * 60)
        res_n = run_methods_seeds(train_pinn_1d_poisson_bc,
                                  ['ad', 'hybrid'],
                                  seeds=seeds_use, tag='neumann',
                                  M=2000, steps=5000, bc_type='neumann',
                                  lambda_bc=10.0)
        plot_convergence_1d(res_n,
                            os.path.join(FIGS_DIR,
                                         'poisson_1d_neumann_convergence.png'))
        r_hyb = train_pinn_1d_poisson_bc('hybrid',
                  dict(seed=42, M=2000, steps=5000, bc_type='neumann',
                       lambda_bc=10.0))
        plot_bc_solution(r_hyb['net'], 'neumann',
                        os.path.join(FIGS_DIR,
                                     'poisson_1d_neumann_solution.png'))
        save_n = {k: {kk: vv for kk, vv in v.items() if kk != 'history'}
                  for k, v in res_n.items()}
        save_n['_histories'] = {k: v['history'] for k, v in res_n.items()}
        with open(os.path.join(RESULTS_DIR,
                               'poisson_1d_neumann_results.json'), 'w') as f:
            json.dump(save_n, f, indent=2)
        print('  Neumann results saved.')

    # ---------- Experiment A7: 1D Poisson Robin BC ----------
    if args.exp in ('all', 'poisson_1d_robin'):
        print('=' * 60)
        print('Experiment A7: 1D Poisson Robin BC')
        print('=' * 60)
        res_r = run_methods_seeds(train_pinn_1d_poisson_bc,
                                  ['ad', 'hybrid'],
                                  seeds=seeds_use, tag='robin',
                                  M=2000, steps=5000, bc_type='robin',
                                  lambda_bc=10.0)
        plot_convergence_1d(res_r,
                            os.path.join(FIGS_DIR,
                                         'poisson_1d_robin_convergence.png'))
        r_hyb = train_pinn_1d_poisson_bc('hybrid',
                  dict(seed=42, M=2000, steps=5000, bc_type='robin',
                       lambda_bc=10.0))
        plot_bc_solution(r_hyb['net'], 'robin',
                        os.path.join(FIGS_DIR,
                                     'poisson_1d_robin_solution.png'))
        save_r = {k: {kk: vv for kk, vv in v.items() if kk != 'history'}
                  for k, v in res_r.items()}
        save_r['_histories'] = {k: v['history'] for k, v in res_r.items()}
        with open(os.path.join(RESULTS_DIR,
                               'poisson_1d_robin_results.json'), 'w') as f:
            json.dump(save_r, f, indent=2)
        print('  Robin results saved.')

    # ---------- Experiment A8: 1D heat equation ----------
    if args.exp in ('all', 'heat_1d'):
        print('=' * 60)
        print('Experiment A8: 1D heat equation')
        print('=' * 60)
        # 3 seeds requested; reduce to 1 if too slow (decided by timing).
        res_h = run_methods_seeds(train_pinn_heat_1d,
                                  ['hybrid', 'ad'],
                                  seeds=seeds_use, tag='heat',
                                  M=1000, N_t=100, steps=3000)
        plot_convergence_1d(res_h,
                            os.path.join(FIGS_DIR,
                                         'heat_1d_convergence.png'))
        save_h = {k: {kk: vv for kk, vv in v.items() if kk != 'history'}
                  for k, v in res_h.items()}
        save_h['_histories'] = {k: v['history'] for k, v in res_h.items()}
        with open(os.path.join(RESULTS_DIR, 'heat_1d_results.json'),
                  'w') as f:
            json.dump(save_h, f, indent=2)
        print('  Heat results saved.')

    print('\nAll requested experiments complete.')
