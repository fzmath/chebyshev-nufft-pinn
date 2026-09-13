"""Generate 2D Poisson convergence curves for Chebyshev vs AD."""
import warnings
warnings.filterwarnings('ignore')
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, r'H:\2026科研\Chebyshev-NUFFT-PINN\code')
import diff_chebyshev as dc

torch.set_default_dtype(torch.float64)
ROOT = r'H:\2026科研\Chebyshev-NUFFT-PINN'
FIGS_DIR = os.path.join(ROOT, 'figures')

class MLP(nn.Module):
    def __init__(self, dim_in=2, hidden=64, layers=3):
        super().__init__()
        mods = [nn.Linear(dim_in, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            mods += [nn.Linear(hidden, hidden), nn.Tanh()]
        mods.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*mods)
    def forward(self, x):
        return self.net(x).squeeze(-1)

def chebyshev_laplacian_2d(N):
    """Build 2D Chebyshev Laplacian on tensor grid."""
    D, _ = dc.chebyshev_diff_matrix(N)
    D2 = D @ D
    I = np.eye(N + 1)
    L2d = np.kron(D2, I) + np.kron(I, D2)
    return torch.tensor(L2d, dtype=torch.float64)

def train_2d(method='cheb', seed=42, N=32, steps=8000):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Chebyshev tensor grid (N intervals = N+1 points per dimension)
    n_pts = N + 1
    x1 = np.cos(np.pi * np.arange(n_pts) / N)
    x2 = np.cos(np.pi * np.arange(n_pts) / N)
    X1, X2 = np.meshgrid(x1, x2, indexing='ij')
    x_flat = np.column_stack([X1.ravel(), X2.ravel()])
    # Map from [-1,1]^2 to [0,1]^2
    x_flat = (x_flat + 1) / 2
    x = torch.tensor(x_flat, dtype=torch.float64)
    M = n_pts * n_pts
    # Exact solution: u = sin(pi*x)*sin(pi*y)
    u_exact = torch.sin(np.pi * x[:, 0]) * torch.sin(np.pi * x[:, 1])
    f_rhs = 2 * (np.pi ** 2) * u_exact
    # Laplacian matrix (with Jacobian scaling: d/dx = 2*d/dxi)
    L2d = chebyshev_laplacian_2d(N) * 4.0
    net = MLP(dim_in=2, hidden=64, layers=3)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    # Boundary mask
    boundary = torch.zeros(M, dtype=torch.bool)
    for i in range(n_pts):
        for j in range(n_pts):
            if i == 0 or i == n_pts-1 or j == 0 or j == n_pts-1:
                boundary[i * n_pts + j] = True
    history = []
    t0 = time.time()
    for step in range(steps + 1):
        if step == 4000:
            for pg in opt.param_groups:
                pg['lr'] = 5e-4
        if step == 6000:
            for pg in opt.param_groups:
                pg['lr'] = 1e-4
        opt.zero_grad()
        u_pred = net(x)
        if method == 'cheb':
            lap_u = L2d @ u_pred
        else:
            x_ad = x.clone().requires_grad_(True)
            u = net(x_ad)
            g1 = torch.autograd.grad(u, x_ad, torch.ones_like(u), create_graph=True)[0]
            ux = g1[:, 0]
            uy = g1[:, 1]
            gxx = torch.autograd.grad(ux, x_ad, torch.ones_like(ux), create_graph=True)[0][:, 0]
            gyy = torch.autograd.grad(uy, x_ad, torch.ones_like(uy), create_graph=True)[0][:, 1]
            lap_u = gxx + gyy
            u_pred = u
        pde_res = lap_u + f_rhs
        loss_pde = torch.mean(pde_res[~boundary] ** 2)
        loss_bc = torch.mean((u_pred[boundary] - u_exact[boundary]) ** 2)
        loss = loss_pde + 100 * loss_bc
        loss.backward()
        opt.step()
        if step % 500 == 0:
            with torch.no_grad():
                l2 = (torch.norm(u_pred - u_exact) / torch.norm(u_exact)).item()
            history.append({'step': step, 'l2': l2})
            if step % 1000 == 0:
                print(f'  [{method}/2d] step={step:5d}: l2={l2:.4e}  loss={loss.item():.4e}')
    elapsed = time.time() - t0
    print(f'  {method}: final L2={history[-1]["l2"]:.4e}, time={elapsed:.1f}s')
    return history

if __name__ == '__main__':
    print('Running Chebyshev 2D for convergence curve...')
    hist_cheb = train_2d('cheb', seed=42, N=32, steps=8000)
    print('Running AD 2D for convergence curve...')
    hist_ad = train_2d('ad', seed=42, N=32, steps=8000)
    # Plot
    plt.figure(figsize=(7, 5))
    steps_cheb = [h['step'] for h in hist_cheb]
    l2_cheb = [h['l2'] for h in hist_cheb]
    steps_ad = [h['step'] for h in hist_ad]
    l2_ad = [h['l2'] for h in hist_ad]
    plt.semilogy(steps_cheb, l2_cheb, 'b-o', label='Chebyshev-PINN (Ours)', markersize=4, linewidth=1.5)
    plt.semilogy(steps_ad, l2_ad, 'r-s', label='AD-PINN', markersize=4, linewidth=1.5)
    plt.xlabel('Training step', fontsize=12)
    plt.ylabel('Relative $L^2$ error', fontsize=12)
    plt.title('2D Poisson equation: training convergence', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, 'poisson_2d_convergence.png'), dpi=150)
    plt.close()
    print('Convergence plot saved to poisson_2d_convergence.png')
    print('Done.')
