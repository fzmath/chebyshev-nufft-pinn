"""
1D Poisson parameter inversion: recover diffusion coefficient a(x) from sparse data.
PDE: -(a(x) u')' = f(x), u(0)=u(1)=0
We observe u at sparse points and invert for a(x) parameterized as a network.
"""
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, r'H:\2026科研\Chebyshev-NUFFT-PINN\code')
import diff_hybrid as dh
import diff_chebyshev as dc

torch.set_default_dtype(torch.float64)

ROOT = r'H:\2026科研\Chebyshev-NUFFT-PINN'
RESULTS_DIR = os.path.join(ROOT, 'results')
FIGS_DIR = os.path.join(ROOT, 'figures')


class MLP(nn.Module):
    def __init__(self, dim_in=1, hidden=64, layers=3):
        super().__init__()
        mods = [nn.Linear(dim_in, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            mods += [nn.Linear(hidden, hidden), nn.Tanh()]
        mods.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def get_lr(step, warmup=500, total=5000, base_lr=1e-3):
    if step < warmup:
        return base_lr * step / warmup
    progress = (step - warmup) / (total - warmup)
    return base_lr * 0.5 * (1 + np.cos(np.pi * progress))


def set_lr(opt, lr):
    for pg in opt.param_groups:
        pg['lr'] = lr


def train_inversion(method='hybrid', seed=42, M=2000, steps=10000, noise=0.0):
    """Invert for diffusion coefficient a(x) in -(a u')' = f.

    True a(x) = 1 + 0.5*sin(2*pi*x)
    True u(x) = sin(pi*x) (approximately, for variable a we use manufactured solution)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    from scipy.stats import qmc
    sampler = qmc.Halton(d=1, scramble=True, seed=seed)
    x_np = sampler.random(M).ravel()
    x = torch.tensor(x_np, dtype=torch.float64)

    # Manufactured solution: u = sin(pi*x), a = 1 + 0.5*sin(2*pi*x)
    # -(a u')' = f => f = -a' u' - a u''
    a_true = 1 + 0.5 * torch.sin(2 * np.pi * x)
    a_true_grad = np.pi * torch.cos(2 * np.pi * x)
    u_true = torch.sin(np.pi * x)
    u_true_grad = np.pi * torch.cos(np.pi * x)
    u_true_lap = -(np.pi ** 2) * torch.sin(np.pi * x)
    f_rhs = -(a_true_grad * u_true_grad + a_true * u_true_lap)

    # Sparse observation points (30% of collocation points)
    n_obs = max(100, int(M * 0.3))
    obs_idx = np.random.choice(M, n_obs, replace=False)
    x_obs = x[obs_idx]
    u_obs = u_true[obs_idx]
    if noise > 0:
        u_obs = u_obs + noise * torch.randn_like(u_obs) * torch.std(u_obs)

    # Networks: u_net predicts u, a_net predicts log(a) for positivity
    u_net = MLP(dim_in=1, hidden=64, layers=3)
    a_net = MLP(dim_in=1, hidden=64, layers=3)
    params = list(u_net.parameters()) + list(a_net.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)

    # Precompute derivative operators
    hp = dict(domain=(0,1), delta=0.15, overlap=0.05, N_cheb=32, N_fourier=16,
              eps_nufft=1e-6, order=1, interior_method='cheb')
    if method == 'hybrid':
        op1 = dh.HybridOperator1D(x_np, hp).A
        hp2 = dict(hp, order=2)
        op2 = dh.HybridOperator1D(x_np, hp2).A

    # Test grid
    x_test = torch.linspace(0, 1, 500, dtype=torch.float64)
    a_test_true = 1 + 0.5 * torch.sin(2 * np.pi * x_test)

    history = []
    t0 = time.time()

    for step in range(steps + 1):
        cur_lr = get_lr(step, warmup=500, total=steps, base_lr=1e-3)
        set_lr(opt, cur_lr)
        opt.zero_grad()

        bc_factor = x * (1.0 - x)
        u_pred = u_net(x.reshape(-1, 1)) * bc_factor
        # Parameterize a via softplus for smooth positivity: a = softplus(log_a) + 0.1
        log_a = a_net(x.reshape(-1, 1))
        a_pred = torch.nn.functional.softplus(log_a) + 0.1

        if method == 'hybrid':
            u_x = op1 @ u_pred
            u_xx = op2 @ u_pred
            a_x = op1 @ a_pred
        else:  # ad
            x_ad = x.reshape(-1, 1).requires_grad_(True)
            u_raw = u_net(x_ad)
            u_pred_ad = u_raw * x_ad[:, 0] * (1.0 - x_ad[:, 0])
            log_a_ad = a_net(x_ad)
            a_pred_ad = torch.nn.functional.softplus(log_a_ad) + 0.1
            u_x = torch.autograd.grad(u_pred_ad, x_ad, torch.ones_like(u_pred_ad),
                                      create_graph=True)[0].squeeze(-1)
            u_xx = torch.autograd.grad(u_x, x_ad, torch.ones_like(u_x),
                                       create_graph=True)[0].squeeze(-1)
            a_x = torch.autograd.grad(a_pred_ad, x_ad, torch.ones_like(a_pred_ad),
                                      create_graph=True)[0].squeeze(-1)
            u_pred = u_pred_ad
            a_pred = a_pred_ad

        # PDE residual: -(a u')' = -a' u' - a u'' = f
        pde_res = -(a_x * u_x + a_pred * u_xx) - f_rhs
        loss_pde = torch.mean(pde_res ** 2)

        # Data loss at observation points
        u_at_obs = u_pred[obs_idx]
        loss_data = torch.mean((u_at_obs - u_obs) ** 2)

        loss = loss_pde + 500.0 * loss_data
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if step % 500 == 0:
            with torch.no_grad():
                log_a_test = a_net(x_test.reshape(-1, 1))
                a_pred_test = torch.nn.functional.softplus(log_a_test) + 0.1
                l2_a = torch.norm(a_pred_test - a_test_true) / torch.norm(a_test_true)
            history.append({'step': step, 'l2_a': l2_a.item(), 'loss': loss.item()})
            if step % 1000 == 0:
                print(f'    [{method}/inversion] seed={seed} step={step:5d}: '
                      f'l2_a={l2_a.item():.4e}  loss={loss.item():.4e}')

    elapsed = time.time() - t0
    with torch.no_grad():
        log_a_test = a_net(x_test.reshape(-1, 1))
        a_pred_test = torch.nn.functional.softplus(log_a_test) + 0.1
        l2_a_final = (torch.norm(a_pred_test - a_test_true) /
                      torch.norm(a_test_true)).item()

    return {'l2_a': l2_a_final, 'time': elapsed, 'history': history,
            'a_pred': a_pred_test.numpy(), 'a_true': a_test_true.numpy(),
            'x_test': x_test.numpy()}


if __name__ == '__main__':
    results = {'hybrid': [], 'ad': []}

    for method in ['hybrid', 'ad']:
        for seed in [42, 123, 456]:
            print(f'\nRunning {method} inversion, seed={seed}...')
            r = train_inversion(method=method, seed=seed, M=2000, steps=10000)
            results[method].append({'l2_a': r['l2_a'], 'time': r['time']})
            print(f'  L2(a)={r["l2_a"]:.4e}, time={r["time"]:.1f}s')

    print('\n=== Inversion Results Summary ===')
    for m in ['hybrid', 'ad']:
        l2s = [r['l2_a'] for r in results[m]]
        times = [r['time'] for r in results[m]]
        print(f'{m}: L2(a)={np.mean(l2s):.4e} +/- {np.std(l2s):.4e}, '
              f'time={np.mean(times):.1f}s')

    with open(os.path.join(RESULTS_DIR, 'inversion_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot inversion result for one seed
    r = train_inversion(method='hybrid', seed=42, M=2000, steps=10000)
    plt.figure(figsize=(6, 4))
    plt.plot(r['x_test'], r['a_true'], 'k-', label='Exact $a(x)$', linewidth=2)
    plt.plot(r['x_test'], r['a_pred'], 'r--', label='Hybrid-PINN', linewidth=1.5)
    plt.xlabel('$x$')
    plt.ylabel('$a(x)$')
    plt.title('Diffusion coefficient inversion')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, 'inversion_result.png'), dpi=150)
    plt.close()
    print('\nInversion plot saved.')
    print('Done.')
