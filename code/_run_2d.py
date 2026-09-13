"""Run full 2D Poisson experiment and save results incrementally."""
import sys, os, json, time
sys.path.insert(0, r'H:\2026科研\Chebyshev-NUFFT-PINN\code')
import numpy as np
import torch
import pinn_experiments as pe

torch.set_default_dtype(torch.float64)

SEEDS = [42, 123, 456]
methods = ['hybrid', 'ad', 'cheb', 'finufft']
M = 5000
steps = 8000

results = {}
for method in methods:
    runs = []
    for seed in SEEDS:
        print(f'  >> 2D [{method}] seed={seed}', flush=True)
        r = pe.train_pinn_2d_poisson(method, dict(seed=seed, M=M, steps=steps))
        runs.append(r)
        print(f'     l2={r["l2"]:.4e}  time={r["time"]:.1f}s', flush=True)
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
    print(f'  << 2D [{method}] L2={results[method]["l2_mean"]:.4e} '
          f'+/- {results[method]["l2_std"]:.4e}  '
          f'time={results[method]["time_mean"]:.1f}s', flush=True)
    # Save incrementally
    with open(os.path.join(pe.RESULTS_DIR, 'poisson_2d_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

# Save the last hybrid run's grid for the solution plot
last_hyb = runs[-1]  # this is from finufft, need hybrid separately
# Re-run hybrid seed=42 briefly to get grid for plot
print('Re-running hybrid seed=42 for solution plot...', flush=True)
r_hyb_plot = pe.train_pinn_2d_poisson('hybrid', dict(seed=42, M=M, steps=steps))

# Generate plots
pe.plot_convergence_2d(results, os.path.join(pe.FIGS_DIR, 'poisson_2d_convergence.png'))
pe.plot_solution_2d(r_hyb_plot, os.path.join(pe.FIGS_DIR, 'poisson_2d_solution.png'))
print('2D plots saved.', flush=True)

# Final summary
print('\n=== 2D Poisson Results ===')
for method in methods:
    r = results[method]
    print('  {:10s}  L2 = {:.4e} +/- {:.4e}  time = {:.1f}s'.format(
        method, r['l2_mean'], r['l2_std'], r['time_mean']))
