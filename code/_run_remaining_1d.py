"""Run remaining 1D methods (cheb, finufft) and save results."""
import sys, os, json, time
sys.path.insert(0, r'H:\2026科研\Chebyshev-NUFFT-PINN\code')
sys.path.insert(0, r'H:\2026科研\JCP-Journal of Computational Physics\code')
import numpy as np
import torch
import pinn_experiments as pe

torch.set_default_dtype(torch.float64)

SEEDS = [42, 123, 456]
methods = ['cheb', 'finufft']

results = {}
for method in methods:
    runs = []
    for seed in SEEDS:
        print(f'  >> 1D [{method}] seed={seed}', flush=True)
        r = pe.train_pinn_1d_poisson(method, dict(seed=seed, M=2000, steps=5000))
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
    print(f'  << 1D [{method}] L2={results[method]["l2_mean"]:.4e} '
          f'+/- {results[method]["l2_std"]:.4e}  '
          f'time={results[method]["time_mean"]:.1f}s', flush=True)

# Save partial results
out = os.path.join(pe.RESULTS_DIR, 'poisson_1d_results_partial.json')
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f'Partial results saved to {out}', flush=True)
