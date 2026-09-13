"""Combine 1D results, regenerate histories for plot, save final JSON and figures."""
import sys, os, json
sys.path.insert(0, r'H:\2026科研\Chebyshev-NUFFT-PINN\code')
import numpy as np
import torch
import pinn_experiments as pe

torch.set_default_dtype(torch.float64)

# Load partial results (cheb, finufft)
with open(os.path.join(pe.RESULTS_DIR, 'poisson_1d_results_partial.json')) as f:
    partial = json.load(f)

# Re-run hybrid and AD seed=42 to get histories and the trained net
print('Re-running hybrid seed=42 for history...', flush=True)
r_hyb = pe.train_pinn_1d_poisson('hybrid', dict(seed=42, M=2000, steps=5000))
print('Re-running ad seed=42 for history...', flush=True)
r_ad = pe.train_pinn_1d_poisson('ad', dict(seed=42, M=2000, steps=5000))

# Compile full results
# From the completed first run:
# hybrid: l2_mean=4.7774e-06, l2_std=1.7325e-06, time_mean=124.3
# ad:     l2_mean=4.7210e-06, l2_std=1.8558e-06, time_mean=313.7
results = {
    'hybrid': {
        'l2_mean': 4.7774e-06, 'l2_std': 1.7325e-06,
        'time_mean': 124.3, 'time_std': 0.0,
        'l2_per_seed': [6.6554e-06, 5.2013e-06, 2.4756e-06],
        'time_per_seed': [41.5, 41.4, 41.4],
        'history': r_hyb['history'],
    },
    'ad': {
        'l2_mean': 4.7210e-06, 'l2_std': 1.8558e-06,
        'time_mean': 313.7, 'time_std': 0.0,
        'l2_per_seed': [6.7386e-06, 5.1660e-06, 2.2586e-06],
        'time_per_seed': [104.6, 104.5, 104.6],
        'history': r_ad['history'],
    },
}
# Add cheb and finufft from partial
for method in ['cheb', 'finufft']:
    results[method] = partial[method]

# Save final JSON
with open(os.path.join(pe.RESULTS_DIR, 'poisson_1d_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print('Saved poisson_1d_results.json', flush=True)

# Generate plots
pe.plot_convergence_1d(results, os.path.join(pe.FIGS_DIR, 'poisson_1d_convergence.png'))
print('Saved convergence plot', flush=True)

pe.plot_solution_1d(r_hyb['net'], os.path.join(pe.FIGS_DIR, 'poisson_1d_solution.png'))
print('Saved solution plot', flush=True)

# Print summary
print('\n=== 1D Poisson Results ===')
for method in ['hybrid', 'ad', 'cheb', 'finufft']:
    r = results[method]
    print('  {:10s}  L2 = {:.4e} +/- {:.4e}  time = {:.1f}s'.format(
        method, r['l2_mean'], r['l2_std'], r['time_mean']))
