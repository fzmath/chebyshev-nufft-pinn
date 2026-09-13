"""Quick comparison: hybrid vs AD, 1000 steps, M=500."""
import sys
sys.path.insert(0, r'H:\2026科研\Chebyshev-NUFFT-PINN\code')
import pinn_experiments as pe

for method in ['hybrid', 'ad', 'cheb', 'finufft']:
    r = pe.train_pinn_1d_poisson(method, dict(seed=42, M=500, steps=1000, warmup=200))
    print('{:10s}  final L2 = {:.4e}  time = {:.1f}s'.format(method, r['l2'], r['time']))
    # show convergence
    for h in r['history']:
        print('    step={:5d}  l2={:.4e}'.format(h['step'], h['l2']))
