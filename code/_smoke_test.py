"""Quick smoke test for pinn_experiments."""
import sys
sys.path.insert(0, r'H:\2026科研\Chebyshev-NUFFT-PINN\code')
import pinn_experiments as pe

print('=== 1D smoke test (M=200, steps=200) ===')
for method in ['hybrid', 'ad', 'cheb', 'finufft']:
    r = pe.train_pinn_1d_poisson(method, dict(seed=42, M=200, steps=200, warmup=50))
    print('  {:10s}  l2={:.4e}  time={:.2f}s'.format(method, r['l2'], r['time']))

print('=== 2D smoke test (M=300, steps=200) ===')
for method in ['hybrid', 'ad']:
    r = pe.train_pinn_2d_poisson(method, dict(seed=42, M=300, steps=200, warmup=50))
    print('  {:10s}  l2={:.4e}  time={:.2f}s'.format(method, r['l2'], r['time']))

print('All smoke tests passed.')
