# Differentiable Chebyshev-NUFFT Hybrid Layer for PINNs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/pytorch-2.6-red.svg)](https://pytorch.org/)

A differentiable hybrid spectral layer for computing spatial derivatives in physics-informed neural networks (PINNs) on **non-periodic domains** with arbitrary non-uniform collocation points.

## Key Features

- **Chebyshev boundary layers**: Enforce Dirichlet, Neumann, and Robin boundary conditions via spectral differentiation
- **Adaptive interior basis**: Choose between Chebyshev (non-periodic) or NUFFT (periodic) basis per subdomain
- **Schwarz-style overlap**: Smoothstep windowing ensures derivative continuity across domain interfaces
- **Exact adjoint**: The entire operator assembles into a single constant matrix $A$, with adjoint $A^\mathsf{T}$, enabling exact backpropagation verified by `gradcheck`
- **Domain decomposition**: Pure Chebyshev and pure Fourier methods both fail on non-uniform non-periodic problems; the hybrid decomposition is essential

## Method Overview

The hybrid layer decomposes the computational domain $\Omega = \Omega_L \cup \Omega_I \cup \Omega_R$:

- **Boundary layers** ($\Omega_L, \Omega_R$): Chebyshev spectral differentiation with SVD-regularized scattered-point regression
- **Interior** ($\Omega_I$): Adaptive choice of Chebyshev or NUFFT basis
- **Overlap**: Schwarz-style extension with smoothstep windowing

The derivative operator is:
$$\mathcal{D}_{\text{hybrid}} = W_L \mathcal{D}_{\text{cheb},L} + W_I \mathcal{D}_{\text{int}} + W_R \mathcal{D}_{\text{cheb},R}$$

## Installation

```bash
# Clone the repository
git clone https://github.com/fzmath/chebyshev-nufft-pinn.git
cd chebyshev-nufft-pinn

# Install dependencies
pip install torch==2.6.0 finufft==2.5.1 numpy scipy matplotlib
```

## Quick Start

```python
import torch
from code.diff_hybrid import HybridSpectralDerivative

# Create hybrid derivative operator on [0, 1] with non-uniform points
x = torch.rand(2000).sort().values  # non-uniform collocation points
deriv = HybridSpectralDerivative(
    points=x,
    boundary_layer_width=0.15,
    overlap_width=0.05,
    n_chebyshev_modes=32,
    interior_method='chebyshev'  # or 'nufft'
)

# Compute first and second derivatives
u = torch.sin(torch.pi * x)
u_x = deriv(u, order=1)
u_xx = deriv(u, order=2)

# The operator is differentiable: backward pass uses exact adjoint A^T
loss = u_xx.sum()
loss.backward()
```

## Project Structure

```
chebyshev-nufft-pinn/
├── code/
│   ├── diff_chebyshev.py      # Differentiable Chebyshev spectral layer
│   ├── diff_hybrid.py         # Differentiable Chebyshev-NUFFT hybrid layer
│   └── pinn_experiments.py    # PINN training and experiments
├── tests/
│   ├── test_diff_chebyshev.py # Unit tests for Chebyshev layer
│   └── test_diff_hybrid.py    # Unit tests for hybrid layer
├── paper/
│   ├── main.tex               # LaTeX source (sn-jnl template)
│   ├── main.pdf               # Compiled paper
│   ├── cover_letter.tex       # Cover letter
│   └── refs.bib               # Bibliography
├── results/                   # Experimental results (JSON)
├── figures/                   # Generated figures
└── docs/                      # Method and experiment design docs
```

## Experimental Results

### 1D Poisson (Dirichlet BC, 3 seeds)

| Method | Rel. L² error | Time (s/seed) |
|--------|--------------|---------------|
| AD-PINN | $4.72\times10^{-6} \pm 1.86\times10^{-6}$ | 104.6 |
| Cheb-PINN | $4.14\times10^{-6} \pm 1.19\times10^{-6}$ | 119.8 |
| **Hybrid-PINN (Ours)** | $\mathbf{4.78\times10^{-6} \pm 1.73\times10^{-6}}$ | **41.4** |
| finufft-PINN | $\approx 14.0$ (diverged) | 473.9 |

### 2D Poisson (Dirichlet BC, 3 seeds)

| Method | Rel. L² error | Time (s/seed) |
|--------|--------------|---------------|
| AD-PINN | $1.41\times10^{-5} \pm 4.70\times10^{-6}$ | 1636.0 |
| **Hybrid-PINN (Ours)** | $\mathbf{2.07\times10^{-4} \pm 1.36\times10^{-4}}$ | **521.3** |
| Cheb-PINN | $\approx 1.48$ (diverged) | 355.3 |
| finufft-PINN | $\approx 0.77$ (diverged) | 1476.5 |

### Heat Equation (parabolic PDE)

| Method | Rel. L² error | Training time | Speedup |
|--------|--------------|---------------|---------|
| AD-PINN | $7.86\times10^{-4}$ | 50 min | — |
| **Hybrid-PINN** | $\mathbf{7.86\times10^{-4}}$ | **18 min** | **2.8×** |

## Running Tests

```bash
# Run all unit tests
python -m pytest tests/ -v

# Run gradcheck for differentiable layers
python tests/test_diff_hybrid.py
python tests/test_diff_chebyshev.py
```

## Running Experiments

```bash
# 1D Poisson experiment
python code/pinn_experiments.py --problem poisson_1d --boundary dirichlet

# 2D Poisson experiment
python code/pinn_experiments.py --problem poisson_2d --boundary dirichlet

# Heat equation experiment
python code/pinn_experiments.py --problem heat_1d

# Ablation studies
python code/pinn_experiments.py --problem ablation --param delta
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{wang2026chebyshev,
  title={Differentiable {C}hebyshev--{NUFFT} Hybrid Layer for Physics-Informed Neural Networks with Non-Periodic Boundary Conditions},
  author={Wang, Fuchang and Cao, Huirong},
  journal={Journal of Scientific Computing},
  year={2026},
  note={Under review}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This work was supported by the Natural Science Foundation of Hebei Province and the Scientific Research Foundation of Emergency Management University.
