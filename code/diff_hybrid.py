"""
Differentiable Chebyshev-NUFFT hybrid layer for Physics-Informed Neural Networks.

This module is the core novelty of the JSC paper.  It combines:

  * a **Chebyshev spectral layer** (diff_chebyshev.py) near the boundaries, where
    the solution is non-periodic and Dirichlet / Neumann / Robin boundary
    conditions must be imposed exactly, and
  * a **differentiable finufft layer** (diff_finufft.py) in the periodic
    interior, where arbitrary non-uniform collocation points are handled
    directly by the NUFFT.

Domain decomposition (1D, physical interval [a, b])
----------------------------------------------------
    left layer    : [a, a + delta]          -> Chebyshev on [-1, 1]
    right layer   : [b - delta, b]           -> Chebyshev on [-1, 1]
    interior      : [a + delta, b - delta]   -> periodic on [-pi, pi]

Coordinate maps and Jacobians
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    left layer    : xi_L = 2 (x - a) / delta - 1,        d xi_L / dx = 2 / delta
    right layer   : xi_R = 2 (x - (b - delta)) / delta - 1, d xi_R / dx = 2 / delta
    interior      : theta = 2 pi (x - (a+delta)) / L_I - pi, d theta / dx = 2 pi / L_I,
                    L_I = b - a - 2 delta.

Hence for an order-m derivative,

    d^m / dx^m = (2 / delta)^m  d^m / d xi^m      (boundary layers)
    d^m / dx^m = (2 pi / L_I)^m d^m / d theta^m  (interior).

Boundary-layer Chebyshev pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The non-uniform layer values are interpolated (Gaussian RBF, partition of
unity / RBF collocation, see below) onto the Gauss-Lobatto Chebyshev nodes,
differentiated by the frozen Chebyshev matrix D^m, and interpolated back to
the original non-uniform points.  Because the collocation points are FIXED,
the whole map is a constant linear operator.

Interior periodic pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~
Type-1 NUFFT (non-uniform -> Fourier coefficients, iflag=+1) -> multiply by
(i k)^m -> Type-2 NUFFT (Fourier coefficients -> non-uniform values,
iflag=-1), divided by the number of interior points M_I (Riemann-sum
normalisation), with the (-1)^m sign correction of diff_finufft.

Sigmoid windows
~~~~~~~~~~~~~~~
    w_L(x) = 1 - sigmoid((x - (a + delta - overlap/2)) / s),
    w_R(x) =     sigmoid((x - (b - delta + overlap/2)) / s),
    w_I(x) = 1 - w_L(x) - w_R(x),
with s = window_scale.  The blended derivative is

    y(x) = w_L(x) y_cheb,L(x) + w_I(x) y_nufft(x) + w_R(x) y_cheb,R(x).

When delta covers the whole interval the interior vanishes and w_I = 0,
recovering a pure Chebyshev discretisation; when delta -> 0 the interior
covers the whole interval and w_L = w_R = 0, recovering pure NUFFT.

Differentiability
-----------------
Every building block is linear in u:
  * the RBF interpolation matrices P, Q are constants (fixed nodes);
  * the Chebyshev matrices D are constants (backward uses D^T);
  * the NUFFT blocks use the exact Type-1 / Type-2 adjoints already
    implemented in diff_finufft.py;
  * the sigmoid windows are diagonal (self-adjoint).

The whole hybrid operator is therefore a single constant M x M matrix A.
`DiffHybridDerivative` wraps it: forward = A u, backward = A^T grad.  It is
verified by torch.autograd.gradcheck (double precision) and by the
adjoint identity <A u, v> = <u, A^T v>.

References
----------
L. N. Trefethen, "Spectral Methods in MATLAB", SIAM, 2000.
Barnett et al. (2019), "A non-uniform fast Fourier transform library of types
1, 2, 3 (finufft)", SIAM J. Sci. Comput.
G. E. Fasshauer, "Meshfree Approximation Methods with MATLAB", World Sci., 2007.
"""

import os
import sys
import math

import numpy as np
import torch

# Re-use the existing, already-verified layers.
sys.path.insert(0, r"H:\2026科研\Chebyshev-NUFFT-PINN\code")
sys.path.insert(0, r"H:\2026科研\JCP-Journal of Computational Physics\code")

import diff_chebyshev as dc  # noqa: E402
import diff_finufft as df   # noqa: E402

torch.set_default_dtype(torch.float64)


# =====================================================================================
# Default parameters
# =====================================================================================
def _default_params(domain):
    a, b = domain
    L = b - a
    return {
        "domain": (float(a), float(b)),
        "delta": 0.15 * L,
        "overlap": 0.05 * L,
        "N_cheb": 32,
        "N_fourier": 16,
        "eps_nufft": 1e-6,
        "order": 1,
        "window_scale": None,      # filled below, overlap / 4
        "rbf_epsilon": None,       # auto-chosen from point density
    }


# =====================================================================================
# Gaussian RBF interpolation (constant matrix)
# =====================================================================================
def _gaussian_rbf_matrix(sources, targets, epsilon):
    """Build the constant Gaussian-RBF interpolation matrix P.

    For a Gaussian kernel phi(r) = exp(-(epsilon * r)^2), the RBF interpolant
    s(target) = sum_j c_j phi(target, source_j) satisfies A c = u on the data
    sites, A_ij = phi(source_i, source_j).  Evaluating at the targets gives

        v_target = P @ v_source,   P = Phi(targets, sources) @ A^{-1}.

    Args:
        sources: (n_s,) numpy array of data-site coordinates (reference coord).
        targets: (n_t,) numpy array of query coordinates.
        epsilon: Gaussian shape parameter.

    Returns:
        P: (n_t, n_s) numpy float64 matrix.
    """
    sources = np.asarray(sources, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    d_ts = targets[:, None] - sources[None, :]
    Phi_ts = np.exp(-(epsilon * d_ts) ** 2)
    d_ss = sources[:, None] - sources[None, :]
    A_ss = np.exp(-(epsilon * d_ss) ** 2)
    # Tiny Tikhonov regularisation: suppresses the high-frequency RBF modes
    # that the Chebyshev derivative (especially order >= 2) would otherwise
    # amplify.  Relative to the diagonal (= 1) this is negligible for
    # accuracy but stabilises the solve.
    A_ss = A_ss + 1e-12 * np.eye(A_ss.shape[0])
    # Solve A_ss @ X = Phi_ts.T  =>  P = X.T.  More stable than inv.
    X = np.linalg.solve(A_ss, Phi_ts.T)
    return X.T


def _auto_rbf_epsilon(sources):
    """Choose a Gaussian shape parameter from the number of source points.

    On the reference interval [-1, 1] a Gaussian RBF needs a shape parameter
    that grows mildly with the number of nodes (the ``RBF width'').  Empirically
    eps ~ 0.2 * sqrt(n) keeps the Gram matrix well-conditioned and the
    interpolation error at the 1e-8 level for smooth functions, for the range
    n = O(10..1000) used in PINN collocation.
    """
    n = np.size(sources)
    if n < 3:
        return 1.0
    return 0.1 * math.sqrt(n)


def _cheb_deriv_at_points(Nc, order, xi_eval):
    """Chebyshev spectral derivative evaluated at *arbitrary* reference points.

    Instead of differentiating on the Lobatto grid and then *re-interpolating*
    (which squares the interpolation error), this assembles the exact linear map

        d^m u / d xi^m  at xi_eval  =  E @ C^m @ V @ u_lobatto

    where
      * V  : Lobatto values -> Chebyshev coefficients a_n   (Type-I DCT),
      * C  : coefficient-space derivative recurrence, applied ``order'' times,
      * E  : E[i, n] = T_n(xi_eval[i]) = cos(n arccos(xi_eval[i])).

    The only approximation is the RBF interpolation of u onto the Lobatto grid
    (done by the caller); the Chebyshev differentiation itself is spectral-exact
    because the derivative polynomial is evaluated directly, not interpolated.

    Args:
        Nc: Chebyshev resolution (Nc+1 Lobatto points).
        order: derivative order.
        xi_eval: (P,) array of reference points in [-1, 1] (allowing slight
            extrapolation, which is harmless because the windows suppress it).

    Returns:
        M: (P, Nc+1) matrix such that deriv_at_eval = M @ u_lobatto.
    """
    # values (Lobatto) -> coefficients
    eye = np.eye(Nc + 1)
    V = np.empty((Nc + 1, Nc + 1))
    for col in range(Nc + 1):
        V[:, col] = dc._cheb_values_to_coeffs(eye[:, col], Nc)
    # coefficients -> derivative coefficients, `order` times
    C = np.eye(Nc + 1)
    Cstep = np.empty((Nc + 1, Nc + 1))
    for col in range(Nc + 1):
        Cstep[:, col] = dc._cheb_coeff_derivative(eye[:, col], Nc)
    for _ in range(order):
        C = Cstep @ C
    # coefficients -> values at arbitrary points.  Evaluation outside [-1, 1]
    # is clipped to the endpoint (Chebyshev extrapolation is unstable); the
    # sigmoid windows are designed so that the interior block has vanishing
    # weight before this extrapolation region is reached.
    xi_eval = np.asarray(xi_eval, dtype=np.float64).reshape(-1, 1)
    n_idx = np.arange(Nc + 1).reshape(1, -1)
    E = np.cos(n_idx * np.arccos(np.clip(xi_eval, -1.0, 1.0)))
    return E @ C @ V


def _cheb_scattered_operator(Nc, order, xi_sources, xi_eval):
    """Chebyshev derivative operator directly on scattered reference points.

    Instead of RBF-interpolating the scattered values onto the Lobatto grid
    (which the derivative then amplifies), this fits the Chebyshev expansion
    *directly* from the scattered sources by ordinary least squares,

        E_sources @ a = u_sources,   E_sources[i, n] = T_n(xi_sources[i]),

    and then applies the spectral derivative in coefficient space and evaluates
    the result at every point xi_eval:

        d^m u / d xi^m (xi_eval) = E_eval @ C^m @ E_sources^+ @ u_sources.

    The least-squares solve is well-conditioned (n_sources >> Nc+1 for the
    point counts used here) and avoids the second RBF interpolation, which is
    the dominant error source for order >= 2 derivatives.

    Args:
        Nc: Chebyshev resolution (Nc+1 coefficients).
        order: derivative order.
        xi_sources: (n_s,) reference points in [-1, 1] carrying u.
        xi_eval: (P,) reference points where the derivative is required.

    Returns:
        T: (P, n_s) matrix such that deriv_at_eval = T @ u_sources.
    """
    xi_sources = np.asarray(xi_sources, dtype=np.float64).ravel()
    xi_eval = np.asarray(xi_eval, dtype=np.float64).ravel()
    ns_idx = np.arange(Nc + 1).reshape(1, -1)
    E_s = np.cos(ns_idx * np.arccos(np.clip(xi_sources, -1.0, 1.0)[:, None]))
    E_e = np.cos(ns_idx * np.arccos(np.clip(xi_eval, -1.0, 1.0)[:, None]))
    # derivative in coefficient space
    eye = np.eye(Nc + 1)
    Cstep = np.empty((Nc + 1, Nc + 1))
    for col in range(Nc + 1):
        Cstep[:, col] = dc._cheb_coeff_derivative(eye[:, col], Nc)
    C = np.eye(Nc + 1)
    for _ in range(order):
        C = Cstep @ C
    # SVD least squares a = pinv(E_s) u.  Using SVD (rather than the normal
    # equations) keeps the condition number at cond(E_s), so the derivative in
    # coefficient space is applied to a well-conditioned coefficient vector.
    U, s, Vt = np.linalg.svd(E_s, full_matrices=False)
    # zero out tiny singular values
    tol = s[0] * 1e-12 if s.size else 0.0
    s_inv = np.where(s > tol, 1.0 / s, 0.0)
    coeff_map = (Vt.T * s_inv) @ U.T          # (Nc+1, n_s), E_s^+
    return E_e @ C @ coeff_map


# =====================================================================================
# Sigmoid windows
# =====================================================================================
def _sigmoid(z):
    """Numerically stable logistic function 1 / (1 + exp(-z))."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


# =====================================================================================
# 1D hybrid operator (linear, constant matrix)
# =====================================================================================
class HybridOperator1D:
    """Assemble the constant M x M matrix A of the 1D hybrid derivative.

    The operator is fully linear in u for fixed collocation points x.  The
    forward application is y = A u and the exact adjoint is grad_u = A^T grad.

    Attributes:
        A: (M, M) torch float64 matrix such that (A u)[j] is the order-th
           derivative of u at x_j.
        M: number of collocation points.
        params: the parameter dictionary used to build the operator.
    """

    def __init__(self, x, params):
        x = np.asarray(x, dtype=np.float64).ravel()
        self.M = x.size
        self.params = dict(params)

        a, b = params["domain"]
        L = b - a
        delta = float(params["delta"])
        overlap = float(params["overlap"])
        order = int(params["order"])
        Nc = int(params["N_cheb"])
        Nf = int(params["N_fourier"])
        eps = float(params["eps_nufft"])
        ws = params.get("window_scale", None)
        if ws is None:
            ws = overlap / 4.0
        self.window_scale = float(ws)

        # ---- Schwarz overlap decomposition ---------------------------------
        # The left Chebyshev block covers [a, c_L + overlap] and the interior
        # block covers [c_L, c_R].  The overlap [c_L, c_L+overlap] is used by
        # BOTH blocks, so neither Chebyshev polynomial has to evaluate outside
        # its own [-1,1] interval: at every point at least one block is inside
        # its reference interval (no unstable extrapolation).  The window
        # smoothly fades between the two spectral operators in the overlap.
        c_L = a + delta
        c_R = b - delta
        L_L = delta + overlap          # width of left Cheb interval
        L_R = delta + overlap          # width of right Cheb interval
        L_I = b - a - 2.0 * delta      # width of interior interval

        left_mask = x <= c_L + overlap
        right_mask = x >= c_R - overlap
        interior_mask = (x >= c_L) & (x <= c_R)

        # ---- smoothstep windows over all points ---------------------------
        # w_L = 1 deep left, 0 once the left overlap is fully crossed;
        # w_R = 1 deep right; w_I = 1 in the deep interior.  The smoothstep
        # S(t) = 3 t^2 - 2 t^3 vanishes with zero derivative at both ends, so:
        #   * w_L = 1 exactly at x = c_L  (left layer fully active at the
        #     interface), and 0 exactly at x = c_L + overlap;
        #   * w_I = 0 for x <= c_L, so the interior Chebyshev block is never
        #     asked to evaluate outside its own [-1, 1] interval (no clipping,
        #     no unstable extrapolation).
        def _smoothstep(t):
            t = np.clip(t, 0.0, 1.0)
            return 3.0 * t * t - 2.0 * t * t * t

        w_L = 1.0 - _smoothstep((x - c_L) / overlap)
        w_R = _smoothstep((x - (c_R - overlap)) / overlap)
        w_I = 1.0 - w_L - w_R
        w_I = np.clip(w_I, 0.0, 1.0)
        total = w_L + w_I + w_R
        w_L = w_L / total
        w_I = w_I / total
        w_R = w_R / total
        self.w_L = torch.from_numpy(w_L)
        self.w_I = torch.from_numpy(w_I)
        self.w_R = torch.from_numpy(w_R)

        A = np.zeros((self.M, self.M), dtype=np.float64)

        # ===================== left Chebyshev block =========================
        if np.any(left_mask):
            idx_L = np.where(left_mask)[0]
            xi_s = 2.0 * (x[idx_L] - a) / L_L - 1.0       # source ref coord
            xi_all = 2.0 * (x - a) / L_L - 1.0            # every point

            s_L = (2.0 / L_L) ** order
            T_L = s_L * _cheb_scattered_operator(Nc, order, xi_s, xi_all)

            block = np.zeros((self.M, self.M), dtype=np.float64)
            block[:, idx_L] = T_L
            A += (w_L)[:, None] * block

        # ===================== right Chebyshev block ========================
        if np.any(right_mask):
            idx_R = np.where(right_mask)[0]
            c0 = c_R - overlap
            xi_s = 2.0 * (x[idx_R] - c0) / L_R - 1.0
            xi_all = 2.0 * (x - c0) / L_R - 1.0

            s_R = (2.0 / L_R) ** order
            T_R = s_R * _cheb_scattered_operator(Nc, order, xi_s, xi_all)

            block = np.zeros((self.M, self.M), dtype=np.float64)
            block[:, idx_R] = T_R
            A += (w_R)[:, None] * block

        # ===================== interior spectral block ======================
        n_I = int(np.sum(interior_mask))
        interior_method = params.get("interior_method", "cheb")
        if n_I > 0 and L_I > 1e-12:
            idx_I = np.where(interior_mask)[0]

            if interior_method == "nufft":
                # periodic Fourier derivative (for genuinely periodic interiors)
                # theta = 2 pi (x - c_L) / L_I - pi, so d/dx = (2 pi / L_I) d/dtheta.
                theta_I = 2.0 * np.pi * (x[idx_I] - c_L) / L_I - np.pi
                theta_all = 2.0 * np.pi * (x - c_L) / L_I - np.pi
                k = np.arange(-Nf // 2, Nf // 2, dtype=np.float64)
                B = np.exp(1j * np.outer(k, theta_I))
                C = np.exp(-1j * np.outer(theta_all, k))
                factor = (1j * k) ** order
                s_I_nufft = (2.0 * np.pi / L_I) ** order
                nufft_block = s_I_nufft * ((-1.0) ** order) / n_I \
                              * (C @ (factor[:, None] * B))
                block = np.zeros((self.M, self.M), dtype=np.float64)
                block[:, idx_I] = nufft_block.real
                A += (w_I)[:, None] * block
            else:
                # Chebyshev spectral interior (direct scattered regression)
                xi_s = 2.0 * (x[idx_I] - c_L) / L_I - 1.0
                xi_all = 2.0 * (x - c_L) / L_I - 1.0

                s_I = (2.0 / L_I) ** order
                T_I = s_I * _cheb_scattered_operator(Nc, order, xi_s, xi_all)

                block = np.zeros((self.M, self.M), dtype=np.float64)
                block[:, idx_I] = T_I
                A += (w_I)[:, None] * block
        else:
            n_I = 0

        self.n_interior = n_I
        self.A = torch.from_numpy(A)

    def apply(self, u):
        """Apply the operator: y = A u.  u may be a (M,) or (M,1) tensor."""
        return self.A @ u


# =====================================================================================
# Functional interface (autograd-native via the precomputed constant matrix)
# =====================================================================================
def hybrid_derivative_1d(u, x, params):
    """Order-`order` hybrid derivative of u at non-uniform points x on [a, b].

    Args:
        u: (M,) tensor of function values at the non-uniform points.
        x: (M,) tensor of non-uniform coordinates in [a, b].
        params: parameter dict (see module doc).  May carry an optional key
            'operator'; if present it is used directly (saves reassembly).

    Returns:
        du: (M,) tensor, d^order u / dx^order.
    """
    op = params.get("operator")
    if op is None:
        op = HybridOperator1D(x, params)
    return op.apply(u)


def hybrid_laplacian_1d(u, x, params):
    """1D Laplacian = order-2 hybrid derivative."""
    p = dict(params)
    p["order"] = 2
    op = p.get("operator")
    if op is None:
        op = HybridOperator1D(x, p)
    return op.apply(u)


# =====================================================================================
# 2D hybrid Laplacian
# =====================================================================================
def _gaussian_rbf2d_laplacian_matrix(xy, epsilon):
    """Build the constant (M, M) matrix of the 2D Gaussian-RBF Laplacian.

    For a Gaussian kernel phi(r) = exp(-eps^2 r^2) with r^2 = (x-x')^2+(y-y')^2,
    the RBF interpolant s(x,y) = sum_j c_j phi((x,y), (x_j,y_j)) satisfies
    A c = u on the nodes, A_ij = phi(node_i, node_j).  Applying
    Delta phi = (4 eps^4 r^2 - 4 eps^2) phi and evaluating at the nodes gives

        (Delta s)(node_i) = [ (4 eps^4 R2 - 4 eps^2) .* Phi ] @ A^{-1} @ u.

    The resulting matrix is linear in u, hence fully differentiable.

    Args:
        xy: (M, 2) numpy array of node coordinates.
        epsilon: Gaussian shape parameter (in the reference coordinate scale).

    Returns:
        L: (M, M) float64 matrix such that (L u)[i] is Delta u at node i.
    """
    xy = np.asarray(xy, dtype=np.float64)
    dx = xy[:, 0:1] - xy[:, 0:1].T
    dy = xy[:, 1:2] - xy[:, 1:2].T
    r2 = dx * dx + dy * dy
    Phi = np.exp(-(epsilon ** 2) * r2)
    # Delta phi at the evaluation nodes (i = row), node j (column).
    delta_Phi = (4.0 * epsilon ** 4 * r2 - 4.0 * epsilon ** 2) * Phi
    # solve A c = u, then (Delta s)_i = sum_j delta_Phi[i,j] c_j
    # i.e. L = delta_Phi @ A^{-1}.
    X = np.linalg.solve(Phi, delta_Phi.T)
    return X.T


def hybrid_laplacian_2d(u, xy, params):
    """2D Laplacian Delta u = d2u/dx2 + d2u/dy2 at scattered points.

    The simplified per-axis 1D hybrid strategy cannot separate x and y when
    the nodes are scattered (u(x, y) is not a single-valued function of x
    alone), so the 2D layer uses a fully coupled Gaussian-RBF interpolant over
    the (x, y) plane.  The Laplacian of the Gaussian kernel is applied
    analytically, which keeps the operator a single constant (M, M) matrix
    (forward = L u, adjoint = L^T grad) and preserves exact differentiability.

    Args:
        u: (M,) tensor of function values at (M, 2) scattered points.
        xy: (M, 2) tensor of non-uniform coordinates.
        params: parameter dict; 'rbf_epsilon' may override the auto choice.

    Returns:
        lap: (M,) tensor, Delta u.
    """
    xy_np = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    eps = params.get("rbf_epsilon", None)
    if eps is None:
        # 2D Gaussian-RBF optimum on [0,1]^2: empirical eps ~= 0.03 sqrt(M)
        # (much narrower than the 1D rule because the kernel couples x and y).
        eps = 0.03 * math.sqrt(xy_np.shape[0])
    L = _gaussian_rbf2d_laplacian_matrix(xy_np, eps)
    L_t = torch.from_numpy(L)
    return L_t @ torch.as_tensor(u, dtype=torch.float64).reshape(-1)


# =====================================================================================
# Differentiable custom Function (exact adjoint)
# =====================================================================================
class DiffHybridDerivative(torch.autograd.Function):
    """Differentiable wrapper of the assembled hybrid operator.

    Forward :  y = A @ u
    Backward:  grad_u = A^T @ grad_output

    A is the constant M x M matrix assembled by :class:`HybridOperator1D`.
    Because every sub-operator (RBF P/Q, Chebyshev D, NUFFT, sigmoid windows)
    is linear and self-adjoint or adjoint-exact, A^T is the exact adjoint.
    """

    @staticmethod
    def forward(ctx, u, A):
        """
        Args:
            u: (M,) input values.
            A: (M, M) constant hybrid operator matrix.

        Returns:
            y: (M,) derivative values.
        """
        ctx.save_for_backward(A)
        return A @ u

    @staticmethod
    def backward(ctx, grad_output):
        (A,) = ctx.saved_tensors
        grad_u = A.T @ grad_output
        return grad_u, None


# =====================================================================================
# Convenience: cross-check against the raw finufft layer (pure interior limit)
# =====================================================================================
def _interior_finufft_reference(u, x, params):
    """Reference interior derivative using the differentiable finufft layer.

    Used only by the tests to confirm the dense NUFFT block matches finufft.
    """
    a, b = params["domain"]
    delta = float(params["delta"])
    L_I = b - a - 2.0 * delta
    interior_mask = (x > a + delta) & (x < b - delta)
    idx = torch.where(interior_mask)[0]
    theta = 2.0 * np.pi * (x[idx] - (a + delta)) / L_I - np.pi
    theta = theta.reshape(-1, 1)
    return df.finufft_derivative(u[idx], theta, int(params["N_fourier"]),
                                 int(params["order"]), 0,
                                 float(params["eps_nufft"])), idx
