"""Numerical kernels for TFA: factors, weights, residual and analytic Jacobian.

Conventions: data ``X`` is (V, T), coordinates ``R`` are (V, D) with D in {2, 3},
centers are (K, D) and widths ``w = 2 sigma**2`` are (K,). The parameter vector of
the center/width solve is ``x = [centers.ravel(), widths]`` (length K * (D + 1)).
All arrays are float64.
"""

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares
from scipy.sparse.linalg import LinearOperator
from sklearn.cluster import KMeans

from htfa._engines import EPS, Engine, NumpyEngine, loss_terms

Array = NDArray[np.float64]

# nlss_method values: the structured bounded Levenberg-Marquardt solver (default),
# the same solver on the variable-projection objective (linear loss only; see
# `ProjectedProblem`) and the scipy least_squares methods kept for cross-checking.
NLSS_METHODS = ("bounded-lm", "varpro", "trf", "dogbox")
# scipy.optimize.least_squares' default tolerances, used by the structured solver
# with the same meaning (see `bounded_lm`).
FTOL = XTOL = GTOL = 1e-8


def ftol_for(eps: float) -> float:
    """``ftol`` for an engine whose arithmetic has machine epsilon ``eps``.

    ``FTOL`` (1e-8) is below float32 resolution: on MPS the relative noise of a
    cost evaluation is of order 1e-8 (float32 sums over samples, float64 over
    voxels), so near convergence steps would be accepted or rejected on noise.
    The tolerance is raised to ``eps`` (1.2e-7 in float32; unchanged in float64).
    """
    return max(FTOL, eps)


# Relative eigenvalue cutoff of the weights' gram matrix in `ProjectedProblem`
# (the square root of float64 epsilon: a condition number above 6.7e7).
GRAM_RCOND = float(np.sqrt(np.finfo(np.float64).eps))

# Initial damping relative to the largest diagonal entry of the scaled Hessian
# (Madsen, Nielsen & Tingleff 2004, sec. 3.2).
LM_TAU = 1e-3

# Relative tolerance (atol = btol) of the LSMR trust-region subproblem solve. scipy's
# default (1e-6) stops the outer solve ~1e-6 away from the minimizer: compressed vs.
# uncompressed linear-loss solves then differed by 1.5e-6 in x. Measured with
# 1e-8: that difference falls to 2e-8, and on V=3000, T=200, K=20 the final cost
# matches 1e-10 to 4e-11 relative, at ~1.7x the time of 1e-6 (0.82 s vs 0.46 s).
LSMR_TOL = 1e-8

# `axis_grid` describes the coordinates by axis when the distinct values, summed
# over the axes, number at most this fraction of the voxels. Each factor is then
# evaluated per axis on the distinct values and gathered: development_fmri has
# 37 + 45 + 32 distinct values for V = 25079 voxels, and factors() took 2.0 ms
# instead of 16 ms (K=100). With arbitrary coordinates (about V distinct values
# per axis) the per-axis tables would cost D times the direct evaluation.
GRID_MAX_FRACTION = 0.5


@dataclass(frozen=True)
class AxisGrid:
    """Distinct coordinates ``values[j]`` along each axis ``j`` and the position
    ``index[j]`` (V,) of every voxel's coordinate among them."""

    values: tuple[Array, ...]
    index: tuple[NDArray[np.intp], ...]


def axis_grid(R: Array) -> AxisGrid | None:
    """`AxisGrid` of the coordinates ``R`` (V, D), or None when they have more
    than ``GRID_MAX_FRACTION * V`` distinct values summed over the axes."""
    columns = [np.unique(R[:, j], return_inverse=True) for j in range(R.shape[1])]
    if sum(values.size for values, _ in columns) > GRID_MAX_FRACTION * R.shape[0]:
        return None
    return AxisGrid(
        tuple(values for values, _ in columns),
        tuple(index.ravel() for _, index in columns),
    )


def sq_distances(R: Array, centers: Array, grid: AxisGrid | None = None) -> Array:
    """Squared distances (V, K) between voxels ``R`` (V, D) and ``centers`` (K, D).

    Summed per dimension rather than via ``|r|^2 - 2 r.c + |c|^2`` so that large
    coordinates do not cancel catastrophically. With ``grid`` (`axis_grid` of
    ``R``) the per-axis squares are computed on the distinct coordinates and
    gathered; the sums are the same, bit for bit.
    """
    if grid is not None:
        d = np.zeros((R.shape[0], centers.shape[0]))
        for j, (values, index) in enumerate(zip(grid.values, grid.index)):
            diff = values[:, None] - centers[None, :, j]
            d += (diff * diff)[index]
        return d
    d = np.zeros((R.shape[0], centers.shape[0]))
    for j in range(R.shape[1]):
        diff = R[:, j, None] - centers[None, :, j]
        d += diff * diff
    return d


def factors(
    R: Array, centers: Array, widths: Array, grid: AxisGrid | None = None
) -> Array:
    """Radial basis factors ``F[v, k] = exp(-||r_v - c_k||^2 / w_k)``, shape (V, K).

    With ``grid`` (`axis_grid` of ``R``) each factor is the product over the axes
    of ``exp(-(r_vj - c_kj)**2 / w_k)``, evaluated on the distinct coordinates and
    gathered. The two forms agree to rounding (measured on development_fmri:
    8.7e-14 relative per entry at most). Where the product underflows, the
    exponential of the sum is below the smallest normal number too.
    """
    if grid is None:
        return np.exp(-sq_distances(R, centers) / widths[None, :])
    F = np.ones((R.shape[0], centers.shape[0]))
    for j, (values, index) in enumerate(zip(grid.values, grid.index)):
        diff = values[:, None] - centers[None, :, j]
        F *= np.exp(-(diff * diff) / widths[None, :])[index]
    return F


def ridge_weights(X: Array, F: Array, method: str = "rr") -> Array:
    """Weights (K, T) solving ``(F^T F + beta I) W = F^T X``.

    ``method="rr"`` uses ``beta = var(X)``; ``method="ols"`` uses ``beta = 0``.
    """
    if method not in ("rr", "ols"):
        raise ValueError(f"weight method must be 'rr' or 'ols', got {method!r}")
    gram = F.T @ F
    if method == "rr":
        gram = gram + np.var(X) * np.eye(F.shape[1])
    result: Array = scipy.linalg.solve(gram, F.T @ X, assume_a="pos")
    return result


def max_sigma(R: Array) -> float:
    """BrainIAK's maximum width ``2 * max(std(R, axis=0))**2``."""
    return 2.0 * float(np.max(np.std(R, axis=0))) ** 2


def bounds(
    R: Array, K: int, lower_ratio: float, upper_ratio: float
) -> tuple[Array, Array]:
    """Lower and upper bounds for ``x = [centers.ravel(), widths]``.

    Centers lie within the per-dimension coordinate range of ``R``; widths lie in
    ``[lower_ratio, upper_ratio] * max_sigma(R)``.
    """
    sigma = max_sigma(R)
    lb = np.concatenate(
        [np.tile(np.nanmin(R, axis=0), K), np.full(K, lower_ratio * sigma)]
    )
    ub = np.concatenate(
        [np.tile(np.nanmax(R, axis=0), K), np.full(K, upper_ratio * sigma)]
    )
    return lb, ub


def init_centers_widths(
    R: Array, K: int, random_state: int | np.random.RandomState | None
) -> tuple[Array, Array]:
    """K-means++ centers (K, D) on ``R`` and widths (K,) equal to ``max_sigma(R)``."""
    km = KMeans(n_clusters=K, init="k-means++", n_init=10, random_state=random_state)
    km.fit(R)
    centers = np.asarray(km.cluster_centers_, dtype=np.float64)
    return centers, np.full(K, max_sigma(R))


@dataclass
class Prior:
    """HTFA template prior on one subject's centers and widths.

    ``centers_cov`` is a full (K, D, D) SPD covariance per factor; ``widths_var`` is
    (K,); ``scale`` multiplies both prior terms of the squared residual.
    """

    centers: Array
    widths: Array
    centers_cov: Array
    widths_var: Array
    scale: float


class ResidualProblem:
    """Residual and matrix-free Jacobian of the center/width least-squares problem.

    Data residual ``s * (X - F W)`` with ``s = 1 / (sqrt(2) std(X))`` (``s = 1`` for
    constant data), followed, when a prior is given, by the K center residuals
    ``sqrt(scale * dc^T Sigma^-1 dc)`` and the K signed width residuals
    ``sqrt(scale / var) * (w - prior.widths)``.

    ``data_on_engine=True`` sends only ``W`` to the engine, whose data must
    already be ``X`` (`ProjectedProblem` sets it once per solve).

    With ``compress=True`` (exact only for a linear loss) ``(X, W)`` is replaced by
    ``(X Q, Rq^T)`` where ``W^T = Q Rq`` is the thin QR, and ``cost_offset`` holds
    ``0.5 s^2 ||X (I - Q Q^T)||_F^2`` so that costs match the uncompressed problem.
    """

    def __init__(
        self,
        X: Array,
        W: Array,
        R: Array,
        engine: Engine,
        prior: Prior | None = None,
        compress: bool = False,
        data_on_engine: bool = False,
    ) -> None:
        if compress and data_on_engine:
            raise ValueError("compress needs the data sent to the engine")
        X = np.asarray(X, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        self.R = np.ascontiguousarray(R, dtype=np.float64)
        self.K = W.shape[0]
        self.D = self.R.shape[1]
        std = float(np.std(X))
        self.s = 1.0 / (np.sqrt(2.0) * std) if std > 0 else 1.0
        self.cost_offset = 0.0
        if compress:
            Q, Rq = np.linalg.qr(W.T)
            XQ = X @ Q
            self.cost_offset = 0.5 * self.s**2 * float(np.sum((X - XQ @ Q.T) ** 2))
            X, W = XQ, Rq.T
        self.engine = engine
        if data_on_engine:
            self.engine.set_weights(W)
        else:
            self.engine.set(X, W)
        self.n_data = X.size
        self._w_sq_norms = np.sum(W * W, axis=1)
        self.prior = prior
        if prior is not None:
            inv = np.linalg.inv(np.asarray(prior.centers_cov, dtype=np.float64))
            # Symmetrize so that Sigma^-1 dc is exactly half the quadratic's gradient.
            self._cov_inv = 0.5 * (inv + np.swapaxes(inv, 1, 2))
            self._width_coef = np.sqrt(prior.scale / np.asarray(prior.widths_var))
        self.n_residuals = self.n_data + (2 * self.K if prior is not None else 0)
        self.n_params = self.K * (self.D + 1)

    def _unpack(self, x: ArrayLike) -> tuple[Array, Array]:
        x = np.asarray(x, dtype=np.float64).ravel()
        n = self.K * self.D
        return x[:n].reshape(self.K, self.D), x[n:]

    @cached_property
    def _grid(self) -> AxisGrid | None:
        """`axis_grid` of the coordinates, for :meth:`cost` and
        :meth:`cost_grad_hess` (computed on first use)."""
        return axis_grid(self.R)

    def _center_prior(self, C: Array) -> tuple[Array, Array]:
        """Center residuals r_c (K,) and their gradients dr_c/dC (K, D)."""
        assert self.prior is not None
        dc = C - self.prior.centers
        sdc = np.einsum("kij,kj->ki", self._cov_inv, dc)
        r_c = np.sqrt(np.maximum(self.prior.scale * np.sum(dc * sdc, axis=1), 0.0))
        inv_r = np.divide(1.0, r_c, out=np.zeros_like(r_c), where=r_c > 0)
        return r_c, (self.prior.scale * inv_r)[:, None] * sdc

    def residual(self, x: ArrayLike) -> Array:
        """Residual vector of length ``V*T`` (+ ``2K`` with a prior)."""
        C, w = self._unpack(x)
        r_d = self.s * self.engine.residual(factors(self.R, C, w))
        if self.prior is None:
            return r_d
        r_c, _ = self._center_prior(C)
        r_w = self._width_coef * (w - self.prior.widths)
        return np.concatenate([r_d, r_c, r_w])

    def jacobian(self, x: ArrayLike) -> LinearOperator:
        """Matrix-free Jacobian of :meth:`residual` at ``x``."""
        C, w = self._unpack(x)
        R, K, s, n_data = self.R, self.K, self.s, self.n_data
        d = sq_distances(R, C)
        Fw = np.exp(-d / w[None, :]) / w[None, :]
        prior = self.prior
        grad_c = self._center_prior(C)[1] if prior is not None else None

        def matvec(p: Array) -> Array:
            pc, pw = self._unpack(p)
            A = Fw * (2.0 * (R @ pc.T - np.sum(C * pc, axis=1)[None, :]) + d * (pw / w))
            data = -s * self.engine.times_w(A)
            if grad_c is None:
                return data
            return np.concatenate(
                [data, np.sum(grad_c * pc, axis=1), self._width_coef * pw]
            )

        def rmatvec(u: Array) -> Array:
            u = np.asarray(u, dtype=np.float64).ravel()
            G = Fw * self.engine.times_wt(u[:n_data])
            g_c = -2.0 * s * (G.T @ R - C * np.sum(G, axis=0)[:, None])
            g_w = -s * np.sum(G * d, axis=0) / w
            if grad_c is not None:
                g_c = g_c + grad_c * u[n_data : n_data + K, None]
                g_w = g_w + self._width_coef * u[n_data + K :]
            return np.concatenate([g_c.ravel(), g_w])

        def matmat(P: Array) -> Array:
            return np.column_stack([matvec(col) for col in np.asarray(P).T])

        def rmatmat(U: Array) -> Array:
            return np.column_stack([rmatvec(col) for col in np.asarray(U).T])

        return LinearOperator(
            (self.n_residuals, self.n_params),
            matvec=matvec,
            rmatvec=rmatvec,
            matmat=matmat,
            rmatmat=rmatmat,
            dtype=np.float64,
        )

    def _prior_terms(
        self, C: Array, w: Array, loss: str, derivatives: bool
    ) -> tuple[float, Array, Array]:
        """Prior cost ``0.5 sum rho(r**2)`` over the 2K prior residuals, and (when
        ``derivatives``) its gradient (P,) and Hessian (P, P).

        The center term is a function of ``q_k = r_ck**2 = scale dc^T S dc``, so its
        exact Hessian ``scale rho' S + 2 rho'' scale**2 (S dc)(S dc)^T`` is used,
        with the eigenvalue ``rho' + 2 q rho''`` along ``dc`` floored at ``EPS`` as
        scipy floors its loss weights. The width residuals are linear in ``w``, so
        their Hessian ``coef**2 (rho' + 2 r**2 rho'')`` is exact (same floor).
        """
        assert self.prior is not None
        K, D, P = self.K, self.D, self.n_params
        dc = C - self.prior.centers
        Sdc = np.einsum("kij,kj->ki", self._cov_inv, dc)
        q = np.maximum(self.prior.scale * np.sum(dc * Sdc, axis=1), 0.0)
        r_w = self._width_coef * (w - self.prior.widths)
        rho_c, rho1_c, rho2_c = loss_terms(loss, q)
        rho_w, rho1_w, rho2_w = loss_terms(loss, r_w * r_w)
        cost = 0.5 * float(np.sum(rho_c) + np.sum(rho_w))
        grad, H = np.zeros(P), np.zeros((P, P))
        if not derivatives:
            return cost, grad, H
        scale = self.prior.scale
        grad[: K * D] = (rho1_c[:, None] * scale * Sdc).ravel()
        grad[K * D :] = rho1_w * r_w * self._width_coef
        along = np.maximum(rho1_c + 2.0 * q * rho2_c, EPS)
        beta = np.divide(along - rho1_c, q, out=np.zeros_like(q), where=q > 0)
        for k in range(K):
            block = scale * (
                rho1_c[k] * self._cov_inv[k]
                + beta[k] * scale * np.outer(Sdc[k], Sdc[k])
            )
            H[k * D : (k + 1) * D, k * D : (k + 1) * D] = block
        width_idx = np.arange(K * D, P)
        H[width_idx, width_idx] = self._width_coef**2 * np.maximum(
            rho1_w + 2.0 * r_w * r_w * rho2_w, EPS
        )
        return cost, grad, H

    def cost(self, x: ArrayLike, loss: str) -> float:
        """Objective ``0.5 * sum(rho(r**2))`` over :meth:`residual` (without
        ``cost_offset``), evaluated without forming the residual vector."""
        C, w = self._unpack(x)
        F = factors(self.R, C, w, self._grid)
        total = self.engine.data_cost(F, self.s, loss)
        if self.prior is not None:
            total += self._prior_terms(C, w, loss, derivatives=False)[0]
        return total

    def cost_grad_hess(
        self,
        x: ArrayLike,
        loss: str,
        derivatives: tuple[Array, Array, Array] | None = None,
    ) -> tuple[float, Array, Array, Array]:
        """Cost (as :meth:`cost`), gradient (P,), positive semidefinite
        Gauss-Newton matrix (P, P) and Hessian (P, P) of the objective.
        ``derivatives`` passes `factor_derivatives` at ``x`` when the caller has
        them already.

        The Gauss-Newton matrix is the Hessian without the data term's
        second-order part ``sum_i rho'_i r_i Hess(r_i)`` (the prior terms keep
        their exact, floored Hessian, which is positive semidefinite).

        The data part of the Hessian is ``J^T diag(u) J + sum_i rho'_i r_i
        Hess(r_i)``, with scipy's loss weights ``u = max(rho' + 2 r**2 rho'', EPS)``
        (exact wherever ``rho' + 2 r**2 rho'' >= 0``: always for the linear,
        soft_l1 and huber losses). ``J^T diag(u) J`` is assembled from the
        per-factor structure ``dr[v, t] / dtheta[k, a] = -s G[k, a, v] W[k, t]``
        (`htfa._engines.Engine.gauss_newton`) instead of the V*T x P Jacobian; the
        second-order term is block diagonal per factor,
        ``-s sum_v M[v, k] d2F[v, k] / dtheta_k^2`` with ``M = (rho' r) W^T``, and
        costs O(V K (D+1)**2). The parameters of factor k are
        ``theta_k = (c_k1, ..., c_kD, w_k)``.
        """
        C, w = self._unpack(x)
        K, D, P = self.K, self.D, self.n_params
        s = self.s
        if derivatives is None:
            derivatives = factor_derivatives(self.R, C, w, self._grid)
        E, F, G = derivatives
        cost, M, H_ka = self.engine.gauss_newton(np.ascontiguousarray(F.T), G, s, loss)
        MT = np.ascontiguousarray(M.T)  # (K, V)
        grad_ka = -s * (G @ MT[:, :, None])[:, :, 0]
        # Second derivatives of F = exp(-d / w) per factor, contracted with M,
        # from the moments m[k, a, b] = sum_v F M e_a e_b of e = (delta, d, 1).
        m = ((F * MT)[:, None, :] * E) @ np.swapaxes(E, 1, 2)  # (K, D+2, D+2)
        wk = w[:, None]
        S = np.empty((K, D + 1, D + 1))
        S[:, :D, :D] = 4.0 / w[:, None, None] ** 2 * m[:, :D, :D]
        S[:, range(D), range(D)] -= 2.0 / wk * m[:, D + 1, D + 1][:, None]
        S[:, :D, D] = 2.0 / wk**2 * (m[:, :D, D] / wk - m[:, :D, D + 1])
        S[:, D, :D] = S[:, :D, D]
        S[:, D, D] = (m[:, D, D] / w - 2.0 * m[:, D, D + 1]) / w**3
        k = np.arange(K)
        pos = param_positions(K, D)
        grad = np.empty(P)
        grad[pos] = grad_ka.ravel()
        gn = np.empty((P, P))
        gn[np.ix_(pos, pos)] = H_ka.reshape(P, P)
        H_ka[:] = 0.0
        H_ka[k, :, k, :] = -s * S
        second = np.empty((P, P))
        second[np.ix_(pos, pos)] = H_ka.reshape(P, P)
        if self.prior is not None:
            c_p, g_p, H_p = self._prior_terms(C, w, loss, derivatives=True)
            cost, grad, gn = cost + c_p, grad + g_p, gn + H_p
        # Exactly symmetric (the two triangles are summed in different orders).
        gn = 0.5 * (gn + gn.T)
        hess = gn + second
        return cost, grad, gn, 0.5 * (hess + hess.T)

    def column_norms(self, x: ArrayLike) -> Array:
        """Exact Euclidean norms (K*(D+1),) of the Jacobian columns at ``x``.

        Each data column is the outer product of a (V,) factor derivative with the
        weights of its factor, so its norm is ``s * ||a_j|| * ||W_k||``.
        """
        C, w = self._unpack(x)
        d = sq_distances(self.R, C)
        Fw = np.exp(-d / w[None, :]) / w[None, :]
        s2w = self.s**2 * self._w_sq_norms
        sq_c = np.empty((self.K, self.D))
        for j in range(self.D):
            a = 2.0 * Fw * (self.R[:, j, None] - C[None, :, j])
            sq_c[:, j] = s2w * np.sum(a * a, axis=0)
        a = Fw * d / w[None, :]
        sq_w = s2w * np.sum(a * a, axis=0)
        if self.prior is not None:
            sq_c += self._center_prior(C)[1] ** 2
            sq_w += self._width_coef**2
        return np.sqrt(np.concatenate([sq_c.ravel(), sq_w]))


def factor_derivatives(
    R: Array, C: Array, w: Array, grid: AxisGrid | None = None
) -> tuple[Array, Array, Array]:
    """Factors and their derivatives in a factor-major, voxel-last layout.

    Returns ``E`` (K, D + 2, V) = ``[delta_1 .. delta_D, d, 1]`` with ``delta =
    r_v - c_k`` and ``d = ||delta||**2``, the factors ``F`` (K, V) and ``G``
    (K, D + 1, V) with ``G[k, a, v] = dF[k, v] / dtheta[k, a]`` for ``theta_k =
    (c_k1, ..., c_kD, w_k)``. The (K, ., V) layout gives contiguous elementwise
    passes and per-factor sums over voxels as batched matrix products.

    ``d`` and ``F`` are, bit for bit, the transposes of `sq_distances` and
    `factors` with the same ``grid``.
    """
    K, D = C.shape
    V = R.shape[0]
    E = np.empty((K, D + 2, V))
    # From a contiguous copy of R.T: the subtraction from the strided view of R
    # took 13 ms of the 18 ms (K=100, V=25k); the values are the same.
    np.subtract(np.ascontiguousarray(R.T)[None], C[:, :, None], out=E[:, :D])
    d = E[:, D]
    if grid is None:
        # The sums of `sq_distances`, in its order.
        np.multiply(E[:, 0], E[:, 0], out=d)
        for j in range(1, D):
            d += E[:, j] * E[:, j]
        F = np.exp(-d / w[:, None])
    else:
        d[:] = sq_distances(R, C, grid).T
        F = np.ascontiguousarray(factors(R, C, w, grid).T)
    E[:, D + 1] = 1.0
    Fw = F / w[:, None]
    G = np.empty((K, D + 1, V))
    np.multiply((2.0 * Fw)[:, None, :], E[:, :D], out=G[:, :D])
    np.multiply(Fw, d / w[:, None], out=G[:, D])
    return E, F, G


def param_positions(K: int, D: int) -> NDArray[np.int64]:
    """Position in ``x = [centers.ravel(), widths]`` of each ``(k, a)``, raveled."""
    k = np.arange(K)
    pos = np.empty((K, D + 1), dtype=np.int64)
    pos[:, :D] = k[:, None] * D + np.arange(D)[None, :]
    pos[:, D] = K * D + k
    return pos.ravel()


class ProjectedProblem:
    """Variable-projection objective of the linear-loss TFA problem.

    The weights are eliminated: ``phi(x) = 0.5 s**2 (||X - F W||_F**2 + beta
    ||W||_F**2)`` with ``W = W(x)`` the ridge solution of `ridge_weights`
    (``beta = var(X)`` for ``'rr'``, 0 for ``'ols'``), plus the prior terms of
    `ResidualProblem`. Its stationary points are exactly the fixed points of
    alternating `ridge_weights` with the linear-loss center/width solve, but a
    Newton method on ``phi`` takes the coupling between the weights and the
    centers/widths into account, which the alternation ignores (it then needs
    hundreds of outer iterations on real data).

    The gradient is the fixed-weight gradient (envelope theorem). The Hessian
    is ``H_xx - H_xW H_WW^-1 H_Wx`` with the exact mixed second derivatives;
    the Gauss-Newton matrix is ``J^T (I - P) J`` (Kaufman's), with ``P`` the
    projection onto the span of the (ridge-augmented) weight columns.
    """

    def __init__(
        self,
        X: Array,
        R: Array,
        K: int,
        engine: Engine,
        prior: Prior | None = None,
        weight_method: str = "rr",
    ) -> None:
        self.X = np.ascontiguousarray(X, dtype=np.float64)
        self.R = np.ascontiguousarray(R, dtype=np.float64)
        self.K, self.D = K, self.R.shape[1]
        self.engine, self.prior = engine, prior
        if weight_method not in ("rr", "ols"):
            raise ValueError(
                f"weight method must be 'rr' or 'ols', got {weight_method!r}"
            )
        self.beta = float(np.var(self.X)) if weight_method == "rr" else 0.0
        self.weight_method = weight_method
        std = float(np.std(self.X))
        self.s = 1.0 / (np.sqrt(2.0) * std) if std > 0 else 1.0
        self.cost_offset = 0.0
        self.n_params = K * (self.D + 1)
        self._prior_only: ResidualProblem | None = None
        self._grid = axis_grid(self.R)
        # The data go to the engine once; every evaluation sends new weights.
        self.engine.set(self.X, np.zeros((K, self.X.shape[1])))

    def _unpack(self, x: ArrayLike) -> tuple[Array, Array]:
        x = np.asarray(x, dtype=np.float64).ravel()
        n = self.K * self.D
        return x[:n].reshape(self.K, self.D), x[n:]

    def _weights(self, F: Array) -> tuple[Array, Array]:
        """Weights (K, T) minimizing ``||X - F W||**2 + beta ||W||**2`` and the
        (pseudo-)inverse (K, K) of ``F^T F + beta I``, from its eigendecomposition.

        Eigenvalues below `GRAM_RCOND` times the largest are dropped (only
        possible with ``'ols'``: a vanishing or a near-duplicate factor). Their
        directions would otherwise give weights of order ``1 / gap`` that cancel
        in ``F W``, and the residual sums would then differ by more than the
        solver's tolerance between evaluation orders (measured: 1.9e-7
        relative for two factors 1e-6 apart). The weights are then the
        minimum-norm minimizer on the kept eigenvectors.
        """
        vals, vecs = np.linalg.eigh(F.T @ F + self.beta * np.eye(self.K))
        keep = vals > GRAM_RCOND * vals[-1]
        inv = (vecs[:, keep] / vals[keep]) @ vecs[:, keep].T
        return inv @ (F.T @ self.X), inv

    def weights(self, x: ArrayLike) -> Array:
        """Ridge weights (K, T) at ``x`` (see `_weights`)."""
        C, w = self._unpack(x)
        return self._weights(factors(self.R, C, w, self._grid))[0]

    def _prior_cost(self, C: Array, w: Array) -> float:
        if self.prior is None:
            return 0.0
        # A one-voxel problem on its own engine carries the prior terms.
        if self._prior_only is None:
            self._prior_only = ResidualProblem(
                self.X[:1, :1],
                np.zeros((self.K, 1)),
                self.R[:1],
                NumpyEngine(1),
                self.prior,
            )
        return self._prior_only._prior_terms(C, w, "linear", derivatives=False)[0]

    def cost(self, x: ArrayLike, loss: str = "linear") -> float:
        """``phi(x)``, from the data residual ``X - F W`` on the engine (as in
        `cost_grad_hess`, so that the solver compares like with like)."""
        C, w = self._unpack(x)
        F = factors(self.R, C, w, self._grid)
        W, _ = self._weights(F)
        self.engine.set_weights(W)
        data = self.engine.data_cost(F, self.s, "linear")
        ridge = 0.5 * self.s**2 * self.beta * float(np.sum(W * W))
        return data + ridge + self._prior_cost(C, w)

    def column_norms(self, x: ArrayLike) -> Array:
        """Fixed-weight Jacobian column norms at ``x`` (for ``x_scale='jac'``)."""
        inner = ResidualProblem(
            self.X, self.weights(x), self.R, self.engine, self.prior,
            data_on_engine=True,
        )  # fmt: skip
        return inner.column_norms(x)

    def cost_grad_hess(
        self, x: ArrayLike, loss: str = "linear"
    ) -> tuple[float, Array, Array, Array]:
        """``phi``, its gradient, Gauss-Newton matrix and Hessian (P, P)."""
        if loss != "linear":
            raise ValueError(f"ProjectedProblem needs the linear loss, got {loss!r}")
        C, w = self._unpack(x)
        K, D = self.K, self.D
        A, P, s2 = D + 1, self.n_params, self.s**2
        derivatives = factor_derivatives(self.R, C, w, self._grid)
        _, F, G = derivatives
        W, Mi = self._weights(F.T)
        inner = ResidualProblem(
            self.X, W, self.R, self.engine, self.prior, data_on_engine=True
        )
        cost, grad, gn, hess = inner.cost_grad_hess(x, "linear", derivatives)
        cost += 0.5 * s2 * self.beta * float(np.sum(W * W))
        G2 = G.reshape(K * A, -1)  # [(k, a), v]
        P2 = F @ G2.T  # (K, KA): P2[l, (k, a)] = sum_v F[l, v] G[k, a, v]
        Bt = G2 @ self.X - P2.T @ W  # (KA, T): G_ka^T (X - F W)
        Y = Mi @ P2  # M^-1 P2
        WW = W @ W.T
        rows = np.repeat(np.arange(K), A)  # factor k of each (k, a)
        term1 = WW[np.ix_(rows, rows)] * (P2.T @ Y)
        Z = (W @ Bt.T)[rows]  # Z[(k, a), (k', b)] = W_k . Bt_k'b
        term2 = Z * Y[rows].T  # times (M^-1 P_ka)[k']
        term4 = Mi[np.ix_(rows, rows)] * (Bt @ Bt.T)
        pos = param_positions(K, D)
        g1 = np.empty((P, P))
        g1[np.ix_(pos, pos)] = s2 * term1
        corr = np.empty((P, P))
        corr[np.ix_(pos, pos)] = s2 * (term1 - term2 - term2.T + term4)
        gn = gn - g1
        hess = hess - corr
        return cost, grad, 0.5 * (gn + gn.T), 0.5 * (hess + hess.T)


def _check_finite(cost: float, g: Array, gn: Array, x: Array) -> None:
    if not (np.isfinite(cost) and np.all(np.isfinite(g)) and np.all(np.isfinite(gn))):
        raise FloatingPointError(
            "bounded-lm: non-finite cost, gradient or Gauss-Newton matrix at "
            f"x = {np.array2string(x, precision=4, threshold=12)}; check the data "
            "and weights for NaN/Inf"
        )


def bounded_lm(
    problem: "ResidualProblem | ProjectedProblem",
    x0: Array,
    bounds: tuple[Array, Array],
    loss: str,
    scale: Array,
    max_nfev: int | None = None,
) -> tuple[Array, float, int, int]:
    """Bounded Levenberg-Marquardt (damped Newton) on the structured Hessian.

    Minimizes ``problem.cost(x, loss)`` over the box ``bounds`` from ``x0``
    (clipped into the box). Every iteration assembles the gradient ``g`` and the
    Hessian ``H`` (`ResidualProblem.cost_grad_hess`, P = K (D + 1) unknowns),
    fixes the variables at a bound whose gradient points out of the box, and
    solves ``(H_ff + lam diag(1 / scale_f**2)) p = -g_f`` on the free ones by
    Cholesky (``scale`` plays scipy's ``x_scale`` role; ``lam`` grows until the
    matrix is positive definite where ``H`` is indefinite). The step
    ``clip(x + p) - x`` is accepted only if it lowers the cost; ``lam`` follows
    Nielsen's rule (divided by up to 3 on success, multiplied by 2, 4, 8, ...
    on failure).

    Stopping rules, with scipy's defaults: ``ftol`` (as scipy: an accepted step
    with ratio > 0.25 lowered the cost by less than ``FTOL`` times the cost
    including ``problem.cost_offset``), ``xtol`` (as scipy: a trial step shorter
    than ``XTOL * (XTOL + ||x||)``), ``gtol`` (the largest unscaled gradient
    entry over the free variables is below ``GTOL``; scipy's trf instead
    multiplies the gradient by its Coleman-Li distance-to-bound vector) and
    ``max_nfev`` (cost evaluations, default ``100 * P`` as for scipy's trf).
    Finite-precision engines (float32 on MPS) use ``ftol`` no smaller than
    `ftol_for` of their dtype.

    Raises
    ------
    FloatingPointError
        When the cost, gradient or Gauss-Newton matrix is not finite (a
        non-finite Hessian alone falls back to the Gauss-Newton model).

    Returns
    -------
    x : ndarray, shape (P,)
        Solution.
    cost : float
        ``problem.cost(x, loss)`` (without ``cost_offset``).
    status : int
        0 ``max_nfev``, 1 ``gtol``, 2 ``ftol``, 3 ``xtol``, 4 ``ftol`` and ``xtol``
        (scipy's codes).
    nfev : int
        Cost evaluations, Hessian assemblies included.
    """
    lb, ub = bounds
    x = np.clip(np.asarray(x0, dtype=np.float64), lb, ub)
    max_nfev = 100 * x.size if max_nfev is None else max_nfev
    ftol = ftol_for(problem.engine.eps)
    cost, g, gn, hess = problem.cost_grad_hess(x, loss)
    nfev, nu = 1, 2.0
    lam = LM_TAU * max(float(np.max(scale**2 * np.diag(gn))), EPS)
    while True:
        _check_finite(cost, g, gn, x)
        free = ~(((x <= lb) & (g > 0)) | ((x >= ub) & (g < 0)))
        if not free.any() or np.max(np.abs(g[free])) < GTOL:
            return x, cost, 1, nfev
        d = scale[free]
        g_s = d * g[free]
        # Newton model where the Hessian is positive definite on the free
        # variables, else the Gauss-Newton model (damping alone does not bound a
        # step along negative curvature, which can jump to another basin).
        H = hess
        H_s = d[:, None] * hess[np.ix_(free, free)] * d[None, :]
        try:
            scipy.linalg.cho_factor(H_s)  # ValueError if not finite
        except (np.linalg.LinAlgError, ValueError):
            H = gn
            H_s = d[:, None] * gn[np.ix_(free, free)] * d[None, :]
        while True:
            if nfev >= max_nfev:
                return x, cost, 0, nfev
            try:
                factor = scipy.linalg.cho_factor(H_s + lam * np.eye(d.size))
            except np.linalg.LinAlgError:
                lam, nu = lam * nu, nu * 2.0
                continue
            p = np.zeros_like(x)
            p[free] = -d * scipy.linalg.cho_solve(factor, g_s)
            x_new = np.clip(x + p, lb, ub)
            step = x_new - x
            predicted = -float(g @ step + 0.5 * step @ H @ step)
            cost_new = problem.cost(x_new, loss)
            nfev += 1
            actual = cost - cost_new
            ratio = float(actual / predicted) if predicted > 0 else 0.0
            # Relative to the full objective: the compressed problem's cost
            # excludes cost_offset.
            ftol_ok = actual < ftol * (cost + problem.cost_offset) and ratio > 0.25
            xtol_ok = np.linalg.norm(step) < XTOL * (XTOL + np.linalg.norm(x))
            accepted = ratio > 0
            if accepted:
                x = x_new
                lam *= max(1.0 / 3.0, 1.0 - (2.0 * ratio - 1.0) ** 3)
                nu = 2.0
            else:
                lam, nu = lam * nu, nu * 2.0
            if ftol_ok or xtol_ok:
                status = 4 if ftol_ok and xtol_ok else (2 if ftol_ok else 3)
                return x, cost_new if accepted else cost, status, nfev
            if accepted:
                break
        cost, g, gn, hess = problem.cost_grad_hess(x, loss)
        nfev += 1


def solve_centers_widths(
    problem: "ResidualProblem | ProjectedProblem",
    x0: Array,
    bounds: tuple[Array, Array],
    method: str = "bounded-lm",
    loss: str = "soft_l1",
    x_scale: str | Array = "jac",
    max_nfev: int | None = None,
) -> tuple[Array, float]:
    """Bounded nonlinear least squares for ``x = [centers.ravel(), widths]``.

    ``method='bounded-lm'`` runs `bounded_lm`; ``'trf'`` and ``'dogbox'`` run
    ``scipy.optimize.least_squares`` with the matrix-free Jacobian and LSMR. Both
    minimize the same objective. ``x_scale="jac"`` uses
    ``1 / problem.column_norms(x0)`` (1 where a norm is 0), since scipy cannot
    compute it for a matrix-free Jacobian. Returns the solution and its cost plus
    ``problem.cost_offset``.
    """
    if method not in NLSS_METHODS:
        raise ValueError(f"method must be one of {NLSS_METHODS}, got {method!r}")
    if (method == "varpro") != isinstance(problem, ProjectedProblem):
        raise ValueError("method 'varpro' takes a ProjectedProblem and only it")
    if isinstance(x_scale, str):
        if x_scale != "jac":
            raise ValueError(f"x_scale must be 'jac' or an array, got {x_scale!r}")
        norms = problem.column_norms(x0)
        x_scale = np.divide(1.0, norms, out=np.ones_like(norms), where=norms > 0)
    if method in ("bounded-lm", "varpro"):
        x, cost, _, _ = bounded_lm(
            problem,
            x0,
            bounds,
            loss,
            np.asarray(x_scale, dtype=np.float64),
            max_nfev,
        )
        return x, cost + problem.cost_offset
    assert isinstance(problem, ResidualProblem)
    result = least_squares(
        problem.residual,
        x0,
        jac=problem.jacobian,
        bounds=bounds,
        method=method,
        loss=loss,
        x_scale=x_scale,
        tr_solver="lsmr",
        tr_options={"atol": LSMR_TOL, "btol": LSMR_TOL},
        max_nfev=max_nfev,
    )
    return np.asarray(result.x, dtype=np.float64), float(
        result.cost
    ) + problem.cost_offset
