"""Numerical kernels for TFA: factors, weights, residual and analytic Jacobian.

Conventions: data ``X`` is (V, T), coordinates ``R`` are (V, D) with D in {2, 3},
centers are (K, D) and widths ``w = 2 sigma**2`` are (K,). The parameter vector of
the center/width solve is ``x = [centers.ravel(), widths]`` (length K * (D + 1)).
All arrays are float64.
"""

from dataclasses import dataclass

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares
from scipy.sparse.linalg import LinearOperator
from sklearn.cluster import KMeans

from htfa._engines import Engine

Array = NDArray[np.float64]

# Relative tolerance (atol = btol) of the LSMR trust-region subproblem solve. scipy's
# default (1e-6) stops the outer solve ~1e-6 away from the minimizer: compressed vs.
# uncompressed linear-loss solves then differed by 1.5e-6 in x. Measured with
# 1e-8: that difference falls to 2e-8, and on V=3000, T=200, K=20 the final cost
# matches 1e-10 to 4e-11 relative, at ~1.7x the time of 1e-6 (0.82 s vs 0.46 s).
LSMR_TOL = 1e-8


def sq_distances(R: Array, centers: Array) -> Array:
    """Squared distances (V, K) between voxels ``R`` (V, D) and ``centers`` (K, D).

    Summed per dimension rather than via ``|r|^2 - 2 r.c + |c|^2`` so that large
    coordinates do not cancel catastrophically.
    """
    d = np.zeros((R.shape[0], centers.shape[0]))
    for j in range(R.shape[1]):
        diff = R[:, j, None] - centers[None, :, j]
        d += diff * diff
    return d


def factors(R: Array, centers: Array, widths: Array) -> Array:
    """Radial basis factors ``F[v, k] = exp(-||r_v - c_k||^2 / w_k)``, shape (V, K)."""
    return np.exp(-sq_distances(R, centers) / widths[None, :])


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
    ) -> None:
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


def solve_centers_widths(
    problem: ResidualProblem,
    x0: Array,
    bounds: tuple[Array, Array],
    method: str = "trf",
    loss: str = "soft_l1",
    x_scale: str | Array = "jac",
) -> tuple[Array, float]:
    """Bounded nonlinear least squares for ``x = [centers.ravel(), widths]``.

    ``x_scale="jac"`` uses ``1 / problem.column_norms(x0)`` (1 where a norm is 0),
    since scipy cannot compute it for a matrix-free Jacobian. Returns the solution
    and its cost plus ``problem.cost_offset``.
    """
    if isinstance(x_scale, str):
        if x_scale != "jac":
            raise ValueError(f"x_scale must be 'jac' or an array, got {x_scale!r}")
        norms = problem.column_norms(x0)
        x_scale = np.divide(1.0, norms, out=np.ones_like(norms), where=norms > 0)
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
    )
    return np.asarray(result.x, dtype=np.float64), float(
        result.cost
    ) + problem.cost_offset
