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

from htfa._engines import EPS, Engine, loss_terms

Array = NDArray[np.float64]

# nlss_method values: the structured bounded Levenberg-Marquardt solver (default)
# and the scipy least_squares methods kept for cross-checking.
NLSS_METHODS = ("bounded-lm", "trf", "dogbox")
# scipy.optimize.least_squares' default tolerances, used by the structured solver
# with the same meaning (see `bounded_lm`).
FTOL = XTOL = GTOL = 1e-8
# Initial damping relative to the largest diagonal entry of the scaled Hessian
# (Madsen, Nielsen & Tingleff 2004, sec. 3.2).
LM_TAU = 1e-3

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
        total = self.engine.data_cost(factors(self.R, C, w), self.s, loss)
        if self.prior is not None:
            total += self._prior_terms(C, w, loss, derivatives=False)[0]
        return total

    def cost_grad_hess(
        self, x: ArrayLike, loss: str
    ) -> tuple[float, Array, Array, Array]:
        """Cost (as :meth:`cost`), gradient (P,), positive semidefinite
        Gauss-Newton matrix (P, P) and Hessian (P, P) of the objective.

        The Gauss-Newton matrix is the Hessian without the data term's
        second-order part ``sum_i rho'_i r_i Hess(r_i)`` (the prior terms keep
        their exact, floored Hessian, which is positive semidefinite).

        The data part of the Hessian is ``J^T diag(u) J + sum_i rho'_i r_i
        Hess(r_i)``, with scipy's loss weights ``u = max(rho' + 2 r**2 rho'', EPS)``
        (exact wherever ``rho' + 2 r**2 rho'' >= 0``: always for the linear,
        soft_l1 and huber losses). ``J^T diag(u) J`` is assembled from the
        per-factor structure ``dr[v, t] / dtheta[k, a] = -s G[k, v, a] W[k, t]``
        (`htfa._engines.Engine.gauss_newton`) instead of the V*T x P Jacobian; the
        second-order term is block diagonal per factor,
        ``-s sum_v M[v, k] d2F[v, k] / dtheta_k^2`` with ``M = (rho' r) W^T``, and
        costs O(V K (D+1)**2). The parameters of factor k are
        ``theta_k = (c_k1, ..., c_kD, w_k)``.
        """
        C, w = self._unpack(x)
        K, D, P = self.K, self.D, self.n_params
        V, s = self.R.shape[0], self.s
        delta = self.R[:, :, None] - C.T[None, :, :]  # (V, D, K)
        d = np.sum(delta * delta, axis=1)
        F = np.exp(-d / w)
        Fw = F / w
        G = np.empty((K, V, D + 1))
        for j in range(D):
            G[:, :, j] = (2.0 * Fw * delta[:, j]).T
        G[:, :, D] = (Fw * d / w).T
        cost, M, H_ak = self.engine.gauss_newton(F, G, s, loss)
        grad_ak = -s * np.einsum("kva,vk->ak", G, M)
        # Second derivatives of F = exp(-d / w) per factor, contracted with M.
        FM = F * M
        S = np.empty((K, D + 1, D + 1))
        for i in range(D):
            for j in range(i, D):
                term = 4.0 * delta[:, i] * delta[:, j] / w**2 - (2.0 / w) * (i == j)
                S[:, i, j] = S[:, j, i] = np.sum(FM * term, axis=0)
            term = 2.0 * delta[:, i] / w**2 * (d / w - 1.0)
            S[:, i, D] = S[:, D, i] = np.sum(FM * term, axis=0)
        S[:, D, D] = np.sum(FM * d / w**3 * (d / w - 2.0), axis=0)
        k = np.arange(K)
        # (a, k) -> position in x = [centers.ravel(), widths].
        pos = np.empty((D + 1, K), dtype=np.int64)
        pos[:D] = (k[None, :] * D) + np.arange(D)[:, None]
        pos[D] = K * D + k
        pos = pos.ravel()
        grad = np.empty(P)
        grad[pos] = grad_ak.ravel()
        gn = np.empty((P, P))
        gn[np.ix_(pos, pos)] = H_ak.reshape(P, P)
        H_ak[:] = 0.0
        H_ak[:, k, :, k] = -s * S
        second = np.empty((P, P))
        second[np.ix_(pos, pos)] = H_ak.reshape(P, P)
        if self.prior is not None:
            c_p, g_p, H_p = self._prior_terms(C, w, loss, derivatives=True)
            cost, grad, gn = cost + c_p, grad + g_p, gn + H_p
        # Exactly symmetric (the two triangles are summed in different orders).
        gn = 0.5 * (gn + gn.T)
        return cost, grad, gn, gn + second

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


def bounded_lm(
    problem: ResidualProblem,
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

    Stopping rules, with scipy's meaning and defaults: ``ftol`` (an accepted step
    with ratio > 0.25 lowered the cost by less than ``FTOL`` times the cost
    including ``problem.cost_offset``), ``xtol``
    (a trial step shorter than ``XTOL * (XTOL + ||x||)``), ``gtol`` (largest
    gradient entry over the free variables below ``GTOL``) and ``max_nfev``
    (cost evaluations, default ``100 * P`` as for scipy's trf).

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
    cost, g, gn, hess = problem.cost_grad_hess(x, loss)
    nfev, nu = 1, 2.0
    lam = LM_TAU * max(float(np.max(scale**2 * np.diag(gn))), EPS)
    while True:
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
            scipy.linalg.cho_factor(H_s)
        except np.linalg.LinAlgError:
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
            ftol_ok = actual < FTOL * (cost + problem.cost_offset) and ratio > 0.25
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
    problem: ResidualProblem,
    x0: Array,
    bounds: tuple[Array, Array],
    method: str = "bounded-lm",
    loss: str = "soft_l1",
    x_scale: str | Array = "jac",
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
    if isinstance(x_scale, str):
        if x_scale != "jac":
            raise ValueError(f"x_scale must be 'jac' or an array, got {x_scale!r}")
        norms = problem.column_norms(x0)
        x_scale = np.divide(1.0, norms, out=np.ones_like(norms), where=norms > 0)
    if method == "bounded-lm":
        x, cost, _, _ = bounded_lm(
            problem, x0, bounds, loss, np.asarray(x_scale, dtype=np.float64)
        )
        return x, cost + problem.cost_offset
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
