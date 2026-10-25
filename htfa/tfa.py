"""Single-subject Topographic Factor Analysis (TFA)."""

import dataclasses
import warnings
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from htfa._engines import LOSS_NAMES, loss_terms, select_engine
from htfa._kernels import (
    NLSS_METHODS,
    Prior,
    ResidualProblem,
    bounds,
    factors,
    init_centers_widths,
    max_sigma,
    ridge_weights,
    solve_centers_widths,
)
from htfa._matching import match_factors

Array = NDArray[np.float64]

INIT_METHODS = ("ncorr", "power", "kmeans")
# Per-voxel power sum_t X**2 this close to uniform (coefficient of variation, before
# median subtraction) makes 'power' weights rounding noise -- always the case for
# z-scored rows -- so 'power' is skipped when 'kmeans' runs too.
UNIFORM_CV = 1e-3
# Single-sample data on a raw scale: fits collapse factors onto the upper width
# bound because the ridge penalty is var(X) in the data's units (BrainIAK's) and
# the model has no intercept. Measured (make_tfa_data 35x53, K=6, T=1): var 0.05
# recovered centers within 1.0; the same data x50 (var 132) missed by 12.2, and
# shifted by +3 sd (|mean| = 3 std) missed by 20; a 47x60 photo (mean 146,
# var 5948) put every width within 2% of the bound. Unit-variance (z-scored)
# data sit an order of magnitude below RAW_VAR_LIMIT.
RAW_VAR_LIMIT = 10.0
# With subsampling, stop when the scaled change has not reached a new minimum for
# this many consecutive iterations (a fresh sample keeps it above the threshold).
PLATEAU_PATIENCE = 5
# Relative floor added to the K-means sample weights so that every voxel counts.
TINY_WEIGHT = 1e-12


def standardize_data(X: ArrayLike, mode: str | bool) -> Array:
    """Z-score data (V, T) before fitting.

    Parameters
    ----------
    X : array_like, shape (V, T)
        Data, one row per voxel and one column per sample.
    mode : {'auto', True, False}
        ``'auto'`` z-scores each row (voxel) across columns when T > 1 and leaves
        single-sample data unchanged. ``True`` z-scores each row when T > 1 and
        z-scores across all voxels when T == 1. ``False`` leaves the data unchanged.
        Constant rows (or a constant single column under ``True``) become 0.

    Returns
    -------
    ndarray, shape (V, T)
        The standardized data as float64.

    Raises
    ------
    ValueError
        If ``mode`` is not ``'auto'``, ``True`` or ``False``.

    Examples
    --------
    >>> import numpy as np
    >>> from htfa.tfa import standardize_data
    >>> standardize_data(np.array([[1.0, 3.0], [2.0, 2.0]]), "auto")
    array([[-1.,  1.],
           [ 0.,  0.]])
    >>> standardize_data(np.array([[1.0], [3.0]]), True)
    array([[-1.],
           [ 1.]])
    """
    is_bool = isinstance(mode, (bool, np.bool_))
    if not is_bool and mode != "auto":
        raise ValueError(f"standardize must be 'auto', True or False, got {mode!r}")
    X = np.asarray(X, dtype=np.float64)
    if is_bool and not mode:
        return X
    if X.shape[1] > 1:
        axis = 1
    elif is_bool:
        axis = 0
    else:
        return X
    mean = X.mean(axis=axis, keepdims=True)
    std = X.std(axis=axis, keepdims=True)
    # Exactly constant slices have a rounding-only std; zero them explicitly.
    constant = np.ptp(X, axis=axis, keepdims=True) == 0
    std[constant] = 1.0
    Z: Array = (X - mean) / std
    Z[np.broadcast_to(constant, Z.shape)] = 0.0
    return Z


def _warn_raw_single_sample(X: Array) -> None:
    """UserWarning when single-sample data (V, 1) are far from unit scale."""
    mean, std = float(np.mean(X)), float(np.std(X))
    if std * std > RAW_VAR_LIMIT or abs(mean) > std:
        warnings.warn(
            f"single-sample data with mean {mean:.4g} and variance {std * std:.4g} "
            "are fitted unstandardized; on such a scale the factors tend to "
            "collapse onto the upper width bound. Pass standardize=True to "
            "z-score them across voxels.",
            UserWarning,
            stacklevel=3,
        )


def _sample_size(n: int, ratio: float, cap: int | None) -> int:
    """BrainIAK's subsample size, clipped to the ``n`` items available."""
    return min(n, cap if cap is not None else n, max(1, int(ratio * n)))


def _check_coords(coords: ArrayLike, V: int | None, D: int | None) -> Array:
    """Validate coordinates (V, D): 2D, matching shape, D in {2, 3}, finite."""
    R = np.asarray(coords, dtype=np.float64)
    if R.ndim != 2 or (V is not None and R.shape[0] != V):
        raise ValueError(
            f"coords must have shape ({V if V is not None else 'V'}, D), got {R.shape}"
        )
    if R.shape[1] not in (2, 3):
        raise ValueError(f"coords must have 2 or 3 dimensions, got D={R.shape[1]}")
    if D is not None and R.shape[1] != D:
        raise ValueError(f"coords must have shape (V, {D}), got {R.shape}")
    if not np.all(np.isfinite(R)):
        raise ValueError("coords contains NaN or Inf")
    return R


def _init_names(init: str | Sequence[str]) -> tuple[str, ...]:
    """Candidate start names for the ``init`` parameter."""
    if isinstance(init, str):
        names: tuple[str, ...] = INIT_METHODS if init == "auto" else (init,)
    elif isinstance(init, (list, tuple)):
        names = tuple(init)
    else:
        names = ()
    if (
        not names
        or len(set(names)) != len(names)
        or any(n not in INIT_METHODS for n in names)
    ):
        raise ValueError(
            f"init must be 'auto', one of {INIT_METHODS} or a sequence of distinct "
            f"names from it, got {init!r}"
        )
    return names


def _floor(p: Array) -> Array:
    return p + TINY_WEIGHT * (float(np.max(p)) or 1.0)


def _power_weights(power: Array) -> Array:
    """Voxel weights ``clip(power - median, 0)`` plus a tiny floor."""
    return _floor(np.clip(power - np.median(power), 0.0, None))


def _ncorr_weights(X: Array, R: Array) -> Array:
    """Squared positive mean correlation of each voxel with its 2D nearest
    spatial neighbours, plus a tiny floor (needs T > 1)."""
    V, D = R.shape
    Z = X - X.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(Z, axis=1, keepdims=True)
    Z = np.divide(Z, norm, out=np.zeros_like(Z), where=norm > 0)
    _, idx = cKDTree(R).query(R, k=min(2 * D + 1, V))
    corr = np.zeros(V)
    for j in range(1, idx.shape[1]):
        corr += np.sum(Z * Z[idx[:, j]], axis=1)
    corr /= idx.shape[1] - 1
    return _floor(np.clip(corr, 0.0, None) ** 2)


def _weighted_start(
    R: Array, K: int, seed: int, weights: Array, w_lo: float, w_hi: float
) -> tuple[Array, Array]:
    """Weighted K-means++ centers and widths ``(4/D)`` times each cluster's
    weighted mean squared distance to its center, clipped to the width bounds."""
    km = KMeans(n_clusters=K, init="k-means++", n_init=10, random_state=seed)
    # Multithreaded weighted K-means sums in a varying order (results differed by
    # ~2e-15 between identical runs); one OpenMP thread makes it reproducible.
    with threadpool_limits(limits=1, user_api="openmp"):
        km.fit(R, sample_weight=weights)
    C = np.asarray(km.cluster_centers_, dtype=np.float64)
    w = np.empty(K)
    for k in range(K):
        m = km.labels_ == k
        d2 = np.sum((R[m] - C[k]) ** 2, axis=1)
        w[k] = 4.0 / R.shape[1] * np.average(d2, weights=weights[m])
    return C, np.clip(w, w_lo, w_hi)


def _initial_starts(
    X: Array,
    R: Array,
    K: int,
    names: tuple[str, ...],
    rng: np.random.Generator,
    w_lo: float,
    w_hi: float,
) -> dict[str, tuple[Array, Array]]:
    """Candidate starts ``name -> (centers, widths)``.

    One K-means seed is drawn per entry of `INIT_METHODS` in a fixed order, so a
    start is the same whether it runs alone or among others.
    """
    seeds = {name: int(rng.integers(np.iinfo(np.int32).max)) for name in INIT_METHODS}
    power = np.sum(X * X, axis=1)
    power_uniform = np.std(power) < UNIFORM_CV * np.mean(power)
    jobs: dict[str, Callable[[], tuple[Array, Array]]] = {}
    for name in names:
        if name == "kmeans":
            jobs[name] = partial(init_centers_widths, R, K, seeds[name])
            continue
        if name == "ncorr" and X.shape[1] > 1:
            weights = _ncorr_weights(X, R)
        elif name == "ncorr" and "power" in names:
            continue  # T == 1: 'ncorr' falls back to 'power', which runs anyway
        elif "kmeans" in names and power_uniform:
            continue
        else:
            weights = _power_weights(power)
        jobs[name] = partial(_weighted_start, R, K, seeds[name], weights, w_lo, w_hi)
    # Each K-means runs on one OpenMP thread (reproducible sums); the candidates
    # run concurrently on threads instead (sklearn's K-means releases the GIL).
    with threadpool_limits(limits=1, user_api="openmp"):
        with ThreadPoolExecutor(max(1, len(jobs))) as pool:
            futures = {name: pool.submit(job) for name, job in jobs.items()}
            return {name: future.result() for name, future in futures.items()}


class TFA(BaseEstimator):
    """Topographic Factor Analysis of one subject's data.

    Models data ``X`` (V, T) as ``F @ W``, where ``F[v, k] =
    exp(-||r_v - c_k||**2 / w_k)`` are radial basis factors with centers ``c_k``
    and widths ``w_k = 2 sigma_k**2`` (BrainIAK's convention). The fit alternates
    ridge-regression weights with a bounded nonlinear least-squares solve for the
    centers and widths, from one or more candidate starts (see ``init``), and keeps
    the candidate with the lowest final objective.

    Parameters
    ----------
    K : int, default=50
        Number of factors; must be smaller than the number of voxels.
    max_iter : int, default=100
        Maximum number of outer iterations per candidate start. ``0`` evaluates
        the start itself (no iterations). BrainIAK's default of 10 together with
        its 1.0 threshold stops after about 3 iterations with roughly 10% width
        error, so both defaults depart from BrainIAK.
    threshold : float, default=1e-3
        Convergence threshold on the scaled change between iterations,
        ``max(max|dc| / L, max|dw| / w)`` with ``L = sqrt(max_sigma(R) / 2)``.
        BrainIAK instead thresholds the raw largest change at 1.0. With subsampling
        (any ratio < 1 or a cap below the data size) the fresh sample of every
        iteration keeps the change from settling, so the fit also stops when the
        change has not reached a new minimum for 5 consecutive iterations
        ("plateau").
    nlss_method : {'bounded-lm', 'trf', 'dogbox'}, default='bounded-lm'
        Center/width solver. ``'bounded-lm'`` is a bounded Levenberg-Marquardt
        method on the explicit K (D + 1)-square Gauss-Newton system, assembled
        from the factor structure of the Jacobian (`htfa._kernels.bounded_lm`).
        ``'trf'`` and ``'dogbox'`` run ``scipy.optimize.least_squares`` with a
        matrix-free Jacobian (BrainIAK's solver; much slower). All minimize the
        same objective with scipy's default tolerances.
    nlss_loss : {'soft_l1', 'linear', 'huber', 'cauchy', 'arctan'}, default='soft_l1'
        Robust loss, as in ``least_squares``. ``'linear'`` enables an exact QR
        compression of the data term.
    x_scale : 'jac' or ndarray, default='jac'
        Variable scaling (``least_squares``' ``x_scale``; the damping scale of
        ``'bounded-lm'``); ``'jac'`` uses the inverse Jacobian column norms at the
        start of each solve.
    weight_method : {'rr', 'ols'}, default='rr'
        Ridge regression (penalty ``var(X)``) or ordinary least squares weights.
    upper_ratio, lower_ratio : float, default=1.8, 0.02
        Widths are bounded to ``[lower_ratio, upper_ratio] * 2 max(std(R))**2``.
    voxel_ratio, tr_ratio : float, default=1.0
        Fraction of voxels and samples drawn (anew each iteration) for the
        center/width solve. 1.0 uses all data.
    max_voxels, max_trs : int, optional
        Caps on the subsample sizes; never more than the data has.
    standardize : {'auto', True, False}, default='auto'
        Applied to ``X`` before anything else; see `standardize_data`. With
        ``False`` the raw values are fitted, and because the data residual is
        divided by ``sqrt(2) std(X)``, an HTFA prior is then relatively stronger
        on high-variance data.
    init : {'auto', 'ncorr', 'power', 'kmeans'} or sequence of those, default='auto'
        Candidate starts; each is fitted on identical subsamples and the one with
        the lowest full-data objective wins (after ``n_screen_iter`` iterations;
        see there). ``'kmeans'`` is BrainIAK's start
        (K-means++ on the coordinates, widths ``max_sigma(R)``). ``'power'`` and
        ``'ncorr'`` run K-means++ weighted per voxel by
        ``clip(sum_t X**2 - median, 0)`` and by the squared positive mean
        correlation with the ``2 D`` nearest spatial neighbours (``'power'`` when
        T == 1), and start each width at ``4 / D`` times its cluster's weighted
        mean squared distance to the center. Weights use the standardized data.
        ``'auto'`` is ``('ncorr', 'power', 'kmeans')``; ``'power'`` is skipped
        when ``sum_t X**2`` is uniform across voxels, as after per-voxel
        z-scoring (it would repeat ``'kmeans'``). Each start gets its own fixed
        K-means seed, so it is identical alone or within ``'auto'``. Ignored when
        a ``prior`` is passed to `fit`.
    n_screen_iter : int, default=0
        With several candidate starts and ``n_screen_iter > 0``, run each start
        for this many iterations, keep the one with the lowest full-data
        objective and continue only that one to convergence (about 1/3 to 2/3 of
        the time of full fits). ``0`` fits every candidate to convergence and
        compares final objectives. Measured with 3 and 5 (`make_tfa_data` 3D
        20^3, K=8 and 2D 40^2, K=6, T=100, seeds 0-5): on raw data screening
        kept the full fits' winner every time, with identical final centers; on
        per-voxel z-scored data it picked a different start in 6 of 12 cases,
        where two starts had reached the same optimum (objectives within 1e-5
        relative, matched centers within 0.03).
    backend : {'auto', 'numpy', 'cpu', 'torch', 'cuda', 'mps'}, default='auto'
        Compute engine for the data-term products; see
        `htfa._engines.select_engine`.
    random_state : None, int or numpy.random.Generator, default=None
        Seeds the K-means initializations and the subsampling.
    verbose : bool, default=False
        Print the scaled change of every iteration.

    Attributes
    ----------
    centers_ : ndarray, shape (K, D)
        Factor centers.
    widths_ : ndarray, shape (K,)
        Factor widths ``2 sigma**2``.
    weights_ : ndarray, shape (K, T)
        Weights fitted on the full (standardized) data.
    n_iter_ : int
        Number of outer iterations run for the winning start.
    converged_ : bool
        Whether the fit stopped before ``max_iter`` (``converged_reason_`` is not
        ``'max_iter'``).
    converged_reason_ : {'threshold', 'plateau', 'max_iter'}
        Why the winning start stopped: the scaled change fell to ``threshold``,
        it plateaued under subsampling, or ``max_iter`` was reached.
    cost_ : float
        Final objective of the winning start: ``0.5 * sum(rho(r**2))`` over the
        full-data residual ``r`` (with the prior penalties when a prior is given).
    init_ : str
        Name of the winning start (``'prior'`` when a prior was given).
    init_objectives_ : dict of str to float
        Full-data objective of every candidate start that was run: final for the
        winner, and after ``n_screen_iter`` iterations for the others (final for
        all with ``n_screen_iter=0`` or a single start).
    engine_ : str
        Name of the compute engine used.
    coords_ : ndarray, shape (V, D)
        Coordinates the model was fitted on.

    Warns
    -----
    ConvergenceWarning
        When the winning start reaches ``max_iter`` (>= 1) without stopping on the
        threshold or a plateau.
    UserWarning
        When single-sample data (T == 1) are fitted unstandardized
        (``standardize`` 'auto' or False) with variance above 10 or ``|mean|``
        above their standard deviation; ``standardize=True`` z-scores them.

    Examples
    --------
    >>> from htfa.synthetic import make_tfa_data
    >>> from htfa.tfa import TFA
    >>> X, R, truth = make_tfa_data((30, 30), 20, K=2, random_state=0)
    >>> model = TFA(K=2, standardize=False, random_state=0).fit(X, R)
    >>> model.centers_.shape, model.weights_.shape
    ((2, 2), (2, 20))
    >>> model.reconstruct().shape
    (900, 20)
    """

    def __init__(
        self,
        K: int = 50,
        max_iter: int = 100,
        threshold: float = 1e-3,
        nlss_method: str = "bounded-lm",
        nlss_loss: str = "soft_l1",
        x_scale: Any = "jac",
        weight_method: str = "rr",
        upper_ratio: float = 1.8,
        lower_ratio: float = 0.02,
        voxel_ratio: float = 1.0,
        tr_ratio: float = 1.0,
        max_voxels: int | None = None,
        max_trs: int | None = None,
        standardize: str | bool = "auto",
        init: str | Sequence[str] = "auto",
        n_screen_iter: int = 0,
        backend: str = "auto",
        random_state: int | np.random.Generator | None = None,
        verbose: bool = False,
    ) -> None:
        self.K = K
        self.max_iter = max_iter
        self.threshold = threshold
        self.nlss_method = nlss_method
        self.nlss_loss = nlss_loss
        self.x_scale = x_scale
        self.weight_method = weight_method
        self.upper_ratio = upper_ratio
        self.lower_ratio = lower_ratio
        self.voxel_ratio = voxel_ratio
        self.tr_ratio = tr_ratio
        self.max_voxels = max_voxels
        self.max_trs = max_trs
        self.standardize = standardize
        self.init = init
        self.n_screen_iter = n_screen_iter
        self.backend = backend
        self.random_state = random_state
        self.verbose = verbose

    def _validate(self, X: ArrayLike, coords: ArrayLike) -> tuple[Array, Array]:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (voxels, samples), got shape {X.shape}")
        R = _check_coords(coords, X.shape[0], None)
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or Inf")
        K, V = self.K, X.shape[0]
        if not isinstance(K, (int, np.integer)) or K < 1:
            raise ValueError(f"K must be a positive integer, got {K!r}")
        if K >= V:
            raise ValueError(f"K={K} must be smaller than the number of voxels ({V})")
        if self.max_iter < 0:
            raise ValueError(f"max_iter must be >= 0, got {self.max_iter}")
        if not isinstance(self.n_screen_iter, (int, np.integer)) or (
            self.n_screen_iter < 0
        ):
            raise ValueError(
                f"n_screen_iter must be an integer >= 0, got {self.n_screen_iter!r}"
            )
        if self.weight_method not in ("rr", "ols"):
            raise ValueError(
                f"weight method must be 'rr' or 'ols', got {self.weight_method!r}"
            )
        if self.nlss_loss not in LOSS_NAMES:
            raise ValueError(
                f"nlss_loss must be one of {LOSS_NAMES}, got {self.nlss_loss!r}"
            )
        if self.nlss_method not in NLSS_METHODS:
            raise ValueError(
                f"nlss_method must be one of {NLSS_METHODS}, got {self.nlss_method!r}"
            )
        return X, R

    def _objective(
        self, X: Array, R: Array, x: Array, engine: Any, prior: Prior | None
    ) -> tuple[float, Array]:
        """Full-data objective at ``x`` and the full-data weights it uses."""
        K, D = self.K, R.shape[1]
        F = factors(R, x[: K * D].reshape(K, D), x[K * D :])
        W = ridge_weights(X, F, self.weight_method)
        r = ResidualProblem(X, W, R, engine, prior).residual(x)
        return 0.5 * float(np.sum(loss_terms(self.nlss_loss, r * r)[0])), W

    def _iterate(
        self,
        X: Array,
        R: Array,
        x: Array,
        rng: np.random.Generator,
        engine: Any,
        prior: Prior | None,
        bnds: tuple[Array, Array],
        sizes: tuple[int, int],
        label: str,
    ) -> Iterator[tuple[Array, int, str | None]]:
        """Outer loop from ``x``, one iteration per step.

        Yields ``(x, n_iter, reason)`` after every iteration; ``reason`` is
        ``None`` until the last one, which carries the stop reason.
        """
        (V, T), (Vs, Ts) = X.shape, sizes
        K, D = self.K, R.shape[1]
        length = np.sqrt(max_sigma(R) / 2.0)
        subsampled = Vs < V or Ts < T
        best, stale = np.inf, 0
        for n_iter in range(1, self.max_iter + 1):
            vi = np.sort(rng.choice(V, Vs, replace=False)) if Vs < V else slice(None)
            ti = np.sort(rng.choice(T, Ts, replace=False)) if Ts < T else slice(None)
            X_sub, R_sub = X[vi][:, ti], R[vi]
            C, w = x[: K * D].reshape(K, D), x[K * D :]
            W = ridge_weights(X_sub, factors(R_sub, C, w), self.weight_method)
            problem = ResidualProblem(
                X_sub, W, R_sub, engine, prior, compress=self.nlss_loss == "linear"
            )
            x_new, _ = solve_centers_widths(
                problem,
                x,
                bnds,
                method=self.nlss_method,
                loss=self.nlss_loss,
                x_scale=self.x_scale,
            )
            C_new = x_new[: K * D].reshape(K, D)
            perm = match_factors(C, C_new)
            C_new, w_new = C_new[perm], x_new[K * D :][perm]
            change = max(
                float(np.max(np.abs(C_new - C))) / length,
                float(np.max(np.abs(w_new - w) / w)),
            )
            x = np.concatenate([C_new.ravel(), w_new])
            if self.verbose:
                print(f"TFA {label} iteration {n_iter}: scaled change {change:.6g}")
            if change <= self.threshold:
                yield x, n_iter, "threshold"
                return
            best, stale = (change, 0) if change < best else (best, stale + 1)
            if subsampled and stale >= PLATEAU_PATIENCE:
                yield x, n_iter, "plateau"
                return
            yield x, n_iter, "max_iter" if n_iter == self.max_iter else None

    def fit(
        self,
        X: ArrayLike,
        coords: ArrayLike,
        prior: Prior | None = None,
        start: tuple[ArrayLike, ArrayLike] | None = None,
    ) -> "TFA":
        """Fit centers, widths and weights.

        Parameters
        ----------
        X : array_like, shape (V, T)
            Data, one row per voxel. Any float dtype or memory layout; computed in
            float64 after `standardize_data`.
        coords : array_like, shape (V, D)
            Voxel coordinates, D in {2, 3}.
        prior : Prior, optional
            HTFA template. When given, the fit starts from ``prior.centers`` and
            ``prior.widths`` (``init`` is ignored), and every solve is penalized
            towards the template with ``scale = 0.5 * (Vs * Ts) / (V * T)``
            (``prior.scale`` is ignored and the caller's prior is not modified).
        start : tuple of (centers, widths), optional
            Explicit starting point, shapes (K, D) and (K,). Overrides both
            ``init`` and the prior start (the prior penalty still applies).

        Returns
        -------
        TFA
            The fitted estimator.

        Raises
        ------
        ValueError
            On invalid shapes, NaN/Inf values, ``K >= V`` or invalid parameters.
        """
        X, R = self._validate(X, coords)
        names = _init_names(self.init)
        if X.shape[1] == 1 and (self.standardize == "auto" or not self.standardize):
            _warn_raw_single_sample(X)
        X = standardize_data(X, self.standardize)
        V, T = X.shape
        K, D = self.K, R.shape[1]
        if prior is not None and np.shape(prior.centers) != (K, D):
            raise ValueError(
                f"prior.centers must have shape ({K}, {D}), "
                f"got {np.shape(prior.centers)}"
            )
        rng = np.random.default_rng(self.random_state)
        sample_seed = int(rng.integers(np.iinfo(np.int32).max))
        sizes = (
            _sample_size(V, self.voxel_ratio, self.max_voxels),
            _sample_size(T, self.tr_ratio, self.max_trs),
        )
        engine = select_engine(self.backend, *sizes, K)
        lb, ub = bounds(R, K, self.lower_ratio, self.upper_ratio)
        if prior is None:
            fit_prior = full_prior = None
        else:
            fit_prior = dataclasses.replace(
                prior, scale=0.5 * (sizes[0] * sizes[1]) / (V * T)
            )
            full_prior = dataclasses.replace(prior, scale=0.5)
        if start is not None:
            if np.shape(start[0]) != (K, D) or np.shape(start[1]) != (K,):
                raise ValueError(
                    f"start must be (centers ({K}, {D}), widths ({K},)), got "
                    f"{np.shape(start[0])} and {np.shape(start[1])}"
                )
            starts: dict[str, tuple[Array, Array]] = {
                "start": (np.asarray(start[0]), np.asarray(start[1]))
            }
        elif prior is not None:
            starts = {"prior": (prior.centers, prior.widths)}
        else:
            starts = _initial_starts(X, R, K, names, rng, lb[-1], ub[-1])
        runs: dict[str, tuple[Array, int, str | None]] = {}
        loops = {}
        for name, (centers, widths) in starts.items():
            x0 = np.clip(np.concatenate([np.ravel(centers), np.ravel(widths)]), lb, ub)
            runs[name] = (x0, 0, None if self.max_iter > 0 else "max_iter")
            loops[name] = self._iterate(
                X,
                R,
                x0,
                np.random.default_rng(sample_seed),
                engine,
                fit_prior,
                (lb, ub),
                sizes,
                name,
            )

        def advance(name: str, n_steps: int) -> None:
            for _ in range(n_steps):
                if runs[name][2] is not None:
                    return
                runs[name] = next(loops[name])

        def objective(name: str) -> tuple[float, Array]:
            return self._objective(X, R, runs[name][0], engine, full_prior)

        screen = self.n_screen_iter if len(starts) > 1 else 0
        for name in starts:
            advance(name, screen if screen > 0 else self.max_iter)
        scores = {name: objective(name) for name in starts}
        self.init_ = min(scores, key=lambda n: scores[n][0])
        if screen > 0:
            advance(self.init_, self.max_iter)
            scores[self.init_] = objective(self.init_)
        self.init_objectives_ = {n: score[0] for n, score in scores.items()}
        self.cost_, self.weights_ = scores[self.init_]
        x, self.n_iter_, reason = runs[self.init_]
        self.converged_reason_ = reason
        self.converged_ = reason != "max_iter"
        if self.max_iter > 0 and not self.converged_:
            warnings.warn(
                f"TFA did not converge in max_iter={self.max_iter} iterations "
                f"(init {self.init_!r}); increase max_iter or threshold",
                ConvergenceWarning,
                stacklevel=2,
            )
        self.centers_ = x[: K * D].reshape(K, D).copy()
        self.widths_ = x[K * D :].copy()
        self.coords_ = R
        self.engine_ = engine.name
        return self

    def factors(self, coords: ArrayLike | None = None) -> Array:
        """Factor images ``F`` (V, K) at ``coords`` (the fitted coords by default).

        Parameters
        ----------
        coords : array_like, shape (V, D), optional
            Coordinates to evaluate the fitted factors at.

        Returns
        -------
        ndarray, shape (V, K)

        Examples
        --------
        >>> import numpy as np
        >>> from htfa.synthetic import make_tfa_data
        >>> X, R, _ = make_tfa_data((12, 12), 10, K=2, random_state=0)
        >>> model = TFA(K=2, init="power", standardize=False, random_state=0)
        >>> model = model.fit(X, R)
        >>> model.factors().shape
        (144, 2)
        >>> model.factors(np.array([[0.0, 0.0], [5.5, 5.5]])).shape
        (2, 2)
        """
        check_is_fitted(self, "centers_")
        D = self.centers_.shape[1]
        R = self.coords_ if coords is None else _check_coords(coords, None, D)
        return factors(R, self.centers_, self.widths_)

    def reconstruct(self) -> Array:
        """Reconstruction ``F @ weights_`` (V, T) at the fitted coords.

        The result is in the standardized space the model was fitted in (see
        ``standardize``), not in the units of the raw input.

        Returns
        -------
        ndarray, shape (V, T)

        Examples
        --------
        >>> import numpy as np
        >>> from htfa.synthetic import make_tfa_data
        >>> X, R, _ = make_tfa_data((12, 12), 10, K=2, random_state=0)
        >>> model = TFA(K=2, init="power", standardize=False, random_state=0)
        >>> recon = model.fit(X, R).reconstruct()
        >>> recon.shape
        (144, 10)
        >>> bool(np.corrcoef(recon.ravel(), X.ravel())[0, 1] > 0.8)
        True
        """
        check_is_fitted(self, "weights_")
        result: Array = self.factors() @ self.weights_
        return result
