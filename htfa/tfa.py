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
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import check_is_fitted

from htfa._engines import LOSS_NAMES, loss_terms, select_engine
from htfa._kernels import (
    NLSS_METHODS,
    SOLVER_MODES,
    Kernel,
    Prior,
    ProjectedProblem,
    ResidualProblem,
    bounds,
    factors,
    init_centers_widths,
    kmeans,
    max_sigma,
    ridge_weights,
    solve_centers_widths,
    x_factors,
)
from htfa._matching import match_factors

from ._validation import real_array

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
# With subsampling, the scaled change has a noise floor (two subsample optima differ:
# 0.56 at caps 3000x300 and 0.31 at 10000x300 on real data, V~70k, T=1137, K=100), so
# the fit stops when the lowest objective so far has fallen by less than
# OBJECTIVE_RTOL per iteration over the last OBJECTIVE_WINDOW iterations. The window
# (and the running minimum) keeps one noisy upward step from stopping a fit that is
# still improving: on make_tfa_data (20, 20), K=4, voxel_ratio=0.5 the objective rose
# by 0.25% at iteration 6 after falling 1.5% at iteration 5. Measured (synthetic, real
# mask 70k x 1137, K=100, caps 3000x300): the one-iteration relative decrease fell
# from 1.5e-2 (iteration 1) to 4.9e-5-1.4e-4 at iterations 12-15; the rule stops at
# iteration 15, median center error 2.20 mm (2.16-2.32 mm from iteration 9 on, 2.18
# at 20).
OBJECTIVE_RTOL = 1e-4
OBJECTIVE_WINDOW = 3
# The objective rule never stops a fit before this many iterations; with a smaller
# max_iter the stopping objective is not evaluated at all.
OBJECTIVE_MIN_ITER = OBJECTIVE_WINDOW + 2
# The stopping objective is evaluated on one fixed random subsample per fit (so that
# its values are comparable across iterations): max(EVAL_VOXEL_FACTOR * Vs,
# EVAL_MIN_VOXELS) voxels (at most V) and all samples unless max_trs caps them.
# Measured on the pilot-shaped synthetic data (69585 x 1137, K=100, caps 3000x300,
# load ~5): with 6000 x 300, soft_l1 stopped at the same iteration (12) on the same
# iterate as with the full-data objective (one-step decreases within 1.2e-4); one
# evaluation took 0.05 s against 1.73 s on all data and ~0.4 s per iteration (12%
# overhead; 10000 x 300: 0.08 s; 20000 x 300: 0.16 s). Linear loss: stop at 14
# instead of 15, final full-data cost 5e-5 higher, median center error 2.17 vs
# 2.20 mm; 0.02 s per evaluation against ~0.06 s per iteration.
EVAL_VOXEL_FACTOR = 2
EVAL_MIN_VOXELS = 6000
# Two-stage loss: the linear-loss variable-projection fit ('varpro') until it
# stops, then the soft_l1 alternation from there.
TWO_STAGE_LOSS = "linear+soft_l1"
# Relative cost difference below which two candidate starts count as tied (they
# reached the same optimum; see `TFA.fit`). Measured: make_htfa_data(3, (12,)*3,
# 40, K=4, seed 1), template subject: 'ncorr' and 'kmeans' reached the same
# centers in a different factor order, and which cost was lower depended on the
# BLAS thread count (serial vs n_jobs=2).
START_TIE_RTOL = 1e-9
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
    X = real_array(X, "X")
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
    R = real_array(coords, "coords")
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
    km = kmeans(R, K, seed, sample_weight=weights)
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
    # The candidates run concurrently on threads (sklearn's K-means releases the
    # GIL), each K-means on one OpenMP thread (see `kmeans`).
    with ThreadPoolExecutor(max(1, len(jobs))) as pool:
        futures = {name: pool.submit(job) for name, job in jobs.items()}
        return {name: future.result() for name, future in futures.items()}


def _kw(kernel: Kernel) -> dict[str, Any]:
    """Kernel keyword arguments; single RBFs use the specialized default path."""
    return {} if kernel.plain_rbf else {"kernel": kernel}


def _x_factors(R: Array, C: Array, w: Array, K: int, kernel: Kernel) -> Array:
    """`factors` for one RBF component (see `_kw`), `x_factors` otherwise."""
    if kernel.plain_rbf:
        result: Array = factors(R, C, w)
        return result
    return x_factors(R, C, w, K, kernel, None)


def component_factors(
    R: Array,
    centers: ArrayLike,
    widths: ArrayLike,
    mixing_weights: ArrayLike,
    kernel: "str | Kernel" = "rbf",
) -> Array:
    """Factors of a fit with any kernel and number of components.

    Factor k is ``sum_m pi_km phi(||r - c_km||**2 / w_km)`` with the profile
    ``phi`` of ``kernel`` (see `TFA`).

    Parameters
    ----------
    R : ndarray, shape (V, D)
        Coordinates to evaluate the factors at.
    centers : array_like, shape (K, M, D)
        Component centers (``component_centers_`` of a fit).
    widths : array_like, shape (K, M)
        Component widths (``component_widths_``).
    mixing_weights : array_like, shape (K, M)
        Positive mixing weights summing to 1 per factor (``mixing_weights_``).
    kernel : {'rbf', 'laplace', 'mexican_hat'}, default='rbf'
        Profile of the components.

    Returns
    -------
    ndarray, shape (V, K)

    Examples
    --------
    >>> import numpy as np
    >>> from htfa.tfa import component_factors
    >>> R = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    >>> F = component_factors(R, [[[0.0, 0.0], [3.0, 0.0]]], [[1.0, 1.0]],
    ...                       [[0.75, 0.25]])
    >>> np.round(F[:, 0], 3)
    array([0.75, 0.28, 0.25])
    """
    C = np.asarray(centers, dtype=np.float64)
    K, M, D = C.shape
    if not isinstance(kernel, Kernel):
        kernel = Kernel(kernel, M)
    w = np.asarray(widths, dtype=np.float64).reshape(K * M)
    tail = np.concatenate([w, Kernel.logits(mixing_weights).ravel()])
    return x_factors(R, C.reshape(K * M, D), tail, K, kernel, None)


def _start_x(kernel: Kernel, start: tuple[ArrayLike, ...], K: int, D: int) -> Array:
    """``x`` of an explicit ``start``: (centers (K, D), widths (K,)) for one
    component, (centers (K, M, D), widths (K, M), mixing weights (K, M))
    for M > 1."""
    M = kernel.n_components
    shapes = [np.shape(a) for a in start]
    start = tuple(real_array(a, "start") for a in start)
    if M == 1:
        if len(start) != 2 or shapes != [(K, D), (K,)]:
            raise ValueError(
                f"start must be (centers ({K}, {D}), widths ({K},)), got {shapes}"
            )
        return np.concatenate([np.ravel(start[0]), np.ravel(start[1])]).astype(float)
    if len(start) != 3 or shapes != [(K, M, D), (K, M), (K, M)]:
        raise ValueError(
            f"start must be (centers ({K}, {M}, {D}), widths ({K}, {M}), "
            f"mixing_weights ({K}, {M})), got {shapes}"
        )
    return kernel.join(start[0], start[1], Kernel.logits(start[2]))


def _matched_step(
    kernel: Kernel, K: int, D: int, x: Array, x_new: Array, length: float
) -> tuple[Array, float]:
    """``x_new`` with its factors matched to those of ``x`` (Hungarian, on
    the component centers), and the scaled change ``max(max|dc| / length,
    max|dw| / w, max|da|)`` over components and logits."""
    M = kernel.n_components
    C, w, a = kernel.split(x, K, D)
    C_new, w_new, a_new = kernel.split(x_new, K, D)
    perm = match_factors(C.reshape(K, M * D), C_new.reshape(K, M * D))
    C_new = C_new.reshape(K, M, D)[perm].reshape(K * M, D)
    w_new = w_new.reshape(K, M)[perm].ravel()
    a_new = a_new[perm]
    change = max(
        float(np.max(np.abs(C_new - C))) / length,
        float(np.max(np.abs(w_new - w) / w)),
    )
    if a.size:
        change = max(change, float(np.max(np.abs(a_new - a))))
    return kernel.join(C_new, w_new, a_new), change


class TFA(BaseEstimator):
    """Topographic Factor Analysis of one subject's data.

    Models data ``X`` (V, T) as ``F @ W``, where ``F[v, k] =
    exp(-||r_v - c_k||**2 / w_k)`` are radial basis factors with centers ``c_k``
    and widths ``w_k = 2 sigma_k**2`` (BrainIAK's convention), or factors of
    another shape or mixtures of several components (``kernel``,
    ``n_components``). The default linear-loss variable-projection solver
    eliminates the temporal weights analytically while optimizing spatial
    parameters. Other spatial solvers alternate weight and spatial updates.
    The fit evaluates one or more candidate starts (see ``init``), keeping the
    candidate with the lowest final objective.

    On macOS, a standalone TFA fit and HTFA's stage-0 fit of the same subject
    can differ at rounding level, because HTFA runs subject fits on one BLAS
    thread and a standalone fit uses the caller's BLAS threads.

    Parameters
    ----------
    K : int, default=50
        Number of factors; must be smaller than the number of voxels.
    max_iter : int, default=100
        Maximum number of outer iterations per candidate start. ``0`` evaluates
        the start itself without spatial optimization; temporal weights are
        still fitted at that starting point.
    threshold : float, default=1e-3
        Convergence threshold on the scaled change between iterations,
        ``max(max|dc| / L, max|dw| / w)`` with ``L = sqrt(max_sigma(R) / 2)``.
        With subsampling (any ratio < 1 or a cap below the data size), fresh
        samples can produce different spatial optima, so successive changes
        need not settle below this threshold.

        With subsampling the fit also evaluates the objective after every
        iteration, on one fixed random subsample of ``max(2 Vs, 6000)`` voxels
        (``Vs`` the per-iteration voxel count, at most V) and all samples
        (``max_trs`` if set), and stops when its lowest value so far has fallen
        by less than a relative 1e-4 per iteration over the last 3 iterations,
        after at least 5 iterations ("objective"), or when the change has not
        reached a new minimum for 5 consecutive iterations ("plateau"); it then
        returns the iterate with the lowest such objective.
    nlss_method : {'auto', 'bounded-lm', 'varpro', 'trf', 'dogbox'}, default='auto'
        Center/width solver. ``'auto'`` is ``'varpro'`` for the linear loss (and
        the first stage of ``'linear+soft_l1'``) and ``'bounded-lm'`` for the
        other losses. ``'bounded-lm'`` is a bounded Levenberg-Marquardt
        method on the explicit K (D + 1)-square Gauss-Newton system, assembled
        from the factor structure of the Jacobian (`htfa._kernels.bounded_lm`).
        ``'trf'`` and ``'dogbox'`` run ``scipy.optimize.least_squares`` with a
        matrix-free Jacobian. All minimize the
        same objective with scipy's default tolerances, for weights held fixed
        within an outer iteration. ``'varpro'`` (linear loss only) runs the
        same solver on the variable-projection objective, with the ridge
        weights eliminated (`htfa._kernels.ProjectedProblem`): its stationary
        points are the fixed points of the alternation. BrainIAK uses
        ``'trf'`` with ``'soft_l1'``.
    nlss_loss : {'linear', 'soft_l1', 'huber', 'cauchy', 'arctan', \
'linear+soft_l1'}, default='linear'
        Loss of the data residual, as in ``least_squares``. ``'linear'`` is
        least squares (with ``'bounded-lm'`` the data term is compressed
        exactly). ``'linear+soft_l1'`` fits in two stages: the linear loss until
        it stops, then ``'soft_l1'`` from there. With ``nlss_method`` ``'auto'``,
        ``'bounded-lm'`` or ``'varpro'`` the first stage uses ``'varpro'`` and
        the second ``'bounded-lm'``; with ``'trf'`` or ``'dogbox'`` both stages
        use that method. ``max_iter`` and ``n_iter_`` count both stages, and
        ``cost_``, ``converged_reason_`` and ``objective_history_`` are the
        second stage's; when the first stage ends at ``max_iter`` the second does
        not run and the fit ends on ``'max_iter'``.
        Robust losses reduce the influence of large residuals and can be useful
        for heavy-tailed noise.
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
        objective and continue only that candidate. ``0`` runs every candidate
        until its stopping rule or iteration limit and compares final objectives.
        Screening reduces work on discarded starts but can select a different
        candidate than fully fitting every start.
    backend : str, default='auto'
        Compute engine: ``'auto'``, ``'numpy'``, ``'cpu'``, ``'torch'``,
        ``'cuda'``, ``'mps'``, or ``'cuda:N'`` for a visible CUDA device index N.
        A single-subject fit uses one device. For data-term products, see
        `htfa._engines.select_engine`.
    mode : {'fast', 'exact'}, default='fast'
        How the ``'bounded-lm'`` and ``'varpro'`` solvers form their steps (no
        effect on ``'trf'`` and ``'dogbox'``; see `htfa._kernels.bounded_lm`).
        Both modes minimize the same objective, with float64 costs and
        gradients (on NumPy) and the same stopping rules. ``'exact'`` takes a
        Gauss-Newton step wherever the Hessian is not positive definite on the
        free variables and assembles the curvature in float64; it is the
        reference behavior, in which the equivalence tests and the BrainIAK
        cross-checks run. ``'fast'`` keeps the Hessian there and raises the
        damping instead, and forms the largest curvature products of the
        linear loss in float32. Different step models and floating-point
        precision can lead to different local minima. Runtime and reconstruction
        quality depend on the workload; see the Performance and Validation
        guides for controlled comparisons and interpretation.
    random_state : None, int or numpy.random.Generator, default=None
        Seeds the K-means initializations and the subsampling.
    verbose : bool, default=False
        Print the scaled change of every iteration (and, with subsampling, the
        relative decrease of the stopping objective), flushed immediately.
    kernel : {'rbf', 'laplace', 'mexican_hat'}, default='rbf'
        Profile ``phi(z)``, ``z = ||r - c||**2 / w``, of each factor (see
        `htfa._kernels.Kernel`): ``'rbf'`` ``exp(-z)``; ``'laplace'``
        ``exp(eps - sqrt(z + eps**2))`` with ``eps = 0.1``, a sharp peak with
        exponential tails ``exp(-r / sqrt(w))``, rounded within about
        ``0.1 sqrt(w)`` of the center so that its derivatives exist;
        ``'mexican_hat'`` ``(1 - 2 z / D) exp(-z)`` in D dimensions, the
        negative Laplacian of a Gaussian: positive within ``r = sqrt(D w /
        2)``, negative outside, with integral 0. All are 1 at the center.
    n_components : int, default=1
        Components per factor. With ``n_components = M > 1`` each factor is
        ``sum_m pi_m phi(||r - c_m||**2 / w_m)``: M components of the kernel's
        shape, each with its own center and width, and mixing weights ``pi =
        softmax(0, a_2, ..., a_M)`` (unbounded logits ``a``)
        fitted in the nonlinear solve with them. The weights sum to 1 and the
        first logit is fixed, so the factor's amplitude stays in its
        time-series weights. Each factor starts with its components at its
        start center, widths ``0.25**m`` times its start width (m = 0, ...,
        M - 1) and equal mixing weights. ``x_scale``, the solvers and both
        modes apply unchanged; the scaled change (see ``threshold``) also
        covers the logits, unscaled.

    Attributes
    ----------
    centers_ : ndarray, shape (K, D)
        Factor centers (with ``n_components > 1``: the mixing-weighted mean
        of the component centers).
    widths_ : ndarray, shape (K,)
        Factor widths ``2 sigma**2`` (with ``n_components > 1``: the
        mixing-weighted mean of the component widths).
    component_centers_ : ndarray, shape (K, M, D)
        Centers of each factor's components (``centers_[:, None]`` for M = 1).
    component_widths_ : ndarray, shape (K, M)
        Widths of each factor's components.
    mixing_weights_ : ndarray, shape (K, M)
        Mixing weights of each factor's components (1 for M = 1).
    weights_ : ndarray, shape (K, T)
        Weights fitted on the full (standardized) data.
    n_iter_ : int
        Number of outer iterations run for the winning start.
    converged_ : bool
        Whether the fit stopped before ``max_iter`` (``converged_reason_`` is not
        ``'max_iter'``).
    converged_reason_ : {'threshold', 'objective', 'plateau', 'max_iter'}
        Why the winning start stopped: the scaled change fell to ``threshold``,
        the stopping objective stopped falling under subsampling (below a
        relative 1e-4 per iteration over 3 iterations), the change plateaued
        under subsampling, or ``max_iter`` was reached.
    cost_ : float
        Final objective of the winning start: ``0.5 * sum(rho(r**2))`` over the
        full-data residual ``r`` (with the prior penalties when a prior is given).
    objective_history_ : list of float
        With subsampling (and ``max_iter >= 5``), the stopping objective of the
        winning start at its start and after every iteration (empty otherwise). The
        returned centers and widths are those of its minimum, unless the fit
        stopped on the threshold (then the last iterate).
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
        threshold, the objective or a plateau. Under subsampling its message
        points to the noise floor and to ``max_voxels``/``max_trs``.
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
        nlss_method: str = "auto",
        nlss_loss: str = "linear",
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
        mode: str = "fast",
        random_state: int | np.random.Generator | None = None,
        verbose: bool = False,
        kernel: str = "rbf",
        n_components: int = 1,
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
        self.mode = mode
        self.random_state = random_state
        self.verbose = verbose
        self.kernel = kernel
        self.n_components = n_components

    def _kernel(self) -> Kernel:
        """The `htfa._kernels.Kernel` of ``kernel`` and ``n_components``."""
        return Kernel(self.kernel, self.n_components)

    def _validate(self, X: ArrayLike, coords: ArrayLike) -> tuple[Array, Array]:
        X = real_array(X, "X")
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
        if self.mode not in SOLVER_MODES:
            raise ValueError(f"mode must be one of {SOLVER_MODES}, got {self.mode!r}")
        self._kernel()  # validates kernel and n_components
        if self.weight_method not in ("rr", "ols"):
            raise ValueError(
                f"weight method must be 'rr' or 'ols', got {self.weight_method!r}"
            )
        if self.nlss_loss not in (*LOSS_NAMES, TWO_STAGE_LOSS):
            raise ValueError(
                f"nlss_loss must be one of {(*LOSS_NAMES, TWO_STAGE_LOSS)}, "
                f"got {self.nlss_loss!r}"
            )
        if self.nlss_method not in ("auto", *NLSS_METHODS):
            raise ValueError(
                f"nlss_method must be one of {('auto', *NLSS_METHODS)}, "
                f"got {self.nlss_method!r}"
            )
        if self.nlss_method == "varpro" and self.nlss_loss not in (
            "linear",
            TWO_STAGE_LOSS,
        ):
            raise ValueError(
                f"nlss_method='varpro' needs nlss_loss='linear', got {self.nlss_loss!r}"
            )
        return X, R

    def _stages(self) -> list[tuple[str, str]]:
        """``(loss, method)`` of each stage of the fit."""
        method = self.nlss_method
        if self.nlss_loss == TWO_STAGE_LOSS:
            first = "varpro" if method in ("auto", "bounded-lm", "varpro") else method
            second = "bounded-lm" if method in ("auto", "varpro") else method
            return [("linear", first), ("soft_l1", second)]
        if method == "auto":
            method = "varpro" if self.nlss_loss == "linear" else "bounded-lm"
        return [(self.nlss_loss, method)]

    def _objective(
        self,
        X: Array,
        R: Array,
        x: Array,
        engine: Any,
        prior: Prior | None,
        loss: str | None = None,
    ) -> tuple[float, Array]:
        """Full-data objective at ``x`` (for ``loss``, by default the final
        stage's) and the full-data weights it uses."""
        K, D = self.K, R.shape[1]
        kernel = self._kernel()
        n = K * kernel.n_components * D
        F = _x_factors(R, x[:n].reshape(-1, D), x[n:], K, kernel)
        W = ridge_weights(X, F, self.weight_method)
        r = ResidualProblem(X, W, R, engine, prior, **_kw(kernel)).residual(x)
        loss = self._stages()[-1][0] if loss is None else loss
        return 0.5 * float(np.sum(loss_terms(loss, r * r)[0])), W

    def _eval_sizes(self, V: int, T: int, sizes: tuple[int, int]) -> tuple[int, int]:
        """Voxels and samples of the fixed subsample for the stopping objective."""
        Ve = min(V, max(EVAL_VOXEL_FACTOR * sizes[0], EVAL_MIN_VOXELS))
        return Ve, T if self.max_trs is None else min(T, self.max_trs)

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
        objective: Callable[[str, Array], float],
        history: list[float],
    ) -> Iterator[tuple[Array, int, str | None]]:
        """Outer loop from ``x`` over the stages of `_stages`.

        A stage that stops for any reason but ``'max_iter'`` hands its result to
        the next one; iterations are counted across stages, and ``history`` and
        the yielded reasons are those of the last stage. See `_iterate_stage`.
        """
        stages = self._stages()
        n_done = 0
        for i, (loss, method) in enumerate(stages):
            last = i == len(stages) - 1
            stage = self._iterate_stage(
                X,
                R,
                x,
                rng,
                engine,
                prior,
                bnds,
                sizes,
                label if last else f"{label} ({loss})",
                partial(objective, loss),
                history if last else [],
                loss,
                method,
                n_done,
            )
            for x, n_done, reason in stage:
                if reason is None or last or reason == "max_iter":
                    yield x, n_done, reason
                if reason == "max_iter":
                    return
            if not last and n_done >= self.max_iter:
                # The stage stopped on its own rule at the last allowed iteration:
                # the next stage has none left.
                yield x, n_done, "max_iter"
                return

    def _iterate_stage(
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
        objective: Callable[[Array], float],
        history: list[float],
        loss: str,
        method: str,
        n_done: int,
    ) -> Iterator[tuple[Array, int, str | None]]:
        """Outer loop of one stage from ``x``, after ``n_done`` iterations.

        Yields ``(x, n_iter, reason)`` after every iteration; ``reason`` is
        ``None`` until the last one, which carries the stop reason. With
        subsampling, ``objective(x)`` is evaluated at the start and after every
        iteration (appended to ``history``) for the ``'objective'`` stopping
        rule, and the yielded ``x`` is the iterate with the lowest objective so
        far (the last one when the fit stops on the threshold).
        """
        (V, T), (Vs, Ts) = X.shape, sizes
        K, D = self.K, R.shape[1]
        kernel = self._kernel()
        n = K * kernel.n_components * D
        length = np.sqrt(max_sigma(R) / 2.0)
        subsampled = Vs < V or Ts < T
        best, stale = np.inf, 0
        # Running minimum of the objective, from the start on, and its iterate.
        if subsampled and self.max_iter - n_done >= OBJECTIVE_MIN_ITER:
            history.append(objective(x))
        lowest, x_best = list(history), x
        for n_iter in range(n_done + 1, self.max_iter + 1):
            vi = np.sort(rng.choice(V, Vs, replace=False)) if Vs < V else slice(None)
            ti = np.sort(rng.choice(T, Ts, replace=False)) if Ts < T else slice(None)
            X_sub, R_sub = X[vi][:, ti], R[vi]
            problem: ResidualProblem | ProjectedProblem
            if method == "varpro":
                problem = ProjectedProblem(
                    X_sub, R_sub, K, engine, prior, self.weight_method, **_kw(kernel)
                )
            else:
                F = _x_factors(R_sub, x[:n].reshape(-1, D), x[n:], K, kernel)
                W = ridge_weights(X_sub, F, self.weight_method)
                problem = ResidualProblem(
                    X_sub, W, R_sub, engine, prior, compress=loss == "linear",
                    **_kw(kernel),
                )  # fmt: skip
            x_new, _ = solve_centers_widths(
                problem,
                x,
                bnds,
                method=method,
                loss=loss,
                x_scale=self.x_scale,
                mode=self.mode,
            )
            x, change = _matched_step(kernel, K, D, x, x_new, length)
            message = f"TFA {label} iteration {n_iter}: scaled change {change:.6g}"
            if lowest:
                cost = objective(x)
                history.append(cost)
                change_obj = (lowest[-1] - cost) / abs(lowest[-1])
                if cost < lowest[-1]:
                    x_best = x
                lowest.append(min(lowest[-1], cost))
                message += f", objective decrease {change_obj:.3g}"
            else:
                x_best = x
            if self.verbose:
                print(message, flush=True)
            if change <= self.threshold:
                yield x, n_iter, "threshold"
                return
            if lowest and n_iter - n_done >= OBJECTIVE_MIN_ITER:
                ref = lowest[-1 - OBJECTIVE_WINDOW]
                window_decrease = (ref - lowest[-1]) / abs(ref)
            else:
                window_decrease = np.inf
            if window_decrease < OBJECTIVE_WINDOW * OBJECTIVE_RTOL:
                yield x_best, n_iter, "objective"
                return
            best, stale = (change, 0) if change < best else (best, stale + 1)
            if subsampled and stale >= PLATEAU_PATIENCE:
                yield x_best, n_iter, "plateau"
                return
            yield x_best, n_iter, "max_iter" if n_iter == self.max_iter else None

    def fit(
        self,
        X: ArrayLike,
        coords: ArrayLike,
        prior: Prior | None = None,
        start: tuple[ArrayLike, ...] | None = None,
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
            ``prior.widths`` (and ``prior.logits``; per component with
            ``n_components > 1``, see `htfa._kernels.Prior`) (``init`` is
            ignored), and every solve is penalized
            towards the template with ``scale = 0.5 * (Vs * Ts) / (V * T)``
            (``prior.scale`` is ignored and the caller's prior is not modified).
        start : tuple of arrays, optional
            Explicit starting point, shapes (K, D) and (K,); with
            ``n_components = M > 1``, component centers (K, M, D), widths
            (K, M) and mixing weights (K, M). Overrides both ``init`` and the
            prior start (the prior penalty still applies).

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
        kernel = self._kernel()
        M = kernel.n_components
        if prior is not None and np.shape(prior.centers) != (K * M, D):
            raise ValueError(
                f"prior.centers must have shape ({K * M}, {D}), "
                f"got {np.shape(prior.centers)}"
            )
        if prior is not None and M > 1:
            expected = {
                "widths": (K * M,),
                "widths_var": (K * M,),
                "centers_cov": (K * M, D, D),
                "logits": (K, M - 1),
                "logits_var": (K, M - 1),
            }
            for field, shape in expected.items():
                if np.shape(getattr(prior, field)) != shape:
                    raise ValueError(
                        f"prior.{field} must have shape {shape}, "
                        f"got {np.shape(getattr(prior, field))}"
                    )
        rng = np.random.default_rng(self.random_state)
        sample_seed = int(rng.integers(np.iinfo(np.int32).max))
        sizes = (
            _sample_size(V, self.voxel_ratio, self.max_voxels),
            _sample_size(T, self.tr_ratio, self.max_trs),
        )
        engine = select_engine(self.backend, *sizes, K)
        lb, ub = bounds(R, K, self.lower_ratio, self.upper_ratio, **_kw(kernel))
        w_lo, w_hi = lb[K * M * D], ub[K * M * D]
        if prior is None:
            fit_prior = full_prior = None
        else:
            fit_prior = dataclasses.replace(
                prior, scale=0.5 * (sizes[0] * sizes[1]) / (V * T)
            )
            full_prior = dataclasses.replace(prior, scale=0.5)
        if start is not None:
            starts: dict[str, Array] = {"start": _start_x(kernel, start, K, D)}
        elif prior is not None:
            logits = () if prior.logits is None else prior.logits
            starts = {"prior": kernel.join(prior.centers, prior.widths, logits)}
        else:
            starts = {
                name: kernel.start(*pair)
                for name, pair in _initial_starts(
                    X, R, K, names, rng, w_lo, w_hi
                ).items()
            }
        Ve, Te = self._eval_sizes(V, T, sizes)
        eval_rng = np.random.default_rng([sample_seed, 1])
        ve = np.sort(eval_rng.choice(V, Ve, replace=False)) if Ve < V else slice(None)
        te = np.sort(eval_rng.choice(T, Te, replace=False)) if Te < T else slice(None)
        X_eval = X if (Ve, Te) == (V, T) else X[ve][:, te]
        R_eval = R[ve]
        eval_prior = (
            None
            if prior is None
            else dataclasses.replace(prior, scale=0.5 * (Ve * Te) / (V * T))
        )
        runs: dict[str, tuple[Array, int, str | None]] = {}
        loops = {}
        histories: dict[str, list[float]] = {name: [] for name in starts}
        for name, x_start in starts.items():
            x0 = np.clip(x_start, lb, ub)
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
                lambda loss, x: self._objective(
                    X_eval, R_eval, x, engine, eval_prior, loss
                )[0],
                histories[name],
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
        # Starts that reach the same optimum differ in cost by rounding only, and
        # rounding differs with the thread count: the first start (in `init`
        # order) within START_TIE_RTOL of the lowest cost wins.
        lowest = min(score[0] for score in scores.values())
        self.init_ = next(
            n for n in starts if scores[n][0] <= lowest + START_TIE_RTOL * abs(lowest)
        )
        if screen > 0:
            advance(self.init_, self.max_iter)
            scores[self.init_] = objective(self.init_)
        self.init_objectives_ = {n: score[0] for n, score in scores.items()}
        self.cost_, self.weights_ = scores[self.init_]
        x, self.n_iter_, reason = runs[self.init_]
        self.objective_history_ = histories[self.init_]
        self.converged_reason_ = reason
        self.converged_ = reason != "max_iter"
        if self.max_iter > 0 and not self.converged_:
            if sizes != (V, T):
                advice = (
                    f"with subsampling ({sizes[0]} of {V} voxels, {sizes[1]} of {T} "
                    "samples) the scaled change has a noise floor, so raise "
                    "max_voxels/max_trs (or voxel_ratio/tr_ratio) or max_iter"
                )
            else:
                advice = "increase max_iter or threshold"
            warnings.warn(
                f"TFA did not converge in max_iter={self.max_iter} iterations "
                f"(init {self.init_!r}); {advice}",
                ConvergenceWarning,
                stacklevel=2,
            )
        if M == 1:
            self.centers_ = x[: K * D].reshape(K, D).copy()
            self.widths_ = x[K * D :].copy()
        self._set_components(x, K, D)
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
        kernel = Kernel(
            getattr(self, "kernel", "rbf"), getattr(self, "n_components", 1)
        )
        if kernel.plain_rbf:
            result: Array = factors(R, self.centers_, self.widths_)
            return result
        return component_factors(
            R,
            self.component_centers_,
            self.component_widths_,
            self.mixing_weights_,
            kernel,
        )

    def _set_components(self, x: Array, K: int, D: int) -> None:
        """``component_centers_``, ``component_widths_`` and
        ``mixing_weights_`` from ``x`` (and, with several components, the
        mixing-weighted ``centers_`` and ``widths_``)."""
        kernel = self._kernel()
        M = kernel.n_components
        C, w, logits = kernel.split(x, K, D)
        pi = kernel.mixing(logits)
        self.component_centers_ = C.reshape(K, M, D).copy()
        self.component_widths_ = w.reshape(K, M).copy()
        self.mixing_weights_ = pi
        if M > 1:
            self.centers_ = np.einsum("km,kmd->kd", pi, self.component_centers_)
            self.widths_ = np.sum(pi * self.component_widths_, axis=1)

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
