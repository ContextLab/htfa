"""Synthetic TFA/HTFA data with known ground truth.

Widths follow the BrainIAK convention ``w = 2 sigma**2``. All generators are
deterministic given ``random_state`` and return float64 arrays.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from htfa._kernels import factors

Array = NDArray[np.float64]

SIGMA_RANGE = (0.06, 0.12)  # fraction of the largest axis extent
CENTER_MARGIN = 0.1  # centers are drawn in the central 80% of each axis
MAX_TRIES = 1000


def _grid(grid_shape: Sequence[int], spacing: float) -> tuple[Array, Array]:
    """Grid coordinates (V, D) and the per-axis extent (D,)."""
    if len(grid_shape) not in (2, 3):
        raise ValueError(f"grid_shape must have 2 or 3 axes, got {tuple(grid_shape)}")
    axes = [np.arange(n, dtype=np.float64) * spacing for n in grid_shape]
    R = np.stack([a.ravel() for a in np.meshgrid(*axes, indexing="ij")], axis=1)
    return R, np.array([a[-1] for a in axes])


def _validate(
    grid_shape: Sequence[int], n_samples: int, K: int, noise: float, spacing: float
) -> None:
    if any(n < 2 for n in grid_shape):
        raise ValueError(f"every grid_shape dimension must be >= 2, got {grid_shape}")
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    if spacing <= 0:
        raise ValueError(f"spacing must be > 0, got {spacing}")
    if noise < 0:
        raise ValueError(f"noise must be >= 0, got {noise}")


def _draw_centers_widths(
    rng: np.random.Generator, extent: Array, K: int, min_separation: float
) -> tuple[Array, Array]:
    """Draw K widths, then K centers at least ``min_separation`` mean sigmas apart."""
    sigma = rng.uniform(*SIGMA_RANGE, size=K) * float(np.max(extent))
    widths = 2.0 * sigma**2
    min_sep = min_separation * np.sqrt(widths.mean() / 2.0)
    lo, hi = CENTER_MARGIN * extent, (1.0 - CENTER_MARGIN) * extent
    centers = np.empty((K, extent.size))
    for k in range(K):
        for _ in range(MAX_TRIES):
            cand = rng.uniform(lo, hi)
            if k == 0 or np.min(np.linalg.norm(centers[:k] - cand, axis=1)) >= min_sep:
                centers[k] = cand
                break
        else:
            raise ValueError(
                f"could not place center {k} of {K} with minimum separation "
                f"{min_sep:.3g} after {MAX_TRIES} tries; use fewer factors K or a "
                "larger grid_shape/spacing"
            )
    return centers, widths


def make_tfa_data(
    grid_shape: Sequence[int] = (20, 20, 20),
    n_samples: int = 100,
    K: int = 10,
    noise: float = 0.1,
    spacing: float = 1.0,
    min_separation: float = 2.0,
    random_state: int | None = None,
) -> tuple[Array, Array, dict[str, Array]]:
    """Simulate single-subject TFA data ``X = F W + noise``.

    Parameters
    ----------
    grid_shape : sequence of int
        Voxels per axis; 2 or 3 axes.
    n_samples : int
        Number of time points T.
    K : int
        Number of factors.
    noise : float
        Standard deviation of the additive Gaussian noise.
    spacing : float
        Distance between neighbouring grid points.
    min_separation : float
        Minimum distance between centers, in units of the mean sigma:
        ``min_separation * sqrt(mean(w) / 2)``.
    random_state : int, optional
        Seed for ``numpy.random.default_rng``.

    Returns
    -------
    X : ndarray, shape (V, T)
        Data.
    R : ndarray, shape (V, D)
        Voxel coordinates (grid index times ``spacing``).
    truth : dict
        ``centers`` (K, D), ``widths`` (K,) with ``w = 2 sigma**2``, and
        ``weights`` (K, T). Centers lie in the central 80% of each axis and are at
        least ``min_separation`` mean sigmas apart; sigma is uniform in
        [0.06, 0.12] times the largest axis extent.

    Raises
    ------
    ValueError
        If centers cannot be placed with the required separation.
    """
    _validate(grid_shape, n_samples, K, noise, spacing)
    rng = np.random.default_rng(random_state)
    R, extent = _grid(grid_shape, spacing)
    centers, widths = _draw_centers_widths(rng, extent, K, min_separation)
    weights = rng.standard_normal((K, n_samples))
    X = factors(R, centers, widths) @ weights
    X += noise * rng.standard_normal(X.shape)
    return X, R, {"centers": centers, "widths": widths, "weights": weights}


def make_htfa_data(
    n_subjects: int = 5,
    grid_shape: Sequence[int] = (20, 20, 20),
    n_samples: int = 100,
    K: int = 10,
    noise: float = 0.1,
    center_jitter: float = 1.0,
    width_jitter: float = 0.05,
    spacing: float = 1.0,
    min_separation: float = 2.0,
    random_state: int | None = None,
) -> tuple[list[Array], list[Array], dict[str, Any]]:
    """Simulate multi-subject HTFA data around a global template.

    Each subject's centers are the global centers plus ``N(0, center_jitter**2)``
    (clipped to the grid) and its widths are the global widths times
    ``exp(N(0, width_jitter))``. Weights and noise are drawn independently per
    subject. Clipped subject centers are not re-checked for separation. Global
    centers and widths are drawn as in `make_tfa_data`.

    Parameters
    ----------
    n_subjects : int
        Number of subjects.
    grid_shape, n_samples, K, noise, spacing, random_state
        As in `make_tfa_data` (``min_separation`` applies to the global centers);
        every subject shares the same grid.
    center_jitter : float
        Standard deviation of subject center offsets, in coordinate units.
    width_jitter : float
        Standard deviation of the log subject-to-global width ratio.

    Returns
    -------
    Xs : list of ndarray, each (V, T)
        Data per subject.
    Rs : list of ndarray, each (V, D)
        Coordinates per subject.
    truth : dict
        ``global_centers`` (K, D), ``global_widths`` (K,), and per-subject lists
        ``centers``, ``widths``, ``weights``.

    Raises
    ------
    ValueError
        If global centers cannot be placed with the required separation.
    """
    if n_subjects < 1:
        raise ValueError(f"n_subjects must be >= 1, got {n_subjects}")
    _validate(grid_shape, n_samples, K, noise, spacing)
    rng = np.random.default_rng(random_state)
    R, extent = _grid(grid_shape, spacing)
    g_centers, g_widths = _draw_centers_widths(rng, extent, K, min_separation)
    Xs, Rs = [], []
    truth: dict[str, Any] = {
        "global_centers": g_centers,
        "global_widths": g_widths,
        "centers": [],
        "widths": [],
        "weights": [],
    }
    for _ in range(n_subjects):
        centers = np.clip(
            g_centers + center_jitter * rng.standard_normal(g_centers.shape), 0, extent
        )
        widths = g_widths * np.exp(width_jitter * rng.standard_normal(K))
        weights = rng.standard_normal((K, n_samples))
        X = factors(R, centers, widths) @ weights
        X += noise * rng.standard_normal(X.shape)
        Xs.append(X)
        Rs.append(R.copy())
        truth["centers"].append(centers)
        truth["widths"].append(widths)
        truth["weights"].append(weights)
    return Xs, Rs, truth


def match_factors(true_centers: Array, est_centers: Array) -> NDArray[np.intp]:
    """Match estimated factors to true ones with the Hungarian algorithm.

    Parameters
    ----------
    true_centers : ndarray, shape (K, D)
        Ground-truth centers.
    est_centers : ndarray, shape (K, D)
        Estimated centers, in arbitrary order.

    Returns
    -------
    perm : ndarray of int, shape (K,)
        Permutation such that ``est_centers[perm][k]`` corresponds to
        ``true_centers[k]`` (minimum total squared Euclidean distance).
    """
    cost = cdist(true_centers, est_centers, metric="sqeuclidean")
    rows, cols = linear_sum_assignment(cost)
    perm = np.empty(len(rows), dtype=np.intp)
    perm[rows] = cols
    return perm
