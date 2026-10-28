"""Synthetic TFA/HTFA data with known ground truth.

Widths follow the BrainIAK convention ``w = 2 sigma**2``. All generators are
deterministic given ``random_state`` and return float64 arrays.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from htfa._kernels import factors
from htfa._matching import match_factors as match_factors  # public re-export

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

    Examples
    --------
    >>> from htfa.synthetic import make_tfa_data
    >>> X, R, truth = make_tfa_data((8, 8, 8), n_samples=20, K=3, random_state=0)
    >>> X.shape, R.shape, truth["centers"].shape
    ((512, 20), (512, 3), (3, 3))
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

    Examples
    --------
    >>> from htfa.synthetic import make_htfa_data
    >>> Xs, Rs, truth = make_htfa_data(
    ...     2, grid_shape=(8, 8, 8), n_samples=20, K=3, random_state=0
    ... )
    >>> len(Xs), Xs[0].shape, truth["global_centers"].shape
    (2, (512, 20), (3, 3))
    """
    if n_subjects < 1:
        raise ValueError(f"n_subjects must be >= 1, got {n_subjects}")
    _validate(grid_shape, n_samples, K, noise, spacing)
    rng = np.random.default_rng(random_state)
    R, extent = _grid(grid_shape, spacing)
    g_centers, g_widths = _draw_centers_widths(rng, extent, K, min_separation)
    Xs: list[Array] = []
    Rs: list[Array] = []
    truth: dict[str, Any] = {
        "global_centers": g_centers,
        "global_widths": g_widths,
        "centers": [],
        "widths": [],
        "weights": [],
    }
    for _ in range(n_subjects):
        _add_subject(
            rng, R, extent, g_centers, g_widths, n_samples, noise, center_jitter,
            width_jitter, Xs, Rs, truth,
        )  # fmt: skip
    return Xs, Rs, truth


def _add_subject(
    rng: np.random.Generator,
    R: Array,
    extent: Array,
    centers: Array,
    widths: Array,
    n_samples: int,
    noise: float,
    center_jitter: float,
    width_jitter: float,
    Xs: list[Array],
    Rs: list[Array],
    truth: dict[str, Any],
) -> None:
    """Draw one subject around template ``centers``/``widths`` (see
    `make_htfa_data`) and append it to ``Xs``, ``Rs`` and ``truth``."""
    K = len(widths)
    c = np.clip(centers + center_jitter * rng.standard_normal(centers.shape), 0, extent)
    w = widths * np.exp(width_jitter * rng.standard_normal(K))
    weights = rng.standard_normal((K, n_samples))
    X = factors(R, c, w) @ weights
    X += noise * rng.standard_normal(X.shape)
    Xs.append(X)
    Rs.append(R.copy())
    truth["centers"].append(c)
    truth["widths"].append(w)
    truth["weights"].append(weights)


def make_group_data(
    group_sizes: Sequence[int] = (4, 4),
    grid_shape: Sequence[int] = (20, 20, 20),
    n_samples: int = 100,
    K: int = 10,
    center_shift: Any = None,
    width_scale: Any = None,
    noise: float = 0.1,
    center_jitter: float = 1.0,
    width_jitter: float = 0.05,
    spacing: float = 1.0,
    min_separation: float = 2.0,
    random_state: int | None = None,
) -> tuple[list[Array], list[Array], list[int], dict[str, Any]]:
    """Simulate groups of subjects whose templates differ from a global one.

    Group ``g``'s centers are the global centers plus ``center_shift[g]`` and
    its widths the global widths times ``width_scale[g]``; its subjects are
    drawn around the group template as in `make_htfa_data`. Global centers and
    widths are drawn as in `make_tfa_data`.

    Parameters
    ----------
    group_sizes : sequence of int
        Number of subjects per group (G groups).
    grid_shape, n_samples, K, noise, spacing, min_separation, random_state
        As in `make_tfa_data`.
    center_shift : array_like, shape (G, K, D), optional
        Offsets of the group centers from the global centers (default 0); group
        centers are clipped to the grid and not re-checked for separation.
    width_scale : array_like, shape (G, K), optional
        Group width over global width (default 1).
    center_jitter, width_jitter
        As in `make_htfa_data`, around each subject's group template.

    Returns
    -------
    Xs, Rs : list of ndarray
        Data (V, T) and coordinates (V, D) per subject, group by group.
    groups : list of int
        Each subject's group index, for ``HTFA.fit(..., groups=groups)``.
    truth : dict
        ``global_centers`` (K, D), ``global_widths`` (K,), ``group_centers``
        (G, K, D), ``group_widths`` (G, K), and per-subject lists ``centers``,
        ``widths``, ``weights``.

    Raises
    ------
    ValueError
        On an empty or non-positive ``group_sizes``, shifts or scales of the
        wrong shape, a non-positive ``width_scale``, or as `make_tfa_data`.

    Examples
    --------
    >>> import numpy as np
    >>> from htfa.synthetic import make_group_data
    >>> shift = np.zeros((2, 3, 2))
    >>> shift[1, 0] = [2.0, 0.0]  # group 1 moves factor 0
    >>> Xs, Rs, groups, truth = make_group_data(
    ...     (2, 3), grid_shape=(10, 10), n_samples=20, K=3, center_shift=shift,
    ...     random_state=0,
    ... )
    >>> len(Xs), groups, truth["group_centers"].shape
    (5, [0, 0, 1, 1, 1], (2, 3, 2))
    """
    sizes = [int(n) for n in group_sizes]
    if not sizes or min(sizes) < 1:
        raise ValueError(f"group_sizes must be positive integers, got {group_sizes}")
    _validate(grid_shape, n_samples, K, noise, spacing)
    rng = np.random.default_rng(random_state)
    R, extent = _grid(grid_shape, spacing)
    g_centers, g_widths = _draw_centers_widths(rng, extent, K, min_separation)
    G, D = len(sizes), len(grid_shape)
    shift = np.zeros((G, K, D)) if center_shift is None else np.asarray(center_shift)
    scale = np.ones((G, K)) if width_scale is None else np.asarray(width_scale)
    if shift.shape != (G, K, D) or scale.shape != (G, K):
        raise ValueError(
            f"center_shift must be ({G}, {K}, {D}) and width_scale ({G}, {K}); "
            f"got {shift.shape} and {scale.shape}"
        )
    if np.any(scale <= 0):
        raise ValueError("width_scale must be positive")
    truth: dict[str, Any] = {
        "global_centers": g_centers,
        "global_widths": g_widths,
        "group_centers": np.clip(g_centers + shift, 0, extent),
        "group_widths": g_widths * scale,
        "centers": [],
        "widths": [],
        "weights": [],
    }
    Xs: list[Array] = []
    Rs: list[Array] = []
    groups: list[int] = []
    for g, n in enumerate(sizes):
        for _ in range(n):
            _add_subject(
                rng, R, extent, truth["group_centers"][g], truth["group_widths"][g],
                n_samples, noise, center_jitter, width_jitter, Xs, Rs, truth,
            )  # fmt: skip
            groups.append(g)
    return Xs, Rs, groups, truth
