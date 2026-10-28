"""Synthetic TFA/HTFA data with known ground truth.

Widths follow the BrainIAK convention ``w = 2 sigma**2``. All generators are
deterministic given ``random_state`` and return float64 arrays. Factors can
have any shape of `htfa._kernels.Kernel` and several components.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from htfa._kernels import Kernel, factors
from htfa._matching import match_factors as match_factors  # public re-export
from htfa.tfa import component_factors

Array = NDArray[np.float64]

SIGMA_RANGE = (0.06, 0.12)  # fraction of the largest axis extent
CENTER_MARGIN = 0.1  # centers are drawn in the central 80% of each axis
MAX_TRIES = 1000
# Mixtures (n_components = M > 1): component m >= 1 of a factor is offset from
# the factor's drawn center by N(0, COMPONENT_OFFSET**2 sigma**2) per axis and
# has COMPONENT_WIDTH_RANGE (uniform) times the previous component's width;
# the mixing weights are Dirichlet(MIXING_CONCENTRATION, ..., ...).
COMPONENT_OFFSET = 0.75
COMPONENT_WIDTH_RANGE = (0.2, 0.5)
MIXING_CONCENTRATION = 3.0


def _draw_components(
    rng: np.random.Generator, centers: Array, widths: Array, M: int
) -> tuple[Array, Array, Array]:
    """Component centers (K, M, D), widths (K, M) and mixing weights (K, M)
    around factor centers and widths (the first component at the drawn center
    with the drawn width; one component: the factors themselves)."""
    K, D = centers.shape
    if M == 1:
        return centers[:, None, :].copy(), widths[:, None].copy(), np.ones((K, 1))
    sigma = np.sqrt(widths / 2.0)
    offsets = COMPONENT_OFFSET * sigma[:, None, None] * rng.standard_normal((K, M, D))
    offsets[:, 0] = 0.0
    ratios = rng.uniform(*COMPONENT_WIDTH_RANGE, size=(K, M))
    ratios[:, 0] = 1.0
    pi = rng.dirichlet(np.full(M, MIXING_CONCENTRATION), size=K)
    return centers[:, None, :] + offsets, widths[:, None] * np.cumprod(ratios, 1), pi


def _shape_factors(
    R: Array, kernel: Kernel, centers: Array, widths: Array, pi: Array
) -> Array:
    """Factors (V, K) of components (K, M, D), (K, M), (K, M); `factors` for
    one RBF component."""
    if kernel.plain_rbf:
        return factors(R, centers[:, 0], widths[:, 0])
    return component_factors(R, centers, widths, pi, kernel)


def _summary(centers: Array, widths: Array, pi: Array) -> tuple[Array, Array]:
    """Factor centers and widths: the mixing-weighted means (as `htfa.TFA`)."""
    return np.einsum("km,kmd->kd", pi, centers), np.sum(pi * widths, axis=1)


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
    kernel: str = "rbf",
    n_components: int = 1,
) -> tuple[Array, Array, dict[str, Any]]:
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
    kernel : {'rbf', 'laplace', 'mexican_hat'}, default='rbf'
        Factor shape (see `htfa.TFA`).
    n_components : int, default=1
        Components per factor. With M > 1 the first component of each factor
        sits at the drawn center with the drawn width; component m >= 1 is
        offset by ``N(0, (0.75 sigma)**2)`` per axis and has a uniform 0.2 to
        0.5 times the previous component's width, and the mixing weights are
        ``Dirichlet(3, ..., 3)``. The draws for one RBF component are those
        of earlier versions.

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
        [0.06, 0.12] times the largest axis extent. Also
        ``component_centers`` (K, M, D), ``component_widths`` (K, M),
        ``mixing_weights`` (K, M) and ``kernel``; with M > 1, ``centers`` and
        ``widths`` are the mixing-weighted means of the components' (as
        `htfa.TFA` reports them).

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
    shape = Kernel(kernel, n_components)
    rng = np.random.default_rng(random_state)
    R, extent = _grid(grid_shape, spacing)
    centers, widths = _draw_centers_widths(rng, extent, K, min_separation)
    Cc, wc, pi = _draw_components(rng, centers, widths, n_components)
    weights = rng.standard_normal((K, n_samples))
    X = _shape_factors(R, shape, Cc, wc, pi) @ weights
    X += noise * rng.standard_normal(X.shape)
    if n_components > 1:
        centers, widths = _summary(Cc, wc, pi)
    return (
        X,
        R,
        {
            "centers": centers,
            "widths": widths,
            "weights": weights,
            "component_centers": Cc,
            "component_widths": wc,
            "mixing_weights": pi,
            "kernel": kernel,
        },
    )


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
    kernel: str = "rbf",
    n_components: int = 1,
) -> tuple[list[Array], list[Array], dict[str, Any]]:
    """Simulate multi-subject HTFA data around a global template.

    Each subject's centers are the global centers plus ``N(0, center_jitter**2)``
    (clipped to the grid) and its widths are the global widths times
    ``exp(N(0, width_jitter))``. Weights and noise are drawn independently per
    subject. Clipped subject centers are not re-checked for separation. Global
    centers and widths are drawn as in `make_tfa_data`. With ``n_components``
    M > 1 the global components are drawn as in `make_tfa_data`; each
    subject's component centers and widths are jittered the same way per
    component, and its mixing logits ``log(pi_m / pi_1)`` get
    ``N(0, width_jitter**2)``.

    Parameters
    ----------
    n_subjects : int
        Number of subjects.
    grid_shape, n_samples, K, noise, spacing, random_state, kernel, n_components
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
        ``centers``, ``widths``, ``weights``; also ``global_component_centers``,
        ``global_component_widths``, ``global_mixing_weights`` and per-subject
        lists ``component_centers``, ``component_widths``, ``mixing_weights``
        (as in `make_tfa_data`), and ``kernel``.

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
    shape = Kernel(kernel, n_components)
    M = n_components
    rng = np.random.default_rng(random_state)
    R, extent = _grid(grid_shape, spacing)
    g_centers, g_widths = _draw_centers_widths(rng, extent, K, min_separation)
    gC, gw, gpi = _draw_components(rng, g_centers, g_widths, M)
    if M > 1:
        g_centers, g_widths = _summary(gC, gw, gpi)
    Xs, Rs = [], []
    truth: dict[str, Any] = {
        "global_centers": g_centers,
        "global_widths": g_widths,
        "centers": [],
        "widths": [],
        "weights": [],
        "global_component_centers": gC,
        "global_component_widths": gw,
        "global_mixing_weights": gpi,
        "component_centers": [],
        "component_widths": [],
        "mixing_weights": [],
        "kernel": kernel,
    }
    for _ in range(n_subjects):
        Cc = np.clip(gC + center_jitter * rng.standard_normal(gC.shape), 0, extent)
        wc = gw * np.exp(width_jitter * rng.standard_normal((K, M)))
        if M > 1:
            a = Kernel.logits(gpi) + width_jitter * rng.standard_normal((K, M - 1))
            pi = Kernel.mixing(a)
        else:
            pi = gpi
        weights = rng.standard_normal((K, n_samples))
        X = _shape_factors(R, shape, Cc, wc, pi) @ weights
        X += noise * rng.standard_normal(X.shape)
        Xs.append(X)
        Rs.append(R.copy())
        centers, widths = (Cc[:, 0], wc[:, 0]) if M == 1 else _summary(Cc, wc, pi)
        truth["centers"].append(centers)
        truth["widths"].append(widths)
        truth["weights"].append(weights)
        truth["component_centers"].append(Cc)
        truth["component_widths"].append(wc)
        truth["mixing_weights"].append(pi)
    return Xs, Rs, truth
