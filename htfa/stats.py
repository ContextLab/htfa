"""Reconstruction scores and nominal conditional Gaussian information criteria.

These are descriptive plug-in criteria, not marginal evidence for HTFA. Spatial
and temporal residual dependence, regularization and mixture singularities break
the usual independent-observation / regular maximum-likelihood justification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np

from .tfa import standardize_data

if TYPE_CHECKING:
    from .results import HTFAResults


@dataclass(frozen=True)
class FitStatistics:
    """Pooled scores; smaller information criteria and larger R² are better.

    ``n_parameters`` includes residual variance. ``n_observations`` counts scalar
    voxel-time measurements, not independent participants. R² uses each voxel's
    temporal mean as its baseline. Constant data have undefined R² (NaN).
    Perfect fits have unbounded Gaussian likelihood and hence -inf criteria.
    AICc is inf when n <= p + 1. No effective-sample-size correction is inferred.
    """

    n_observations: int
    n_parameters: int
    rss: float
    tss: float
    r2: float
    log_likelihood: float
    aic: float
    aicc: float
    bic: float

    def to_dict(self) -> dict:
        """Return scalar values suitable for a comparison table."""
        return asdict(self)


def gaussian_fit_statistics(observed, predicted, *, n_parameters: int) -> FitStatistics:
    """Score equally shaped (voxels, times) arrays or lists of subject arrays.

    Supply the total nominal parameter count, INCLUDING one pooled residual
    variance. Likelihood uses sigma² = RSS/n and independent Gaussian residuals.
    Compare only the identical observations, masks, scaling and preprocessing.
    No intercept is fitted here; the temporal mean is only the R² baseline.
    Complex inputs and unrepresentable squared-error scales raise ValueError
    rather than silently discarding imaginary parts or reporting a perfect fit.
    """
    if (
        isinstance(n_parameters, (bool, np.bool_))
        or not isinstance(n_parameters, (int, np.integer))
        or n_parameters < 1
    ):
        raise ValueError("n_parameters must be a positive integer including variance")
    xs = [observed] if isinstance(observed, np.ndarray) else list(observed)
    ys = [predicted] if isinstance(predicted, np.ndarray) else list(predicted)
    if not xs or len(xs) != len(ys):
        raise ValueError("supply equally many nonempty subject arrays")
    rss = tss = 0.0
    n = 0
    for x, y in zip(xs, ys, strict=True):
        if np.iscomplexobj(x) or np.iscomplexobj(y):
            raise ValueError("observations and predictions must be real-valued")
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if x.ndim != 2 or x.size == 0 or x.shape != y.shape:
            raise ValueError(
                "each pair must have the same nonempty (voxels, times) shape"
            )
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("observations and predictions must be finite")
        try:
            with np.errstate(over="raise", invalid="raise"):
                residual = x - y
                centered = x - x.mean(axis=1, keepdims=True)
                residual_sum = float(np.sum(residual**2))
                centered_sum = float(np.sum(centered**2))
        except FloatingPointError as exc:
            raise ValueError("squared errors overflow; rescale the data") from exc
        if (residual_sum == 0 and np.any(residual)) or (
            centered_sum == 0 and np.any(centered)
        ):
            raise ValueError("squared errors underflow; rescale the data")
        rss += residual_sum
        tss += centered_sum
        if not np.isfinite(rss) or not np.isfinite(tss):
            raise ValueError("pooled squared errors overflow; rescale the data")
        n += x.size
    p = int(n_parameters)
    ll = (
        float(-0.5 * n * (np.log(2 * np.pi) + 1 + (np.log(rss) - np.log(n))))
        if rss
        else np.inf
    )
    aic = 2 * p - 2 * ll
    return FitStatistics(
        n,
        p,
        rss,
        tss,
        1 - rss / tss if tss else np.nan,
        ll,
        aic,
        aic + 2 * p * (p + 1) / (n - p - 1) if n > p + 1 else np.inf,
        p * np.log(n) - 2 * ll,
    )


def parameter_count(results: HTFAResults) -> dict[str, int]:
    """Nominal conditional reconstruction parameter budget for fitted TFA/HTFA.

    Per subject: K*M*(D+1) centers/widths, K*(M-1) independent mixture weights,
    and K*T temporal coefficients. One shared residual variance is added.
    The global/group templates and their dispersions are conditioned on, NOT
    counted as independently fitted reconstruction parameters. This is not a
    count for the joint hierarchical likelihood or an effective degrees of
    freedom estimate; shrinkage and redundant components are not adjusted.
    """
    k, m = results.model.K, results.model.n_components
    spatial = sum(k * (m * (s.coords.shape[1] + 1) + m - 1) for s in results.subjects)
    temporal = sum(w.size for w in results.weights)
    return {
        "spatial": spatial,
        "temporal": temporal,
        "variance": 1,
        "total": spatial + temporal + 1,
    }


def reconstruction_statistics(results: HTFAResults) -> FitStatistics:
    """Score full fitted subjects in the model's standardized fitting space.

    Uses :func:`parameter_count` and :func:`gaussian_fit_statistics`. Includes
    every supplied voxel, even if optimization used a voxel subsample.
    """
    observed = [
        standardize_data(s.data, results.model.standardize) for s in results.subjects
    ]
    predicted = [results.reconstruct(i) for i in range(len(observed))]
    return gaussian_fit_statistics(
        observed, predicted, n_parameters=parameter_count(results)["total"]
    )
