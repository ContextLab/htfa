"""Factor-level connectivity from factor weight time courses."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

Array = np.ndarray

__all__ = ["static_connectivity", "dynamic_connectivity", "isfc"]

_FISHER_CLIP = 1.0 - 1e-7


def _weights(weights: Array) -> Array:
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 2:
        raise ValueError(f"weights must have shape (K, T); got {w.shape}")
    return w


def _window(window: int, T: int) -> None:
    if not 2 <= window <= T:
        raise ValueError(f"window must be in [2, T={T}], got {window}")


def static_connectivity(weights: Array) -> Array:
    """Pearson correlation between factor time courses.

    Parameters
    ----------
    weights : ndarray, shape (K, T)
        Factor weights over samples.

    Returns
    -------
    ndarray, shape (K, K)

    Examples
    --------
    >>> import numpy as np
    >>> w = np.random.default_rng(0).standard_normal((3, 50))
    >>> static_connectivity(w).shape
    (3, 3)
    """
    return np.atleast_2d(np.corrcoef(_weights(weights)))


def dynamic_connectivity(weights: Array, window: int) -> Array:
    """Sliding-window Pearson correlation between factor time courses.

    Parameters
    ----------
    weights : ndarray, shape (K, T)
        Factor weights over samples.
    window : int
        Window length in samples (2 <= window <= T).

    Returns
    -------
    ndarray, shape (T - window + 1, K, K)

    Examples
    --------
    >>> import numpy as np
    >>> w = np.random.default_rng(0).standard_normal((3, 50))
    >>> dynamic_connectivity(w, 20).shape
    (31, 3, 3)
    """
    w = _weights(weights)
    _window(window, w.shape[1])
    return np.stack(
        [
            static_connectivity(w[:, i : i + window])
            for i in range(w.shape[1] - window + 1)
        ]
    )


def isfc(weights: Sequence[Array], window: int | None = None) -> Array:
    """Inter-subject functional correlation between factors.

    For each subject, its factor time courses are correlated with the mean of the
    other subjects' time courses; the result is symmetrized as ``(C + C.T) / 2`` and
    averaged over subjects in Fisher-z space.

    Parameters
    ----------
    weights : list of ndarray, each shape (K, T)
        One weight matrix per subject; all must share ``K`` and ``T``.
    window : int, optional
        If given, sliding-window ISFC.

    Returns
    -------
    ndarray, shape (K, K) or (T - window + 1, K, K)

    Examples
    --------
    >>> import numpy as np
    >>> w = np.random.default_rng(0).standard_normal((3, 50))
    >>> isfc([w, w + 0.1], window=20).shape
    (31, 3, 3)
    """
    ws = [_weights(w) for w in weights]
    if len(ws) < 2:
        raise ValueError("isfc needs at least 2 subjects")
    if len({w.shape for w in ws}) != 1:
        shapes = sorted({w.shape for w in ws})
        raise ValueError(f"all subjects must share K and T; got shapes {shapes}")
    total = np.sum(ws, axis=0)
    K, T = ws[0].shape
    if window is not None:
        _window(window, T)
    zs = []
    for w in ws:
        other = (total - w) / (len(ws) - 1)
        spans = (
            [slice(None)]
            if window is None
            else [slice(i, i + window) for i in range(T - window + 1)]
        )
        cs = []
        for s in spans:
            c = np.corrcoef(w[:, s], other[:, s])[:K, K:]
            cs.append((c + c.T) / 2)
        c_all = np.stack(cs)
        zs.append(np.arctanh(np.clip(c_all, -_FISHER_CLIP, _FISHER_CLIP)))
    out = np.tanh(np.mean(zs, axis=0))
    return out[0] if window is None else out
