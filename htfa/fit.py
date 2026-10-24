"""Main entry point: load data and fit TFA or HTFA."""

from __future__ import annotations

from typing import Any

from .htfa import HTFA
from .io import load
from .results import HTFAResults
from .tfa import TFA

__all__ = ["fit"]


def fit(
    data: Any,
    K: int = 50,
    coords: Any = None,
    mask: Any = None,
    stack: bool = False,
    channels: str = "gray",
    spacing: Any = None,
    **model_kwargs: Any,
) -> HTFAResults:
    """Load ``data`` and fit TFA (one subject) or HTFA (several).

    Parameters
    ----------
    data, coords, mask, stack, channels, spacing
        As in `htfa.io.load`.
    K : int
        Number of factors.
    **model_kwargs
        Passed to `htfa.TFA` or `htfa.HTFA`; an unknown key raises ``TypeError``.
        One subject fits a TFA and several fit an HTFA, so HTFA-only kwargs
        (e.g. ``max_global_iter``) fail for a single subject.

    Returns
    -------
    HTFAResults

    Examples
    --------
    >>> import numpy as np
    >>> from htfa import fit
    >>> img = np.random.default_rng(0).random((6, 6, 6, 8)) + 1
    >>> fit(img, K=2, standardize=False).connectivity().shape
    (2, 2)
    """
    subjects = load(
        data, coords=coords, mask=mask, stack=stack, channels=channels, spacing=spacing
    )
    if len(subjects) == 1:
        tfa = TFA(K=K, **model_kwargs).fit(subjects[0].data, subjects[0].coords)
        return HTFAResults(tfa, subjects)
    htfa = HTFA(K=K, **model_kwargs).fit(
        [s.data for s in subjects], [s.coords for s in subjects]
    )
    return HTFAResults(htfa, subjects)
