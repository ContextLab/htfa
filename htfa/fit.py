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
    groups: Any = None,
    subject_ids: Any = None,
    **model_kwargs: Any,
) -> HTFAResults:
    """Load ``data`` and fit TFA (one subject) or HTFA (several).

    Parameters
    ----------
    data, coords, mask, stack, channels, spacing
        As in `htfa.io.load`.
    K : int
        Number of factors.
    groups, subject_ids : optional
        Nested groups of subjects and subject identifiers, passed to
        `htfa.HTFA.fit` (several subjects only); ``groups`` needs
        ``template_update='fixed'`` or ``'empirical'``.
    **model_kwargs
        Passed to `htfa.TFA` or `htfa.HTFA`; an unknown key raises ``TypeError``.
        One subject fits a TFA and several fit an HTFA, so HTFA-only kwargs
        (e.g. ``max_global_iter``) fail for a single subject. Both take
        ``kernel`` and ``n_components`` (factor shapes and mixtures).

    Returns
    -------
    HTFAResults

    Examples
    --------
    >>> import numpy as np
    >>> from htfa import fit
    >>> img = np.random.default_rng(0).random((6, 6, 6, 8)) + 1
    >>> res = fit(img, K=2, standardize=False, init="power", max_iter=5)
    >>> res.connectivity().shape
    (2, 2)
    """
    subjects = load(
        data, coords=coords, mask=mask, stack=stack, channels=channels, spacing=spacing
    )
    if len(subjects) == 1:
        if groups is not None or subject_ids is not None:
            raise ValueError("groups and subject_ids need several subjects")
        tfa = TFA(K=K, **model_kwargs).fit(subjects[0].data, subjects[0].coords)
        return HTFAResults(tfa, subjects)
    htfa = HTFA(K=K, **model_kwargs).fit(
        [s.data for s in subjects],
        [s.coords for s in subjects],
        groups=groups,
        subject_ids=subject_ids,
    )
    return HTFAResults(htfa, subjects)
