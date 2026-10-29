"""Main entry point: load data and fit TFA or HTFA."""

from __future__ import annotations

import os
from typing import Any

from .groups import GroupTree
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
        ``template_update='fixed'`` or ``'empirical'``. With ``groups=None``,
        nested folders supply group paths automatically and the template update
        defaults to ``'fixed'``. Explicit groups override folder metadata;
        ``groups=False`` disables inference. A single subject still fits TFA.
        A ``GroupTree.from_paths`` tree supports unequal hierarchy depths.
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
    if groups is False:
        groups = None
        infer_groups = False
    else:
        infer_groups = groups is None
    if len(subjects) == 1:
        if groups is not None or subject_ids is not None:
            raise ValueError("groups and subject_ids need several subjects")
        tfa = TFA(K=K, **model_kwargs).fit(subjects[0].data, subjects[0].coords)
        return HTFAResults(tfa, subjects)
    if infer_groups and any(s.group_path for s in subjects):
        directories: dict[tuple[str, ...], str] = {}
        sources = set()
        for subject in subjects:
            if subject.source is None:
                continue
            source = os.path.realpath(subject.source)
            if source in sources:
                raise ValueError(
                    "folder hierarchy contains duplicate input files; select each "
                    "subject once or supply explicit groups"
                )
            sources.add(source)
            if not subject.group_path:
                continue
            directory = os.path.normcase(
                os.path.abspath(os.path.dirname(subject.source))
            )
            # Every prefix is a template node, even when that folder has no
            # files directly inside it. Check ancestors as well as leaf groups
            # so two distinct roots named "control" cannot silently merge.
            for depth in range(len(subject.group_path), 0, -1):
                path = subject.group_path[:depth]
                previous = directories.setdefault(path, directory)
                if previous != directory:
                    raise ValueError(
                        "folder names identify different directories with the same "
                        "group path; supply explicit groups or groups=False"
                    )
                directory = os.path.dirname(directory)
        groups = GroupTree.from_paths([s.group_path for s in subjects])
        model_kwargs.setdefault("template_update", "fixed")
    htfa = HTFA(K=K, **model_kwargs).fit(
        [s.data for s in subjects],
        [s.coords for s in subjects],
        groups=groups,
        subject_ids=subject_ids,
    )
    return HTFAResults(htfa, subjects)
