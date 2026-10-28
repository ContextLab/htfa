"""Hungarian matching of factors by their centers."""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

Array = NDArray[np.float64]


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

    Examples
    --------
    >>> import numpy as np
    >>> from htfa.synthetic import match_factors
    >>> true = np.array([[0.0, 0.0], [5.0, 5.0]])
    >>> est = np.array([[5.1, 4.9], [0.1, -0.1]])
    >>> match_factors(true, est)
    array([1, 0])
    """
    cost = cdist(true_centers, est_centers, metric="sqeuclidean")
    rows, cols = linear_sum_assignment(cost)
    perm = np.empty(len(rows), dtype=np.intp)
    perm[rows] = cols
    return perm


def match_components(template_centers: Array, est_centers: Array) -> NDArray[np.intp]:
    """Match each factor's components to a template's, by their centers.

    Parameters
    ----------
    template_centers, est_centers : ndarray, shape (K, M, D)
        Component centers of K factors with M components each, the factors
        already matched (`match_factors`).

    Returns
    -------
    perm : ndarray of int, shape (K, M)
        ``est_centers[k, perm[k]]`` corresponds to ``template_centers[k]``
        (Hungarian, per factor).

    Examples
    --------
    >>> import numpy as np
    >>> from htfa._matching import match_components
    >>> t = np.array([[[0.0, 0.0], [5.0, 5.0]]])
    >>> match_components(t, t[:, ::-1])
    array([[1, 0]])
    """
    return np.stack(
        [
            match_factors(t, e)
            for t, e in zip(template_centers, est_centers, strict=True)
        ]
    )
