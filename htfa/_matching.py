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
    """
    cost = cdist(true_centers, est_centers, metric="sqeuclidean")
    rows, cols = linear_sum_assignment(cost)
    perm = np.empty(len(rows), dtype=np.intp)
    perm[rows] = cols
    return perm
