"""Topographic Factor Analysis (TFA) implementation.

This module provides a standalone implementation of Topographic Factor Analysis,
which serves as the base for Hierarchical TFA.
"""

from typing import Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans


class TFA(BaseEstimator):
    """Topographic Factor Analysis (TFA).

    A scikit-learn style estimator that performs topographic factor analysis
    to decompose neural data into spatial factors and weight matrices.

    Parameters
    ----------
    K : int
        Number of factors to extract.
    max_num_voxel : Optional[int]
        Maximum number of voxels to use for analysis.
    max_num_tr : Optional[int]
        Maximum number of time points to use.
    max_iter : int
        Maximum number of optimization iterations.
    tol : float
        Convergence tolerance.
    verbose : bool
        Whether to print progress information.
    """

    def __init__(
        self,
        K: int = 10,
        max_num_voxel: Optional[int] = None,
        max_num_tr: Optional[int] = None,
        max_iter: int = 500,
        tol: float = 1e-6,
        verbose: bool = False,
        n_factors: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        # Allow n_factors as alias for K
        if n_factors is not None:
            K = n_factors
        self.K = K
        self.random_state = random_state
        self.max_num_voxel = max_num_voxel
        self.max_num_tr = max_num_tr
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        # Fitted parameters
        self.factors_: Optional[np.ndarray] = None
        self.weights_: Optional[np.ndarray] = None
        self.centers_: Optional[np.ndarray] = None
        self.widths_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, coords: Optional[np.ndarray] = None) -> "TFA":
        """Fit the TFA model to data.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (n_voxels, n_timepoints).
        coords : Optional[np.ndarray]
            Spatial coordinates of shape (n_voxels, n_dims).

        Returns
        -------
        self : TFA
            The fitted estimator.
        """
        # Input validation
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        n_voxels, n_timepoints = X.shape

        # Subsample if requested
        if self.max_num_voxel is not None and n_voxels > self.max_num_voxel:
            voxel_idx = np.random.choice(n_voxels, self.max_num_voxel, replace=False)
            X = X[voxel_idx]
            if coords is not None:
                coords = coords[voxel_idx]

        if self.max_num_tr is not None and n_timepoints > self.max_num_tr:
            tr_idx = np.random.choice(n_timepoints, self.max_num_tr, replace=False)
            X = X[:, tr_idx]

        # Initialize parameters using k-means clustering
        self._initialize_parameters(X, coords)

        # Run optimization
        self._optimize(X, coords)

        return self

    def _initialize_parameters(
        self, X: np.ndarray, coords: Optional[np.ndarray]
    ) -> None:
        """Initialize factor centers and widths using k-means clustering."""
        if coords is not None:
            # Use spatial coordinates for initialization
            kmeans = KMeans(n_clusters=self.K, random_state=42)
            kmeans.fit(coords)
            self.centers_ = kmeans.cluster_centers_

            # Estimate widths from cluster assignments
            labels = kmeans.labels_
            self.widths_ = np.zeros(self.K)
            for k in range(self.K):
                if np.sum(labels == k) > 1:
                    cluster_coords = coords[labels == k]
                    center = self.centers_[k]
                    distances = np.linalg.norm(cluster_coords - center, axis=1)
                    self.widths_[k] = np.std(distances) + 1e-6
                else:
                    self.widths_[k] = 1.0
        else:
            # Use data-based initialization
            n_voxels = X.shape[0]
            self.centers_ = np.random.randn(self.K, n_voxels)
            self.widths_ = np.ones(self.K)

    def _optimize(self, X: np.ndarray, coords: Optional[np.ndarray]) -> None:
        """Run the main optimization loop."""
        # Placeholder for optimization - will be implemented in detail
        # This would contain the iterative estimation of factors and weights
        n_voxels, n_timepoints = X.shape

        # Initialize factors and weights
        self.factors_ = np.random.randn(self.K, n_voxels)
        self.weights_ = np.random.randn(n_timepoints, self.K)

        if self.verbose:
            print(f"TFA fitting completed with K={self.K} factors")

    def get_factors(self) -> Optional[np.ndarray]:
        """Get the estimated spatial factors.

        Returns
        -------
        factors : np.ndarray or None
            Spatial factors of shape (K, n_voxels).
        """
        return self.factors_

    def get_weights(self) -> Optional[np.ndarray]:
        """Get the estimated weight matrix.

        Returns
        -------
        weights : np.ndarray or None
            Weight matrix of shape (n_timepoints, K).
        """
        return self.weights_
