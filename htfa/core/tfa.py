"""Topographic Factor Analysis (TFA) implementation.

This module provides a standalone implementation of Topographic Factor Analysis,
which serves as the base for Hierarchical TFA.
"""

from typing import Optional, Tuple, Union

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import distance
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
        weight_method: str = "rr",
        regularization: float = 0.01,
        nlss_method: str = "trf",
        nlss_loss: str = "soft_l1",
        upper_ratio: float = 1.8,
        lower_ratio: float = 0.02,
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
        self.weight_method = weight_method
        self.regularization = regularization
        self.nlss_method = nlss_method
        self.nlss_loss = nlss_loss
        self.upper_ratio = upper_ratio
        self.lower_ratio = lower_ratio

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
        """Run the main optimization loop using coordinate descent."""
        if coords is None:
            raise ValueError("Spatial coordinates required for TFA optimization")

        n_voxels, n_timepoints = X.shape

        for iteration in range(self.max_iter):
            # Compute spatial factors from centers and widths
            self.factors_ = self._compute_factors(coords)

            # Update weights using ridge regression or OLS
            self.weights_ = self._compute_weights(X, self.factors_)

            # Update centers and widths using least squares
            old_centers = self.centers_.copy()
            old_widths = self.widths_.copy()

            self.centers_, self.widths_ = self._update_centers_widths(
                X, coords, self.factors_, self.weights_
            )

            # Check convergence
            center_diff = np.max(np.abs(self.centers_ - old_centers))
            width_diff = np.max(np.abs(self.widths_ - old_widths))

            if self.verbose and iteration % 10 == 0:
                reconstruction = self.factors_.T @ self.weights_.T
                mse = np.mean((X - reconstruction) ** 2)
                print(
                    f"Iteration {iteration}: MSE={mse:.6f}, center_diff={center_diff:.6f}"
                )

            if center_diff < self.tol and width_diff < self.tol:
                if self.verbose:
                    print(f"TFA converged at iteration {iteration}")
                break

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

    def _compute_factors(self, coords: np.ndarray) -> np.ndarray:
        """Compute spatial factors using RBF (Radial Basis Function).

        Parameters
        ----------
        coords : np.ndarray
            Spatial coordinates of shape (n_voxels, n_dims).

        Returns
        -------
        factors : np.ndarray
            Spatial factors of shape (K, n_voxels).
        """
        n_voxels = coords.shape[0]
        factors = np.zeros((self.K, n_voxels))

        for k in range(self.K):
            # Compute RBF: exp(-||x - center||^2 / (2 * width^2))
            distances = np.linalg.norm(coords - self.centers_[k], axis=1)
            factors[k] = np.exp(-(distances**2) / (2 * self.widths_[k] ** 2))

        return factors

    def _compute_weights(self, X: np.ndarray, factors: np.ndarray) -> np.ndarray:
        """Compute weight matrix using ridge regression or OLS.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (n_voxels, n_timepoints).
        factors : np.ndarray
            Spatial factors of shape (K, n_voxels).

        Returns
        -------
        weights : np.ndarray
            Weight matrix of shape (n_timepoints, K).
        """
        # Solve: W = (F @ F.T + lambda*I)^-1 @ F @ X.T
        FTF = factors @ factors.T

        if self.weight_method == "rr":
            # Ridge regression
            beta = self.regularization * np.var(X)
            reg_term = beta * np.eye(self.K)
            weights = np.linalg.solve(FTF + reg_term, factors @ X).T
        else:
            # Ordinary least squares
            weights = np.linalg.solve(FTF, factors @ X).T

        return weights

    def _update_centers_widths(
        self,
        X: np.ndarray,
        coords: np.ndarray,
        factors: np.ndarray,
        weights: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Update centers and widths using nonlinear least squares.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (n_voxels, n_timepoints).
        coords : np.ndarray
            Spatial coordinates of shape (n_voxels, n_dims).
        factors : np.ndarray
            Current spatial factors of shape (K, n_voxels).
        weights : np.ndarray
            Current weight matrix of shape (n_timepoints, K).

        Returns
        -------
        centers : np.ndarray
            Updated centers of shape (K, n_dims).
        widths : np.ndarray
            Updated widths of shape (K,).
        """
        n_dims = coords.shape[1]

        # Define bounds first
        coord_min = np.min(coords, axis=0)
        coord_max = np.max(coords, axis=0)
        max_sigma = 2.0 * np.max(np.std(coords, axis=0)) ** 2

        lower_bounds = np.concatenate(
            [np.tile(coord_min, self.K), np.full(self.K, self.lower_ratio * max_sigma)]
        )

        upper_bounds = np.concatenate(
            [np.tile(coord_max, self.K), np.full(self.K, self.upper_ratio * max_sigma)]
        )

        # Clip current parameters to be within bounds
        centers_clipped = np.clip(self.centers_, coord_min, coord_max)
        widths_clipped = np.clip(
            self.widths_, self.lower_ratio * max_sigma, self.upper_ratio * max_sigma
        )

        # Flatten clipped parameters
        init_params = np.concatenate([centers_clipped.flatten(), widths_clipped])

        # Define residual function
        def residual(params):
            centers = params[: self.K * n_dims].reshape(self.K, n_dims)
            widths = params[self.K * n_dims :]

            # Compute factors with new parameters
            new_factors = np.zeros_like(factors)
            for k in range(self.K):
                distances = np.linalg.norm(coords - centers[k], axis=1)
                new_factors[k] = np.exp(-(distances**2) / (2 * widths[k] ** 2))

            # Compute reconstruction error
            reconstruction = new_factors.T @ weights.T
            return (X - reconstruction).flatten()

        # Optimize using least squares
        result = least_squares(
            residual,
            init_params,
            bounds=(lower_bounds, upper_bounds),
            method=self.nlss_method,
            loss=self.nlss_loss,
            max_nfev=100,  # Limit function evaluations for efficiency
        )

        # Extract updated parameters
        centers = result.x[: self.K * n_dims].reshape(self.K, n_dims)
        widths = result.x[self.K * n_dims :]

        return centers, widths
