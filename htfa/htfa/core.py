"""Core HTFA (Hierarchical Topographic Factor Analysis) implementation.

This module provides the main HTFA algorithm with support for modern ML backends
and performance optimizations.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from abc import ABC, abstractmethod

import numpy as np


class HTFABackend(ABC):
    """Abstract base class for HTFA backends."""

    @abstractmethod
    def array(self, data: Any) -> Any:
        """Create array from data."""
        pass

    @abstractmethod
    def zeros(self, shape: Tuple[int, ...], dtype: Any = None) -> Any:
        """Create zeros array."""
        pass

    @abstractmethod
    def ones(self, shape: Tuple[int, ...], dtype: Any = None) -> Any:
        """Create ones array."""
        pass

    @abstractmethod
    def random(self, shape: Tuple[int, ...], dtype: Any = None) -> Any:
        """Create random array."""
        pass

    @abstractmethod
    def matmul(self, a: Any, b: Any) -> Any:
        """Matrix multiplication."""
        pass

    @abstractmethod
    def transpose(self, a: Any, axes: Optional[Tuple[int, ...]] = None) -> Any:
        """Transpose array."""
        pass

    @abstractmethod
    def svd(self, a: Any, full_matrices: bool = True) -> Tuple[Any, Any, Any]:
        """Singular Value Decomposition."""
        pass

    @abstractmethod
    def norm(self, a: Any, axis: Optional[int] = None, keepdims: bool = False) -> Any:
        """Compute norm."""
        pass

    @abstractmethod
    def mean(self, a: Any, axis: Optional[int] = None, keepdims: bool = False) -> Any:
        """Compute mean."""
        pass

    @abstractmethod
    def to_numpy(self, a: Any) -> np.ndarray:
        """Convert to numpy array."""
        pass


class NumPyBackend(HTFABackend):
    """NumPy backend for HTFA."""

    def array(self, data: Any) -> np.ndarray:
        return np.array(data)

    def zeros(self, shape: Tuple[int, ...], dtype: Any = None) -> np.ndarray:
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape: Tuple[int, ...], dtype: Any = None) -> np.ndarray:
        return np.ones(shape, dtype=dtype)

    def random(self, shape: Tuple[int, ...], dtype: Any = None) -> np.ndarray:
        return np.random.random(shape).astype(dtype if dtype else np.float32)

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.matmul(a, b)

    def transpose(
        self, a: np.ndarray, axes: Optional[Tuple[int, ...]] = None
    ) -> np.ndarray:
        return np.transpose(a, axes)

    def svd(
        self, a: np.ndarray, full_matrices: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.linalg.svd(a, full_matrices=full_matrices)

    def norm(
        self, a: np.ndarray, axis: Optional[int] = None, keepdims: bool = False
    ) -> np.ndarray:
        return np.linalg.norm(a, axis=axis, keepdims=keepdims)

    def mean(
        self, a: np.ndarray, axis: Optional[int] = None, keepdims: bool = False
    ) -> np.ndarray:
        return np.mean(a, axis=axis, keepdims=keepdims)

    def to_numpy(self, a: np.ndarray) -> np.ndarray:
        return a


class HTFA:
    """Hierarchical Topographic Factor Analysis.

    A dimensionality reduction technique that learns hierarchical representations
    with topographic organization.

    Parameters
    ----------
    n_factors : int
        Number of factors to extract at each level.
    n_levels : int, default=2
        Number of hierarchical levels.
    regularization : float, default=0.01
        Regularization strength.
    max_iter : int, default=1000
        Maximum number of iterations for optimization.
    tol : float, default=1e-6
        Tolerance for convergence.
    backend : str or HTFABackend, default='numpy'
        Backend to use for computations ('numpy', 'jax', 'pytorch', or custom backend).
    random_state : int, optional
        Random state for reproducibility.
    """

    def __init__(
        self,
        n_factors: int,
        n_levels: int = 2,
        regularization: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-6,
        backend: Union[str, HTFABackend] = "numpy",
        random_state: Optional[int] = None,
    ):
        self.n_factors = n_factors
        self.n_levels = n_levels
        self.regularization = regularization
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        # Set up backend
        if isinstance(backend, str):
            self.backend = self._create_backend(backend)
        else:
            self.backend = backend

        # Initialize state
        self.is_fitted_ = False
        self.factors_ = None
        self.loadings_ = None
        self.reconstruction_error_ = None

        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)

    def _create_backend(self, backend_name: str) -> HTFABackend:
        """Create backend from string name."""
        if backend_name == "numpy":
            return NumPyBackend()
        elif backend_name == "jax":
            try:
                from .backends.jax_backend import JAXBackend

                return JAXBackend()
            except ImportError:
                raise ImportError(
                    "JAX backend not available. Install JAX with: pip install jax jaxlib"
                )
        elif backend_name == "pytorch":
            try:
                from .backends.pytorch_backend import PyTorchBackend

                return PyTorchBackend()
            except ImportError:
                raise ImportError(
                    "PyTorch backend not available. Install PyTorch: pip install torch"
                )
        else:
            raise ValueError(f"Unknown backend: {backend_name}")

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "HTFA":
        """Fit HTFA model to data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,), optional
            Target values (ignored for unsupervised learning).

        Returns
        -------
        self : HTFA
            Returns the instance itself.
        """
        X = self.backend.array(X)
        n_samples, n_features = X.shape

        # Initialize factors and loadings for each level
        self.factors_ = []
        self.loadings_ = []

        current_data = X

        for level in range(self.n_levels):
            # Set random seed if provided for reproducible initialization
            if self.random_state is not None:
                np.random.seed(self.random_state + level)  # Different seed per level

            # Initialize random factors and loadings
            factors = self.backend.random(
                (n_samples, self.n_factors),
                dtype=X.dtype if hasattr(X, "dtype") else np.float32,
            )
            loadings = self.backend.random(
                (self.n_factors, current_data.shape[1]),
                dtype=X.dtype if hasattr(X, "dtype") else np.float32,
            )

            # Optimize factors and loadings using alternating least squares
            prev_error = float("inf")

            for iteration in range(self.max_iter):
                # Update factors (fix loadings)
                factors = self._update_factors(current_data, loadings)

                # Update loadings (fix factors)
                loadings = self._update_loadings(current_data, factors)

                # Compute reconstruction error
                reconstruction = self.backend.matmul(factors, loadings)
                error = self.backend.norm(current_data - reconstruction)

                # Check for convergence
                if abs(prev_error - error) < self.tol:
                    break

                prev_error = error

            self.factors_.append(factors)
            self.loadings_.append(loadings)

            # Prepare data for next level (use factors as input)
            if level < self.n_levels - 1:
                current_data = factors

        # Compute final reconstruction error
        self._compute_reconstruction_error(X)
        self.is_fitted_ = True

        return self

    def _update_factors(self, X: Any, loadings: Any) -> Any:
        """Update factors using least squares with regularization."""
        # F = X @ L^T @ (L @ L^T + λI)^{-1}
        LLT = self.backend.matmul(loadings, self.backend.transpose(loadings))

        # Add regularization
        reg_term = self.regularization * self.backend.array(np.eye(LLT.shape[0]))
        LLT_reg = LLT + reg_term

        # Solve for factors
        XLT = self.backend.matmul(X, self.backend.transpose(loadings))

        # Use SVD for numerical stability (pseudo-inverse)
        U, s, Vt = self.backend.svd(LLT_reg, full_matrices=False)
        s_inv = 1.0 / (s + 1e-12)  # Add small value for numerical stability
        LLT_inv = self.backend.matmul(
            self.backend.matmul(
                self.backend.transpose(Vt), self.backend.array(np.diag(s_inv))
            ),
            self.backend.transpose(U),
        )

        factors = self.backend.matmul(XLT, LLT_inv)
        return factors

    def _update_loadings(self, X: Any, factors: Any) -> Any:
        """Update loadings using least squares with regularization."""
        # L = (F^T @ F + λI)^{-1} @ F^T @ X
        FTF = self.backend.matmul(self.backend.transpose(factors), factors)

        # Add regularization
        reg_term = self.regularization * self.backend.array(np.eye(FTF.shape[0]))
        FTF_reg = FTF + reg_term

        # Solve for loadings
        FTX = self.backend.matmul(self.backend.transpose(factors), X)

        # Use SVD for numerical stability
        U, s, Vt = self.backend.svd(FTF_reg, full_matrices=False)
        s_inv = 1.0 / (s + 1e-12)
        FTF_inv = self.backend.matmul(
            self.backend.matmul(
                self.backend.transpose(Vt), self.backend.array(np.diag(s_inv))
            ),
            self.backend.transpose(U),
        )

        loadings = self.backend.matmul(FTF_inv, FTX)
        return loadings

    def _compute_reconstruction_error(self, X: Any) -> None:
        """Compute reconstruction error."""
        # Direct reconstruction from fitted factors and loadings
        # Start from bottom level and work upward
        reconstruction = self.factors_[0]  # Bottom level factors
        reconstruction = self.backend.matmul(reconstruction, self.loadings_[0])

        error = self.backend.norm(X - reconstruction)
        self.reconstruction_error_ = self.backend.to_numpy(error).item()

    def transform(self, X: np.ndarray, inverse: bool = False) -> np.ndarray:
        """Transform data using fitted HTFA model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.
        inverse : bool, default=False
            If True, perform inverse transform (reconstruction from fitted factors).

        Returns
        -------
        X_transformed : ndarray
            Transformed data.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before transform.")

        if inverse:
            # Reconstruct original data from fitted factors
            # Use bottom-level factors to reconstruct original data
            reconstruction = self.factors_[0]  # Bottom level factors
            reconstruction = self.backend.matmul(reconstruction, self.loadings_[0])
            return self.backend.to_numpy(reconstruction)
        else:
            # Transform new data to hierarchical factors
            X = self.backend.array(X)
            current_data = X

            for level in range(len(self.loadings_)):
                # Project data onto loadings to get factors
                factors = self._update_factors(current_data, self.loadings_[level])
                current_data = factors

            return self.backend.to_numpy(current_data)  # Return top-level factors

    def fit_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Fit model and transform data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,), optional
            Target values (ignored).

        Returns
        -------
        X_transformed : ndarray
            Transformed data.
        """
        return self.fit(X, y).transform(X)

    def get_reconstruction_error(self) -> float:
        """Get reconstruction error."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted first.")
        return self.reconstruction_error_

    def get_factors_all_levels(self) -> List[np.ndarray]:
        """Get factors from all hierarchical levels."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted first.")
        return [self.backend.to_numpy(factors) for factors in self.factors_]

    def get_loadings_all_levels(self) -> List[np.ndarray]:
        """Get loadings from all hierarchical levels."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted first.")
        return [self.backend.to_numpy(loadings) for loadings in self.loadings_]
