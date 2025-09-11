"""Hierarchical Topographic Factor Analysis (HTFA) implementation.

This module provides a standalone implementation of Hierarchical Topographic Factor Analysis,
based on the BrainIAK implementation but with minimal dependencies.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.base import BaseEstimator

from htfa.core.tfa import TFA
from htfa.validation import validate_random_state
from htfa.backend_base import HTFABackend
from htfa.backends.numpy_backend import NumPyBackend


class HTFA(BaseEstimator):
    """Hierarchical Topographic Factor Analysis (HTFA).

    HTFA extends TFA to analyze multi-subject neuroimaging data by:
    1. Estimating a global template across subjects
    2. Finding spatial factors (F) and subject-specific weight matrices (W)
    3. Using iterative optimization with MAP estimation

    Parameters
    ----------
    K : int
        Number of factors to extract.
    max_num_voxel : Optional[int]
        Maximum number of voxels to use for analysis.
    max_num_tr : Optional[int]
        Maximum number of time points to use per subject.
    max_global_iter : int
        Maximum number of global iterations across all subjects.
    max_local_iter : int
        Maximum number of local iterations per subject.
    tol : float
        Convergence tolerance.
    verbose : bool
        Whether to print progress information.
    n_levels : int, default=2
        Number of hierarchical levels in the model.
    backend : str, HTFABackend, or None, default=None
        Computational backend to use ('numpy', 'jax', 'pytorch', custom backend, or None for numpy).
    random_state : int, RandomState instance or None, default=None
        Random state for reproducible results.
    max_iter : int, default=100
        Maximum number of iterations for optimization.
    """

    def __init__(
        self,
        K: int = 10,
        max_num_voxel: Optional[int] = None,
        max_num_tr: Optional[int] = None,
        max_global_iter: int = 10,
        max_local_iter: int = 50,
        tol: float = 1e-6,
        verbose: bool = False,
        n_levels: int = 2,
        backend: Optional[Union[str, HTFABackend]] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        max_iter: int = 100,
        n_factors: Optional[int] = None,
    ):
        # Allow n_factors as alias for K (backward compatibility)
        if n_factors is not None:
            K = n_factors
        
        # Store all parameters as instance attributes
        self.K = K
        self.max_num_voxel = max_num_voxel
        self.max_num_tr = max_num_tr
        self.max_global_iter = max_global_iter
        self.max_local_iter = max_local_iter
        self.tol = tol
        self.verbose = verbose
        self.n_levels = n_levels
        # Validate and store random_state
        self.random_state = validate_random_state(random_state)
        self.max_iter = max_iter

        # Initialize backend
        if isinstance(backend, str):
            self.backend = self._create_backend(backend)
        elif backend is None:
            self.backend = NumPyBackend()
        else:
            self.backend = backend

        # Fitted parameters
        self.global_template_: Optional[np.ndarray] = None
        self.factors_: Optional[List[np.ndarray]] = None
        self.weights_: Optional[List[np.ndarray]] = None
        self.subject_templates_: Optional[List[np.ndarray]] = None
        self.convergence_info_: Optional[Dict[str, Any]] = None

    def _create_backend(self, backend_name: str) -> HTFABackend:
        """Create backend from string name."""
        if backend_name == "numpy":
            return NumPyBackend()
        elif backend_name == "jax":
            try:
                from htfa.backends.jax_backend import JAXBackend
                return JAXBackend()
            except ImportError:
                raise ImportError(
                    "JAX backend not available. Install JAX with: pip install jax jaxlib"
                )
        elif backend_name == "pytorch":
            try:
                from htfa.backends.pytorch_backend import PyTorchBackend
                return PyTorchBackend()
            except ImportError:
                raise ImportError(
                    "PyTorch backend not available. Install PyTorch: pip install torch"
                )
        else:
            raise ValueError(f"Unknown backend: {backend_name}")

    def fit(
        self, X: List[np.ndarray], coords: Optional[List[np.ndarray]] = None
    ) -> "HTFA":
        """Fit the HTFA model to multi-subject data.

        Parameters
        ----------
        X : List[np.ndarray]
            List of data matrices, each of shape (n_voxels, n_timepoints).
            One matrix per subject.
        coords : Optional[List[np.ndarray]]
            List of spatial coordinates, each of shape (n_voxels, n_dims).
            One coordinate array per subject. If None, uses voxel indices.

        Returns
        -------
        self : HTFA
            The fitted estimator.
        """
        # Input validation
        if not isinstance(X, list):
            raise ValueError("X must be a list of arrays (one per subject)")

        n_subjects = len(X)
        if n_subjects == 0:
            raise ValueError("X cannot be empty")

        # Validate coordinate data if provided
        if coords is not None:
            if len(coords) != n_subjects:
                raise ValueError("coords must have same length as X")

        if self.verbose:
            print(f"Fitting HTFA with {n_subjects} subjects and K={self.K} factors")

        # Initialize subject-specific TFA models
        self._initialize_subject_models(X, coords)

        # Run hierarchical optimization
        self._hierarchical_optimization(X, coords)

        return self

    def _initialize_subject_models(
        self, X: List[np.ndarray], coords: Optional[List[np.ndarray]]
    ) -> None:
        """Initialize TFA models for each subject."""
        self.subject_models_ = []

        for i, subject_data in enumerate(X):
            subject_coords = coords[i] if coords is not None else None

            # Create and fit individual TFA model
            tfa = TFA(
                K=self.K,
                max_num_voxel=self.max_num_voxel,
                max_num_tr=self.max_num_tr,
                max_iter=self.max_local_iter,
                tol=self.tol,
                verbose=False,  # Suppress individual subject verbose output
                random_state=self.random_state,
                backend=self.backend,  # Pass backend to TFA
            )

            tfa.fit(subject_data, subject_coords)
            self.subject_models_.append(tfa)

        if self.verbose:
            print("Individual subject TFA models initialized")

    def _hierarchical_optimization(
        self, X: List[np.ndarray], coords: Optional[List[np.ndarray]]
    ) -> None:
        """Run the hierarchical optimization algorithm."""
        n_subjects = len(X)

        # Initialize global template
        self._compute_global_template()

        # Main hierarchical optimization loop
        converged = False
        for global_iter in range(self.max_global_iter):
            if self.verbose:
                print(f"Global iteration {global_iter + 1}/{self.max_global_iter}")

            # Update subject-specific models based on global template
            old_template = (
                self.global_template_.copy()
                if self.global_template_ is not None
                else None
            )

            for i, (subject_data, tfa_model) in enumerate(zip(X, self.subject_models_)):
                subject_coords = coords[i] if coords is not None else None
                self._update_subject_model(
                    tfa_model, subject_data, subject_coords, global_iter
                )

            # Update global template
            self._compute_global_template()

            # Check convergence
            if self._check_convergence(old_template, self.global_template_):
                converged = True
                self.convergence_info_ = {
                    'converged': True,
                    'n_iterations': global_iter + 1,
                    'subject_convergence': [m.convergence_info_ for m in self.subject_models_ if hasattr(m, 'convergence_info_')]
                }
                if self.verbose:
                    print(f"Converged after {global_iter + 1} global iterations")
                break
        
        # Record convergence status if not converged
        if not converged:
            self.convergence_info_ = {
                'converged': False,
                'n_iterations': self.max_global_iter,
                'subject_convergence': [m.convergence_info_ for m in self.subject_models_ if hasattr(m, 'convergence_info_')]
            }
            warnings.warn(
                f"HTFA did not converge after {self.max_global_iter} global iterations. "
                f"Consider increasing max_global_iter or adjusting tolerance.",
                UserWarning
            )

        # Extract final parameters
        self._extract_final_parameters()

    def _compute_global_template(self) -> None:
        """Compute the global template from all subject models using MAP estimation."""
        if not hasattr(self, "subject_models_"):
            return

        # Collect centers and widths from all subjects
        all_centers = []
        all_widths = []

        for model in self.subject_models_:
            if hasattr(model, "centers_") and model.centers_ is not None:
                all_centers.append(model.centers_)
                all_widths.append(model.widths_)

        if len(all_centers) > 0:
            # Stack all subject parameters
            all_centers = np.array(all_centers)  # Shape: (n_subjects, K, n_dims)
            all_widths = np.array(all_widths)  # Shape: (n_subjects, K)

            # Compute mean and variance for hierarchical prior
            self.template_centers_ = np.mean(all_centers, axis=0)
            self.template_widths_ = np.mean(all_widths, axis=0)

            # Compute covariance for centers (simplified - using identity scaled by variance)
            centers_variance = np.var(all_centers, axis=0)
            self.template_centers_cov_ = np.mean(centers_variance) * np.eye(
                all_centers.shape[2]
            )

            # Compute variance for widths
            self.template_widths_var_ = np.var(all_widths, axis=0)

            # Store as global template
            n_dims = self.template_centers_.shape[1]
            self.global_template_ = {
                "centers": self.template_centers_,
                "widths": self.template_widths_,
                "centers_cov": self.template_centers_cov_,
                "widths_var": self.template_widths_var_,
            }
        else:
            # Initialize with default template
            self.global_template_ = None

    def _update_subject_model(
        self,
        tfa_model: TFA,
        subject_data: np.ndarray,
        subject_coords: Optional[np.ndarray],
        global_iter: int,
    ) -> None:
        """Update a single subject's TFA model based on global template.

        This implements the MAP estimation with hierarchical prior from the global template.
        """
        if self.global_template_ is None or subject_coords is None:
            # No template yet, just run standard TFA
            tfa_model.fit(subject_data, subject_coords)
            return

        # Extract template parameters
        template_centers = self.global_template_["centers"]
        template_widths = self.global_template_["widths"]

        # Align subject's factors with template using Hungarian algorithm
        if hasattr(tfa_model, "centers_") and tfa_model.centers_ is not None:
            # Compute cost matrix based on center distances
            cost_matrix = np.zeros((self.K, self.K))
            for i in range(self.K):
                for j in range(self.K):
                    cost_matrix[i, j] = np.linalg.norm(
                        tfa_model.centers_[i] - template_centers[j]
                    )

            # Find optimal assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Reorder subject's parameters to match template
            tfa_model.centers_ = tfa_model.centers_[col_ind]
            tfa_model.widths_ = tfa_model.widths_[col_ind]

        # Re-fit with template as prior (simplified version)
        # In full implementation, this would modify the optimization objective
        # to include the prior terms
        tfa_model.centers_ = 0.7 * tfa_model.centers_ + 0.3 * template_centers
        tfa_model.widths_ = 0.7 * tfa_model.widths_ + 0.3 * template_widths

        # Run a few iterations of optimization with the prior
        for _ in range(min(10, self.max_local_iter)):
            tfa_model.factors_ = tfa_model._compute_factors(subject_coords)
            tfa_model.weights_ = tfa_model._compute_weights(
                subject_data, tfa_model.factors_
            )

    def _check_convergence(
        self,
        old_template: Union[dict, None],
        new_template: Union[dict, None],
    ) -> bool:
        """Check if global template has converged."""
        if old_template is None or new_template is None:
            return False

        # Handle dict template structure
        if isinstance(old_template, dict) and isinstance(new_template, dict):
            center_diff = np.linalg.norm(
                new_template["centers"] - old_template["centers"]
            )
            width_diff = np.linalg.norm(new_template["widths"] - old_template["widths"])
            total_diff = center_diff + width_diff
        else:
            # Fallback for array templates
            total_diff = np.linalg.norm(new_template - old_template)

        return total_diff < self.tol

    def _extract_final_parameters(self) -> None:
        """Extract final fitted parameters from subject models."""
        self.factors_ = [model.get_factors() for model in self.subject_models_]
        self.weights_ = [model.get_weights() for model in self.subject_models_]

        # Compute subject-specific templates
        self.subject_templates_ = []
        for factors, weights in zip(self.factors_, self.weights_):
            if factors is not None and weights is not None:
                template = weights @ factors
                self.subject_templates_.append(template)
            else:
                self.subject_templates_.append(None)

    def get_global_template(self) -> Optional[np.ndarray]:
        """Get the global template.

        Returns
        -------
        template : np.ndarray or None
            Global template of shape (K, n_voxels).
        """
        return self.global_template_

    def get_subject_factors(self, subject_idx: int) -> Optional[np.ndarray]:
        """Get spatial factors for a specific subject.

        Parameters
        ----------
        subject_idx : int
            Subject index.

        Returns
        -------
        factors : np.ndarray or None
            Spatial factors of shape (K, n_voxels).
        """
        if self.factors_ is None or subject_idx >= len(self.factors_):
            return None
        return self.factors_[subject_idx]

    def get_subject_weights(self, subject_idx: int) -> Optional[np.ndarray]:
        """Get weight matrix for a specific subject.

        Parameters
        ----------
        subject_idx : int
            Subject index.

        Returns
        -------
        weights : np.ndarray or None
            Weight matrix of shape (n_timepoints, K).
        """
        if self.weights_ is None or subject_idx >= len(self.weights_):
            return None
        return self.weights_[subject_idx]
