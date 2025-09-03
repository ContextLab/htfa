"""Hierarchical Topographic Factor Analysis (HTFA) implementation.

This module provides a standalone implementation of Hierarchical Topographic Factor Analysis,
based on the BrainIAK implementation but with minimal dependencies.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator

from htfa.core.tfa import TFA


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
        n_factors: Optional[int] = None,
        max_iter: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        # Allow n_factors as alias for K
        if n_factors is not None:
            K = n_factors
        # Allow max_iter as alias for max_local_iter
        if max_iter is not None:
            max_local_iter = max_iter
        self.K = K
        self.random_state = random_state
        self.max_num_voxel = max_num_voxel
        self.max_num_tr = max_num_tr
        self.max_global_iter = max_global_iter
        self.max_local_iter = max_local_iter
        self.tol = tol
        self.verbose = verbose

        # Fitted parameters
        self.global_template_: Optional[np.ndarray] = None
        self.factors_: Optional[List[np.ndarray]] = None
        self.weights_: Optional[List[np.ndarray]] = None
        self.subject_templates_: Optional[List[np.ndarray]] = None

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
        for global_iter in range(self.max_global_iter):
            if self.verbose:
                print(f"Global iteration {global_iter + 1}/{self.max_global_iter}")

            # Update subject-specific models based on global template
            old_template = self.global_template_.copy()

            for i, (subject_data, tfa_model) in enumerate(zip(X, self.subject_models_)):
                subject_coords = coords[i] if coords is not None else None
                self._update_subject_model(
                    tfa_model, subject_data, subject_coords, global_iter
                )

            # Update global template
            self._compute_global_template()

            # Check convergence
            if self._check_convergence(old_template, self.global_template_):
                if self.verbose:
                    print(f"Converged after {global_iter + 1} global iterations")
                break

        # Extract final parameters
        self._extract_final_parameters()

    def _compute_global_template(self) -> None:
        """Compute the global template from all subject models."""
        if not hasattr(self, "subject_models_"):
            return

        # Average factors across subjects (placeholder implementation)
        subject_factors = [model.get_factors() for model in self.subject_models_]

        if all(f is not None for f in subject_factors):
            self.global_template_ = np.mean(subject_factors, axis=0)
        else:
            # Initialize with random template if factors not available
            n_voxels = (
                self.subject_models_[0].factors_.shape[1]
                if self.subject_models_[0].factors_ is not None
                else 100
            )
            self.global_template_ = np.random.randn(self.K, n_voxels)

    def _update_subject_model(
        self,
        tfa_model: TFA,
        subject_data: np.ndarray,
        subject_coords: Optional[np.ndarray],
        global_iter: int,
    ) -> None:
        """Update a single subject's TFA model based on global template."""
        # Placeholder for subject-specific update
        # In full implementation, this would incorporate global template information
        # into the subject-specific optimization
        pass

    def _check_convergence(
        self, old_template: np.ndarray, new_template: np.ndarray
    ) -> bool:
        """Check if global template has converged."""
        if old_template is None or new_template is None:
            return False

        diff = np.linalg.norm(new_template - old_template)
        return diff < self.tol

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
