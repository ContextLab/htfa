"""HTFAResults class for comprehensive result storage and visualization.

This module defines the HTFAResults class that provides an intuitive interface
for accessing HTFA analysis results, including built-in visualization methods
and NIfTI reconstruction capabilities.
"""

from typing import Any, Dict, List, Optional, Union

import warnings

import numpy as np

# Placeholder imports - will be implemented in Phase 2
# import nibabel as nib
# import pandas as pd
# import matplotlib.pyplot as plt
# from nilearn import plotting


class HTFAResults:
    """Comprehensive results from HTFA analysis with built-in visualization and export.

    This class provides an intuitive interface to HTFA analysis results, including
    global and subject-specific parameters, built-in plotting methods, and the
    ability to reconstruct NIfTI images from the fitted factors.

    Attributes
    ----------
    global_template : np.ndarray
        Global spatial factors of shape (K, n_voxels).
    subject_factors : List[np.ndarray]
        Per-subject spatial factors, each of shape (K, n_voxels).
    subject_weights : List[np.ndarray]
        Per-subject temporal weights, each of shape (n_timepoints, K).
    bids_info : dict
        Information about the original BIDS dataset.
    preprocessing : dict
        Applied preprocessing steps and parameters.
    model_params : dict
        Model hyperparameters used for fitting.
    fit_info : dict
        Convergence and optimization details.
    template_img : nibabel.Nifti1Image
        Template image for spatial reconstruction.
    brain_mask : np.ndarray
        Brain mask used for analysis.
    coordinates : np.ndarray
        Voxel coordinates in template space.
    """

    def __init__(
        self,
        global_template: np.ndarray,
        subject_factors: List[np.ndarray],
        subject_weights: List[np.ndarray],
        bids_info: Dict[str, Any],
        preprocessing: Dict[str, Any],
        model_params: Dict[str, Any],
        fit_info: Dict[str, Any],
        template_img: Optional["nib.Nifti1Image"] = None,
        brain_mask: Optional[np.ndarray] = None,
        coordinates: Optional[np.ndarray] = None,
    ):
        self.global_template = global_template
        self.subject_factors = subject_factors
        self.subject_weights = subject_weights
        self.bids_info = bids_info
        self.preprocessing = preprocessing
        self.model_params = model_params
        self.fit_info = fit_info
        self.template_img = template_img
        self.brain_mask = brain_mask
        self.coordinates = coordinates

        # Placeholder warning
        warnings.warn(
            "HTFAResults class is not fully implemented yet. "
            "Full functionality will be completed in Phase 2.",
            UserWarning,
        )

    def plot_global_factors(
        self,
        factors: Optional[List[int]] = None,
        display_mode: str = "mosaic",
        colorbar: bool = True,
        threshold: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Plot global spatial factors as brain maps.

        Parameters
        ----------
        factors : List[int], optional
            Which factors to plot. If None, plots all factors.
        display_mode : str, default='mosaic'
            Display mode for nilearn plotting.
        colorbar : bool, default=True
            Whether to show colorbar.
        threshold : float, optional
            Threshold for displaying activation.
        **kwargs
            Additional arguments passed to nilearn plotting functions.
        """
        raise NotImplementedError("Will be implemented in Phase 2 (Issue #66)")

    def plot_subject_factors(
        self, subject_id: str, factors: Optional[List[int]] = None, **kwargs
    ) -> None:
        """Plot subject-specific factors as brain maps.

        Parameters
        ----------
        subject_id : str
            Subject identifier.
        factors : List[int], optional
            Which factors to plot. If None, plots all factors.
        **kwargs
            Additional arguments passed to plotting functions.
        """
        raise NotImplementedError("Will be implemented in Phase 2 (Issue #66)")

    def plot_temporal_weights(
        self, subject_id: str, networks: Optional[List[str]] = None, **kwargs
    ) -> None:
        """Plot temporal weight timeseries.

        Parameters
        ----------
        subject_id : str
            Subject identifier.
        networks : List[str], optional
            Which networks/factors to plot. If None, plots all.
        **kwargs
            Additional arguments passed to plotting functions.
        """
        raise NotImplementedError("Will be implemented in Phase 2 (Issue #66)")

    def plot_network_summary(
        self,
        include_timeseries: bool = True,
        include_connectivity: bool = False,
        save_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Create summary visualization of all factors and networks.

        Parameters
        ----------
        include_timeseries : bool, default=True
            Whether to include temporal weight plots.
        include_connectivity : bool, default=False
            Whether to include factor connectivity analysis.
        save_path : str, optional
            Path to save the figure.
        **kwargs
            Additional arguments passed to plotting functions.
        """
        raise NotImplementedError("Will be implemented in Phase 2 (Issue #66)")

    def to_nifti(
        self, factor_idx: Optional[int] = None, subject_id: Optional[str] = None
    ) -> "nib.Nifti1Image":
        """Reconstruct NIfTI images from HTFA factors.

        Parameters
        ----------
        factor_idx : int, optional
            Which factor to reconstruct. If None, reconstructs all factors.
        subject_id : str, optional
            Subject identifier for subject-specific reconstruction.
            If None, uses global template.

        Returns
        -------
        nibabel.Nifti1Image
            Reconstructed NIfTI image.
        """
        raise NotImplementedError("Will be implemented in Phase 2 (Issue #66)")

    def save_results(self, output_dir: str) -> None:
        """Save all results in BIDS-derivatives format.

        Parameters
        ----------
        output_dir : str
            Output directory for saving results.

        Creates
        -------
        - desc-global_factors.nii.gz : Global spatial factors
        - desc-{subject}_factors.nii.gz : Subject-specific factors
        - desc-{subject}_weights.tsv : Temporal weights
        - desc-model_params.json : Model parameters and metadata
        """
        raise NotImplementedError("Will be implemented in Phase 2 (Issue #66)")

    def get_network_timeseries(self, subject_id: str) -> "pd.DataFrame":
        """Extract network timeseries for further analysis.

        Parameters
        ----------
        subject_id : str
            Subject identifier.

        Returns
        -------
        pd.DataFrame
            Network timeseries with columns for each factor/network.
        """
        raise NotImplementedError("Will be implemented in Phase 2 (Issue #66)")

    def __repr__(self) -> str:
        """String representation of HTFAResults."""
        n_subjects = len(self.subject_factors)
        n_factors = (
            self.global_template.shape[0]
            if self.global_template is not None
            else "Unknown"
        )

        return (
            f"HTFAResults(\n"
            f"  n_subjects={n_subjects},\n"
            f"  n_factors={n_factors},\n"
            f"  bids_dataset='{self.bids_info.get('dataset_name', 'Unknown')}'\n"
            f")"
        )
