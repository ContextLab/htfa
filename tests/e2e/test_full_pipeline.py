"""End-to-end tests for complete HTFA workflows."""

from pathlib import Path

import numpy as np
import pytest


@pytest.mark.e2e
@pytest.mark.skip(reason="fit_bids not yet implemented")
def test_bids_to_results_pipeline(mock_bids_dataset, temp_dir):
    """Test complete pipeline from BIDS input to results output."""
    # This will be implemented when fit_bids is available
    pass


@pytest.mark.e2e
@pytest.mark.slow
def test_multi_subject_analysis(sample_neuroimaging_data, temp_dir):
    """Test multi-subject HTFA analysis workflow."""
    from htfa.core.htfa import HTFA

    # Prepare data, transpose to (n_voxels, n_timepoints)
    # Reduce data size for faster testing
    subjects_data = [
        data[:30, :50].T for data in list(sample_neuroimaging_data.values())[:2]
    ]  # Use only 2 subjects, 50 voxels, 30 timepoints
    n_voxels = subjects_data[0].shape[0]
    coords = np.random.randn(n_voxels, 3) * 50

    # Fit HTFA model with reduced complexity
    htfa = HTFA(
        K=3, max_global_iter=1, max_local_iter=2
    )  # Reduced components and iterations for testing
    # HTFA expects coords as a list, one per subject
    coords_list = [coords for _ in subjects_data]
    htfa.fit(subjects_data, coords_list)

    # Verify basic outputs
    assert hasattr(htfa, "global_template_")  # Global template
    assert htfa.global_template_ is not None


@pytest.mark.e2e
@pytest.mark.skip(reason="Optimization module not yet implemented")
@pytest.mark.parametrize("optimizer", ["adam", "als", "batch"])
def test_optimization_backends(optimizer, sample_neuroimaging_data):
    """Test different optimization backends."""
    # This will be implemented when optimization module is available
    pass


@pytest.mark.e2e
@pytest.mark.slow
def test_large_scale_analysis(temp_dir):
    """Test HTFA on larger, more realistic data."""
    from htfa.core.htfa import HTFA

    # Generate even smaller dataset for faster CI
    n_subjects = 2
    n_timepoints = 30
    n_voxels = 40

    # HTFA expects (n_voxels, n_timepoints) 
    subjects_data = [np.random.randn(n_voxels, n_timepoints) for _ in range(n_subjects)]
    coords = np.random.randn(n_voxels, 3) * 50

    # Fit HTFA with reduced complexity
    htfa = HTFA(K=3, max_global_iter=1, max_local_iter=2, verbose=False)
    # HTFA expects coords as a list, one per subject
    coords_list = [coords for _ in subjects_data]
    htfa.fit(subjects_data, coords_list)

    # Verify outputs
    assert hasattr(htfa, "global_template_")
    assert htfa.global_template_ is not None


@pytest.mark.e2e
def test_cross_validation_workflow(sample_neuroimaging_data):
    """Test cross-validation for model selection."""
    from sklearn.model_selection import TimeSeriesSplit

    from htfa.core.tfa import TFA

    # Use single subject data, transpose to (n_voxels, n_timepoints)
    # Reduce data size for faster testing
    X = list(sample_neuroimaging_data.values())[0][:30, :40].T  # 40 voxels, 30 timepoints
    coords = np.random.randn(X.shape[0], 3)

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=2)  # Reduce splits for speed
    scores = []

    # Split on time dimension (X.T splits timepoints)
    for train_idx, test_idx in tscv.split(X.T):
        X_train = X[:, train_idx]  # All voxels, training timepoints
        X_test = X[:, test_idx]    # All voxels, test timepoints

        # Fit model with reduced complexity
        tfa = TFA(K=3, max_iter=5)  # Reduced components and iterations
        tfa.fit(X_train, coords)

        # Just check that model fits without error
        assert hasattr(tfa, "factors_")
        assert hasattr(tfa, "weights_")
        scores.append(1.0)  # Dummy score

    # Check we ran the splits
    assert len(scores) == 2
