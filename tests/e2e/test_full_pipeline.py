"""End-to-end tests for complete HTFA workflows."""

from pathlib import Path

import numpy as np
import pytest


@pytest.mark.e2e
def test_bids_to_results_pipeline(mock_bids_dataset, temp_dir):
    """Test complete pipeline from BIDS input to results output."""
    from htfa.fit import fit_bids
    from htfa.results import HTFAResults

    # Run HTFA on mock BIDS dataset
    results = fit_bids(
        bids_dir=mock_bids_dataset, output_dir=temp_dir, n_components=5, task="rest"
    )

    # Verify results
    assert isinstance(results, HTFAResults)
    assert results.factors is not None
    assert results.weights is not None

    # Check output files were created
    output_files = list(temp_dir.glob("*"))
    assert len(output_files) > 0

    # Verify results can be saved
    results_file = temp_dir / "htfa_results.pkl"
    results.save(results_file)
    assert results_file.exists()

    # Verify results can be loaded
    loaded_results = HTFAResults.load(results_file)
    assert np.array_equal(loaded_results.factors, results.factors)


@pytest.mark.e2e
def test_multi_subject_analysis(sample_neuroimaging_data, temp_dir):
    """Test multi-subject HTFA analysis workflow."""
    from htfa.core.htfa import HTFA
    from htfa.results import HTFAResults

    # Prepare data
    subjects_data = list(sample_neuroimaging_data.values())
    n_voxels = subjects_data[0].shape[1]
    coords = np.random.randn(n_voxels, 3) * 50

    # Fit HTFA model
    htfa = HTFA(n_components=10, max_iter=50)
    htfa.fit(subjects_data, coords)

    # Create results object
    results = HTFAResults(
        factors=htfa.global_factors_,
        weights=[htfa.subject_weights_[i] for i in range(len(subjects_data))],
        coordinates=coords,
        subject_ids=[f"sub-{i:02d}" for i in range(len(subjects_data))],
    )

    # Generate visualizations
    fig = results.plot_factors(n_factors=3)
    assert fig is not None

    # Export to NIfTI (if affine provided)
    affine = np.eye(4)
    nifti_file = temp_dir / "factors.nii.gz"
    results.to_nifti(nifti_file, affine=affine, shape=(50, 50, 50))
    assert nifti_file.exists()


@pytest.mark.e2e
@pytest.mark.parametrize("optimizer", ["adam", "als", "batch"])
def test_optimization_backends(optimizer, sample_neuroimaging_data):
    """Test different optimization backends."""
    from htfa.optimization import OptimizedHTFA

    # Use smaller data for faster testing
    X = list(sample_neuroimaging_data.values())[0][:50, :500]
    coords = np.random.randn(500, 3)

    # Create optimized HTFA with specific optimizer
    model = OptimizedHTFA(n_components=5, optimizer=optimizer, max_iter=20)

    # Fit model
    model.fit([X], coords)

    # Verify convergence
    assert hasattr(model, "convergence_")
    assert len(model.convergence_) > 0

    # Check factors are valid
    assert not np.any(np.isnan(model.global_factors_))
    assert not np.any(np.isinf(model.global_factors_))


@pytest.mark.e2e
@pytest.mark.slow
def test_large_scale_analysis(temp_dir):
    """Test HTFA on larger, more realistic data."""
    from htfa.core.htfa import HTFA

    # Generate larger dataset
    n_subjects = 10
    n_timepoints = 500
    n_voxels = 10000

    subjects_data = [np.random.randn(n_timepoints, n_voxels) for _ in range(n_subjects)]
    coords = np.random.randn(n_voxels, 3) * 50

    # Fit HTFA
    htfa = HTFA(
        n_components=20, max_iter=10, verbose=True  # Fewer iterations for testing
    )

    htfa.fit(subjects_data, coords)

    # Verify outputs
    assert htfa.global_factors_.shape == (20, n_voxels)
    assert len(htfa.subject_weights_) == n_subjects
    assert all(w.shape == (n_timepoints, 20) for w in htfa.subject_weights_)

    # Check reconstruction error
    for i, X in enumerate(subjects_data):
        X_reconstructed = htfa.subject_weights_[i] @ htfa.subject_factors_[i]
        reconstruction_error = np.mean((X - X_reconstructed) ** 2)
        assert reconstruction_error < 10  # Reasonable threshold


@pytest.mark.e2e
def test_cross_validation_workflow(sample_neuroimaging_data):
    """Test cross-validation for model selection."""
    from sklearn.model_selection import TimeSeriesSplit

    from htfa.core.tfa import TFA

    # Use single subject data
    X = list(sample_neuroimaging_data.values())[0]
    coords = np.random.randn(X.shape[1], 3)

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]

        # Fit model
        tfa = TFA(n_components=10)
        tfa.fit(X_train, coords)

        # Evaluate on test set
        X_reconstructed = tfa.transform(X_test) @ tfa.factors_
        mse = np.mean((X_test - X_reconstructed) ** 2)
        scores.append(mse)

    # Check scores are reasonable
    assert len(scores) == 3
    assert all(s < 10 for s in scores)
    assert np.std(scores) < 5  # Consistent across folds
