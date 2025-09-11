"""Performance benchmark tests for HTFA ecosystem."""

import numpy as np
import pytest


@pytest.mark.benchmark
@pytest.mark.slow
def test_tfa_fit_performance(benchmark, sample_neuroimaging_data):
    """Benchmark TFA fitting performance."""
    from htfa.core.tfa import TFA

    # Use first subject's data and transpose to (n_voxels, n_timepoints)
    # Reduce size for faster benchmarking
    X = list(sample_neuroimaging_data.values())[0][
        :20, :30
    ].T  # 30 voxels, 20 timepoints
    coords = np.random.randn(X.shape[0], 3)

    def run_tfa_fit():
        tfa = TFA(K=3, max_iter=5)  # Reduced components and iterations
        tfa.fit(X, coords)
        return tfa

    result = benchmark(run_tfa_fit)
    assert result is not None
    assert hasattr(result, "factors_")
    assert hasattr(result, "weights_")


@pytest.mark.benchmark
def test_htfa_fit_performance(benchmark, sample_neuroimaging_data):
    """Benchmark HTFA fitting performance for multiple subjects."""
    from htfa.core.htfa import HTFA

    # Prepare multi-subject data, transpose to (n_voxels, n_timepoints)
    # Reduce size for faster benchmarking
    subjects_data = [
        data[:20, :30].T for data in list(sample_neuroimaging_data.values())[:2]
    ]
    coords = np.random.randn(subjects_data[0].shape[0], 3)

    def run_htfa_fit():
        htfa = HTFA(K=3, max_global_iter=1, max_local_iter=2)
        # HTFA expects coords as a list, one per subject
        coords_list = [coords for _ in subjects_data]
        htfa.fit(subjects_data, coords_list)
        return htfa

    result = benchmark(run_htfa_fit)
    assert result is not None
    assert hasattr(result, "global_template_")


@pytest.mark.benchmark
def test_memory_usage(benchmark, capture_metrics):
    """Benchmark memory usage during HTFA operations."""
    from htfa.core.tfa import TFA

    # Generate smaller dataset for faster testing
    # TFA expects (n_voxels, n_timepoints)
    X = np.random.randn(100, 50)  # 100 voxels, 50 timepoints
    coords = np.random.randn(100, 3)

    def memory_intensive_operation():
        tfa = TFA(K=5, max_iter=5)
        tfa.fit(X, coords)
        return tfa

    benchmark(memory_intensive_operation)

    # Check memory usage from capture_metrics
    memory_delta = capture_metrics.get("memory_delta", 0)
    assert memory_delta < 1024  # Less than 1GB increase


@pytest.mark.benchmark
@pytest.mark.parametrize("n_components", [5, 10, 20])
def test_scaling_with_components(benchmark, n_components):
    """Test performance scaling with number of components."""
    from htfa.core.tfa import TFA

    # TFA expects (n_voxels, n_timepoints)
    X = np.random.randn(50, 30)  # 50 voxels, 30 timepoints
    coords = np.random.randn(50, 3)

    def run_with_n_components():
        tfa = TFA(K=n_components, max_iter=5)
        tfa.fit(X, coords)
        return tfa

    result = benchmark(run_with_n_components)
    assert result.K == n_components


@pytest.mark.benchmark
@pytest.mark.parametrize("n_voxels", [1000, 5000])
def test_scaling_with_voxels(benchmark, n_voxels):
    """Test performance scaling with number of voxels."""
    from htfa.core.tfa import TFA

    # TFA expects (n_voxels, n_timepoints)
    # Use much smaller timepoints for faster tests
    X = np.random.randn(n_voxels, 20)  # n_voxels voxels, 20 timepoints
    coords = np.random.randn(n_voxels, 3)

    def run_with_n_voxels():
        tfa = TFA(K=3, max_iter=5)  # Reduced components and iterations
        tfa.fit(X, coords)
        return tfa

    result = benchmark(run_with_n_voxels)
    assert result.factors_.shape[1] == n_voxels
