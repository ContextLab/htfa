"""Performance benchmark tests for HTFA ecosystem."""

import numpy as np
import pytest


@pytest.mark.benchmark
def test_tfa_fit_performance(benchmark, sample_neuroimaging_data):
    """Benchmark TFA fitting performance."""
    from htfa.core.tfa import TFA

    # Use first subject's data
    X = list(sample_neuroimaging_data.values())[0]
    coords = np.random.randn(X.shape[1], 3)

    def run_tfa_fit():
        tfa = TFA(K=10)
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

    # Prepare multi-subject data
    subjects_data = list(sample_neuroimaging_data.values())[:3]
    coords = np.random.randn(subjects_data[0].shape[1], 3)

    def run_htfa_fit():
        htfa = HTFA(K=10, max_global_iter=2, max_local_iter=5)
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

    # Generate larger dataset for memory testing
    X = np.random.randn(200, 5000)
    coords = np.random.randn(5000, 3)

    def memory_intensive_operation():
        tfa = TFA(K=20)
        tfa.fit(X, coords)
        return tfa

    result = benchmark(memory_intensive_operation)

    # Check memory usage from capture_metrics
    memory_delta = capture_metrics.get("memory_delta", 0)
    assert memory_delta < 1024  # Less than 1GB increase


@pytest.mark.benchmark
@pytest.mark.parametrize("n_components", [5, 10, 20])
def test_scaling_with_components(benchmark, n_components):
    """Test performance scaling with number of components."""
    from htfa.core.tfa import TFA

    X = np.random.randn(100, 1000)
    coords = np.random.randn(1000, 3)

    def run_with_n_components():
        tfa = TFA(K=n_components)
        tfa.fit(X, coords)
        return tfa

    result = benchmark(run_with_n_components)
    assert result.K == n_components


@pytest.mark.benchmark
@pytest.mark.parametrize("n_voxels", [1000, 5000, 10000])
def test_scaling_with_voxels(benchmark, n_voxels):
    """Test performance scaling with number of voxels."""
    from htfa.core.tfa import TFA

    # TFA expects (n_voxels, n_timepoints)
    X = np.random.randn(n_voxels, 100)  # n_voxels voxels, 100 timepoints
    coords = np.random.randn(n_voxels, 3)

    def run_with_n_voxels():
        tfa = TFA(K=10)
        tfa.fit(X, coords)
        return tfa

    result = benchmark(run_with_n_voxels)
    assert result.factors_.shape[1] == n_voxels
