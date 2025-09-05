"""Tests for HTFA core functionality."""

import numpy as np
import pytest

from htfa.htfa import HTFA, HTFABackend, NumPyBackend, HTFABenchmark


class TestHTFA:
    """Test HTFA core functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_samples = 100
        self.n_features = 50
        self.n_factors = 5
        self.X = np.random.randn(self.n_samples, self.n_features).astype(np.float32)
    
    def test_htfa_initialization(self):
        """Test HTFA initialization."""
        model = HTFA(n_factors=self.n_factors)
        assert model.n_factors == self.n_factors
        assert model.n_levels == 2  # default
        assert not model.is_fitted_
        assert isinstance(model.backend, NumPyBackend)
    
    def test_htfa_fit(self):
        """Test HTFA fitting."""
        model = HTFA(n_factors=self.n_factors, max_iter=10)
        model.fit(self.X)
        
        assert model.is_fitted_
        assert len(model.factors_) == model.n_levels
        assert len(model.loadings_) == model.n_levels
        assert model.reconstruction_error_ is not None
    
    def test_htfa_transform(self):
        """Test HTFA transformation."""
        model = HTFA(n_factors=self.n_factors, max_iter=10)
        model.fit(self.X)
        
        # Forward transform
        transformed = model.transform(self.X)
        assert transformed.shape == (self.n_samples, self.n_factors)
        
        # Inverse transform (reconstruction)
        reconstructed = model.transform(self.X, inverse=True)
        assert reconstructed.shape == self.X.shape
    
    def test_htfa_fit_transform(self):
        """Test HTFA fit_transform method."""
        model = HTFA(n_factors=self.n_factors, max_iter=10)
        transformed = model.fit_transform(self.X)
        
        assert model.is_fitted_
        assert transformed.shape == (self.n_samples, self.n_factors)
    
    def test_htfa_different_levels(self):
        """Test HTFA with different number of levels."""
        for n_levels in [1, 2, 3]:
            model = HTFA(n_factors=self.n_factors, n_levels=n_levels, max_iter=5)
            model.fit(self.X)
            
            assert len(model.factors_) == n_levels
            assert len(model.loadings_) == n_levels
    
    def test_htfa_reproducibility(self):
        """Test HTFA reproducibility with random_state."""
        model1 = HTFA(n_factors=self.n_factors, random_state=42, max_iter=5)
        model2 = HTFA(n_factors=self.n_factors, random_state=42, max_iter=5)
        
        transformed1 = model1.fit_transform(self.X)
        transformed2 = model2.fit_transform(self.X)
        
        np.testing.assert_allclose(transformed1, transformed2, rtol=1e-5)
    
    def test_htfa_get_factors_loadings(self):
        """Test getting factors and loadings from all levels."""
        model = HTFA(n_factors=self.n_factors, n_levels=2, max_iter=10)
        model.fit(self.X)
        
        factors_all = model.get_factors_all_levels()
        loadings_all = model.get_loadings_all_levels()
        
        assert len(factors_all) == 2
        assert len(loadings_all) == 2
        assert all(isinstance(f, np.ndarray) for f in factors_all)
        assert all(isinstance(l, np.ndarray) for l in loadings_all)
    
    def test_htfa_reconstruction_error(self):
        """Test reconstruction error computation."""
        model = HTFA(n_factors=self.n_factors, max_iter=10)
        model.fit(self.X)
        
        error = model.get_reconstruction_error()
        assert isinstance(error, float)
        assert error >= 0
    
    def test_unfitted_model_errors(self):
        """Test errors when using unfitted model."""
        model = HTFA(n_factors=self.n_factors)
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.transform(self.X)
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.get_reconstruction_error()
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.get_factors_all_levels()
    
    def test_invalid_backend(self):
        """Test invalid backend specification."""
        with pytest.raises(ValueError, match="Unknown backend"):
            HTFA(n_factors=self.n_factors, backend='invalid_backend')


class TestHTFABackend:
    """Test HTFA backend functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.backend = NumPyBackend()
        self.data = np.random.randn(10, 5).astype(np.float32)
    
    def test_backend_array_operations(self):
        """Test basic backend array operations."""
        # Test array creation
        arr = self.backend.array(self.data)
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, self.data)
        
        # Test zeros and ones
        zeros = self.backend.zeros((3, 3))
        ones = self.backend.ones((3, 3))
        assert zeros.shape == (3, 3)
        assert ones.shape == (3, 3)
        
        # Test random
        random_arr = self.backend.random((2, 2))
        assert random_arr.shape == (2, 2)
    
    def test_backend_linear_algebra(self):
        """Test backend linear algebra operations."""
        A = self.backend.array([[1, 2], [3, 4]])
        B = self.backend.array([[5, 6], [7, 8]])
        
        # Test matmul
        C = self.backend.matmul(A, B)
        expected = np.array([[19, 22], [43, 50]])
        np.testing.assert_array_equal(C, expected)
        
        # Test transpose
        AT = self.backend.transpose(A)
        expected_T = np.array([[1, 3], [2, 4]])
        np.testing.assert_array_equal(AT, expected_T)
        
        # Test SVD
        U, s, Vt = self.backend.svd(A)
        assert U.shape[0] == A.shape[0]
        assert len(s) == min(A.shape)
        assert Vt.shape[1] == A.shape[1]
        
        # Test norm and mean
        norm_val = self.backend.norm(A)
        mean_val = self.backend.mean(A)
        assert isinstance(norm_val, (np.ndarray, float, int))  # Can be scalar or array
        assert isinstance(mean_val, (np.ndarray, float, int))  # Can be scalar or array
    
    def test_backend_to_numpy(self):
        """Test conversion to numpy arrays."""
        arr = self.backend.array(self.data)
        numpy_arr = self.backend.to_numpy(arr)
        assert isinstance(numpy_arr, np.ndarray)
        np.testing.assert_array_equal(numpy_arr, self.data)


class TestHTFABenchmark:
    """Test HTFA benchmarking functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.benchmark = HTFABenchmark()
        self.data = np.random.randn(50, 20).astype(np.float32)
    
    def test_benchmark_htfa(self):
        """Test basic HTFA benchmarking."""
        result = self.benchmark.benchmark_htfa(
            self.data, n_factors=5, max_iter=5
        )
        
        assert result.execution_time > 0
        assert result.memory_usage >= 0
        assert result.reconstruction_error >= 0
        assert result.data_shape == self.data.shape
        assert result.backend == 'numpy'
        assert result.parameters['n_factors'] == 5
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        data, ground_truth = self.benchmark.generate_synthetic_data(
            n_samples=100, n_features=50, n_factors_true=5
        )
        
        assert data.shape == (100, 50)
        assert ground_truth['factors'].shape == (100, 5)
        assert ground_truth['loadings'].shape == (5, 50)
        assert 'noise_level' in ground_truth
    
    def test_benchmark_results_storage(self):
        """Test benchmark results are stored correctly."""
        assert len(self.benchmark.results) == 0
        
        self.benchmark.benchmark_htfa(self.data, n_factors=3, max_iter=5)
        assert len(self.benchmark.results) == 1
        
        self.benchmark.benchmark_htfa(self.data, n_factors=4, max_iter=5)
        assert len(self.benchmark.results) == 2
        
        self.benchmark.clear_results()
        assert len(self.benchmark.results) == 0
    
    def test_compare_backends_numpy_only(self):
        """Test backend comparison (numpy only)."""
        results = self.benchmark.compare_backends(
            self.data, n_factors=3, backends=['numpy'], max_iter=5
        )
        
        assert len(results) == 1
        assert results[0].backend == 'numpy'


@pytest.mark.parametrize("n_factors", [1, 5, 10])
@pytest.mark.parametrize("n_levels", [1, 2, 3])
def test_htfa_parameter_combinations(n_factors, n_levels):
    """Test HTFA with different parameter combinations."""
    X = np.random.randn(30, 20).astype(np.float32)
    model = HTFA(n_factors=n_factors, n_levels=n_levels, max_iter=5)
    
    transformed = model.fit_transform(X)
    assert transformed.shape == (30, n_factors)
    assert len(model.factors_) == n_levels


def test_htfa_edge_cases():
    """Test HTFA edge cases."""
    # Test with minimal data
    X_small = np.random.randn(5, 3).astype(np.float32)
    model = HTFA(n_factors=2, max_iter=5)
    model.fit(X_small)
    assert model.is_fitted_
    
    # Test with single factor
    model_single = HTFA(n_factors=1, max_iter=5)
    transformed = model_single.fit_transform(X_small)
    assert transformed.shape == (5, 1)