"""Comprehensive tests for HTFA backend functionality."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from htfa.backend_base import HTFA as BaseHTFA
from htfa.backend_base import HTFABackend
from htfa.backends.numpy_backend import NumPyBackend
from htfa.core.htfa import HTFA
from htfa.core.tfa import TFA


class TestNumPyBackend:
    """Test NumPy backend implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.backend = NumPyBackend()
        self.test_data = np.random.randn(10, 5)
        self.test_shape = (5, 3)

    def test_array_creation(self):
        """Test array creation from data."""
        result = self.backend.array(self.test_data)
        np.testing.assert_array_equal(result, self.test_data)
        assert isinstance(result, np.ndarray)

    def test_zeros_creation(self):
        """Test zeros array creation."""
        result = self.backend.zeros(self.test_shape)
        expected = np.zeros(self.test_shape)
        np.testing.assert_array_equal(result, expected)
        assert result.shape == self.test_shape

    def test_ones_creation(self):
        """Test ones array creation."""
        result = self.backend.ones(self.test_shape)
        expected = np.ones(self.test_shape)
        np.testing.assert_array_equal(result, expected)
        assert result.shape == self.test_shape

    def test_random_creation(self):
        """Test random array creation."""
        result = self.backend.random(self.test_shape)
        assert result.shape == self.test_shape
        assert isinstance(result, np.ndarray)
        # Check that values are different (not all zeros)
        assert not np.allclose(result, 0)

    def test_matmul(self):
        """Test matrix multiplication."""
        a = np.random.randn(3, 4)
        b = np.random.randn(4, 5)
        result = self.backend.matmul(a, b)
        expected = np.matmul(a, b)
        np.testing.assert_array_almost_equal(result, expected)

    def test_transpose(self):
        """Test array transpose."""
        result = self.backend.transpose(self.test_data)
        expected = np.transpose(self.test_data)
        np.testing.assert_array_equal(result, expected)

        # Test with axes
        result_axes = self.backend.transpose(self.test_data, axes=(1, 0))
        expected_axes = np.transpose(self.test_data, axes=(1, 0))
        np.testing.assert_array_equal(result_axes, expected_axes)

    def test_svd(self):
        """Test singular value decomposition."""
        test_matrix = np.random.randn(5, 4)
        U, s, Vt = self.backend.svd(test_matrix, full_matrices=True)

        # Check reconstruction
        reconstruction = U[:, : len(s)] @ np.diag(s) @ Vt
        np.testing.assert_array_almost_equal(reconstruction, test_matrix)

        # Check shapes
        assert U.shape == (5, 5)
        assert s.shape == (4,)
        assert Vt.shape == (4, 4)

    def test_norm(self):
        """Test norm computation."""
        result = self.backend.norm(self.test_data)
        expected = np.linalg.norm(self.test_data)
        np.testing.assert_almost_equal(result, expected)

        # Test with axis
        result_axis = self.backend.norm(self.test_data, axis=0)
        expected_axis = np.linalg.norm(self.test_data, axis=0)
        np.testing.assert_array_almost_equal(result_axis, expected_axis)

    def test_mean(self):
        """Test mean computation."""
        result = self.backend.mean(self.test_data)
        expected = np.mean(self.test_data)
        np.testing.assert_almost_equal(result, expected)

        # Test with axis
        result_axis = self.backend.mean(self.test_data, axis=0)
        expected_axis = np.mean(self.test_data, axis=0)
        np.testing.assert_array_almost_equal(result_axis, expected_axis)

    def test_to_numpy(self):
        """Test conversion to numpy array."""
        result = self.backend.to_numpy(self.test_data)
        np.testing.assert_array_equal(result, self.test_data)
        assert isinstance(result, np.ndarray)


class TestBackendFactory:
    """Test backend factory and selection functionality."""

    def test_numpy_backend_creation(self):
        """Test NumPy backend creation from string."""
        htfa = BaseHTFA(n_factors=5, backend="numpy")
        assert isinstance(htfa.backend, NumPyBackend)

    def test_numpy_backend_default(self):
        """Test default backend is NumPy."""
        htfa = BaseHTFA(n_factors=5)  # No backend specified
        assert isinstance(htfa.backend, NumPyBackend)

    def test_custom_backend_object(self):
        """Test passing custom backend object."""
        custom_backend = NumPyBackend()
        htfa = BaseHTFA(n_factors=5, backend=custom_backend)
        assert htfa.backend is custom_backend

    @patch("htfa.backends.jax_backend.HAS_JAX", False)
    def test_jax_backend_unavailable(self):
        """Test error when JAX backend is not available."""
        with pytest.raises(ImportError, match="JAX backend not available"):
            BaseHTFA(n_factors=5, backend="jax")

    @patch("htfa.backends.pytorch_backend.HAS_TORCH", False)
    def test_pytorch_backend_unavailable(self):
        """Test error when PyTorch backend is not available."""
        with pytest.raises(ImportError, match="PyTorch backend not available"):
            BaseHTFA(n_factors=5, backend="pytorch")

    def test_unknown_backend_error(self):
        """Test error for unknown backend string."""
        with pytest.raises(ValueError, match="Unknown backend: unknown"):
            BaseHTFA(n_factors=5, backend="unknown")


class TestTFABackendIntegration:
    """Test TFA with different backends."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.X = np.random.randn(20, 10)  # 20 voxels, 10 timepoints
        self.coords = np.random.randn(20, 3)  # 3D coordinates

    def test_tfa_numpy_backend_default(self):
        """Test TFA with default NumPy backend."""
        tfa = TFA(K=3, max_iter=10, verbose=False)
        assert isinstance(tfa.backend, NumPyBackend)

        # Test fitting
        tfa.fit(self.X, self.coords)
        assert tfa.factors_ is not None
        assert tfa.weights_ is not None
        assert tfa.factors_.shape == (3, 20)
        assert tfa.weights_.shape == (10, 3)

    def test_tfa_explicit_numpy_backend(self):
        """Test TFA with explicit NumPy backend."""
        tfa = TFA(K=3, backend="numpy", max_iter=10, verbose=False)
        assert isinstance(tfa.backend, NumPyBackend)

        tfa.fit(self.X, self.coords)
        assert tfa.factors_ is not None
        assert tfa.weights_ is not None

    def test_tfa_custom_backend_object(self):
        """Test TFA with custom backend object."""
        backend = NumPyBackend()
        tfa = TFA(K=3, backend=backend, max_iter=10, verbose=False)
        assert tfa.backend is backend

        tfa.fit(self.X, self.coords)
        assert tfa.factors_ is not None
        assert tfa.weights_ is not None

    def test_tfa_backend_factory_methods(self):
        """Test TFA backend factory methods."""
        tfa = TFA(K=3, max_iter=10, verbose=False)

        # Test NumPy backend creation
        numpy_backend = tfa._create_backend("numpy")
        assert isinstance(numpy_backend, NumPyBackend)

        # Test unknown backend error
        with pytest.raises(ValueError, match="Unknown backend: unknown"):
            tfa._create_backend("unknown")


class TestHTFABackendIntegration:
    """Test HTFA with different backends."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        # Multi-subject data
        self.X = [
            np.random.randn(15, 8),  # Subject 1: 15 voxels, 8 timepoints
            np.random.randn(15, 8),  # Subject 2
            np.random.randn(15, 8),  # Subject 3
        ]
        self.coords = [
            np.random.randn(15, 3),  # 3D coordinates for each subject
            np.random.randn(15, 3),
            np.random.randn(15, 3),
        ]

    def test_htfa_numpy_backend_default(self):
        """Test HTFA with default NumPy backend."""
        htfa = HTFA(K=2, max_global_iter=2, max_local_iter=5, verbose=False)
        assert isinstance(htfa.backend, NumPyBackend)

        # Test fitting
        htfa.fit(self.X, self.coords)
        assert htfa.factors_ is not None
        assert htfa.weights_ is not None
        assert len(htfa.factors_) == 3  # One per subject
        assert len(htfa.weights_) == 3

    def test_htfa_explicit_numpy_backend(self):
        """Test HTFA with explicit NumPy backend."""
        htfa = HTFA(
            K=2, backend="numpy", max_global_iter=2, max_local_iter=5, verbose=False
        )
        assert isinstance(htfa.backend, NumPyBackend)

        htfa.fit(self.X, self.coords)
        assert htfa.factors_ is not None
        assert htfa.weights_ is not None

    def test_htfa_custom_backend_object(self):
        """Test HTFA with custom backend object."""
        backend = NumPyBackend()
        htfa = HTFA(
            K=2, backend=backend, max_global_iter=2, max_local_iter=5, verbose=False
        )
        assert htfa.backend is backend

        htfa.fit(self.X, self.coords)
        assert htfa.factors_ is not None
        assert htfa.weights_ is not None

    def test_htfa_backend_propagation_to_tfa(self):
        """Test that HTFA passes backend to TFA models."""
        custom_backend = NumPyBackend()
        htfa = HTFA(
            K=2,
            backend=custom_backend,
            max_global_iter=2,
            max_local_iter=5,
            verbose=False,
        )

        htfa.fit(self.X, self.coords)

        # Check that all TFA models received the same backend
        for tfa_model in htfa.subject_models_:
            assert tfa_model.backend is custom_backend

    def test_htfa_backend_factory_methods(self):
        """Test HTFA backend factory methods."""
        htfa = HTFA(K=2, max_global_iter=2, max_local_iter=5, verbose=False)

        # Test NumPy backend creation
        numpy_backend = htfa._create_backend("numpy")
        assert isinstance(numpy_backend, NumPyBackend)

        # Test unknown backend error
        with pytest.raises(ValueError, match="Unknown backend: unknown"):
            htfa._create_backend("unknown")


class TestBackendSwapping:
    """Test dynamic backend swapping functionality."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.X = np.random.randn(10, 5)
        self.coords = np.random.randn(10, 3)

    def test_tfa_backend_swapping(self):
        """Test switching TFA backend after initialization."""
        tfa = TFA(K=2, max_iter=5, verbose=False)
        original_backend = tfa.backend
        assert isinstance(original_backend, NumPyBackend)

        # Swap to a new backend
        new_backend = NumPyBackend()
        tfa.backend = new_backend
        assert tfa.backend is new_backend
        assert tfa.backend is not original_backend

    def test_htfa_backend_swapping(self):
        """Test switching HTFA backend after initialization."""
        htfa = HTFA(K=2, max_global_iter=1, max_local_iter=3, verbose=False)
        original_backend = htfa.backend
        assert isinstance(original_backend, NumPyBackend)

        # Swap to a new backend
        new_backend = NumPyBackend()
        htfa.backend = new_backend
        assert htfa.backend is new_backend
        assert htfa.backend is not original_backend

    def test_base_htfa_backend_swapping(self):
        """Test switching base HTFA backend after initialization."""
        htfa = BaseHTFA(n_factors=2, max_iter=5)
        original_backend = htfa.backend
        assert isinstance(original_backend, NumPyBackend)

        # Swap to a new backend
        new_backend = NumPyBackend()
        htfa.backend = new_backend
        assert htfa.backend is new_backend
        assert htfa.backend is not original_backend


class TestBackendCompatibility:
    """Test backend compatibility and consistency."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.X = np.random.randn(8, 6)
        self.coords = np.random.randn(8, 3)

    def test_numpy_backend_consistency(self):
        """Test that NumPy backend produces consistent results."""
        # Create two identical TFA instances with NumPy backend
        tfa1 = TFA(K=2, random_state=42, max_iter=10, verbose=False, backend="numpy")
        tfa2 = TFA(K=2, random_state=42, max_iter=10, verbose=False, backend="numpy")

        # Fit both models
        tfa1.fit(self.X, self.coords)
        tfa2.fit(self.X, self.coords)

        # Results should be identical (or very close due to randomness in optimization)
        np.testing.assert_array_almost_equal(tfa1.factors_, tfa2.factors_, decimal=3)
        np.testing.assert_array_almost_equal(tfa1.weights_, tfa2.weights_, decimal=3)

    def test_backend_interface_compliance(self):
        """Test that all backends implement required interface methods."""
        backends = [NumPyBackend()]

        for backend in backends:
            assert hasattr(backend, "array")
            assert hasattr(backend, "zeros")
            assert hasattr(backend, "ones")
            assert hasattr(backend, "random")
            assert hasattr(backend, "matmul")
            assert hasattr(backend, "transpose")
            assert hasattr(backend, "svd")
            assert hasattr(backend, "norm")
            assert hasattr(backend, "mean")
            assert hasattr(backend, "to_numpy")

            # Test that all methods are callable
            assert callable(getattr(backend, "array"))
            assert callable(getattr(backend, "zeros"))
            assert callable(getattr(backend, "ones"))
            assert callable(getattr(backend, "random"))
            assert callable(getattr(backend, "matmul"))
            assert callable(getattr(backend, "transpose"))
            assert callable(getattr(backend, "svd"))
            assert callable(getattr(backend, "norm"))
            assert callable(getattr(backend, "mean"))
            assert callable(getattr(backend, "to_numpy"))


class TestBackendErrorHandling:
    """Test error handling in backend operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.backend = NumPyBackend()

    def test_invalid_matmul_dimensions(self):
        """Test error handling for invalid matrix multiplication dimensions."""
        a = np.random.randn(3, 4)
        b = np.random.randn(5, 2)  # Incompatible dimensions

        with pytest.raises(ValueError):
            self.backend.matmul(a, b)

    def test_invalid_transpose_axes(self):
        """Test error handling for invalid transpose axes."""
        data = np.random.randn(3, 4)

        with pytest.raises((ValueError, IndexError)):
            self.backend.transpose(data, axes=(0, 1, 2))  # Too many axes

    def test_svd_singular_matrix_handling(self):
        """Test SVD on near-singular matrices."""
        # Create a near-singular matrix
        singular_matrix = np.array([[1, 2], [1, 2 + 1e-15]])

        # Should not raise an error, but might produce warnings
        U, s, Vt = self.backend.svd(singular_matrix)
        assert U is not None
        assert s is not None
        assert Vt is not None


class TestPerformanceRegression:
    """Test for performance regressions with backend integration."""

    def setup_method(self):
        """Set up larger test data for performance testing."""
        np.random.seed(42)
        self.large_X = np.random.randn(50, 20)
        self.large_coords = np.random.randn(50, 3)

    def test_tfa_performance_with_backend(self):
        """Test that TFA performance doesn't degrade significantly with backend."""
        import time

        # Test with explicit NumPy backend
        start_time = time.time()
        tfa_backend = TFA(K=5, max_iter=20, verbose=False, backend="numpy")
        tfa_backend.fit(self.large_X, self.large_coords)
        backend_time = time.time() - start_time

        # Test with default backend (should be similar)
        start_time = time.time()
        tfa_default = TFA(K=5, max_iter=20, verbose=False)
        tfa_default.fit(self.large_X, self.large_coords)
        default_time = time.time() - start_time

        # Backend overhead should be minimal (less than 50% increase)
        assert (
            backend_time < default_time * 1.5
        ), f"Backend time {backend_time:.3f}s vs default {default_time:.3f}s"

    def test_htfa_performance_with_backend(self):
        """Test that HTFA performance doesn't degrade significantly with backend."""
        import time

        # Multi-subject data
        X_multi = [self.large_X[:30], self.large_X[30:], self.large_X[20:]]
        coords_multi = [
            self.large_coords[:30],
            self.large_coords[30:],
            self.large_coords[20:],
        ]

        # Test with explicit NumPy backend
        start_time = time.time()
        htfa_backend = HTFA(
            K=3, backend="numpy", max_global_iter=2, max_local_iter=10, verbose=False
        )
        htfa_backend.fit(X_multi, coords_multi)
        backend_time = time.time() - start_time

        # Test with default backend
        start_time = time.time()
        htfa_default = HTFA(K=3, max_global_iter=2, max_local_iter=10, verbose=False)
        htfa_default.fit(X_multi, coords_multi)
        default_time = time.time() - start_time

        # Backend overhead should be minimal
        assert (
            backend_time < default_time * 1.5
        ), f"Backend time {backend_time:.3f}s vs default {default_time:.3f}s"


# Test for optional backends (only run if available)
class TestJAXBackendIntegration:
    """Test JAX backend integration (if available)."""

    def test_jax_backend_creation(self):
        """Test JAX backend creation when available."""
        try:
            import jax

            htfa = BaseHTFA(n_factors=3, backend="jax")
            from htfa.backends.jax_backend import JAXBackend

            assert isinstance(htfa.backend, JAXBackend)
        except ImportError:
            pytest.skip("JAX not available")


class TestPyTorchBackendIntegration:
    """Test PyTorch backend integration (if available)."""

    def test_pytorch_backend_creation(self):
        """Test PyTorch backend creation when available."""
        try:
            import torch

            htfa = BaseHTFA(n_factors=3, backend="pytorch")
            from htfa.backends.pytorch_backend import PyTorchBackend

            assert isinstance(htfa.backend, PyTorchBackend)
        except ImportError:
            pytest.skip("PyTorch not available")
