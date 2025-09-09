"""Test BrainIAK algorithm implementations."""

import numpy as np
import pytest

from htfa.core.htfa import HTFA
from htfa.core.tfa import TFA


def generate_synthetic_data(
    n_voxels=100, n_timepoints=50, n_factors=5, noise_level=0.1
):
    """Generate synthetic neuroimaging data with known factor structure."""
    # Generate spatial coordinates
    coords = np.random.randn(n_voxels, 3) * 10

    # Generate true factors using RBF
    true_centers = np.random.randn(n_factors, 3) * 5
    true_widths = np.random.uniform(1, 3, n_factors)

    factors = np.zeros((n_factors, n_voxels))
    for k in range(n_factors):
        distances = np.linalg.norm(coords - true_centers[k], axis=1)
        factors[k] = np.exp(-(distances**2) / (2 * true_widths[k] ** 2))

    # Generate weights
    weights = np.random.randn(n_timepoints, n_factors)

    # Generate data as factor @ weight + noise
    data = factors.T @ weights.T
    noise = np.random.randn(*data.shape) * noise_level * np.std(data)
    data += noise

    return data, coords, factors, weights, true_centers, true_widths


class TestTFA:
    """Test TFA implementation."""

    def test_tfa_initialization(self):
        """Test TFA can be initialized."""
        tfa = TFA(K=10, max_iter=10)
        assert tfa.K == 10
        assert tfa.max_iter == 10

    def test_tfa_fit_synthetic(self):
        """Test TFA fitting on synthetic data."""
        # Generate synthetic data
        data, coords, true_factors, true_weights, true_centers, true_widths = (
            generate_synthetic_data(n_voxels=50, n_timepoints=30, n_factors=3)
        )

        # Fit TFA model
        tfa = TFA(K=3, max_iter=20, verbose=True)
        tfa.fit(data, coords)

        # Check outputs exist
        assert tfa.centers_ is not None
        assert tfa.widths_ is not None
        assert tfa.factors_ is not None
        assert tfa.weights_ is not None

        # Check shapes
        assert tfa.centers_.shape == (3, 3)  # K x n_dims
        assert tfa.widths_.shape == (3,)  # K
        assert tfa.factors_.shape == (3, 50)  # K x n_voxels
        assert tfa.weights_.shape == (30, 3)  # n_timepoints x K

    def test_tfa_compute_factors(self):
        """Test RBF factor computation."""
        tfa = TFA(K=2)
        coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        tfa.centers_ = np.array([[0.5, 0.5], [0, 0]])
        tfa.widths_ = np.array([1.0, 0.5])

        factors = tfa._compute_factors(coords)

        assert factors.shape == (2, 4)
        # Check RBF properties
        assert np.all(factors >= 0)
        assert np.all(factors <= 1)

    def test_tfa_compute_weights(self):
        """Test weight computation using ridge regression."""
        tfa = TFA(K=2, weight_method="rr", regularization=0.01)

        X = np.random.randn(10, 5)
        factors = np.random.randn(2, 10)

        weights = tfa._compute_weights(X, factors)

        assert weights.shape == (5, 2)  # n_timepoints x K

    def test_tfa_convergence(self):
        """Test that TFA converges on simple data."""
        # Create simple data with clear structure
        n_voxels, n_timepoints = 30, 20
        coords = np.random.randn(n_voxels, 2)

        # Create data with known structure
        true_factors = np.random.randn(2, n_voxels)
        true_weights = np.random.randn(n_timepoints, 2)
        data = true_factors.T @ true_weights.T

        tfa = TFA(K=2, max_iter=50, tol=1e-4, verbose=True)
        tfa.fit(data, coords)

        # Check reconstruction error
        reconstruction = tfa.factors_.T @ tfa.weights_.T
        mse = np.mean((data - reconstruction) ** 2)

        # Should achieve reasonable reconstruction (relaxed threshold for simple test)
        assert mse < 2.0  # Reasonable threshold for convergence test


class TestHTFA:
    """Test HTFA implementation."""

    def test_htfa_initialization(self):
        """Test HTFA can be initialized."""
        htfa = HTFA(K=10, max_global_iter=5)
        assert htfa.K == 10
        assert htfa.max_global_iter == 5

    def test_htfa_fit_multi_subject(self):
        """Test HTFA fitting on multi-subject synthetic data."""
        n_subjects = 3
        subjects_data = []
        subjects_coords = []

        # Generate data for each subject
        for _ in range(n_subjects):
            data, coords, _, _, _, _ = generate_synthetic_data(
                n_voxels=40, n_timepoints=25, n_factors=3
            )
            subjects_data.append(data)
            subjects_coords.append(coords)

        # Fit HTFA model
        htfa = HTFA(K=3, max_global_iter=3, max_local_iter=10, verbose=True)
        htfa.fit(subjects_data, subjects_coords)

        # Check outputs exist
        assert htfa.global_template_ is not None
        assert htfa.factors_ is not None
        assert htfa.weights_ is not None
        assert len(htfa.factors_) == n_subjects
        assert len(htfa.weights_) == n_subjects

    def test_htfa_template_estimation(self):
        """Test global template estimation."""
        htfa = HTFA(K=2)

        # Create mock subject models
        from unittest.mock import Mock

        subject1 = Mock()
        subject1.centers_ = np.array([[0, 0], [1, 1]])
        subject1.widths_ = np.array([1.0, 1.5])
        subject1.get_factors = lambda: np.random.randn(2, 10)

        subject2 = Mock()
        subject2.centers_ = np.array([[0.1, 0.1], [0.9, 0.9]])
        subject2.widths_ = np.array([0.9, 1.6])
        subject2.get_factors = lambda: np.random.randn(2, 10)

        htfa.subject_models_ = [subject1, subject2]
        htfa._compute_global_template()

        assert htfa.global_template_ is not None
        assert "centers" in htfa.global_template_
        assert "widths" in htfa.global_template_

        # Check template is average of subjects
        expected_centers = np.array([[0.05, 0.05], [0.95, 0.95]])
        np.testing.assert_allclose(
            htfa.global_template_["centers"], expected_centers, rtol=1e-5
        )

    def test_htfa_convergence_check(self):
        """Test convergence checking."""
        htfa = HTFA(K=2, tol=1e-3)

        template1 = {
            "centers": np.array([[0, 0], [1, 1]]),
            "widths": np.array([1.0, 1.0]),
        }

        template2 = {
            "centers": np.array([[0, 0], [1, 1]]),
            "widths": np.array([1.0, 1.0]),
        }

        # Same templates should converge
        assert htfa._check_convergence(template1, template2) == True

        template3 = {
            "centers": np.array([[0.5, 0.5], [2, 2]]),
            "widths": np.array([2.0, 2.0]),
        }

        # Different templates should not converge
        assert htfa._check_convergence(template1, template3) == False


if __name__ == "__main__":
    # Run basic tests
    print("Testing TFA implementation...")
    test_tfa = TestTFA()
    test_tfa.test_tfa_initialization()
    test_tfa.test_tfa_fit_synthetic()
    test_tfa.test_tfa_compute_factors()
    test_tfa.test_tfa_compute_weights()
    test_tfa.test_tfa_convergence()
    print("TFA tests passed!")

    print("\nTesting HTFA implementation...")
    test_htfa = TestHTFA()
    test_htfa.test_htfa_initialization()
    test_htfa.test_htfa_fit_multi_subject()
    test_htfa.test_htfa_template_estimation()
    test_htfa.test_htfa_convergence_check()
    print("HTFA tests passed!")

    print("\nAll tests completed successfully!")
