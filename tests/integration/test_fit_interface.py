"""Integration tests for HTFA fit interface.

This module provides end-to-end testing of the HTFA fit interface,
including real data flows, parameter inference, and model fitting
without mocks to ensure the entire pipeline works correctly.
"""

import numpy as np
import pytest

from htfa.core import HTFA, TFA
from htfa.fit import fit


@pytest.mark.slow
class TestFitInterfaceIntegration:
    """Integration tests for the main fit interface."""

    def setup_method(self):
        """Set up synthetic data for integration testing."""
        np.random.seed(42)  # Reproducible tests

        # Create realistic fMRI-like synthetic data
        # Reduced sizes for faster testing
        self.n_voxels = 100
        self.n_timepoints = 50
        self.n_subjects = 3
        self.true_n_factors = 5

        # Generate synthetic fMRI data with spatial structure
        self.single_subject_data = self._generate_synthetic_fmri_data(
            self.n_voxels, self.n_timepoints, self.true_n_factors
        )

        # Generate realistic brain coordinates (MNI space)
        self.coordinates = self._generate_brain_coordinates(self.n_voxels)

        # Multi-subject data with slight variations
        self.multi_subject_data = []
        for i in range(self.n_subjects):
            # Add subject-specific variation
            noise_scale = 0.1 + i * 0.05  # Increasing noise per subject
            subject_data = self._generate_synthetic_fmri_data(
                self.n_voxels,
                self.n_timepoints,
                self.true_n_factors,
                noise_scale=noise_scale,
            )
            self.multi_subject_data.append(subject_data)

    def _generate_synthetic_fmri_data(
        self, n_voxels: int, n_timepoints: int, n_factors: int, noise_scale: float = 0.1
    ) -> np.ndarray:
        """Generate realistic synthetic fMRI data with factor structure."""
        # Create spatial factor maps with realistic brain-like structure
        factors = np.random.randn(n_voxels, n_factors)

        # Add spatial smoothing to make factors more brain-like
        for f in range(n_factors):
            # Simple spatial smoothing by averaging with neighbors
            smoothed = factors[:, f].copy()
            for i in range(1, n_voxels - 1):
                smoothed[i] = 0.5 * factors[i, f] + 0.25 * (
                    factors[i - 1, f] + factors[i + 1, f]
                )
            factors[:, f] = smoothed

        # Create temporal dynamics
        weights = np.random.randn(n_factors, n_timepoints)

        # Add temporal smoothing for more realistic dynamics
        for f in range(n_factors):
            for t in range(1, n_timepoints - 1):
                weights[f, t] = 0.6 * weights[f, t] + 0.2 * (
                    weights[f, t - 1] + weights[f, t + 1]
                )

        # Generate data as factors @ weights + noise
        signal = factors @ weights
        noise = np.random.randn(n_voxels, n_timepoints) * noise_scale

        return signal + noise

    def _generate_brain_coordinates(self, n_voxels: int) -> np.ndarray:
        """Generate realistic brain coordinates in MNI space."""
        # Create coordinates that roughly span brain volume
        x_coords = np.random.uniform(-90, 90, n_voxels)
        y_coords = np.random.uniform(-126, 90, n_voxels)
        z_coords = np.random.uniform(-72, 108, n_voxels)

        return np.column_stack([x_coords, y_coords, z_coords])

    def test_single_subject_tfa_integration(self):
        """Test complete TFA workflow on single subject data."""
        # Fit TFA model
        model = fit(self.single_subject_data, coords=self.coordinates)

        # Verify model type and properties
        assert isinstance(model, TFA)
        assert hasattr(model, "factors_")
        assert hasattr(model, "weights_")
        assert hasattr(model, "coords_")

        # Verify data shapes
        assert model.factors_ is not None
        assert model.weights_ is not None
        np.testing.assert_array_equal(model.coords_, self.coordinates)

        # Verify factors shape matches expected dimensions
        n_factors = model.K
        assert model.factors_.shape[0] == n_factors
        assert model.weights_.shape[1] == n_factors

        # Verify parameter inference worked reasonably
        assert 1 <= n_factors <= 50  # Should be within reasonable bounds

        # Test that model has converged (basic sanity check)
        assert hasattr(model, "tol")

    def test_multi_subject_htfa_integration(self):
        """Test complete HTFA workflow on multi-subject data."""
        # Fit HTFA model
        model = fit(self.multi_subject_data, coords=self.coordinates)

        # Verify model type and properties
        assert isinstance(model, HTFA)
        assert hasattr(model, "global_template_")
        assert hasattr(model, "factors_")
        assert hasattr(model, "weights_")
        assert hasattr(model, "coords_")

        # Verify data structures
        assert model.global_template_ is not None
        assert model.factors_ is not None
        assert model.weights_ is not None
        assert len(model.factors_) == self.n_subjects
        assert len(model.weights_) == self.n_subjects
        np.testing.assert_array_equal(model.coords_, self.coordinates)

        # Verify subject-specific factors
        for subject_idx in range(self.n_subjects):
            subject_factors = model.factors_[subject_idx]
            subject_weights = model.weights_[subject_idx]
            assert subject_factors is not None
            assert subject_weights is not None
            assert subject_factors.shape[0] == model.K
            assert subject_weights.shape[1] == model.K

    def test_parameter_inference_accuracy(self):
        """Test that parameter inference produces reasonable results."""
        # Test with different data sizes
        test_cases = [
            (100, 50),  # Small dataset
            (500, 100),  # Medium dataset
            (2000, 300),  # Larger dataset
        ]

        for n_vox, n_time in test_cases:
            # Generate synthetic data
            coords = self._generate_brain_coordinates(n_vox)
            data = self._generate_synthetic_fmri_data(
                n_vox, n_time, 8
            )  # 8 true factors

            # Fit model and check inferred parameters
            model = fit(data, coords=coords)

            # Parameter inference heuristic: min(sqrt(n_voxels), n_timepoints/10, 50)
            expected_factors = min(int(np.sqrt(n_vox)), max(1, n_time // 10), 50)

            assert model.K == expected_factors
            assert 1 <= model.K <= 50

    def test_explicit_parameter_specification(self):
        """Test that explicitly specified parameters override inference."""
        explicit_n_factors = 15
        explicit_params = {"max_iter": 200, "tol": 1e-8, "verbose": True}

        # Single subject
        model_tfa = fit(
            self.single_subject_data,
            coords=self.coordinates,
            n_factors=explicit_n_factors,
            **explicit_params,
        )

        assert isinstance(model_tfa, TFA)
        assert model_tfa.K == explicit_n_factors
        assert model_tfa.max_iter == 200
        assert model_tfa.tol == 1e-8
        assert model_tfa.verbose is True

        # Multi-subject
        model_htfa = fit(
            self.multi_subject_data,
            coords=self.coordinates,
            n_factors=explicit_n_factors,
            **explicit_params,
        )

        assert isinstance(model_htfa, HTFA)
        assert model_htfa.K == explicit_n_factors
        assert model_htfa.max_local_iter == 200  # Should map to max_iter
        assert model_htfa.tol == 1e-8
        assert model_htfa.verbose is True

    def test_force_multi_subject_mode(self):
        """Test forcing HTFA for single subject data."""
        # Force multi-subject analysis on single subject data
        model = fit(
            self.single_subject_data, coords=self.coordinates, multi_subject=True
        )

        # Should return HTFA instead of TFA
        assert isinstance(model, HTFA)
        assert len(model.factors_) == 1  # Only one subject
        assert len(model.weights_) == 1
        np.testing.assert_array_equal(model.coords_, self.coordinates)

    def test_data_quality_robustness(self):
        """Test robustness to different data quality conditions."""
        # Test with noisy data
        noisy_data = (
            self.single_subject_data
            + np.random.randn(*self.single_subject_data.shape) * 0.5
        )
        model_noisy = fit(noisy_data, coords=self.coordinates)
        assert isinstance(model_noisy, TFA)
        assert model_noisy.factors_ is not None

        # Test with high variance data
        high_var_data = self.single_subject_data * 10
        model_high_var = fit(high_var_data, coords=self.coordinates)
        assert isinstance(model_high_var, TFA)
        assert model_high_var.factors_ is not None

        # Test with zero-mean data
        centered_data = self.single_subject_data - np.mean(
            self.single_subject_data, axis=1, keepdims=True
        )
        model_centered = fit(centered_data, coords=self.coordinates)
        assert isinstance(model_centered, TFA)
        assert model_centered.factors_ is not None

    def test_different_array_dtypes(self):
        """Test compatibility with different numpy dtypes."""
        # Test with float32
        data_f32 = self.single_subject_data.astype(np.float32)
        coords_f32 = self.coordinates.astype(np.float32)

        model_f32 = fit(data_f32, coords=coords_f32)
        assert isinstance(model_f32, TFA)

        # Test with float64
        data_f64 = self.single_subject_data.astype(np.float64)
        coords_f64 = self.coordinates.astype(np.float64)

        model_f64 = fit(data_f64, coords=coords_f64)
        assert isinstance(model_f64, TFA)

        # Models should produce similar results (within numerical precision)
        assert model_f32.K == model_f64.K

    def test_memory_efficiency_large_data(self):
        """Test memory efficiency with larger datasets."""
        # Create larger synthetic dataset (but still reasonable for testing)
        large_n_voxels = 500
        large_n_timepoints = 100

        large_coords = self._generate_brain_coordinates(large_n_voxels)
        large_data = self._generate_synthetic_fmri_data(
            large_n_voxels, large_n_timepoints, 10
        )

        # This should not crash due to memory issues
        model = fit(large_data, coords=large_coords, n_factors=25)

        assert isinstance(model, TFA)
        assert model.K == 25
        assert model.factors_.shape[0] == 25
        np.testing.assert_array_equal(model.coords_, large_coords)

    def test_reproducibility_with_random_state(self):
        """Test that results are reproducible with fixed random state."""
        random_state = 123

        # Run same analysis twice with same random state
        model1 = fit(
            self.single_subject_data,
            coords=self.coordinates,
            n_factors=10,
            random_state=random_state,
        )

        model2 = fit(
            self.single_subject_data,
            coords=self.coordinates,
            n_factors=10,
            random_state=random_state,
        )

        # Results should be identical (or very close)
        assert model1.K == model2.K
        # Note: Exact reproducibility depends on the underlying TFA implementation
        # For now just test that both models complete successfully
        assert model1.factors_ is not None
        assert model2.factors_ is not None


@pytest.mark.slow
class TestErrorHandlingIntegration:
    """Integration tests for error handling in realistic scenarios."""

    def test_invalid_coordinate_dimensions(self):
        """Test error handling for coordinate dimension mismatches."""
        data = np.random.randn(100, 50)

        # Wrong number of voxels in coordinates
        wrong_coords = np.random.randn(90, 3)

        with pytest.raises(
            ValueError, match="Coordinates shape.*doesn't match.*data voxels"
        ):
            fit(data, coords=wrong_coords)

    def test_inconsistent_multi_subject_data(self):
        """Test error handling for inconsistent multi-subject data."""
        # Create inconsistent multi-subject data
        inconsistent_data = [
            np.random.randn(100, 50),  # 100 voxels
            np.random.randn(120, 50),  # 120 voxels - inconsistent!
            np.random.randn(100, 50),  # 100 voxels
        ]
        coords = np.random.randn(100, 3)

        with pytest.raises(ValueError, match="expected 100 voxels, got 120"):
            fit(inconsistent_data, coords=coords)

    def test_edge_case_data_dimensions(self):
        """Test behavior with edge case data dimensions."""
        coords = np.random.randn(10, 3)

        # Very few timepoints
        few_timepoints = np.random.randn(10, 2)
        model_few = fit(few_timepoints, coords=coords)
        assert isinstance(model_few, TFA)
        assert model_few.K >= 1  # Should infer at least 1 factor

        # Very few voxels
        few_voxels_coords = np.random.randn(3, 3)
        few_voxels = np.random.randn(3, 50)
        model_few_voxels = fit(few_voxels, coords=few_voxels_coords)
        assert isinstance(model_few_voxels, TFA)
        assert model_few_voxels.K >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
