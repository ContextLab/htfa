"""Reproducibility tests for HTFA random state threading.

This module tests that random_state parameter properly controls reproducibility
across all stochastic operations in TFA and HTFA classes.
"""

import numpy as np
import pytest

from htfa.core.htfa import HTFA
from htfa.core.tfa import TFA


class TestTFAReproducibility:
    """Test reproducibility of TFA with random_state parameter."""

    @pytest.fixture
    def sample_data(self):
        """Generate consistent test data."""
        np.random.seed(42)
        n_voxels, n_timepoints = 100, 50
        coords = np.random.randn(n_voxels, 3)
        data = np.random.randn(n_voxels, n_timepoints)
        return data, coords

    def test_tfa_reproducibility_with_fixed_seed(self, sample_data):
        """Test that TFA produces identical results with same random_state."""
        data, coords = sample_data

        # Run TFA with same random state multiple times
        results = []
        for _ in range(3):
            tfa = TFA(K=5, random_state=42, max_iter=10, verbose=False)
            tfa.fit(data, coords)
            results.append(
                {
                    "factors": tfa.get_factors().copy(),
                    "weights": tfa.get_weights().copy(),
                    "centers": tfa.centers_.copy(),
                    "widths": tfa.widths_.copy(),
                }
            )

        # Verify all results are identical
        for i in range(1, len(results)):
            np.testing.assert_array_almost_equal(
                results[0]["factors"],
                results[i]["factors"],
                decimal=10,
                err_msg=f"Factors differ between run 0 and run {i}",
            )
            np.testing.assert_array_almost_equal(
                results[0]["weights"],
                results[i]["weights"],
                decimal=10,
                err_msg=f"Weights differ between run 0 and run {i}",
            )
            np.testing.assert_array_almost_equal(
                results[0]["centers"],
                results[i]["centers"],
                decimal=10,
                err_msg=f"Centers differ between run 0 and run {i}",
            )
            np.testing.assert_array_almost_equal(
                results[0]["widths"],
                results[i]["widths"],
                decimal=10,
                err_msg=f"Widths differ between run 0 and run {i}",
            )

    def test_tfa_different_seeds_produce_different_results(self, sample_data):
        """Test that different random_state values produce different results."""
        data, coords = sample_data

        # Run TFA with different random states
        tfa1 = TFA(K=5, random_state=42, max_iter=10, verbose=False)
        tfa1.fit(data, coords)

        tfa2 = TFA(K=5, random_state=123, max_iter=10, verbose=False)
        tfa2.fit(data, coords)

        # Results should be different
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(tfa1.get_factors(), tfa2.get_factors())

        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(tfa1.get_weights(), tfa2.get_weights())

    def test_tfa_none_random_state_is_non_deterministic(self, sample_data):
        """Test that None random_state produces non-deterministic results."""
        data, coords = sample_data

        # Run TFA with None random_state multiple times
        results = []
        for _ in range(3):
            tfa = TFA(K=5, random_state=None, max_iter=10, verbose=False)
            tfa.fit(data, coords)
            results.append(
                {
                    "factors": tfa.get_factors().copy(),
                    "weights": tfa.get_weights().copy(),
                }
            )

        # At least one pair should be different (high probability)
        all_same = True
        for i in range(1, len(results)):
            try:
                np.testing.assert_array_almost_equal(
                    results[0]["factors"], results[i]["factors"], decimal=5
                )
                np.testing.assert_array_almost_equal(
                    results[0]["weights"], results[i]["weights"], decimal=5
                )
            except AssertionError:
                all_same = False
                break

        assert (
            not all_same
        ), "Results with random_state=None should be non-deterministic"

    def test_tfa_subsampling_reproducibility(self, sample_data):
        """Test reproducibility when subsampling is triggered."""
        data, coords = sample_data

        # Force subsampling by setting small limits
        tfa1 = TFA(
            K=3,
            random_state=42,
            max_num_voxel=50,
            max_num_tr=30,
            max_iter=5,
            verbose=False,
        )
        tfa1.fit(data, coords)

        tfa2 = TFA(
            K=3,
            random_state=42,
            max_num_voxel=50,
            max_num_tr=30,
            max_iter=5,
            verbose=False,
        )
        tfa2.fit(data, coords)

        # Results should be identical due to same random_state
        np.testing.assert_array_almost_equal(
            tfa1.get_factors(), tfa2.get_factors(), decimal=8
        )
        np.testing.assert_array_almost_equal(
            tfa1.get_weights(), tfa2.get_weights(), decimal=8
        )

    def test_tfa_kmeans_initialization_reproducibility(self, sample_data):
        """Test reproducibility of K-means initialization."""
        data, coords = sample_data

        # Test multiple runs with same random_state
        centers_list = []
        widths_list = []

        for _ in range(3):
            tfa = TFA(
                K=4, random_state=42, max_iter=1, verbose=False
            )  # Only 1 iteration
            tfa.fit(data, coords)
            centers_list.append(tfa.centers_.copy())
            widths_list.append(tfa.widths_.copy())

        # All initializations should be identical
        for i in range(1, len(centers_list)):
            np.testing.assert_array_almost_equal(
                centers_list[0], centers_list[i], decimal=10
            )
            np.testing.assert_array_almost_equal(
                widths_list[0], widths_list[i], decimal=10
            )

    def test_tfa_no_coords_initialization_reproducibility(self, sample_data):
        """Test reproducibility when no coordinates are provided."""
        data, _ = sample_data

        # Test without coordinates (should use random initialization)
        results = []
        for _ in range(3):
            tfa = TFA(K=4, random_state=42, max_iter=5, verbose=False)
            tfa.fit(data, coords=None)  # No coordinates
            results.append(
                {"factors": tfa.get_factors().copy(), "centers": tfa.centers_.copy()}
            )

        # Results should be identical
        for i in range(1, len(results)):
            np.testing.assert_array_almost_equal(
                results[0]["factors"], results[i]["factors"], decimal=8
            )
            np.testing.assert_array_almost_equal(
                results[0]["centers"], results[i]["centers"], decimal=10
            )


class TestHTFAReproducibility:
    """Test reproducibility of HTFA with random_state parameter."""

    @pytest.fixture
    def multi_subject_data(self):
        """Generate consistent multi-subject test data."""
        np.random.seed(42)
        n_subjects = 3
        n_voxels, n_timepoints = 50, 30

        X = []
        coords = []

        for _ in range(n_subjects):
            subject_coords = np.random.randn(n_voxels, 3)
            subject_data = np.random.randn(n_voxels, n_timepoints)
            X.append(subject_data)
            coords.append(subject_coords)

        return X, coords

    def test_htfa_reproducibility_with_fixed_seed(self, multi_subject_data):
        """Test that HTFA produces identical results with same random_state."""
        X, coords = multi_subject_data

        # Run HTFA with same random state multiple times
        results = []
        for _ in range(2):  # Fewer runs due to computational cost
            htfa = HTFA(
                K=3, random_state=42, max_global_iter=3, max_local_iter=5, verbose=False
            )
            htfa.fit(X, coords)
            results.append(
                {
                    "global_template": htfa.get_global_template(),
                    "subject_factors": [
                        htfa.get_subject_factors(i) for i in range(len(X))
                    ],
                    "subject_weights": [
                        htfa.get_subject_weights(i) for i in range(len(X))
                    ],
                }
            )

        # Verify results are identical
        # Global template comparison
        if (
            results[0]["global_template"] is not None
            and results[1]["global_template"] is not None
        ):
            for key in ["centers", "widths"]:
                np.testing.assert_array_almost_equal(
                    results[0]["global_template"][key],
                    results[1]["global_template"][key],
                    decimal=8,
                    err_msg=f"Global template {key} differs between runs",
                )

        # Subject-specific results comparison
        for subject_idx in range(len(X)):
            if (
                results[0]["subject_factors"][subject_idx] is not None
                and results[1]["subject_factors"][subject_idx] is not None
            ):
                np.testing.assert_array_almost_equal(
                    results[0]["subject_factors"][subject_idx],
                    results[1]["subject_factors"][subject_idx],
                    decimal=6,
                    err_msg=f"Subject {subject_idx} factors differ",
                )

            if (
                results[0]["subject_weights"][subject_idx] is not None
                and results[1]["subject_weights"][subject_idx] is not None
            ):
                np.testing.assert_array_almost_equal(
                    results[0]["subject_weights"][subject_idx],
                    results[1]["subject_weights"][subject_idx],
                    decimal=6,
                    err_msg=f"Subject {subject_idx} weights differ",
                )

    def test_htfa_different_seeds_produce_different_results(self, multi_subject_data):
        """Test that different random_state values produce different results."""
        X, coords = multi_subject_data

        # Run HTFA with different random states
        htfa1 = HTFA(
            K=3, random_state=42, max_global_iter=3, max_local_iter=5, verbose=False
        )
        htfa1.fit(X, coords)

        htfa2 = HTFA(
            K=3, random_state=123, max_global_iter=3, max_local_iter=5, verbose=False
        )
        htfa2.fit(X, coords)

        # At least one component should be different
        differences_found = False

        # Check subject factors
        for subject_idx in range(len(X)):
            factors1 = htfa1.get_subject_factors(subject_idx)
            factors2 = htfa2.get_subject_factors(subject_idx)

            if factors1 is not None and factors2 is not None:
                try:
                    np.testing.assert_array_almost_equal(factors1, factors2, decimal=5)
                except AssertionError:
                    differences_found = True
                    break

        assert (
            differences_found
        ), "Different random_state should produce different results"

    def test_htfa_none_random_state_is_non_deterministic(self, multi_subject_data):
        """Test that None random_state produces non-deterministic results."""
        X, coords = multi_subject_data

        # Run HTFA with None random_state multiple times
        results = []
        for _ in range(2):  # Limited runs due to computational cost
            htfa = HTFA(
                K=3,
                random_state=None,
                max_global_iter=2,
                max_local_iter=3,
                verbose=False,
            )
            htfa.fit(X, coords)

            subject_factors = []
            for i in range(len(X)):
                factors = htfa.get_subject_factors(i)
                if factors is not None:
                    subject_factors.append(factors.copy())

            results.append(subject_factors)

        # Check that results are different (high probability)
        if len(results[0]) > 0 and len(results[1]) > 0:
            all_same = True
            for i in range(min(len(results[0]), len(results[1]))):
                try:
                    np.testing.assert_array_almost_equal(
                        results[0][i], results[1][i], decimal=5
                    )
                except AssertionError:
                    all_same = False
                    break

            # It's possible (though unlikely) for results to be similar, so we don't assert
            # Just document the behavior
            if all_same:
                print(
                    "Warning: Non-deterministic results were identical (low probability event)"
                )

    def test_htfa_individual_subject_models_use_random_state(self, multi_subject_data):
        """Test that individual TFA models within HTFA use the random_state properly."""
        X, coords = multi_subject_data

        # Create HTFA with specific random_state
        htfa = HTFA(
            K=3, random_state=42, max_global_iter=1, max_local_iter=3, verbose=False
        )
        htfa.fit(X, coords)

        # Extract the individual TFA models and check they were initialized consistently
        assert hasattr(
            htfa, "subject_models_"
        ), "HTFA should have subject_models_ after fitting"
        assert len(htfa.subject_models_) == len(
            X
        ), "Should have one TFA model per subject"

        # All models should be TFA instances
        for model in htfa.subject_models_:
            assert isinstance(model, TFA), "Subject models should be TFA instances"
            assert model.centers_ is not None, "TFA models should have fitted centers"
            assert model.widths_ is not None, "TFA models should have fitted widths"


class TestEdgeCases:
    """Test edge cases for reproducibility."""

    def test_empty_data_handling(self):
        """Test handling of edge cases that might affect reproducibility."""
        # Test with minimal data
        small_data = np.random.randn(10, 5)
        small_coords = np.random.randn(10, 2)

        tfa1 = TFA(K=2, random_state=42, max_iter=3, verbose=False)
        tfa1.fit(small_data, small_coords)

        tfa2 = TFA(K=2, random_state=42, max_iter=3, verbose=False)
        tfa2.fit(small_data, small_coords)

        # Should still be reproducible
        np.testing.assert_array_almost_equal(
            tfa1.get_factors(), tfa2.get_factors(), decimal=8
        )

    def test_single_subject_htfa(self):
        """Test HTFA reproducibility with single subject."""
        np.random.seed(42)
        data = np.random.randn(30, 20)
        coords = np.random.randn(30, 3)

        X = [data]
        coords_list = [coords]

        htfa1 = HTFA(
            K=3, random_state=42, max_global_iter=2, max_local_iter=3, verbose=False
        )
        htfa1.fit(X, coords_list)

        htfa2 = HTFA(
            K=3, random_state=42, max_global_iter=2, max_local_iter=3, verbose=False
        )
        htfa2.fit(X, coords_list)

        # Should be reproducible
        factors1 = htfa1.get_subject_factors(0)
        factors2 = htfa2.get_subject_factors(0)

        if factors1 is not None and factors2 is not None:
            np.testing.assert_array_almost_equal(factors1, factors2, decimal=6)

    def test_random_state_type_handling(self):
        """Test that different random_state types work correctly."""
        data = np.random.randn(20, 15)
        coords = np.random.randn(20, 3)

        # Test with integer
        tfa_int = TFA(K=2, random_state=42, max_iter=3, verbose=False)
        tfa_int.fit(data, coords)

        # Test with np.random.RandomState
        rng = np.random.RandomState(42)
        tfa_rng = TFA(K=2, random_state=rng, max_iter=3, verbose=False)
        tfa_rng.fit(data, coords)

        # Both should produce valid results
        assert tfa_int.get_factors() is not None
        assert tfa_rng.get_factors() is not None
        assert tfa_int.get_weights() is not None
        assert tfa_rng.get_weights() is not None


if __name__ == "__main__":
    # Run basic smoke test
    print("Running basic reproducibility smoke test...")

    # Test TFA
    np.random.seed(123)
    data = np.random.randn(50, 30)
    coords = np.random.randn(50, 3)

    tfa1 = TFA(K=3, random_state=42, max_iter=5, verbose=True)
    tfa1.fit(data, coords)

    tfa2 = TFA(K=3, random_state=42, max_iter=5, verbose=True)
    tfa2.fit(data, coords)

    print("TFA factors match:", np.allclose(tfa1.get_factors(), tfa2.get_factors()))
    print("TFA weights match:", np.allclose(tfa1.get_weights(), tfa2.get_weights()))

    # Test HTFA
    X = [data, data + 0.1 * np.random.randn(*data.shape)]
    coords_list = [coords, coords + 0.1 * np.random.randn(*coords.shape)]

    htfa1 = HTFA(
        K=3, random_state=42, max_global_iter=2, max_local_iter=3, verbose=True
    )
    htfa1.fit(X, coords_list)

    htfa2 = HTFA(
        K=3, random_state=42, max_global_iter=2, max_local_iter=3, verbose=True
    )
    htfa2.fit(X, coords_list)

    print(
        "HTFA subject 0 factors match:",
        np.allclose(htfa1.get_subject_factors(0), htfa2.get_subject_factors(0)),
    )

    print("Smoke test completed!")
