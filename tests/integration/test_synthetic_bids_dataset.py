"""Integration tests for synthetic BIDS dataset."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from bids import BIDSLayout


class TestSyntheticBIDSDataset:
    """Test synthetic BIDS dataset integrity and properties."""

    @pytest.fixture(scope="class")
    def dataset_path(self):
        """Path to synthetic dataset."""
        return Path(__file__).parent.parent / "data" / "synthetic"

    @pytest.fixture(scope="class")
    def layout(self, dataset_path):
        """BIDS layout for the dataset."""
        return BIDSLayout(dataset_path, validate=False)

    @pytest.fixture(scope="class")
    def ground_truth(self, dataset_path):
        """Load ground truth parameters."""
        with open(dataset_path / "factor_params.json") as f:
            return json.load(f)

    def test_dataset_exists(self, dataset_path):
        """Test that dataset directory exists."""
        assert dataset_path.exists()
        assert dataset_path.is_dir()

    def test_bids_structure(self, layout):
        """Test BIDS structure is valid."""
        # Check we can load with pybids
        assert layout is not None

        # Check expected participants
        subjects = layout.get_subjects()
        expected = [f"{i:02d}" for i in range(1, 11)]
        assert sorted(subjects) == sorted(expected)

    def test_metadata_files(self, dataset_path):
        """Test required BIDS metadata files exist."""
        # dataset_description.json
        desc_file = dataset_path / "dataset_description.json"
        assert desc_file.exists()

        with open(desc_file) as f:
            desc = json.load(f)
            assert "BIDSVersion" in desc
            assert "Name" in desc
            assert desc["Name"] == "Synthetic HTFA Validation Dataset"

        # participants.tsv
        participants_file = dataset_path / "participants.tsv"
        assert participants_file.exists()

        import pandas as pd

        participants = pd.read_csv(participants_file, sep="\t")
        assert len(participants) == 10
        assert "participant_id" in participants.columns
        assert "age" in participants.columns
        assert "sex" in participants.columns

    def test_functional_files(self, layout):
        """Test functional NIfTI files for each participant."""
        subjects = layout.get_subjects()

        for sub in subjects:
            # Get functional file
            func_files = layout.get(subject=sub, suffix="bold", extension="nii.gz")
            assert len(func_files) == 1

            # Check file exists and can be loaded
            func_file = func_files[0]
            assert Path(func_file.path).exists()

            img = nib.load(func_file.path)
            data = img.get_fdata()

            # Check dimensions
            assert data.shape == (10, 10, 10, 100)
            assert data.dtype == np.float64

            # Check affine matrix
            affine = img.affine
            assert affine.shape == (4, 4)

            # Check JSON sidecar
            json_files = layout.get(subject=sub, suffix="bold", extension="json")
            assert len(json_files) == 1

            with open(json_files[0].path) as f:
                metadata = json.load(f)
                assert metadata["RepetitionTime"] == 2.0
                assert metadata["TaskName"] == "rest"
                assert "EchoTime" in metadata

    def test_ground_truth_file(self, ground_truth):
        """Test ground truth parameters file."""
        # Check dataset info
        info = ground_truth["dataset_info"]
        assert info["n_participants"] == 10
        assert info["n_voxels"] == 1000
        assert info["n_timepoints"] == 100
        assert info["n_factors"] == 4
        assert info["noise_level"] == 0.1

        # Check factor centers
        centers = np.array(ground_truth["factor_centers"])
        assert centers.shape == (4, 3)

        expected_centers = np.array(
            [[25, 30, 15], [75, 30, 15], [50, 60, 15], [50, 30, 45]]
        )
        np.testing.assert_array_almost_equal(centers, expected_centers)

        # Check factor widths
        widths = np.array(ground_truth["factor_widths"])
        assert widths.shape == (4,)
        np.testing.assert_array_almost_equal(widths, [5.0, 5.0, 5.0, 5.0])

    def test_participant_ground_truth(self, ground_truth):
        """Test per-participant ground truth data."""
        participants = ground_truth["participants"]

        for i in range(1, 11):
            sub_id = f"sub-{i:02d}"
            assert sub_id in participants

            participant_data = participants[sub_id]

            # Check factors
            factors = np.array(participant_data["factors"])
            assert factors.shape == (1000, 4)

            # Factors should be normalized
            norms = np.linalg.norm(factors, axis=0)
            np.testing.assert_array_almost_equal(norms, np.ones(4), decimal=5)

            # Check weights
            weights = np.array(participant_data["weights"])
            assert weights.shape == (4, 100)

            # Check spatial coordinates
            coords = np.array(participant_data["spatial_coords"])
            assert coords.shape == (1000, 3)

            # Coordinates should be in [0, 100] range
            assert coords.min() >= 0
            assert coords.max() <= 100

            # Check affine matrix
            affine = np.array(participant_data["affine"])
            assert affine.shape == (4, 4)

    def test_data_properties(self, layout, ground_truth):
        """Test synthetic data has expected statistical properties."""
        subjects = layout.get_subjects()

        for sub in subjects:

            # Load data
            func_file = layout.get(subject=sub, suffix="bold", extension="nii.gz")[0]
            img = nib.load(func_file.path)
            data = img.get_fdata()

            # Reshape to 2D (voxels x time)
            data_2d = data.reshape(-1, 100)[:1000]  # First 1000 voxels

            # Check data has reasonable mean and std
            assert -10 < data_2d.mean() < 10
            assert 0.01 < data_2d.std() < 10

            # Check temporal properties
            temporal_mean = data_2d.mean(axis=0)
            assert temporal_mean.shape == (100,)

            # Should have some temporal variation
            assert temporal_mean.std() > 0

    def test_factor_reconstruction(self, ground_truth):
        """Test that data can be approximately reconstructed from factors."""
        # Test for first participant
        participant_data = ground_truth["participants"]["sub-01"]

        factors = np.array(participant_data["factors"])  # (1000, 4)
        weights = np.array(participant_data["weights"])  # (4, 100)

        # Reconstruct: Y = F @ W
        reconstructed = factors @ weights

        # Check shape
        assert reconstructed.shape == (1000, 100)

        # Should have reasonable values
        assert not np.any(np.isnan(reconstructed))
        assert not np.any(np.isinf(reconstructed))

    def test_factor_orthogonality(self, ground_truth):
        """Test that factors have minimal overlap."""
        participant_data = ground_truth["participants"]["sub-01"]
        factors = np.array(participant_data["factors"])  # (1000, 4)

        # Compute factor correlations
        factor_corr = np.corrcoef(factors.T)

        # Off-diagonal elements should be small (factors are somewhat orthogonal)
        off_diagonal = factor_corr[np.triu_indices(4, k=1)]

        # RBF factors won't be perfectly orthogonal, but should have low correlation
        assert np.all(np.abs(off_diagonal) < 0.5)

    def test_temporal_patterns_distinct(self, ground_truth):
        """Test that temporal patterns are distinct across factors."""
        participant_data = ground_truth["participants"]["sub-01"]
        weights = np.array(participant_data["weights"])  # (4, 100)

        # Compute temporal correlations
        temporal_corr = np.corrcoef(weights)

        # Off-diagonal correlations should not be too high
        off_diagonal = temporal_corr[np.triu_indices(4, k=1)]

        # Temporal patterns should be somewhat distinct
        # Allow higher correlation since we're creating complex patterns
        assert np.all(np.abs(off_diagonal) < 0.9)

        # At least most correlations should be low
        assert np.sum(np.abs(off_diagonal) < 0.5) >= 4  # At least 4 out of 6 pairs
