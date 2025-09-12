"""Tests for htfa.bids module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from htfa.bids import (
    build_bids_query,
    extract_bids_metadata,
    fit_bids,
    load_bids_data,
    load_bids_for_htfa,
    parse_bids_dataset,
    validate_bids_structure,
)


class TestParseBidsDataset:
    """Test parse_bids_dataset function."""

    @patch("htfa.bids.BIDSLayout")
    @patch("htfa.bids.validate_bids_path")
    def test_parse_bids_basic(self, mock_validate, mock_layout_class):
        """Test basic BIDS dataset parsing."""
        mock_validate.return_value = Path("/path/to/dataset")

        mock_layout = MagicMock()
        mock_layout.get_subjects.return_value = ["01", "02", "03"]
        mock_layout.get_sessions.return_value = ["1", "2"]
        mock_layout.get_tasks.return_value = ["rest", "task"]
        mock_layout_class.return_value = mock_layout

        result = parse_bids_dataset("/path/to/dataset")

        assert result == mock_layout
        mock_validate.assert_called_once_with("/path/to/dataset")
        mock_layout_class.assert_called_once_with(
            "/path/to/dataset", derivatives=True, validate=False, absolute_paths=True
        )

    @patch("htfa.bids.BIDSLayout")
    @patch("htfa.bids.validate_bids_path")
    def test_parse_bids_with_options(self, mock_validate, mock_layout_class):
        """Test BIDS parsing with options."""
        mock_validate.return_value = Path("/path/to/dataset")
        mock_layout = MagicMock()
        mock_layout_class.return_value = mock_layout

        # Set up mock returns
        mock_layout.get_subjects.return_value = ["01"]
        mock_layout.get_sessions.return_value = []
        mock_layout.get_tasks.return_value = ["rest"]

        result = parse_bids_dataset(
            "/path/to/dataset", derivatives=False, validate=True
        )

        mock_layout_class.assert_called_once_with(
            "/path/to/dataset", derivatives=False, validate=True, absolute_paths=True
        )

    @patch("htfa.bids.BIDSLayout")
    @patch("htfa.bids.validate_bids_path")
    def test_parse_bids_error_handling(self, mock_validate, mock_layout_class):
        """Test error handling in BIDS parsing."""
        mock_validate.return_value = Path("/path/to/dataset")
        mock_layout_class.side_effect = Exception("BIDS parsing error")

        with pytest.raises(RuntimeError, match="Failed to parse BIDS dataset"):
            parse_bids_dataset("/path/to/dataset")


class TestValidateBidsStructure:
    """Test validate_bids_structure function."""

    def test_validate_bids_valid_directory(self):
        """Test validation of valid BIDS directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            # Create minimal BIDS structure
            (path / "dataset_description.json").write_text('{"Name": "Test"}')
            (path / "participants.tsv").touch()
            (path / "sub-01").mkdir()

            result = validate_bids_structure(path)

            assert result["is_valid"] is True
            assert result["has_participants"] is True
            assert result["has_description"] is True
            assert result["n_subjects"] == 1

    def test_validate_bids_missing_description(self):
        """Test validation with missing dataset_description.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / "participants.tsv").touch()
            (path / "sub-01").mkdir()

            result = validate_bids_structure(path)

            assert result["is_valid"] is False
            assert result["has_description"] is False
            assert "missing dataset_description.json" in result["warnings"][0]

    def test_validate_bids_no_subjects(self):
        """Test validation with no subject directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / "dataset_description.json").write_text('{"Name": "Test"}')
            (path / "participants.tsv").touch()

            result = validate_bids_structure(path)

            assert result["n_subjects"] == 0
            assert "No subject directories found" in result["warnings"][0]


class TestExtractBidsMetadata:
    """Test extract_bids_metadata function."""

    @patch("htfa.bids.BIDSLayout")
    def test_extract_metadata_basic(self, mock_layout_class):
        """Test basic metadata extraction."""
        mock_layout = MagicMock()
        mock_layout.get_subjects.return_value = ["01", "02"]
        mock_layout.get_sessions.return_value = ["1"]
        mock_layout.get_tasks.return_value = ["rest", "task"]
        mock_layout.get_runs.return_value = []
        mock_layout.get.return_value = [MagicMock(), MagicMock()]
        mock_layout.description = {"Name": "TestDataset"}
        mock_layout_class.return_value = mock_layout

        metadata = extract_bids_metadata(mock_layout)

        assert metadata["n_subjects"] == 2
        assert metadata["n_sessions"] == 1
        assert metadata["n_tasks"] == 2
        assert metadata["n_runs"] == 0
        assert metadata["dataset_name"] == "TestDataset"
        assert "subjects" in metadata
        assert "tasks" in metadata

    @patch("htfa.bids.BIDSLayout")
    def test_extract_metadata_with_filters(self, mock_layout_class):
        """Test metadata extraction with filters."""
        mock_layout = MagicMock()
        mock_layout.get_subjects.return_value = ["01"]
        mock_layout.get_sessions.return_value = []
        mock_layout.get_tasks.return_value = ["rest"]
        mock_layout.get_runs.return_value = ["1", "2"]
        mock_layout.get.return_value = [MagicMock()]
        mock_layout.description = None
        mock_layout_class.return_value = mock_layout

        metadata = extract_bids_metadata(mock_layout, subject="01", task="rest")

        assert metadata["n_subjects"] == 1
        assert metadata["n_tasks"] == 1
        assert metadata["n_runs"] == 2
        mock_layout.get.assert_called_with(
            return_type="object", subject="01", task="rest"
        )


class TestBuildBidsQuery:
    """Test build_bids_query function."""

    def test_build_query_empty(self):
        """Test building empty query."""
        query = build_bids_query()
        assert query == {}

    def test_build_query_with_filters(self):
        """Test building query with filters."""
        query = build_bids_query(subject="01", task="rest", session="1", run="2")

        assert query["subject"] == "01"
        assert query["task"] == "rest"
        assert query["session"] == "1"
        assert query["run"] == "2"

    def test_build_query_with_lists(self):
        """Test building query with list values."""
        query = build_bids_query(subject=["01", "02"], task=["rest", "task"])

        assert query["subject"] == ["01", "02"]
        assert query["task"] == ["rest", "task"]

    def test_build_query_filters_none(self):
        """Test that None values are filtered out."""
        query = build_bids_query(subject="01", task=None, session="1")

        assert "subject" in query
        assert "task" not in query
        assert "session" in query


class TestLoadBidsData:
    """Test load_bids_data function."""

    @patch("htfa.bids.nib.load")
    @patch("htfa.bids.BIDSLayout")
    def test_load_bids_data_single_subject(self, mock_layout_class, mock_nib_load):
        """Test loading data for single subject."""
        # Setup mock layout
        mock_layout = MagicMock()
        mock_file = MagicMock()
        mock_file.path = "/data/sub-01_bold.nii.gz"
        mock_file.get_entities.return_value = {"subject": "01"}
        mock_layout.get.return_value = [mock_file]
        mock_layout_class.return_value = mock_layout

        # Setup mock image
        mock_img = MagicMock()
        mock_data = np.random.randn(10, 10, 10, 50)
        mock_img.get_fdata.return_value = mock_data
        mock_img.affine = np.eye(4)
        mock_nib_load.return_value = mock_img

        result = load_bids_data(mock_layout, subject="01")

        assert len(result) == 1
        assert "data" in result[0]
        assert "metadata" in result[0]
        assert result[0]["data"].shape == (10, 10, 10, 50)

    @patch("htfa.bids.nib.load")
    @patch("htfa.bids.BIDSLayout")
    def test_load_bids_data_multiple_subjects(self, mock_layout_class, mock_nib_load):
        """Test loading data for multiple subjects."""
        mock_layout = MagicMock()

        # Create mock files for two subjects
        mock_files = []
        for i in range(2):
            mock_file = MagicMock()
            mock_file.path = f"/data/sub-0{i+1}_bold.nii.gz"
            mock_file.get_entities.return_value = {"subject": f"0{i+1}"}
            mock_files.append(mock_file)

        mock_layout.get.return_value = mock_files
        mock_layout_class.return_value = mock_layout

        # Mock image loading
        mock_img = MagicMock()
        mock_img.get_fdata.return_value = np.random.randn(10, 10, 10, 50)
        mock_img.affine = np.eye(4)
        mock_nib_load.return_value = mock_img

        result = load_bids_data(mock_layout)

        assert len(result) == 2

    @patch("htfa.bids.BIDSLayout")
    def test_load_bids_data_no_files(self, mock_layout_class):
        """Test when no files are found."""
        mock_layout = MagicMock()
        mock_layout.get.return_value = []
        mock_layout_class.return_value = mock_layout

        with pytest.raises(ValueError, match="No BOLD files found"):
            load_bids_data(mock_layout, subject="99")


class TestFitBids:
    """Test fit_bids function."""

    @patch("htfa.bids.HTFA")
    @patch("htfa.bids.load_bids_for_htfa")
    @patch("htfa.bids.parse_bids_dataset")
    def test_fit_bids_basic(self, mock_parse, mock_load, mock_htfa_class):
        """Test basic BIDS fitting."""
        # Setup mocks
        mock_layout = MagicMock()
        mock_parse.return_value = mock_layout

        mock_data = [np.random.randn(100, 50) for _ in range(3)]
        mock_coords = [np.random.randn(100, 3) for _ in range(3)]
        mock_load.return_value = (mock_data, mock_coords, {})

        mock_model = MagicMock()
        mock_htfa_class.return_value = mock_model

        result = fit_bids("/path/to/dataset", n_factors=5, max_iter=10)

        assert result == mock_model
        mock_parse.assert_called_once()
        mock_load.assert_called_once()
        mock_htfa_class.assert_called_once_with(n_factors=5, max_iter=10)
        mock_model.fit.assert_called_once()

    @patch("htfa.bids.TFA")
    @patch("htfa.bids.load_bids_for_htfa")
    @patch("htfa.bids.parse_bids_dataset")
    def test_fit_bids_single_subject(self, mock_parse, mock_load, mock_tfa_class):
        """Test BIDS fitting with single subject."""
        mock_layout = MagicMock()
        mock_parse.return_value = mock_layout

        # Single subject data
        mock_data = [np.random.randn(100, 50)]
        mock_coords = [np.random.randn(100, 3)]
        mock_load.return_value = (mock_data, mock_coords, {})

        mock_model = MagicMock()
        mock_tfa_class.return_value = mock_model

        result = fit_bids("/path/to/dataset", n_factors=5)

        # Should use TFA for single subject
        mock_tfa_class.assert_called_once_with(n_factors=5)
        mock_model.fit.assert_called_once()


class TestLoadBidsForHtfa:
    """Test load_bids_for_htfa function."""

    @patch("htfa.bids.nib.load")
    def test_load_bids_for_htfa(self, mock_nib_load):
        """Test loading BIDS data for HTFA."""
        mock_layout = MagicMock()

        # Create mock files
        mock_files = []
        for i in range(2):
            mock_file = MagicMock()
            mock_file.path = f"/data/sub-0{i+1}_bold.nii.gz"
            mock_file.get_entities.return_value = {"subject": f"0{i+1}"}
            mock_files.append(mock_file)

        mock_layout.get.return_value = mock_files

        # Mock image with 4D data
        mock_img = MagicMock()
        mock_data_4d = np.random.randn(10, 10, 10, 50)
        mock_img.get_fdata.return_value = mock_data_4d
        mock_img.affine = np.eye(4)
        mock_img.shape = (10, 10, 10, 50)
        mock_nib_load.return_value = mock_img

        data_list, coords_list, metadata = load_bids_for_htfa(mock_layout)

        assert len(data_list) == 2
        assert len(coords_list) == 2
        assert data_list[0].shape == (1000, 50)  # Flattened spatial dims
        assert coords_list[0].shape == (1000, 3)

    @patch("htfa.bids.nib.load")
    def test_load_bids_for_htfa_with_mask(self, mock_nib_load):
        """Test loading with brain mask."""
        mock_layout = MagicMock()

        mock_file = MagicMock()
        mock_file.path = "/data/sub-01_bold.nii.gz"
        mock_file.get_entities.return_value = {"subject": "01"}
        mock_layout.get.return_value = [mock_file]

        # Mock BOLD image
        mock_bold_data = np.random.randn(10, 10, 10, 50)
        mock_bold_img = MagicMock()
        mock_bold_img.get_fdata.return_value = mock_bold_data
        mock_bold_img.affine = np.eye(4)
        mock_bold_img.shape = (10, 10, 10, 50)

        # Mock mask
        mask = np.ones((10, 10, 10), dtype=bool)
        mask[5:, :, :] = False  # Half mask

        mock_nib_load.return_value = mock_bold_img

        data_list, coords_list, metadata = load_bids_for_htfa(mock_layout, mask=mask)

        assert data_list[0].shape == (500, 50)  # Only masked voxels
        assert coords_list[0].shape == (500, 3)
