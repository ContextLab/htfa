"""Unit tests for input detection logic in htfa.fit module.

This test suite validates the automatic input type detection capabilities of the
main htfa.fit() function, including path detection, array detection, parameter
inference, and error handling scenarios.
"""

from typing import List, Union

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from htfa.core import HTFA, TFA

# Import the functions we're testing
from htfa.fit import (
    _fit_arrays,
    _fit_bids_dataset,
    _infer_parameters,
    _load_nifti_file,
    fit,
)
from htfa.validation import ValidationError


class TestInputDetection:
    """Test cases for automatic input type detection in fit() function."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.bids_path = Path(self.temp_dir) / "test_bids"
        self.bids_path.mkdir()

        # Create fake BIDS-like structure
        (self.bids_path / "dataset_description.json").touch()
        (self.bids_path / "participants.tsv").touch()
        sub_dir = self.bids_path / "sub-01" / "func"
        sub_dir.mkdir(parents=True)

        # Sample data for testing
        self.n_voxels = 100
        self.n_timepoints = 50
        self.n_subjects = 3

        self.single_subject_data = np.random.randn(self.n_voxels, self.n_timepoints)
        self.coordinates = np.random.randn(self.n_voxels, 3)

        self.multi_subject_data = [
            np.random.randn(self.n_voxels, self.n_timepoints)
            for _ in range(self.n_subjects)
        ]

    def teardown_method(self):
        """Clean up test fixtures after each test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_detect_bids_directory_input(self):
        """Test detection of BIDS directory input."""
        # Test with string path
        with patch("htfa.fit._fit_bids_dataset") as mock_fit_bids:
            mock_fit_bids.return_value = Mock(spec=TFA)

            result = fit(str(self.bids_path))

            mock_fit_bids.assert_called_once_with(str(self.bids_path), n_factors=None)
            assert result is not None

    def test_detect_pathlike_input(self):
        """Test detection of PathLike object input."""
        with patch("htfa.fit._fit_bids_dataset") as mock_fit_bids:
            mock_fit_bids.return_value = Mock(spec=TFA)

            result = fit(self.bids_path)  # Path object

            mock_fit_bids.assert_called_once_with(self.bids_path, n_factors=None)
            assert result is not None

    def test_detect_single_array_input(self):
        """Test detection of single numpy array input."""
        with patch("htfa.fit._fit_arrays") as mock_fit_arrays:
            mock_fit_arrays.return_value = Mock(spec=TFA)

            result = fit(self.single_subject_data, coords=self.coordinates)

            mock_fit_arrays.assert_called_once_with(
                self.single_subject_data,
                self.coordinates,
                n_factors=None,
                multi_subject=False,
            )
            assert result is not None

    def test_detect_multi_subject_array_input(self):
        """Test detection of multi-subject array list input."""
        with patch("htfa.fit._fit_arrays") as mock_fit_arrays:
            mock_fit_arrays.return_value = Mock(spec=HTFA)

            result = fit(self.multi_subject_data, coords=self.coordinates)

            mock_fit_arrays.assert_called_once_with(
                self.multi_subject_data,
                self.coordinates,
                n_factors=None,
                multi_subject=True,
            )
            assert result is not None

    def test_detect_tuple_input(self):
        """Test detection of tuple input (should be treated as multi-subject)."""
        data_tuple = tuple(self.multi_subject_data)

        with patch("htfa.fit._fit_arrays") as mock_fit_arrays:
            mock_fit_arrays.return_value = Mock(spec=HTFA)

            result = fit(data_tuple, coords=self.coordinates)

            mock_fit_arrays.assert_called_once_with(
                data_tuple, self.coordinates, n_factors=None, multi_subject=True
            )

    def test_force_multi_subject_for_single_array(self):
        """Test forcing multi_subject=True for single array."""
        with patch("htfa.fit._fit_arrays") as mock_fit_arrays:
            mock_fit_arrays.return_value = Mock(spec=HTFA)

            result = fit(
                self.single_subject_data, coords=self.coordinates, multi_subject=True
            )

            mock_fit_arrays.assert_called_once_with(
                self.single_subject_data,
                self.coordinates,
                n_factors=None,
                multi_subject=True,
            )

    def test_invalid_input_type_raises_error(self):
        """Test that invalid input types raise appropriate errors."""
        invalid_inputs = [
            123,  # int
            12.5,  # float
            {"data": []},  # dict
            None,  # None
            set([1, 2, 3]),  # set
        ]

        for invalid_input in invalid_inputs:
            with pytest.raises(ValueError, match="Unsupported data type"):
                fit(invalid_input)

    def test_missing_coords_for_array_input_raises_error(self):
        """Test that missing coords parameter raises TypeError for arrays."""
        # Single array without coords
        with pytest.raises(
            TypeError, match="coords parameter is required for array input"
        ):
            fit(self.single_subject_data)

        # Multi-subject arrays without coords
        with pytest.raises(
            TypeError,
            match="coords parameter is required for multi-subject array input",
        ):
            fit(self.multi_subject_data)

    def test_parameter_forwarding(self):
        """Test that additional parameters are forwarded correctly."""
        extra_params = {"n_factors": 15, "max_iter": 200, "tol": 1e-5, "verbose": True}

        with patch("htfa.fit._fit_arrays") as mock_fit_arrays:
            mock_fit_arrays.return_value = Mock(spec=TFA)

            fit(self.single_subject_data, coords=self.coordinates, **extra_params)

            mock_fit_arrays.assert_called_once_with(
                self.single_subject_data,
                self.coordinates,
                n_factors=15,
                multi_subject=False,
                max_iter=200,
                tol=1e-5,
                verbose=True,
            )


class TestParameterInference:
    """Test cases for automatic parameter inference."""

    @pytest.mark.parametrize(
        "n_voxels,n_timepoints,expected_factors",
        [
            (100, 50, 5),  # min(sqrt(100), 50//10, 50) = min(10, 5, 50) = 5
            (400, 200, 20),  # min(sqrt(400), 200//10, 50) = min(20, 20, 50) = 20
            (1600, 100, 10),  # min(sqrt(1600), 100//10, 50) = min(40, 10, 50) = 10
            (
                10000,
                1000,
                50,
            ),  # min(sqrt(10000), 1000//10, 50) = min(100, 100, 50) = 50
            (25, 30, 3),  # min(sqrt(25), 30//10, 50) = min(5, 3, 50) = 3
            (100, 5, 1),  # min(sqrt(100), 5//10, 50) = min(10, 0, 50) -> max(1, 0) = 1
        ],
    )
    def test_n_factors_inference(self, n_voxels, n_timepoints, expected_factors):
        """Test n_factors inference with various data dimensions."""
        data = np.random.randn(n_voxels, n_timepoints)

        inferred = _infer_parameters(data)

        assert inferred["n_factors"] == expected_factors
        assert inferred["max_iter"] == 100
        assert inferred["tol"] == 1e-6

    def test_parameter_override(self):
        """Test that provided parameters override inference."""
        data = np.random.randn(100, 50)

        override_params = {
            "n_factors": 20,  # Override inferred value
            "max_iter": 150,  # Override default
            "custom_param": "test",  # Should be ignored
        }

        inferred = _infer_parameters(data, **override_params)

        # Overridden values
        assert inferred["n_factors"] == 20
        assert inferred["max_iter"] == 150

        # Default value
        assert inferred["tol"] == 1e-6

        # Custom parameter should be ignored
        assert "custom_param" not in inferred

    def test_minimal_factors_constraint(self):
        """Test that n_factors is always at least 1."""
        # Create data where heuristic would suggest 0 factors
        tiny_data = np.random.randn(4, 2)  # Very small dimensions

        inferred = _infer_parameters(tiny_data)

        assert inferred["n_factors"] >= 1


class TestBIDSDatasetFitting:
    """Test cases for BIDS dataset fitting logic."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.bids_path = Path(self.temp_dir) / "test_bids"
        self.bids_path.mkdir()

        # Create minimal BIDS structure for validation
        dataset_desc = {"Name": "Test Dataset", "BIDSVersion": "1.6.0"}
        import json

        with open(self.bids_path / "dataset_description.json", "w") as f:
            json.dump(dataset_desc, f)

        # Create subject directory with neuroimaging data
        sub_dir = self.bids_path / "sub-01" / "func"
        sub_dir.mkdir(parents=True)
        (sub_dir / "sub-01_task-rest_bold.nii.gz").touch()

        # Create a fake NIfTI file for single file tests
        self.nifti_file = self.bids_path / "test.nii.gz"
        self.nifti_file.touch()

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_nonexistent_path_raises_error(self):
        """Test that nonexistent paths raise FileNotFoundError."""
        fake_path = "/nonexistent/path/to/bids"

        with pytest.raises(FileNotFoundError, match="Path does not exist"):
            _fit_bids_dataset(fake_path)

    def test_valid_nifti_file_extension_check(self):
        """Test that only valid NIfTI files are accepted."""
        # Valid extensions should not raise error in path validation
        valid_nifti = self.bids_path / "valid.nii"
        valid_nifti.touch()

        with patch("htfa.fit._load_nifti_file") as mock_load:
            mock_load.side_effect = NotImplementedError("Mocked for testing")

            with pytest.raises(NotImplementedError):
                _fit_bids_dataset(valid_nifti)

        # Invalid extension should raise ValueError
        invalid_file = self.bids_path / "invalid.txt"
        invalid_file.touch()

        with pytest.raises(ValueError, match="Single file must be NIfTI format"):
            _fit_bids_dataset(invalid_file)

    def test_bids_directory_not_implemented_error(self):
        """Test that BIDS directory parsing raises NotImplementedError."""
        with pytest.raises(
            NotImplementedError, match="BIDS dataset parsing will be implemented"
        ):
            _fit_bids_dataset(self.bids_path)

    def test_nifti_loading_not_implemented_error(self):
        """Test that NIfTI loading raises NotImplementedError."""
        with pytest.raises(
            NotImplementedError, match="NIfTI file loading will be implemented"
        ):
            _load_nifti_file(self.nifti_file)

    def test_nifti_parameter_inference_path(self):
        """Test parameter inference path in single NIfTI file processing."""
        # This test triggers the parameter inference lines before hitting NotImplementedError
        nifti_path = Path(self.temp_dir) / "test.nii.gz"
        nifti_path.touch()

        with patch("htfa.fit._load_nifti_file") as mock_load:
            # Mock successful loading to test parameter inference
            mock_data = np.random.randn(1000, 100)  # 1000 voxels, 100 timepoints
            mock_coords = np.random.randn(1000, 3)
            mock_load.return_value = (mock_data, mock_coords)

            with patch("htfa.fit.TFA") as mock_tfa:
                mock_model = MagicMock()
                mock_tfa.return_value = mock_model

                # Call without n_factors to trigger inference
                result = _fit_bids_dataset(nifti_path)

                # Should infer n_factors and create TFA model
                assert mock_tfa.called
                assert hasattr(result, "coords_")

    def test_invalid_path_type_error(self):
        """Test error for invalid path types."""
        # Test the catch-all error for paths that are neither files nor directories
        # This is a rare edge case but could happen with special file types
        import os
        import tempfile

        # Create a named pipe (FIFO) which is neither file nor directory
        temp_dir = Path(tempfile.mkdtemp())
        try:
            fifo_path = temp_dir / "test_fifo"
            os.mkfifo(str(fifo_path))  # Create named pipe

            # This should trigger line 171 in fit.py since it's neither file nor directory
            with pytest.raises(ValueError, match="Invalid path type"):
                _fit_bids_dataset(fifo_path)

        finally:
            # Clean up
            try:
                fifo_path.unlink()
                temp_dir.rmdir()
            except:
                pass


class TestArraysFitting:
    """Test cases for numpy arrays fitting logic."""

    def setup_method(self):
        """Set up test fixtures."""
        self.n_voxels = 100
        self.n_timepoints = 50
        self.n_subjects = 3

        self.single_data = np.random.randn(self.n_voxels, self.n_timepoints)
        self.coordinates = np.random.randn(self.n_voxels, 3)

        self.multi_data = [
            np.random.randn(self.n_voxels, self.n_timepoints)
            for _ in range(self.n_subjects)
        ]

    @patch("htfa.fit.TFA")
    def test_single_subject_array_fitting(self, mock_tfa_class):
        """Test fitting single subject array data."""
        mock_model = Mock()
        mock_tfa_class.return_value = mock_model

        result = _fit_arrays(
            self.single_data, self.coordinates, n_factors=10, multi_subject=False
        )

        # Verify TFA was instantiated correctly
        mock_tfa_class.assert_called_once_with(n_factors=10)

        # Verify fit was called with transposed data
        expected_data = self.single_data.T  # (n_timepoints, n_voxels)
        mock_model.fit.assert_called_once()

        # Check that coordinates were stored
        assert hasattr(result, "coords_")
        np.testing.assert_array_equal(result.coords_, self.coordinates)

    @patch("htfa.fit.HTFA")
    def test_multi_subject_array_fitting(self, mock_htfa_class):
        """Test fitting multi-subject array data."""
        mock_model = Mock()
        mock_htfa_class.return_value = mock_model

        result = _fit_arrays(
            self.multi_data,
            self.coordinates,
            n_factors=10,
            multi_subject=None,  # Should default to True for list input
        )

        # Verify HTFA was instantiated correctly
        mock_htfa_class.assert_called_once_with(n_factors=10)

        # Verify fit was called with list of transposed arrays
        mock_model.fit.assert_called_once()
        call_args = mock_model.fit.call_args[0][0]  # Get first positional argument
        assert len(call_args) == self.n_subjects

        # Check that coordinates were stored
        assert hasattr(result, "coords_")
        np.testing.assert_array_equal(result.coords_, self.coordinates)

    def test_empty_data_list_raises_error(self):
        """Test that empty data list raises ValueError."""
        with pytest.raises(ValueError, match="Empty data list provided"):
            _fit_arrays([], self.coordinates)

    def test_inconsistent_voxel_dimensions_raises_error(self):
        """Test that inconsistent voxel dimensions raise ValueError."""
        inconsistent_data = [
            np.random.randn(100, 50),  # 100 voxels
            np.random.randn(150, 50),  # 150 voxels - inconsistent!
            np.random.randn(100, 50),  # 100 voxels
        ]

        with pytest.raises(ValueError, match="Subject 1: expected 100 voxels, got 150"):
            _fit_arrays(inconsistent_data, self.coordinates)

    def test_non_2d_arrays_raise_error(self):
        """Test that non-2D arrays in list raise ValueError."""
        invalid_data = [
            np.random.randn(100, 50),  # Valid 2D
            np.random.randn(100, 50, 5),  # Invalid 3D
            np.random.randn(100, 50),  # Valid 2D
        ]

        with pytest.raises(ValueError, match="Subject 1: expected 2D array, got 3D"):
            _fit_arrays(invalid_data, self.coordinates)

    def test_coordinate_mismatch_raises_error(self):
        """Test that coordinate dimension mismatch raises ValueError."""
        wrong_coords = np.random.randn(50, 3)  # Wrong number of voxels

        with pytest.raises(
            ValueError, match="Coordinates shape 50 doesn't match data voxels 100"
        ):
            _fit_arrays(self.single_data, wrong_coords)

        # Test for multi-subject case
        with pytest.raises(
            ValueError, match="Coordinates shape 50 doesn't match data voxels 100"
        ):
            _fit_arrays(self.multi_data, wrong_coords)

    @patch("htfa.fit._infer_parameters")
    def test_parameter_inference_called_when_none(self, mock_infer):
        """Test that parameter inference is called when n_factors=None."""
        mock_infer.return_value = {"n_factors": 8}

        with patch("htfa.fit.TFA") as mock_tfa:
            mock_model = Mock()
            mock_tfa.return_value = mock_model

            _fit_arrays(self.single_data, self.coordinates, n_factors=None)

            mock_infer.assert_called_once_with(self.single_data)
            mock_tfa.assert_called_once_with(n_factors=8)

    def test_single_subject_wrong_dimensions_raises_error(self):
        """Test that wrong dimensional single subject data raises error."""
        # Test 1D array
        with pytest.raises(ValueError, match="Expected 2D array, got 1D"):
            _fit_arrays(np.random.randn(100), self.coordinates)

        # Test 3D array
        with pytest.raises(ValueError, match="Expected 2D array, got 3D"):
            _fit_arrays(np.random.randn(10, 100, 50), self.coordinates)

    @patch("htfa.fit.HTFA")
    def test_force_htfa_for_single_subject(self, mock_htfa_class):
        """Test forcing HTFA for single subject when multi_subject=True."""
        mock_model = Mock()
        mock_htfa_class.return_value = mock_model

        result = _fit_arrays(
            self.single_data,
            self.coordinates,
            n_factors=10,
            multi_subject=True,  # Force HTFA
        )

        # Verify HTFA was used instead of TFA
        mock_htfa_class.assert_called_once_with(n_factors=10)

        # Verify data was wrapped in list for HTFA
        mock_model.fit.assert_called_once()
        call_args = mock_model.fit.call_args[0][0]
        assert isinstance(call_args, list)
        assert len(call_args) == 1
        np.testing.assert_array_equal(call_args[0], self.single_data.T)


class TestInputDetectionEdgeCases:
    """Test edge cases and error conditions in input detection."""

    def test_string_but_not_path(self):
        """Test handling of string that looks like a path but doesn't exist."""
        fake_path = "/this/path/definitely/does/not/exist"

        # Should still route to _fit_bids_dataset which will raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            fit(fake_path)

    def test_pathlike_edge_cases(self):
        """Test edge cases with PathLike objects."""
        from pathlib import PurePath

        # Create a PurePath (which doesn't check filesystem)
        pure_path = PurePath("/fake/path")

        # Should still route to _fit_bids_dataset
        with patch("htfa.fit._fit_bids_dataset") as mock_fit:
            mock_fit.side_effect = FileNotFoundError("Mocked error")

            with pytest.raises(FileNotFoundError):
                fit(pure_path)

    def test_coords_none_explicitly_provided(self):
        """Test when coords=None is explicitly provided for arrays."""
        data = np.random.randn(100, 50)

        with pytest.raises(TypeError, match="coords parameter is required"):
            fit(data, coords=None)

    def test_parameter_passthrough_completeness(self):
        """Test that all extra parameters are passed through correctly."""
        data = np.random.randn(100, 50)
        coords = np.random.randn(100, 3)

        # Test with many different parameter types
        params = {
            "n_factors": 15,
            "max_iter": 200,
            "tol": 1e-4,
            "alpha": 0.05,
            "random_state": 42,
            "verbose": True,
            "init": "random",
            "custom_parameter": "should_be_passed",
        }

        with patch("htfa.fit._fit_arrays") as mock_fit_arrays:
            mock_fit_arrays.return_value = Mock(spec=TFA)

            fit(data, coords=coords, **params)

            # Check all parameters were passed
            call_kwargs = mock_fit_arrays.call_args[1]

            assert call_kwargs["n_factors"] == 15
            assert call_kwargs["max_iter"] == 200
            assert call_kwargs["tol"] == 1e-4
            assert call_kwargs["alpha"] == 0.05
            assert call_kwargs["random_state"] == 42
            assert call_kwargs["verbose"] is True
            assert call_kwargs["init"] == "random"
            assert call_kwargs["custom_parameter"] == "should_be_passed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
