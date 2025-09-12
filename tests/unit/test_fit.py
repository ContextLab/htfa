"""Tests for htfa.fit module."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from htfa.core import HTFA, TFA
from htfa.fit import _fit_arrays, _fit_bids_dataset, _infer_parameters, fit


class TestFitFunction:
    """Test the main fit() function."""

    def test_fit_with_single_array(self):
        """Test fit with single subject array input."""
        data = np.random.randn(100, 50)  # 100 voxels, 50 timepoints
        coords = np.random.randn(100, 3)

        model = fit(data, coords=coords, n_factors=5, max_iter=5)

        assert isinstance(model, TFA)
        assert model.K == 5
        assert hasattr(model, "coords_")
        assert np.array_equal(model.coords_, coords)

    def test_fit_with_multi_subject_arrays(self):
        """Test fit with multi-subject array input."""
        data = [np.random.randn(100, 50) for _ in range(3)]
        coords = np.random.randn(100, 3)

        model = fit(data, coords=coords, n_factors=5, max_iter=5)

        assert isinstance(model, HTFA)
        assert model.K == 5
        assert hasattr(model, "coords_")
        assert np.array_equal(model.coords_, coords)

    def test_fit_single_array_force_multi_subject(self):
        """Test forcing HTFA for single subject."""
        data = np.random.randn(100, 50)
        coords = np.random.randn(100, 3)

        model = fit(data, coords=coords, n_factors=5, multi_subject=True, max_iter=5)

        assert isinstance(model, HTFA)
        assert model.K == 5

    def test_fit_without_coords_raises_error(self):
        """Test that array input without coords raises error."""
        data = np.random.randn(100, 50)

        with pytest.raises(TypeError, match="coords parameter is required"):
            fit(data)

    def test_fit_multi_array_without_coords_raises_error(self):
        """Test that multi-array input without coords raises error."""
        data = [np.random.randn(100, 50) for _ in range(3)]

        with pytest.raises(TypeError, match="coords parameter is required"):
            fit(data)

    def test_fit_with_invalid_data_type(self):
        """Test fit with invalid data type."""
        with pytest.raises(ValueError, match="Unsupported data type"):
            fit({"invalid": "data"})

    def test_fit_infers_n_factors(self):
        """Test that n_factors is inferred when not provided."""
        data = np.random.randn(100, 50)
        coords = np.random.randn(100, 3)

        model = fit(data, coords=coords, max_iter=5)

        assert isinstance(model, TFA)
        assert model.K > 0
        assert model.K <= 50  # Should be capped

    def test_fit_routes_to_bids_for_path(self):
        """Test that path input routes to BIDS handler."""
        # Test that passing a path raises appropriate error
        # since BIDS parsing is not yet implemented
        with pytest.raises(FileNotFoundError):
            fit("/nonexistent/path/to/dataset", n_factors=10)

    def test_fit_routes_to_bids_for_pathlike(self):
        """Test that PathLike input routes to BIDS handler."""
        # Test that passing a Path raises appropriate error
        # since BIDS parsing is not yet implemented
        with pytest.raises(FileNotFoundError):
            fit(Path("/nonexistent/path/to/dataset"), n_factors=10)


class TestFitArrays:
    """Test the _fit_arrays function."""

    def test_fit_arrays_single_subject(self):
        """Test fitting single subject array."""
        data = np.random.randn(100, 50)
        coords = np.random.randn(100, 3)

        model = _fit_arrays(data, coords, n_factors=5, multi_subject=False, max_iter=5)

        assert isinstance(model, TFA)
        assert model.K == 5
        assert np.array_equal(model.coords_, coords)

    def test_fit_arrays_multi_subject(self):
        """Test fitting multi-subject arrays."""
        data = [np.random.randn(100, 50) for _ in range(3)]
        coords = np.random.randn(100, 3)

        model = _fit_arrays(data, coords, n_factors=5, max_iter=5)

        assert isinstance(model, HTFA)
        assert model.K == 5

    def test_fit_arrays_empty_list_raises_error(self):
        """Test that empty data list raises error."""
        coords = np.random.randn(100, 3)

        with pytest.raises(ValueError, match="Empty data list"):
            _fit_arrays([], coords)

    def test_fit_arrays_mismatched_voxels_raises_error(self):
        """Test that mismatched voxel counts raise error."""
        data = [
            np.random.randn(100, 50),
            np.random.randn(90, 50),  # Different number of voxels
        ]
        coords = np.random.randn(100, 3)

        with pytest.raises(ValueError, match="expected 100 voxels, got 90"):
            _fit_arrays(data, coords)

    def test_fit_arrays_wrong_dimensions_raises_error(self):
        """Test that wrong array dimensions raise error."""
        data = np.random.randn(100, 50, 10)  # 3D array
        coords = np.random.randn(100, 3)

        with pytest.raises(ValueError, match="Expected 2D array, got 3D"):
            _fit_arrays(data, coords)

    def test_fit_arrays_coords_mismatch_raises_error(self):
        """Test that coordinate shape mismatch raises error."""
        data = np.random.randn(100, 50)
        coords = np.random.randn(90, 3)  # Wrong number of coordinates

        with pytest.raises(ValueError, match="Coordinates shape 90 doesn't match"):
            _fit_arrays(data, coords)

    def test_fit_arrays_multi_subject_wrong_dims(self):
        """Test multi-subject with wrong dimensions."""
        data = [np.random.randn(100, 50, 10), np.random.randn(100, 50)]  # 3D array
        coords = np.random.randn(100, 3)

        with pytest.raises(ValueError, match="Subject 0: expected 2D array"):
            _fit_arrays(data, coords)


class TestFitBidsDataset:
    """Test the _fit_bids_dataset function."""

    def test_fit_bids_nonexistent_path_raises_error(self):
        """Test that nonexistent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Path does not exist"):
            _fit_bids_dataset("/nonexistent/path")

    def test_fit_bids_invalid_file_extension(self):
        """Test that non-NIfTI file raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
            with pytest.raises(ValueError, match="Single file must be NIfTI format"):
                _fit_bids_dataset(tmp.name)

    def test_fit_bids_single_nifti_file(self):
        """Test fitting a single NIfTI file."""
        # Now that NIfTI loading is implemented, test with a proper NIfTI file
        # For unit test, we'll create a minimal NIfTI file
        import nibabel as nib

        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
            try:
                # Create a minimal 4D NIfTI file
                data = np.random.randn(10, 10, 10, 20)  # Small 4D array
                img = nib.Nifti1Image(data, np.eye(4))
                nib.save(img, tmp.name)

                # Now fit should work
                model = _fit_bids_dataset(tmp.name, n_factors=3, max_iter=5)
                assert model is not None
                assert hasattr(model, "factors_")
            finally:
                # Clean up
                import os

                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)

    def test_fit_bids_single_nifti_gz_file(self):
        """Test fitting a single .nii.gz file."""
        # Now that NIfTI loading is implemented, test with a proper NIfTI file
        import nibabel as nib

        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
            try:
                # Create a minimal 4D NIfTI file
                data = np.random.randn(10, 10, 10, 20)  # Small 4D array
                img = nib.Nifti1Image(data, np.eye(4))
                nib.save(img, tmp.name)

                # Now fit should work
                model = _fit_bids_dataset(tmp.name, n_factors=3, max_iter=5)
                assert model is not None
                assert hasattr(model, "factors_")
            finally:
                # Clean up
                import os

                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)

    def test_fit_bids_directory_not_implemented(self):
        """Test that BIDS directory raises NotImplementedError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal BIDS structure
            Path(tmpdir, "dataset_description.json").write_text('{"Name": "Test"}')
            with pytest.raises(NotImplementedError, match="BIDS dataset parsing"):
                _fit_bids_dataset(tmpdir)

    def test_fit_bids_invalid_path_type(self):
        """Test with a path that's neither file nor directory."""
        # Create a named pipe (FIFO) which is neither file nor directory
        with tempfile.TemporaryDirectory() as tmpdir:
            fifo_path = Path(tmpdir) / "test.fifo"
            os.mkfifo(fifo_path)

            with pytest.raises(ValueError, match="Invalid path type"):
                _fit_bids_dataset(fifo_path)


class TestInferParameters:
    """Test the _infer_parameters function."""

    def test_infer_parameters_basic(self):
        """Test basic parameter inference."""
        data = np.random.randn(100, 50)

        params = _infer_parameters(data)

        assert "n_factors" in params
        assert "max_iter" in params
        assert "tol" in params
        assert params["n_factors"] > 0
        assert params["n_factors"] <= 50
        assert params["max_iter"] == 100
        assert params["tol"] == 1e-6

    def test_infer_parameters_large_data(self):
        """Test parameter inference with large data."""
        data = np.random.randn(10000, 500)

        params = _infer_parameters(data)

        # Should be capped at 50
        assert params["n_factors"] == 50

    def test_infer_parameters_small_timepoints(self):
        """Test with very few timepoints."""
        data = np.random.randn(1000, 5)

        params = _infer_parameters(data)

        # Should be max(1, 5//10) = 1
        assert params["n_factors"] >= 1

    def test_infer_parameters_override(self):
        """Test that kwargs override inferred values."""
        data = np.random.randn(100, 50)

        params = _infer_parameters(data, n_factors=15, max_iter=200)

        assert params["n_factors"] == 15
        assert params["max_iter"] == 200
        assert params["tol"] == 1e-6  # Not overridden

    def test_infer_parameters_ignore_extra_kwargs(self):
        """Test that extra kwargs are ignored."""
        data = np.random.randn(100, 50)

        params = _infer_parameters(data, extra_param=123)

        assert "extra_param" not in params


class TestLoadNiftiFile:
    """Test the _load_nifti_file function."""

    def test_load_nifti_file_not_found(self):
        """Test that _load_nifti_file handles missing files."""
        from htfa.fit import _load_nifti_file

        with pytest.raises(FileNotFoundError):
            _load_nifti_file(Path("/fake/nonexistent/path.nii"))
