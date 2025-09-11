"""Comprehensive tests for HTFA validation framework."""


import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from htfa.validation import (
    ValidationError,
    check_data_quality,
    format_validation_error,
    validate_arrays,
    validate_bids_path,
    validate_parameters,
)


class TestValidationError:
    """Test ValidationError custom exception."""

    def test_basic_creation(self):
        """Test basic ValidationError creation."""
        error = ValidationError("Test message")
        assert str(error) == "Test message"
        assert error.error_type == "validation"
        assert error.context == {}

    def test_with_error_type_and_context(self):
        """Test ValidationError with error type and context."""
        context = {"key": "value", "number": 42}
        error = ValidationError("Test message", error_type="custom", context=context)

        assert str(error) == "Test message"
        assert error.error_type == "custom"
        assert error.context == context


class TestValidateArrays:
    """Test array validation functionality."""

    def test_valid_2d_array(self):
        """Test validation of valid 2D array."""
        data = np.random.randn(100, 50)  # 100 voxels, 50 timepoints
        coords = np.random.randn(100, 3)  # 3D coordinates

        data_val, coords_val = validate_arrays(data, coords)

        assert data_val.shape == (100, 50)
        assert data_val.dtype == np.float64
        assert coords_val.shape == (100, 3)
        assert coords_val.dtype == np.float64

    def test_valid_3d_array(self):
        """Test validation of valid 3D array."""
        data = np.random.randn(5, 100, 50)  # 5 subjects, 100 voxels, 50 timepoints

        data_val, coords_val = validate_arrays(data)

        assert data_val.shape == (5, 100, 50)
        assert data_val.dtype == np.float64
        assert coords_val is None

    def test_valid_4d_array(self):
        """Test validation of valid 4D array."""
        data = np.random.randn(
            3, 2, 100, 50
        )  # 3 subjects, 2 runs, 100 voxels, 50 timepoints

        data_val, coords_val = validate_arrays(data)

        assert data_val.shape == (3, 2, 100, 50)
        assert data_val.dtype == np.float64
        assert coords_val is None

    def test_list_of_arrays(self):
        """Test validation of list of arrays."""
        data_list = [
            np.random.randn(100, 50),
            np.random.randn(100, 60),
            np.random.randn(100, 55),
        ]

        data_val, coords_val = validate_arrays(data_list)

        assert data_val.shape == (3, 100, 50)  # Should pad to minimum timepoints
        assert data_val.dtype == np.float64

    def test_list_with_coords(self):
        """Test validation of list with coordinate arrays."""
        data_list = [np.random.randn(100, 50), np.random.randn(100, 60)]
        coords_list = [np.random.randn(100, 3), np.random.randn(100, 3)]

        data_val, coords_val = validate_arrays(data_list, coords_list)

        assert data_val.shape[0] == 2  # 2 subjects
        assert data_val.shape[1] == 100  # 100 voxels
        assert coords_val.shape == (2, 100, 3)

    def test_empty_list_error(self):
        """Test error on empty data list."""
        with pytest.raises(ValidationError) as exc_info:
            validate_arrays([])

        assert exc_info.value.error_type == "empty_input"

    def test_1d_array_error(self):
        """Test error on 1D array input."""
        data = np.random.randn(100)

        with pytest.raises(ValidationError) as exc_info:
            validate_arrays(data)

        assert exc_info.value.error_type == "dimension_error"

    def test_5d_array_error(self):
        """Test error on 5D array input."""
        data = np.random.randn(2, 3, 4, 100, 50)

        with pytest.raises(ValidationError) as exc_info:
            validate_arrays(data)

        assert exc_info.value.error_type == "dimension_error"

    def test_nan_values_error(self):
        """Test error on NaN values in data."""
        data = np.random.randn(100, 50)
        data[10, 20] = np.nan
        data[5, 15] = np.nan  # Add multiple NaN values

        with pytest.raises(ValidationError) as exc_info:
            validate_arrays(data)

        assert exc_info.value.error_type == "nan_values"
        assert exc_info.value.context["n_nan"] == 2  # Check count calculation

    def test_inf_values_error(self):
        """Test error on infinite values in data."""
        data = np.random.randn(100, 50)
        data[5, 10] = np.inf
        data[15, 25] = -np.inf  # Add multiple inf values including negative

        with pytest.raises(ValidationError) as exc_info:
            validate_arrays(data)

        assert exc_info.value.error_type == "inf_values"
        assert exc_info.value.context["n_inf"] == 2  # Check count calculation

    def test_sklearn_validation_error(self):
        """Test error when sklearn check_array fails."""
        # Create data that will fail sklearn validation but pass initial checks
        # Use a list with mixed types that can't be converted consistently
        data = [[1.0, 2.0], [3.0, "invalid"]]  # Mixed numeric/string

        with pytest.raises(ValidationError) as exc_info:
            validate_arrays(data)

        assert exc_info.value.error_type == "array_validation"
        assert "failed validation" in str(exc_info.value)
        assert "array_index" in exc_info.value.context

    def test_insufficient_voxels_error(self):
        """Test error on insufficient voxels."""
        data = np.random.randn(1, 50)  # Only 1 voxel

        with pytest.raises(ValidationError) as exc_info:
            validate_arrays(data)

        assert exc_info.value.error_type == "insufficient_voxels"

    def test_insufficient_timepoints_error(self):
        """Test error on insufficient timepoints."""
        data = np.random.randn(100, 1)  # Only 1 timepoint

        with pytest.raises(ValidationError) as exc_info:
            validate_arrays(data)

        assert exc_info.value.error_type == "insufficient_timepoints"

    def test_inconsistent_voxels_in_list(self):
        """Test error on inconsistent voxel counts in list."""
        data_list = [
            np.random.randn(100, 50),
            np.random.randn(120, 50),  # Different number of voxels
        ]

        with pytest.raises(ValidationError) as exc_info:
            validate_arrays(data_list)

        assert exc_info.value.error_type == "shape_mismatch"

    def test_coords_shape_mismatch(self):
        """Test error on coordinate shape mismatch."""
        data = np.random.randn(100, 50)
        coords = np.random.randn(120, 3)  # Wrong number of voxels

        with pytest.raises(ValidationError) as exc_info:
            validate_arrays(data, coords)

        assert exc_info.value.error_type == "coords_shape_mismatch"

    def test_coords_count_mismatch(self):
        """Test error on coordinate count mismatch."""
        data_list = [np.random.randn(100, 50), np.random.randn(100, 60)]
        coords_list = [
            np.random.randn(100, 3)
        ]  # Only one coord array for two data arrays

        with pytest.raises(ValidationError) as exc_info:
            validate_arrays(data_list, coords_list)

        assert exc_info.value.error_type == "coords_count_mismatch"


class TestValidateBidsPath:
    """Test BIDS path validation functionality."""

    def test_valid_bids_directory(self):
        """Test validation of a valid BIDS directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            bids_dir = Path(temp_dir)

            # Create required BIDS structure
            (bids_dir / "dataset_description.json").write_text(
                json.dumps({"Name": "Test Dataset", "BIDSVersion": "1.6.0"})
            )

            # Create subject directory with neuroimaging data
            sub_dir = bids_dir / "sub-01" / "anat"
            sub_dir.mkdir(parents=True)
            (sub_dir / "sub-01_T1w.nii.gz").write_text("")

            path_val = validate_bids_path(bids_dir)
            assert path_val == bids_dir.resolve()

    def test_none_path_error(self):
        """Test error on None path."""
        with pytest.raises(ValidationError) as exc_info:
            validate_bids_path(None)

        assert exc_info.value.error_type == "null_path"

    def test_invalid_path_format_error(self):
        """Test error on invalid path format."""
        with pytest.raises(ValidationError) as exc_info:
            validate_bids_path(123)  # Invalid path type

        assert exc_info.value.error_type == "invalid_path"

    def test_nonexistent_path_error(self):
        """Test error on nonexistent path."""
        nonexistent_path = "/this/path/does/not/exist"

        with pytest.raises(ValidationError) as exc_info:
            validate_bids_path(nonexistent_path)

        assert exc_info.value.error_type == "path_not_exists"

    def test_not_directory_error(self):
        """Test error when path is not a directory."""
        with tempfile.NamedTemporaryFile() as temp_file:
            with pytest.raises(ValidationError) as exc_info:
                validate_bids_path(temp_file.name)

            assert exc_info.value.error_type == "not_directory"

    def test_missing_dataset_description_error(self):
        """Test error on missing dataset_description.json."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValidationError) as exc_info:
                validate_bids_path(temp_dir)

            assert exc_info.value.error_type == "invalid_bids_structure"
            assert "dataset_description.json" in exc_info.value.context["missing_files"]

    def test_no_subjects_error(self):
        """Test error when no subject directories found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            bids_dir = Path(temp_dir)

            # Create dataset description but no subjects
            (bids_dir / "dataset_description.json").write_text(
                json.dumps({"Name": "Test Dataset", "BIDSVersion": "1.6.0"})
            )

            with pytest.raises(ValidationError) as exc_info:
                validate_bids_path(bids_dir)

            assert exc_info.value.error_type == "no_subjects"

    def test_warning_no_neuroimaging_data(self):
        """Test warning when no neuroimaging data found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            bids_dir = Path(temp_dir)

            # Create required BIDS structure
            (bids_dir / "dataset_description.json").write_text(
                json.dumps({"Name": "Test Dataset", "BIDSVersion": "1.6.0"})
            )

            # Create subject directory without neuroimaging data
            sub_dir = bids_dir / "sub-01"
            sub_dir.mkdir(parents=True)
            (sub_dir / "README.txt").write_text("")

            with pytest.warns(UserWarning, match="No neuroimaging files found"):
                validate_bids_path(bids_dir)


class TestValidateParameters:
    """Test parameter validation functionality."""

    def test_valid_parameters(self):
        """Test validation of valid parameters."""
        params = {
            "n_factors": 10,
            "max_iter": 100,
            "tol": 1e-6,
            "alpha": 0.01,
            "verbose": False,
            "random_state": 42,
        }

        validated = validate_parameters(**params)

        for key, value in params.items():
            assert validated[key] == value

    def test_unknown_parameter_error(self):
        """Test error on unknown parameter."""
        with pytest.raises(ValidationError) as exc_info:
            validate_parameters(unknown_param=123)

        assert exc_info.value.error_type == "unknown_parameter"
        assert "unknown_param" in exc_info.value.context["parameter"]

    def test_multiple_type_error_validation(self):
        """Test type error for parameters that accept multiple types."""
        # random_state accepts (int, None) but we pass a string
        with pytest.raises(ValidationError) as exc_info:
            validate_parameters(random_state="invalid_string")

        assert exc_info.value.error_type == "type_error"
        assert "must be one of types" in str(exc_info.value)
        assert exc_info.value.context["parameter"] == "random_state"

    def test_n_factors_type_error(self):
        """Test error on wrong n_factors type."""
        with pytest.raises(ValidationError) as exc_info:
            validate_parameters(n_factors="10")  # String instead of int

        assert exc_info.value.error_type == "type_error"
        assert exc_info.value.context["parameter"] == "n_factors"

    def test_n_factors_range_error(self):
        """Test error on n_factors out of range."""
        with pytest.raises(ValidationError) as exc_info:
            validate_parameters(n_factors=0)  # Must be >= 1

        assert exc_info.value.error_type == "value_range_error"

    def test_max_iter_range_error(self):
        """Test error on max_iter out of range."""
        with pytest.raises(ValidationError) as exc_info:
            validate_parameters(max_iter=-1)

        assert exc_info.value.error_type == "value_range_error"

        with pytest.raises(ValidationError) as exc_info:
            validate_parameters(max_iter=20000)  # Too large

        assert exc_info.value.error_type == "value_range_error"

    def test_tol_range_error(self):
        """Test error on tolerance out of range."""
        with pytest.raises(ValidationError) as exc_info:
            validate_parameters(tol=0.0)  # Must be > 0

        assert exc_info.value.error_type == "value_range_error"

        with pytest.raises(ValidationError) as exc_info:
            validate_parameters(tol=2.0)  # Too large

        assert exc_info.value.error_type == "value_range_error"

    def test_init_choice_error(self):
        """Test error on invalid init choice."""
        with pytest.raises(ValidationError) as exc_info:
            validate_parameters(init="invalid")

        assert exc_info.value.error_type == "invalid_choice"
        assert "invalid" in exc_info.value.context["value"]
        assert "k-means" in exc_info.value.context["valid_choices"]
        assert "random" in exc_info.value.context["valid_choices"]

    def test_random_state_none_allowed(self):
        """Test that random_state can be None."""
        validated = validate_parameters(random_state=None)
        assert validated["random_state"] is None

    def test_verbose_type_error(self):
        """Test error on wrong verbose type."""
        with pytest.raises(ValidationError) as exc_info:
            validate_parameters(verbose="True")  # String instead of bool

        assert exc_info.value.error_type == "type_error"

    def test_array_with_nan_values(self):
        """Test detection of NaN values in arrays."""
        data = np.array([[1.0, 2.0], [np.nan, 4.0]])  # Contains NaN
        with pytest.raises(ValidationError) as exc_info:
            validate_arrays([data])

        assert exc_info.value.error_type == "nan_values"
        assert "contains 1 NaN values" in str(exc_info.value)
        assert exc_info.value.context["n_nan"] == 1
        assert exc_info.value.context["array_index"] == 0

    def test_array_with_infinite_values(self):
        """Test detection of infinite values in arrays."""
        data = np.array([[1.0, 2.0], [np.inf, 4.0]])  # Contains infinity
        with pytest.raises(ValidationError) as exc_info:
            validate_arrays([data])

        assert exc_info.value.error_type == "inf_values"
        assert "contains 1 infinite values" in str(exc_info.value)
        assert exc_info.value.context["n_inf"] == 1
        assert exc_info.value.context["array_index"] == 0


class TestSklearnValidationErrors:
    """Test cases for sklearn validation error paths."""

    def test_invalid_data_array_validation_error(self):
        """Test sklearn validation error for malformed data."""
        # This should trigger the sklearn check_array exception in validate_arrays around line 180
        # Create data that contains complex numbers, which sklearn can't convert to float64
        import numpy as np

        invalid_data = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=complex)

        with pytest.raises(ValidationError) as exc_info:
            validate_arrays(invalid_data)

        assert exc_info.value.error_type == "array_validation"
        assert "Data array validation failed" in str(exc_info.value)

    def test_invalid_coords_validation_error(self):
        """Test sklearn validation error for invalid coordinate array."""
        # Valid data but invalid coordinates that can't be converted
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        invalid_coords = np.array([["x", "y", "z"], ["a", "b", "c"]], dtype=object)

        with pytest.raises(ValidationError) as exc_info:
            validate_arrays(data, coords=invalid_coords)

        assert exc_info.value.error_type == "coords_validation"
        assert "validation failed" in str(exc_info.value)

    def test_invalid_multi_coords_validation_error(self):
        """Test sklearn validation error for multi-subject coordinate arrays."""
        # Valid data but invalid coordinates in list format
        data = [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[5.0, 6.0], [7.0, 8.0]])]
        invalid_coords = [
            np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64),  # Valid
            np.array([["x", "y", "z"], ["a", "b", "c"]], dtype=object),  # Invalid
        ]

        with pytest.raises(ValidationError) as exc_info:
            validate_arrays(data, coords=invalid_coords)

        assert exc_info.value.error_type == "coords_validation"
        assert "failed validation" in str(exc_info.value)


class TestFormatValidationError:
    """Test error message formatting functionality."""

    def test_dimension_error_formatting(self):
        """Test formatting of dimension error."""
        context = {"shape": (10, 5), "ndim": 2, "expected": "≥ 3D"}

        message = format_validation_error("dimension_error", context)

        assert "Invalid array dimensions" in message
        assert "(10, 5)" in message
        assert "2D" in message
        assert "≥ 3D" in message
        assert "Action:" in message

    def test_nan_values_formatting(self):
        """Test formatting of NaN values error."""
        context = {
            "n_nan": 5,
            "total_elements": 1000,
            "suggestion": "Remove affected data points",
        }

        message = format_validation_error("nan_values", context)

        assert "5 NaN values" in message
        assert "1000 total elements" in message
        assert "Action:" in message
        assert "Remove affected data points" in message

    def test_type_error_formatting(self):
        """Test formatting of type error."""
        context = {
            "parameter": "n_factors",
            "expected_type": "int",
            "actual_type": "str",
            "description": "Number of factors to extract",
        }

        message = format_validation_error("type_error", context)

        assert "n_factors" in message
        assert "expected int" in message
        assert "got str" in message
        assert "Number of factors to extract" in message
        assert "Action:" in message

    def test_unknown_error_type_fallback(self):
        """Test fallback for unknown error type."""
        context = {"test": "value"}

        message = format_validation_error("unknown_type", context)

        assert "unknown_type" in message
        assert str(context) in message

    def test_template_formatting_error_fallback(self):
        """Test fallback when template formatting fails."""
        context = {"wrong_key": "value"}  # Missing expected keys

        message = format_validation_error("dimension_error", context)

        # Should fall back to basic format
        assert "dimension_error" in message
        assert "missing context key" in message


class TestCheckDataQuality:
    """Test data quality checking functionality."""

    def test_good_quality_data(self):
        """Test quality check on good data."""
        np.random.seed(42)
        data = np.random.randn(100, 50)  # Good quality data

        report = check_data_quality(data)

        assert len(report["warnings"]) == 0
        assert "mean" in report["metrics"]
        assert "std" in report["metrics"]
        assert "outlier_ratio" in report["metrics"]

    def test_low_variance_warning(self):
        """Test warning on low variance data."""
        data = np.ones((100, 50)) * 1e-10  # Very low variance

        report = check_data_quality(data)

        warning_found = any(
            "low signal variance" in w.lower() for w in report["warnings"]
        )
        assert warning_found

        recommendation_found = any(
            "preprocessing" in r.lower() for r in report["recommendations"]
        )
        assert recommendation_found

    def test_high_outlier_warning(self):
        """Test warning on high outlier ratio."""
        np.random.seed(42)
        data = np.random.randn(100, 50)
        # Add many outliers
        data[:20, :] = 100  # Make first 20 rows outliers

        report = check_data_quality(data)

        warning_found = any("outliers" in w.lower() for w in report["warnings"])
        assert warning_found

    def test_large_range_warning(self):
        """Test warning on large data range."""
        data = np.random.randn(100, 50)
        data[0, 0] = 10000  # Create large range

        report = check_data_quality(data)

        warning_found = any("large data range" in w.lower() for w in report["warnings"])
        assert warning_found

        recommendation_found = any(
            "normalization" in r.lower() for r in report["recommendations"]
        )
        assert recommendation_found

    def test_large_mean_warning(self):
        """Test warning on large mean."""
        data = np.random.randn(100, 50) + 5000  # Large mean

        report = check_data_quality(data)

        warning_found = any("large data mean" in w.lower() for w in report["warnings"])
        assert warning_found

        recommendation_found = any(
            "mean-centering" in r.lower() for r in report["recommendations"]
        )
        assert recommendation_found

    def test_quality_metrics(self):
        """Test that all expected metrics are included."""
        np.random.seed(42)
        data = np.random.randn(100, 50)

        report = check_data_quality(data)

        expected_metrics = ["mean", "std", "min", "max", "shape", "outlier_ratio"]
        for metric in expected_metrics:
            assert metric in report["metrics"]

        assert report["metrics"]["shape"] == (100, 50)


class TestIntegration:
    """Integration tests for validation framework."""

    def test_validate_arrays_then_check_quality(self):
        """Test integration between array validation and quality checking."""
        # Create synthetic fMRI-like data
        np.random.seed(42)
        n_voxels, n_timepoints = 1000, 200

        # Simulate fMRI signal with some structure
        signal = np.random.randn(n_voxels, n_timepoints) * 0.1
        trend = np.linspace(0, 1, n_timepoints)
        signal += trend[np.newaxis, :] * 0.02  # Add slight trend

        # Validate arrays first
        data_val, _ = validate_arrays(signal)
        assert data_val.shape == (n_voxels, n_timepoints)

        # Then check quality
        quality_report = check_data_quality(data_val)
        assert isinstance(quality_report, dict)
        assert "metrics" in quality_report
        assert "warnings" in quality_report

    def test_error_chain_validation(self):
        """Test that validation errors chain properly."""
        # Create data with multiple issues
        data = np.random.randn(10, 5)
        data[0, 0] = np.nan
        data[1, 1] = np.inf

        # Should catch first error (NaN) before second (inf)
        with pytest.raises(ValidationError) as exc_info:
            validate_arrays(data)

        # Should be NaN error since it's checked first
        assert exc_info.value.error_type == "nan_values"

    def test_parameter_validation_comprehensive(self):
        """Test comprehensive parameter validation."""
        # Test valid combination
        params = validate_parameters(
            n_factors=20,
            max_iter=500,
            tol=1e-8,
            alpha=0.1,
            init="random",
            random_state=123,
            verbose=True,
            n_levels=3,  # HTFA parameter
        )

        assert len(params) == 8
        assert params["n_factors"] == 20
        assert params["init"] == "random"
        assert params["n_levels"] == 3
