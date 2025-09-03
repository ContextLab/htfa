"""Comprehensive input validation framework for HTFA.

This module provides input validation functions for arrays, BIDS paths, parameters,
and error formatting to ensure data integrity and provide clear error messages.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_array


class ValidationError(ValueError):
    """Custom exception for validation errors with enhanced error messages."""
    
    def __init__(self, message: str, error_type: str = "validation", context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_type = error_type
        self.context = context or {}


def validate_arrays(
    data: Union[np.ndarray, List[np.ndarray]], 
    coords: Optional[Union[np.ndarray, List[np.ndarray]]] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Validate input arrays for shape, type, and dimensional consistency.
    
    Parameters
    ----------
    data : array-like or list of array-like
        Neural data arrays. Can be:
        - Single array: (n_voxels, n_timepoints) 
        - Multiple subjects: list of (n_voxels, n_timepoints) arrays
        - 3D array: (n_subjects, n_voxels, n_timepoints)
        - 4D array: (n_subjects, n_runs, n_voxels, n_timepoints)
    coords : array-like or list of array-like, optional
        Coordinate arrays corresponding to data. Must match data structure.
        
    Returns
    -------
    data_validated : ndarray
        Validated and standardized data array.
    coords_validated : ndarray or None
        Validated coordinate array if provided.
        
    Raises
    ------
    ValidationError
        If arrays fail validation checks.
        
    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.randn(1000, 100)  # 1000 voxels, 100 timepoints
    >>> coords = np.random.randn(1000, 3)  # 3D coordinates for each voxel
    >>> data_val, coords_val = validate_arrays(data, coords)
    """
    # Handle list inputs
    if isinstance(data, list):
        if len(data) == 0:
            raise ValidationError(
                "Empty data list provided",
                error_type="empty_input",
                context={"input_type": "list", "length": 0}
            )
        
        # Validate each array in the list
        validated_arrays = []
        for i, arr in enumerate(data):
            # Convert to numpy array first for validation checks
            arr_np = np.asarray(arr)
            
            # Check for NaN values before sklearn validation
            if np.any(np.isnan(arr_np)):
                n_nan = np.sum(np.isnan(arr_np))
                raise ValidationError(
                    f"Array {i} contains {n_nan} NaN values",
                    error_type="nan_values",
                    context={
                        "array_index": i,
                        "n_nan": n_nan,
                        "total_elements": arr_np.size,
                        "suggestion": "Use preprocessing to handle NaN values or remove affected timepoints/voxels"
                    }
                )
            
            # Check for infinite values before sklearn validation
            if np.any(np.isinf(arr_np)):
                n_inf = np.sum(np.isinf(arr_np))
                raise ValidationError(
                    f"Array {i} contains {n_inf} infinite values",
                    error_type="inf_values",
                    context={
                        "array_index": i,
                        "n_inf": n_inf,
                        "total_elements": arr_np.size,
                        "suggestion": "Check for division by zero or overflow in preprocessing"
                    }
                )
            
            try:
                arr_validated = check_array(
                    arr_np, dtype=np.float64, ensure_2d=True, allow_nd=False
                )
                validated_arrays.append(arr_validated)
            except ValueError as e:
                raise ValidationError(
                    f"Array {i} failed validation: {str(e)}",
                    error_type="array_validation",
                    context={"array_index": i, "original_error": str(e)}
                )
        
        # Check shape consistency
        shapes = [arr.shape for arr in validated_arrays]
        n_voxels_list = [s[0] for s in shapes]
        n_timepoints_list = [s[1] for s in shapes]
        
        if len(set(n_voxels_list)) > 1:
            raise ValidationError(
                f"Inconsistent number of voxels across arrays: {n_voxels_list}",
                error_type="shape_mismatch",
                context={
                    "shapes": shapes,
                    "n_voxels": n_voxels_list,
                    "suggestion": "Ensure all subjects have the same spatial dimensions"
                }
            )
        
        # Pad arrays to match the minimum number of timepoints for stacking
        # Find minimum timepoints
        min_timepoints = min(n_timepoints_list)
        
        # Truncate all arrays to minimum timepoints
        truncated_arrays = []
        for arr in validated_arrays:
            truncated_arrays.append(arr[:, :min_timepoints])
        
        # Stack arrays into 3D format: (n_subjects, n_voxels, n_timepoints)
        data_validated = np.stack(truncated_arrays, axis=0)
        
    else:
        # Single array input
        data_np = np.asarray(data)
        
        # Check for NaN values before sklearn validation
        if np.any(np.isnan(data_np)):
            n_nan = np.sum(np.isnan(data_np))
            raise ValidationError(
                f"Data contains {n_nan} NaN values",
                error_type="nan_values",
                context={
                    "n_nan": n_nan,
                    "total_elements": data_np.size,
                    "suggestion": "Use preprocessing to handle NaN values or remove affected timepoints/voxels"
                }
            )
        
        # Check for infinite values before sklearn validation
        if np.any(np.isinf(data_np)):
            n_inf = np.sum(np.isinf(data_np))
            raise ValidationError(
                f"Data contains {n_inf} infinite values",
                error_type="inf_values",
                context={
                    "n_inf": n_inf,
                    "total_elements": data_np.size,
                    "suggestion": "Check for division by zero or overflow in preprocessing"
                }
            )
        
        try:
            data_validated = check_array(
                data_np, dtype=np.float64, ensure_2d=False, allow_nd=True
            )
        except ValueError as e:
            raise ValidationError(
                f"Data array validation failed: {str(e)}",
                error_type="array_validation",
                context={"original_error": str(e)}
            )
    
    # Validate dimensions
    if data_validated.ndim < 2:
        raise ValidationError(
            f"Data must be at least 2D, got shape {data_validated.shape}",
            error_type="dimension_error",
            context={
                "shape": data_validated.shape,
                "ndim": data_validated.ndim,
                "expected": "≥ 2D"
            }
        )
    
    if data_validated.ndim > 4:
        raise ValidationError(
            f"Data cannot exceed 4D, got shape {data_validated.shape}",
            error_type="dimension_error",
            context={
                "shape": data_validated.shape,
                "ndim": data_validated.ndim,
                "expected": "≤ 4D"
            }
        )
    
    # NaN and Inf values are already checked before sklearn validation
    
    # Check for minimum data requirements
    if data_validated.ndim >= 2:
        voxel_dim = -2  # Second to last dimension is typically voxels
        timepoint_dim = -1  # Last dimension is typically timepoints
        
        n_voxels = data_validated.shape[voxel_dim]
        n_timepoints = data_validated.shape[timepoint_dim]
        
        if n_voxels < 2:
            raise ValidationError(
                f"Need at least 2 voxels for analysis, got {n_voxels}",
                error_type="insufficient_voxels",
                context={
                    "n_voxels": n_voxels,
                    "suggestion": "Increase spatial resolution or check data dimensions"
                }
            )
        
        if n_timepoints < 2:
            raise ValidationError(
                f"Need at least 2 timepoints for analysis, got {n_timepoints}",
                error_type="insufficient_timepoints",
                context={
                    "n_timepoints": n_timepoints,
                    "suggestion": "Increase temporal resolution or check data dimensions"
                }
            )
    
    # Validate coordinates if provided
    coords_validated = None
    if coords is not None:
        if isinstance(coords, list):
            if len(coords) != len(data) if isinstance(data, list) else 1:
                raise ValidationError(
                    f"Mismatch between data arrays ({len(data) if isinstance(data, list) else 1}) and coordinate arrays ({len(coords)})",
                    error_type="coords_count_mismatch",
                    context={
                        "n_data_arrays": len(data) if isinstance(data, list) else 1,
                        "n_coord_arrays": len(coords)
                    }
                )
            
            validated_coord_arrays = []
            for i, coord_arr in enumerate(coords):
                try:
                    coord_validated = check_array(
                        coord_arr, dtype=np.float64, ensure_2d=True
                    )
                    validated_coord_arrays.append(coord_validated)
                except ValueError as e:
                    raise ValidationError(
                        f"Coordinate array {i} failed validation: {str(e)}",
                        error_type="coords_validation",
                        context={"coord_index": i, "original_error": str(e)}
                    )
            
            coords_validated = np.stack(validated_coord_arrays, axis=0)
            
        else:
            try:
                coords_validated = check_array(
                    coords, dtype=np.float64, ensure_2d=True
                )
            except ValueError as e:
                raise ValidationError(
                    f"Coordinates validation failed: {str(e)}",
                    error_type="coords_validation",
                    context={"original_error": str(e)}
                )
        
        # Check coordinate-data consistency
        if isinstance(data, list):
            expected_n_voxels = data_validated.shape[1]  # After stacking: (n_subjects, n_voxels, n_timepoints)
        else:
            expected_n_voxels = data_validated.shape[-2]  # Voxel dimension
        
        if coords_validated.shape[-2] != expected_n_voxels:
            raise ValidationError(
                f"Coordinate array has {coords_validated.shape[-2]} rows but data has {expected_n_voxels} voxels",
                error_type="coords_shape_mismatch",
                context={
                    "coord_shape": coords_validated.shape,
                    "data_shape": data_validated.shape,
                    "expected_coord_voxels": expected_n_voxels
                }
            )
    
    return data_validated, coords_validated


def validate_bids_path(path: Union[str, Path]) -> Path:
    """Validate BIDS directory structure and accessibility.
    
    Parameters
    ----------
    path : str or Path
        Path to BIDS dataset directory.
        
    Returns
    -------
    path_validated : Path
        Validated Path object to BIDS directory.
        
    Raises
    ------
    ValidationError
        If path fails BIDS validation checks.
        
    Examples
    --------
    >>> path = validate_bids_path('/path/to/bids/dataset')
    """
    if path is None:
        raise ValidationError(
            "BIDS path cannot be None",
            error_type="null_path",
            context={"suggestion": "Provide a valid path to BIDS dataset"}
        )
    
    # Convert to Path object
    try:
        path_obj = Path(path).resolve()
    except (TypeError, ValueError) as e:
        raise ValidationError(
            f"Invalid path format: {str(e)}",
            error_type="invalid_path",
            context={
                "provided_path": str(path),
                "original_error": str(e),
                "suggestion": "Provide a valid file system path"
            }
        )
    
    # Check if path exists
    if not path_obj.exists():
        raise ValidationError(
            f"BIDS path does not exist: {path_obj}",
            error_type="path_not_exists",
            context={
                "path": str(path_obj),
                "suggestion": "Check that the path is correct and accessible"
            }
        )
    
    # Check if it's a directory
    if not path_obj.is_dir():
        raise ValidationError(
            f"BIDS path is not a directory: {path_obj}",
            error_type="not_directory",
            context={
                "path": str(path_obj),
                "suggestion": "Provide path to a directory, not a file"
            }
        )
    
    # Check for basic BIDS structure
    required_bids_files = ['dataset_description.json']
    missing_files = []
    
    for req_file in required_bids_files:
        if not (path_obj / req_file).exists():
            missing_files.append(req_file)
    
    if missing_files:
        raise ValidationError(
            f"Missing required BIDS files: {missing_files}",
            error_type="invalid_bids_structure",
            context={
                "path": str(path_obj),
                "missing_files": missing_files,
                "suggestion": "Ensure this is a valid BIDS dataset with required metadata files"
            }
        )
    
    # Check for subjects directory
    subjects_dir = path_obj / 'sub-*'
    subject_dirs = list(path_obj.glob('sub-*'))
    
    if not subject_dirs:
        raise ValidationError(
            "No subject directories found (sub-*)",
            error_type="no_subjects",
            context={
                "path": str(path_obj),
                "suggestion": "Ensure BIDS dataset contains subject directories named 'sub-<label>'"
            }
        )
    
    # Validate at least one subject has neuroimaging data
    has_neuroimaging_data = False
    neuroimaging_extensions = ['.nii', '.nii.gz']
    
    for subject_dir in subject_dirs[:5]:  # Check first 5 subjects for efficiency
        # Check if any neuroimaging files exist by converting generators to lists
        if any(list(subject_dir.rglob(f'*{ext}')) for ext in neuroimaging_extensions):
            has_neuroimaging_data = True
            break
    
    if not has_neuroimaging_data:
        warnings.warn(
            f"No neuroimaging files found in first 5 subjects at {path_obj}. "
            "This may not be a neuroimaging BIDS dataset.",
            UserWarning
        )
    
    return path_obj


def validate_parameters(**kwargs) -> Dict[str, Any]:
    """Validate algorithm parameters for TFA/HTFA.
    
    Parameters
    ----------
    **kwargs : dict
        Algorithm parameters to validate.
        
    Returns
    -------
    validated_params : dict
        Dictionary of validated parameters with type conversion applied.
        
    Raises
    ------
    ValidationError
        If parameters fail validation.
        
    Examples
    --------
    >>> params = validate_parameters(n_factors=10, max_iter=100, tol=1e-6)
    """
    validated = {}
    
    # Define parameter specifications
    param_specs = {
        'n_factors': {
            'type': int,
            'min': 1,
            'max': None,
            'default': 10,
            'description': 'Number of factors to extract'
        },
        'max_iter': {
            'type': int,
            'min': 1,
            'max': 10000,
            'default': 100,
            'description': 'Maximum number of optimization iterations'
        },
        'tol': {
            'type': float,
            'min': 1e-12,
            'max': 1.0,
            'default': 1e-6,
            'description': 'Convergence tolerance'
        },
        'alpha': {
            'type': float,
            'min': 0.0,
            'max': None,
            'default': 0.01,
            'description': 'Regularization parameter'
        },
        'random_state': {
            'type': (int, type(None)),
            'min': None,
            'max': None,
            'default': None,
            'description': 'Random seed for reproducibility'
        },
        'verbose': {
            'type': bool,
            'min': None,
            'max': None,
            'default': False,
            'description': 'Enable verbose output'
        },
        'init': {
            'type': str,
            'min': None,
            'max': None,
            'default': 'k-means',
            'choices': ['k-means', 'random'],
            'description': 'Initialization method'
        },
        'n_levels': {
            'type': int,
            'min': 1,
            'max': 10,
            'default': 2,
            'description': 'Number of hierarchical levels (HTFA only)'
        }
    }
    
    # Validate each provided parameter
    for param_name, param_value in kwargs.items():
        if param_name not in param_specs:
            raise ValidationError(
                f"Unknown parameter: '{param_name}'",
                error_type="unknown_parameter",
                context={
                    "parameter": param_name,
                    "valid_parameters": list(param_specs.keys()),
                    "suggestion": f"Use one of: {', '.join(param_specs.keys())}"
                }
            )
        
        spec = param_specs[param_name]
        
        # Type validation
        expected_type = spec['type']
        if isinstance(expected_type, tuple):
            # Multiple allowed types
            if not isinstance(param_value, expected_type):
                raise ValidationError(
                    f"Parameter '{param_name}' must be one of types {expected_type}, got {type(param_value).__name__}",
                    error_type="type_error",
                    context={
                        "parameter": param_name,
                        "expected_types": [t.__name__ if t is not type(None) else 'None' for t in expected_type],
                        "actual_type": type(param_value).__name__,
                        "description": spec['description']
                    }
                )
        else:
            # Single expected type
            if expected_type != type(None) and not isinstance(param_value, expected_type):
                raise ValidationError(
                    f"Parameter '{param_name}' must be of type {expected_type.__name__}, got {type(param_value).__name__}",
                    error_type="type_error",
                    context={
                        "parameter": param_name,
                        "expected_type": expected_type.__name__,
                        "actual_type": type(param_value).__name__,
                        "description": spec['description']
                    }
                )
        
        # Range validation for numeric types
        if spec['min'] is not None and param_value is not None:
            if param_value < spec['min']:
                raise ValidationError(
                    f"Parameter '{param_name}' must be >= {spec['min']}, got {param_value}",
                    error_type="value_range_error",
                    context={
                        "parameter": param_name,
                        "value": param_value,
                        "min_allowed": spec['min'],
                        "max_allowed": spec['max'],
                        "description": spec['description']
                    }
                )
        
        if spec['max'] is not None and param_value is not None:
            if param_value > spec['max']:
                raise ValidationError(
                    f"Parameter '{param_name}' must be <= {spec['max']}, got {param_value}",
                    error_type="value_range_error",
                    context={
                        "parameter": param_name,
                        "value": param_value,
                        "min_allowed": spec['min'],
                        "max_allowed": spec['max'],
                        "description": spec['description']
                    }
                )
        
        # Choice validation for string parameters
        if 'choices' in spec and param_value is not None:
            if param_value not in spec['choices']:
                raise ValidationError(
                    f"Parameter '{param_name}' must be one of {spec['choices']}, got '{param_value}'",
                    error_type="invalid_choice",
                    context={
                        "parameter": param_name,
                        "value": param_value,
                        "valid_choices": spec['choices'],
                        "description": spec['description']
                    }
                )
        
        validated[param_name] = param_value
    
    return validated


def format_validation_error(error_type: str, context: Dict[str, Any]) -> str:
    """Format validation errors with descriptive, actionable messages.
    
    Parameters
    ----------
    error_type : str
        Type of validation error.
    context : dict
        Context information about the error.
        
    Returns
    -------
    formatted_message : str
        Formatted error message with actionable guidance.
        
    Examples
    --------
    >>> context = {'shape': (10, 5), 'expected': '≥ 2D'}
    >>> message = format_validation_error('dimension_error', context)
    """
    templates = {
        'array_validation': (
            "Array validation failed: {original_error}\n"
            "Action: Ensure input is a valid numeric array with no missing dimensions."
        ),
        'dimension_error': (
            "Invalid array dimensions: got shape {shape} ({ndim}D), expected {expected}\n"
            "Action: Reshape your data or check input format. "
            "Expected formats:\n"
            "  - 2D: (n_voxels, n_timepoints)\n"
            "  - 3D: (n_subjects, n_voxels, n_timepoints)\n"
            "  - 4D: (n_subjects, n_runs, n_voxels, n_timepoints)"
        ),
        'shape_mismatch': (
            "Inconsistent array shapes across inputs: {shapes}\n"
            "Action: {suggestion}"
        ),
        'nan_values': (
            "Data contains {n_nan} NaN values out of {total_elements} total elements\n"
            "Action: {suggestion}"
        ),
        'inf_values': (
            "Data contains {n_inf} infinite values out of {total_elements} total elements\n"
            "Action: {suggestion}"
        ),
        'insufficient_voxels': (
            "Insufficient spatial resolution: {n_voxels} voxels\n"
            "Action: {suggestion}"
        ),
        'insufficient_timepoints': (
            "Insufficient temporal resolution: {n_timepoints} timepoints\n"
            "Action: {suggestion}"
        ),
        'path_not_exists': (
            "BIDS dataset path does not exist: {path}\n"
            "Action: {suggestion}"
        ),
        'not_directory': (
            "BIDS path must be a directory: {path}\n"
            "Action: {suggestion}"
        ),
        'invalid_bids_structure': (
            "Invalid BIDS dataset structure at {path}\n"
            "Missing required files: {missing_files}\n"
            "Action: {suggestion}"
        ),
        'no_subjects': (
            "No subject directories found in BIDS dataset at {path}\n"
            "Action: {suggestion}"
        ),
        'type_error': (
            "Parameter '{parameter}' type error: expected {expected_type}, got {actual_type}\n"
            "Description: {description}\n"
            "Action: Convert parameter to the correct type before passing to function."
        ),
        'value_range_error': (
            "Parameter '{parameter}' value out of range: {value}\n"
            "Allowed range: {min_allowed} to {max_allowed}\n"
            "Description: {description}\n"
            "Action: Choose a value within the valid range."
        ),
        'invalid_choice': (
            "Parameter '{parameter}' has invalid value: '{value}'\n"
            "Valid choices: {valid_choices}\n"
            "Description: {description}\n"
            "Action: Use one of the valid choices listed above."
        ),
        'unknown_parameter': (
            "Unknown parameter: '{parameter}'\n"
            "Valid parameters: {valid_parameters}\n"
            "Action: {suggestion}"
        ),
        'coords_validation': (
            "Coordinate array validation failed: {original_error}\n"
            "Action: Ensure coordinates are numeric arrays with proper shape."
        ),
        'coords_shape_mismatch': (
            "Coordinate array shape mismatch:\n"
            "  Coordinate shape: {coord_shape}\n"
            "  Data shape: {data_shape}\n"
            "  Expected coordinate voxels: {expected_coord_voxels}\n"
            "Action: Ensure coordinate array has same number of voxels as data."
        ),
        'coords_count_mismatch': (
            "Mismatch between number of data arrays ({n_data_arrays}) and coordinate arrays ({n_coord_arrays})\n"
            "Action: Provide either one coordinate array per data array, or a single coordinate array for all."
        ),
    }
    
    if error_type not in templates:
        return f"Validation error of type '{error_type}': {context}"
    
    template = templates[error_type]
    
    try:
        return template.format(**context)
    except KeyError as e:
        # Fallback if template formatting fails
        return f"Validation error of type '{error_type}': missing context key {e}. Context: {context}"


def check_data_quality(
    data: np.ndarray,
    min_signal_std: float = 1e-6,
    max_outlier_ratio: float = 0.05
) -> Dict[str, Any]:
    """Check data quality and provide warnings for potential issues.
    
    Parameters
    ----------
    data : ndarray
        Input data to check.
    min_signal_std : float, default=1e-6
        Minimum standard deviation threshold for signal detection.
    max_outlier_ratio : float, default=0.05
        Maximum allowed ratio of outlier values.
        
    Returns
    -------
    quality_report : dict
        Dictionary containing quality metrics and warnings.
    """
    report = {
        'warnings': [],
        'metrics': {},
        'recommendations': []
    }
    
    # Flatten data for analysis
    flat_data = data.flatten()
    
    # Basic statistics
    report['metrics']['mean'] = np.mean(flat_data)
    report['metrics']['std'] = np.std(flat_data)
    report['metrics']['min'] = np.min(flat_data)
    report['metrics']['max'] = np.max(flat_data)
    report['metrics']['shape'] = data.shape
    
    # Check for low signal variance
    if report['metrics']['std'] < min_signal_std:
        report['warnings'].append(
            f"Very low signal variance (std={report['metrics']['std']:.2e}). "
            "Data may be mostly constant or heavily filtered."
        )
        report['recommendations'].append(
            "Consider checking preprocessing steps or data scaling."
        )
    
    # Check for outliers using IQR method
    q25, q75 = np.percentile(flat_data, [25, 75])
    iqr = q75 - q25
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr
    
    outliers = np.sum((flat_data < lower_bound) | (flat_data > upper_bound))
    outlier_ratio = outliers / len(flat_data)
    report['metrics']['outlier_ratio'] = outlier_ratio
    
    if outlier_ratio > max_outlier_ratio:
        report['warnings'].append(
            f"High proportion of outliers ({outlier_ratio:.1%}). "
            "Data may contain artifacts or need preprocessing."
        )
        report['recommendations'].append(
            "Consider outlier removal, artifact detection, or robust preprocessing methods."
        )
    
    # Check for data scaling issues
    data_range = report['metrics']['max'] - report['metrics']['min']
    if data_range > 1000:
        report['warnings'].append(
            f"Large data range ({data_range:.2f}). "
            "Consider standardization or normalization for numerical stability."
        )
        report['recommendations'].append(
            "Apply z-score normalization or min-max scaling before analysis."
        )
    
    # Check for potential precision issues
    if np.abs(report['metrics']['mean']) > 1000:
        report['warnings'].append(
            f"Large data mean ({report['metrics']['mean']:.2f}). "
            "Consider mean-centering for numerical stability."
        )
        report['recommendations'].append(
            "Apply mean-centering: data = data - np.mean(data, axis=-1, keepdims=True)"
        )
    
    return report


# Export main validation functions
__all__ = [
    'ValidationError',
    'validate_arrays', 
    'validate_bids_path',
    'validate_parameters',
    'format_validation_error',
    'check_data_quality'
]