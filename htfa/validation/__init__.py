"""Validation and cross-epic compatibility checking for HTFA ecosystem."""

# Import existing validation functions from parent module
from ..validation import (
    ValidationError,
    check_data_quality,
    format_validation_error,
    validate_arrays,
    validate_bids_path,
    validate_parameters,
)

# Import new cross-epic validation
from .cross_epic_check import ComponentStatus, CrossEpicValidator

__all__ = [
    "ValidationError",
    "check_data_quality", 
    "format_validation_error",
    "validate_arrays",
    "validate_bids_path",
    "validate_parameters",
    "CrossEpicValidator",
    "ComponentStatus",
]
