"""BIDS dataset integration for HTFA.

This module provides functionality for parsing and working with BIDS
(Brain Imaging Data Structure) datasets, including:
- BIDS dataset parsing with pybids integration
- Subject/task/session filtering capabilities
- Metadata extraction and aggregation
- Derivative detection and loading
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import os
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import numpy.typing as npt
import pandas as pd
from bids import BIDSLayout
from bids.layout import BIDSFile

from .validation import validate_bids_path


def parse_bids_dataset(
    path: Union[str, Path],
    derivatives: bool = True,
    validate: bool = False,
    **filters: Any,
) -> BIDSLayout:
    """Parse BIDS dataset and return BIDSLayout with optional filtering.

    This is the main entry point for BIDS dataset parsing. It creates a
    BIDSLayout object and applies any filtering criteria.

    Parameters
    ----------
    path : str or Path
        Path to the BIDS dataset root directory.
    derivatives : bool, default=True
        Whether to include derivatives in the layout.
    validate : bool, default=False
        Whether to validate BIDS compliance (can be slow for large datasets).
    **filters
        Additional filters to apply during parsing. Common filters include:
        - subject : str or list of str
        - session : str or list of str
        - task : str or list of str
        - datatype : str or list of str (e.g., 'func', 'anat')
        - extension : str or list of str (e.g., '.nii.gz')
        - return_type : str, default='filename'

    Returns
    -------
    BIDSLayout
        Parsed BIDS layout object with applied filters.

    Raises
    ------
    ValidationError
        If the path fails BIDS validation checks.
    RuntimeError
        If BIDS parsing fails due to structural issues.

    Examples
    --------
    >>> layout = parse_bids_dataset('/path/to/bids/dataset')
    >>> layout = parse_bids_dataset(
    ...     '/path/to/bids/dataset',
    ...     subject=['01', '02'],
    ...     task='rest',
    ...     datatype='func'
    ... )
    """
    # Use comprehensive BIDS validation from validation framework
    validated_path = validate_bids_path(path)

    try:
        # Create BIDSLayout with error handling
        layout = BIDSLayout(
            str(validated_path),
            derivatives=derivatives,
            validate=validate,
            absolute_paths=True,
        )

        # Log basic dataset info
        n_subjects = len(layout.get_subjects())
        n_sessions = len(layout.get_sessions())
        n_tasks = len(layout.get_tasks())

        print(
            f"Parsed BIDS dataset: {n_subjects} subjects, "
            f"{n_sessions} sessions, {n_tasks} tasks"
        )

    except Exception as e:
        raise RuntimeError(f"Failed to parse BIDS dataset at {validated_path}: {e}")

    return layout


def validate_bids_structure(path: Union[str, Path]) -> Dict[str, Any]:
    """Validate BIDS dataset structure and return compliance report.

    Performs comprehensive BIDS compliance checking including:
    - Required files and directories
    - Naming conventions
    - JSON sidecar files
    - TSV metadata files

    Parameters
    ----------
    path : str or Path
        Path to the BIDS dataset root directory.

    Returns
    -------
    dict
        Validation report containing:
        - 'valid': bool - Overall compliance status
        - 'errors': list - Critical compliance errors
        - 'warnings': list - Non-critical warnings
        - 'summary': dict - Dataset summary statistics

    Examples
    --------
    >>> report = validate_bids_structure('/path/to/bids/dataset')
    >>> if not report['valid']:
    ...     print("BIDS errors:", report['errors'])
    """
    path = Path(path)

    # Initialize report
    report: Dict[str, Any] = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "summary": {},
    }

    # Check if path exists
    if not path.exists():
        report["valid"] = False
        report["errors"].append(f"Dataset path does not exist: {path}")
        return report

    # Check for required files
    required_files = ["dataset_description.json"]
    for req_file in required_files:
        if not (path / req_file).exists():
            report["errors"].append(f"Missing required file: {req_file}")
            report["valid"] = False

    # Check for participants.tsv (recommended)
    if not (path / "participants.tsv").exists():
        report["warnings"].append("Missing participants.tsv file (recommended)")

    try:
        # Use pybids validation if available
        layout = BIDSLayout(str(path), validate=True)

        # Collect summary statistics
        report["summary"] = {
            "n_subjects": len(layout.get_subjects()),
            "n_sessions": len(layout.get_sessions()),
            "n_tasks": len(layout.get_tasks()),
            "datatypes": layout.get_datatypes(),
            "modalities": layout.get_modalities(),
        }

    except Exception as e:
        report["valid"] = False
        report["errors"].append(f"BIDS validation failed: {e}")

    return report


def extract_bids_metadata(
    files: List[Union[str, BIDSFile]],
    include_events: bool = True,
    include_physio: bool = False,
) -> pd.DataFrame:
    """Extract and aggregate metadata from BIDS files.

    Extracts metadata from JSON sidecar files, TSV files, and file paths
    to create a comprehensive metadata table for analysis.

    Parameters
    ----------
    files : list of str or BIDSFile
        List of BIDS files to extract metadata from.
    include_events : bool, default=True
        Whether to include events.tsv data.
    include_physio : bool, default=False
        Whether to include physiological data metadata.

    Returns
    -------
    pd.DataFrame
        Metadata table with columns for file paths, entities,
        and extracted JSON/TSV metadata.

    Examples
    --------
    >>> layout = parse_bids_dataset('/path/to/bids')
    >>> func_files = layout.get(datatype='func', extension='.nii.gz')
    >>> metadata = extract_bids_metadata(func_files)
    """
    if not files:
        return pd.DataFrame()

    metadata_records = []

    for file in files:
        if isinstance(file, str):
            file_path = file
            # Extract entities from filename using BIDS conventions
            entities = _parse_bids_filename(file_path)
        elif hasattr(file, "path"):
            file_path = file.path
            entities = file.get_entities() if hasattr(file, "get_entities") else {}
        else:
            continue

        # Initialize record with file info
        record = {"file_path": file_path, **entities}

        # Extract JSON metadata from sidecar
        json_path = str(file_path).replace(".nii.gz", ".json").replace(".nii", ".json")
        if os.path.exists(json_path):
            try:
                import json

                with open(json_path) as f:
                    json_metadata = json.load(f)
                    record.update(json_metadata)
            except Exception as e:
                warnings.warn(f"Failed to read JSON metadata from {json_path}: {e}")

        # Extract events data if requested
        if include_events and "task" in entities:
            events_path = (
                str(file_path)
                .replace(".nii.gz", "_events.tsv")
                .replace(".nii", "_events.tsv")
            )
            if os.path.exists(events_path):
                try:
                    events_df = pd.read_csv(events_path, sep="\t")
                    record["n_events"] = str(len(events_df))
                    record["event_types"] = str(list(events_df.get("trial_type", [])))
                except Exception as e:
                    warnings.warn(f"Failed to read events from {events_path}: {e}")

        metadata_records.append(record)

    # Convert to DataFrame
    metadata_df = pd.DataFrame(metadata_records)

    return metadata_df


def build_bids_query(**filters: Any) -> Dict[str, Any]:
    """Build and validate BIDS query parameters.

    Constructs query parameters for BIDSLayout.get() calls with
    validation and normalization of filter criteria.

    Parameters
    ----------
    **filters
        Filter criteria including:
        - subject : str or list of str
        - session : str or list of str
        - task : str or list of str
        - run : int or list of int
        - datatype : str or list of str
        - suffix : str or list of str
        - extension : str or list of str

    Returns
    -------
    dict
        Normalized query parameters suitable for BIDSLayout.get().

    Examples
    --------
    >>> query = build_bids_query(subject=['01', '02'], task='rest')
    >>> files = layout.get(**query)
    """
    query = {}

    # Normalize filter values
    for key, value in filters.items():
        if value is None:
            continue

        # Handle special cases
        if key == "subject":
            # Remove 'sub-' prefix if present
            if isinstance(value, str):
                value = value.replace("sub-", "")
            elif isinstance(value, list):
                value = [v.replace("sub-", "") for v in value]

        elif key == "session":
            # Remove 'ses-' prefix if present
            if isinstance(value, str):
                value = value.replace("ses-", "")
            elif isinstance(value, list):
                value = [v.replace("ses-", "") for v in value]

        elif key == "run":
            # Convert to integer if string
            if isinstance(value, str) and value.isdigit():
                value = int(value)
            elif isinstance(value, list):
                value = [
                    int(v) if isinstance(v, str) and v.isdigit() else v for v in value
                ]

        query[key] = value

    return query


def load_bids_data(
    layout: BIDSLayout,
    mask: Optional[Union[str, npt.NDArray[np.floating[Any]]]] = None,
    standardize: bool = True,
    **query_filters: Any,
) -> Tuple[npt.NDArray[np.floating[Any]], pd.DataFrame]:
    """Load and preprocess BIDS neuroimaging data.

    Loads neuroimaging data files from a BIDS dataset with optional
    masking, standardization, and filtering.

    Parameters
    ----------
    layout : BIDSLayout
        BIDS layout object from parse_bids_dataset.
    mask : str, array-like, or None
        Brain mask to apply. If str, path to mask file. If array,
        boolean or integer mask. If None, no masking applied.
    standardize : bool, default=True
        Whether to standardize data (z-score) across time.
    **query_filters
        Additional query filters (passed to build_bids_query).

    Returns
    -------
    data : ndarray of shape (n_files, n_voxels, n_timepoints) or (n_voxels, n_timepoints)
        Loaded and preprocessed neuroimaging data.
    metadata : pd.DataFrame
        Metadata for loaded files.

    Examples
    --------
    >>> layout = parse_bids_dataset('/path/to/bids')
    >>> data, meta = load_bids_data(
    ...     layout,
    ...     subject=['01', '02'],
    ...     task='rest',
    ...     datatype='func'
    ... )
    """
    # Build query and get files
    query = build_bids_query(**query_filters)
    files = layout.get(return_type="filename", **query)

    if not files:
        raise ValueError(
            f"No files found matching query: {query}. "
            f"Available subjects: {layout.get_subjects()}, "
            f"tasks: {layout.get_tasks()}, "
            f"datatypes: {layout.get_datatypes()}"
        )

    print(f"Found {len(files)} files matching query")

    # Validate mask if provided
    if mask is not None and isinstance(mask, str):
        mask_path = Path(mask)
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file does not exist: {mask}")
        print(f"Using brain mask: {mask}")

    # Extract metadata
    metadata = extract_bids_metadata(files)

    # Load data
    data_list = []
    valid_files = []

    for i, file_path in enumerate(files):
        try:
            # Load NIfTI data with validation
            img = nib.load(file_path)
            data = img.get_fdata()

            # Validate data dimensions
            if data.ndim < 3:
                warnings.warn(
                    f"Skipping {file_path}: expected 3D or 4D data, got {data.ndim}D"
                )
                continue
            elif data.ndim > 4:
                warnings.warn(
                    f"Skipping {file_path}: expected 3D or 4D data, got {data.ndim}D"
                )
                continue

            # Check for empty data
            if data.size == 0:
                warnings.warn(f"Skipping {file_path}: empty data array")
                continue

            # Check for all-zero or all-NaN data
            if np.all(data == 0) or np.all(np.isnan(data)):
                warnings.warn(f"Skipping {file_path}: data is all zeros or NaN")
                continue

            print(f"Processing file {i+1}/{len(files)}: {Path(file_path).name}")

            # Apply mask if provided
            if mask is not None:
                if isinstance(mask, str):
                    mask_img = nib.load(mask)
                    mask_data = mask_img.get_fdata().astype(bool)
                else:
                    mask_data = np.asarray(mask).astype(bool)

                # Validate mask dimensions match data
                data_spatial_shape = data.shape[:3]
                mask_shape = (
                    mask_data.shape[:3] if mask_data.ndim >= 3 else mask_data.shape
                )

                if data_spatial_shape != mask_shape:
                    warnings.warn(
                        f"Skipping {file_path}: mask shape {mask_shape} "
                        f"doesn't match data spatial shape {data_spatial_shape}"
                    )
                    continue

                # Reshape data for masking
                if data.ndim == 4:  # 4D data (x, y, z, t)
                    data_2d = data.reshape(-1, data.shape[-1])  # (voxels, time)
                    mask_1d = mask_data.reshape(-1)

                    # Check if mask has any True values
                    if not np.any(mask_1d):
                        warnings.warn(
                            f"Skipping {file_path}: mask contains no valid voxels"
                        )
                        continue

                    data_masked = data_2d[mask_1d, :]
                else:  # 3D data
                    data_1d = data.reshape(-1)
                    mask_1d = mask_data.reshape(-1)

                    if not np.any(mask_1d):
                        warnings.warn(
                            f"Skipping {file_path}: mask contains no valid voxels"
                        )
                        continue

                    data_masked = data_1d[mask_1d]
            else:
                # No masking - flatten spatial dimensions
                if data.ndim == 4:
                    data_masked = data.reshape(-1, data.shape[-1])  # (voxels, time)
                else:
                    data_masked = data.reshape(-1)  # (voxels,)

            # Standardize if requested
            if standardize and data_masked.ndim == 2:
                # Standardize across time (axis=1)
                data_masked = (
                    data_masked - data_masked.mean(axis=1, keepdims=True)
                ) / (data_masked.std(axis=1, keepdims=True) + 1e-8)

            data_list.append(data_masked)
            valid_files.append(file_path)

        except MemoryError:
            warnings.warn(f"Skipping {file_path}: insufficient memory to load file")
        except Exception as e:
            warnings.warn(f"Failed to load {file_path}: {type(e).__name__}: {e}")

    # Validate we have at least some valid data
    if not data_list:
        raise RuntimeError(
            f"Failed to load any valid data files from {len(files)} candidates. "
            f"Check file formats, paths, and masks."
        )

    print(f"Successfully loaded {len(data_list)} out of {len(files)} files")

    # Stack data arrays with shape validation
    if len(data_list) == 1:
        final_data = data_list[0]
    else:
        # Check if all arrays have same shape
        shapes = [arr.shape for arr in data_list]
        unique_shapes = set(shapes)

        if len(unique_shapes) == 1:
            final_data = np.stack(data_list, axis=0)  # (files, voxels, time)
        else:
            # Try to find common dimension issues
            voxel_dims = [s[0] for s in shapes]
            time_dims = [s[1] if len(s) > 1 else 1 for s in shapes]

            shape_summary = {
                "voxel_dimensions": list(set(voxel_dims)),
                "time_dimensions": list(set(time_dims)),
                "all_shapes": shapes,
            }

            raise ValueError(
                f"Inconsistent data shapes across files. "
                f"Cannot stack arrays with different dimensions.\n"
                f"Shape summary: {shape_summary}\n"
                f"Suggestion: Ensure all files use the same mask and preprocessing"
            )

    # Filter metadata to match valid files
    valid_metadata = metadata[metadata["file_path"].isin(valid_files)].reset_index(
        drop=True
    )

    return final_data, valid_metadata


def _parse_bids_filename(filename: str) -> Dict[str, Any]:
    """Parse BIDS filename to extract entities.

    Internal helper function to extract BIDS entities from filename
    using standard BIDS naming conventions.

    Parameters
    ----------
    filename : str
        BIDS filename to parse.

    Returns
    -------
    dict
        Dictionary of extracted entities.
    """
    entities: Dict[str, Any] = {}

    # Get basename without path and extension
    basename = Path(filename).stem
    if basename.endswith(".nii"):
        basename = basename[:-4]  # Remove .nii from .nii.gz

    # Split by underscores and parse key-value pairs
    parts = basename.split("_")

    for part in parts:
        if "-" in part:
            key, value = part.split("-", 1)
            # Try to convert numbers
            try:
                if "." in value:
                    entities[key] = float(value)
                else:
                    entities[key] = int(value)
            except ValueError:
                entities[key] = value
        else:
            # This might be the suffix (e.g., 'bold', 'T1w')
            entities["suffix"] = part

    return entities


# Main BIDS fitting interface
def fit_bids(
    bids_path: Union[str, Path],
    mask_path: Optional[str] = None,
    n_factors: Optional[int] = None,
    **filters: Any,
) -> Any:
    """Fit HTFA model on BIDS dataset with automatic configuration.

    This is the main entry point for HTFA analysis on BIDS datasets.
    It handles parsing, data loading, model fitting, and result generation
    automatically.

    Parameters
    ----------
    bids_path : str or Path
        Path to BIDS dataset root directory.
    mask_path : str or None
        Path to brain mask file. If None, no masking applied.
    n_factors : int or None
        Number of factors to extract. If None, automatically determined.
    **filters
        BIDS query filters (subject, task, session, etc.).

    Returns
    -------
    model : TFA or HTFA
        Fitted model instance with BIDS metadata attached.

    Examples
    --------
    >>> model = fit_bids('/path/to/bids', subject=['01', '02'], task='rest')
    >>> factors = model.get_factors()
    >>> weights = model.get_weights()
    """
    # Import here to avoid circular imports
    from . import fit as htfa_fit

    # Prepare filter arguments - convert mask_path to mask
    fit_kwargs = dict(filters)
    if mask_path is not None:
        fit_kwargs["mask"] = mask_path

    # Delegate to the main fit function which handles BIDS detection
    return htfa_fit.fit(bids_path, n_factors=n_factors, **fit_kwargs)


# Convenience function for common HTFA workflow
def load_bids_for_htfa(
    bids_path: Union[str, Path], mask_path: Optional[str] = None, **filters: Any
) -> Tuple[npt.NDArray[np.floating[Any]], pd.DataFrame, BIDSLayout]:
    """Load BIDS data optimized for HTFA analysis.

    Convenience function that combines parsing, validation, and loading
    for a typical HTFA workflow.

    Parameters
    ----------
    bids_path : str or Path
        Path to BIDS dataset root directory.
    mask_path : str or None
        Path to brain mask file.
    **filters
        BIDS query filters (subject, task, etc.).

    Returns
    -------
    data : ndarray
        Loaded and preprocessed neuroimaging data.
    metadata : pd.DataFrame
        File metadata and experimental parameters.
    layout : BIDSLayout
        BIDS layout object for further queries.

    Examples
    --------
    >>> data, metadata, layout = load_bids_for_htfa(
    ...     '/path/to/bids',
    ...     mask_path='/path/to/mask.nii.gz',
    ...     subject=['01', '02'],
    ...     task='rest',
    ...     datatype='func'
    ... )
    """
    # Parse dataset (includes comprehensive BIDS validation via validate_bids_path)
    layout = parse_bids_dataset(bids_path, **filters)

    # Additional structure validation (non-blocking) using our validation framework
    validation = validate_bids_structure(bids_path)
    if validation["errors"]:
        warnings.warn(f"BIDS validation errors: {validation['errors']}")

    # Load data
    data, metadata = load_bids_data(layout, mask=mask_path, standardize=True, **filters)

    return data, metadata, layout
