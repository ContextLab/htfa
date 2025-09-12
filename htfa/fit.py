"""Main HTFA fitting interface with automatic input type detection.

This module provides the primary public API for HTFA fitting, automatically
detecting input types (BIDS datasets vs numpy arrays) and routing to
appropriate fitting functions.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import os
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .core import HTFA, TFA


def fit(
    data: Union[str, os.PathLike[str], npt.NDArray[np.floating[Any]]],
    coords: Optional[npt.NDArray[np.floating[Any]]] = None,
    n_factors: Optional[int] = None,
    multi_subject: Optional[bool] = None,
    **kwargs: Any,
) -> Union[TFA, HTFA]:
    """Fit HTFA model with automatic input type detection.

    This is the main public API for HTFA that automatically detects whether
    the input is a BIDS dataset path or numpy arrays and routes to the
    appropriate fitting function.

    Parameters
    ----------
    data : str, PathLike, or ndarray
        Input data. Can be:
        - Path to BIDS dataset directory
        - Path to single NIfTI file
        - Numpy array of shape (n_voxels, n_timepoints) for single subject
        - List of arrays for multi-subject analysis
    coords : ndarray, optional
        Voxel coordinates array of shape (n_voxels, 3). Required for array input,
        ignored for BIDS input (coordinates extracted from NIfTI headers).
    n_factors : int, optional
        Number of factors to extract. If None, will be inferred from data.
    multi_subject : bool, optional
        Whether to use HTFA (multi-subject) or TFA (single-subject).
        If None, will be automatically determined from data.
    **kwargs
        Additional parameters passed to TFA/HTFA constructors.

    Returns
    -------
    TFA or HTFA
        Fitted model instance.

    Raises
    ------
    ValueError
        If input data format is not recognized or invalid.
    FileNotFoundError
        If specified BIDS path does not exist.
    TypeError
        If coords are required but not provided for array input.

    Examples
    --------
    >>> # BIDS dataset
    >>> model = fit('/path/to/bids/dataset')

    >>> # Single subject array
    >>> data = np.random.randn(1000, 200)  # 1000 voxels, 200 timepoints
    >>> coords = np.random.randn(1000, 3)  # voxel coordinates
    >>> model = fit(data, coords=coords)

    >>> # Multi-subject arrays
    >>> data_list = [np.random.randn(1000, 200) for _ in range(5)]
    >>> model = fit(data_list, coords=coords)
    """
    # Detect input type and route to appropriate function
    if isinstance(data, (str, os.PathLike)):
        # Path input - check if it's BIDS dataset or single file
        return _fit_bids_dataset(data, n_factors=n_factors, **kwargs)
    elif isinstance(data, (list, tuple)) and not isinstance(data, np.ndarray):
        # Multi-subject array input
        if coords is None:
            raise TypeError(
                "coords parameter is required for multi-subject array input"
            )
        return _fit_arrays(
            data, coords, n_factors=n_factors, multi_subject=True, **kwargs
        )
    elif isinstance(data, np.ndarray):
        # Single subject array input
        if coords is None:
            raise TypeError("coords parameter is required for array input")
        # Default to False for single arrays if not specified
        use_multi = multi_subject if multi_subject is not None else False
        return _fit_arrays(
            data, coords, n_factors=n_factors, multi_subject=use_multi, **kwargs
        )
    else:
        raise ValueError(
            f"Unsupported data type: {type(data)}. "
            "Expected str/PathLike for BIDS paths or ndarray/list for arrays."
        )


def _fit_bids_dataset(
    path: Union[str, os.PathLike[str]], n_factors: Optional[int] = None, **kwargs: Any
) -> Union[TFA, HTFA]:
    """Fit HTFA model on BIDS dataset.

    Parameters
    ----------
    path : str or PathLike
        Path to BIDS dataset directory or single NIfTI file.
    n_factors : int, optional
        Number of factors to extract. If None, will be inferred.
    **kwargs
        Additional parameters for TFA/HTFA.

    Returns
    -------
    TFA or HTFA
        Fitted model instance.

    Raises
    ------
    FileNotFoundError
        If path does not exist.
    ValueError
        If single file is not a valid NIfTI format.
    NotImplementedError
        For BIDS dataset directories (to be implemented by Stream B).
    """
    path = Path(path)

    # Check if path exists first (matches test expectations)
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    if path.is_file():
        # Single NIfTI file - validate file extension
        # Handle .nii.gz files properly (path.suffix only gets final extension)
        if str(path).lower().endswith(".nii.gz") or path.suffix.lower() == ".nii":
            # Valid NIfTI file
            pass
        else:
            raise ValueError(f"Single file must be NIfTI format, got: {path.name}")

        # Load single file and extract data/coords
        data, coords = _load_nifti_file(path)

        # Infer parameters if not provided
        if n_factors is None:
            n_factors = _infer_parameters(data, **kwargs)["n_factors"]

        # Use TFA for single subject
        model = TFA(n_factors=n_factors, **kwargs)
        model.fit(data.T)  # TFA expects (n_timepoints, n_voxels)

        # Store coordinates for spatial mapping
        model.coords_ = coords
        return model

    elif path.is_dir():
        # BIDS dataset directory - validate BIDS structure first
        from .validation import validate_bids_path

        validate_bids_path(path)  # This will raise ValidationError if invalid

        # This will be implemented by Stream B (BIDS utilities)
        # For now, raise NotImplementedError with clear message
        raise NotImplementedError(
            "BIDS dataset parsing will be implemented by Stream B. "
            "Currently only single NIfTI files are supported."
        )
    else:
        raise ValueError(f"Invalid path type: {path}")


def _fit_arrays(
    data: Union[npt.NDArray[np.floating[Any]], List[npt.NDArray[np.floating[Any]]]],
    coords: npt.NDArray[np.floating[Any]],
    n_factors: Optional[int] = None,
    multi_subject: Optional[bool] = False,
    **kwargs: Any,
) -> Union[TFA, HTFA]:
    """Fit HTFA model on numpy arrays.

    Parameters
    ----------
    data : ndarray or list of ndarrays
        Input data:
        - Single array: (n_voxels, n_timepoints)
        - List of arrays: each (n_voxels, n_timepoints) for multi-subject
    coords : ndarray
        Voxel coordinates of shape (n_voxels, 3).
    n_factors : int, optional
        Number of factors to extract. If None, will be inferred.
    multi_subject : bool, default=False
        Whether to use HTFA (True) or TFA (False).
    **kwargs
        Additional parameters for TFA/HTFA.

    Returns
    -------
    TFA or HTFA
        Fitted model instance.

    Raises
    ------
    ValueError
        If data shapes are inconsistent or invalid.
    """
    if isinstance(data, list):
        # Multi-subject data
        if not data:
            raise ValueError("Empty data list provided")

        # Validate all arrays have same number of voxels and are 2D
        n_voxels = data[0].shape[0]
        for i, arr in enumerate(data):
            if arr.ndim != 2:
                raise ValueError(f"Subject {i}: expected 2D array, got {arr.ndim}D")
            if arr.shape[0] != n_voxels:
                raise ValueError(
                    f"Subject {i}: expected {n_voxels} voxels, got {arr.shape[0]}"
                )

        # Check coordinates match
        if coords.shape[0] != n_voxels:
            raise ValueError(
                f"Coordinates shape {coords.shape[0]} doesn't match "
                f"data voxels {n_voxels}"
            )

        # Infer parameters if needed
        if n_factors is None:
            inferred = _infer_parameters(data[0], **kwargs)
            n_factors = inferred["n_factors"]

        # Use HTFA for multi-subject (override multi_subject parameter)
        model = HTFA(n_factors=n_factors, **kwargs)

        # HTFA expects list of (n_voxels, n_timepoints) - data is already in this format
        # Create coordinate list (same coords for all subjects)
        coords_list = [coords for _ in data] if coords is not None else None
        model.fit(data, coords_list)

        # Store coordinates
        model.coords_ = coords
        return model

    else:
        # Single subject data
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got {data.ndim}D")

        n_voxels, n_timepoints = data.shape
        if coords.shape[0] != n_voxels:
            raise ValueError(
                f"Coordinates shape {coords.shape[0]} doesn't match "
                f"data voxels {n_voxels}"
            )

        # Infer parameters if needed
        if n_factors is None:
            inferred = _infer_parameters(data, **kwargs)
            n_factors = inferred["n_factors"]

        if multi_subject:
            # Force HTFA even for single subject
            model = HTFA(n_factors=n_factors, **kwargs)
            # Wrap data and coords in lists for HTFA
            coords_list = [coords] if coords is not None else None
            model.fit([data], coords_list)  # HTFA expects (n_voxels, n_timepoints)
        else:
            # Use TFA for single subject
            model = TFA(n_factors=n_factors, **kwargs)
            model.fit(data, coords)  # TFA expects (n_voxels, n_timepoints)

        # Store coordinates
        model.coords_ = coords
        return model


def _infer_parameters(
    data: npt.NDArray[np.floating[Any]], **kwargs: Any
) -> Dict[str, Any]:
    """Infer optimal parameters from data characteristics.

    Parameters
    ----------
    data : ndarray
        Input data of shape (n_voxels, n_timepoints).
    **kwargs
        Additional parameters that override inference.

    Returns
    -------
    dict
        Dictionary of inferred parameters.
    """
    n_voxels, n_timepoints = data.shape

    # Infer number of factors based on data dimensions
    # Use heuristic: min(sqrt(n_voxels), n_timepoints/10, 50)
    n_factors_heuristic = min(
        int(np.sqrt(n_voxels)),
        max(1, n_timepoints // 10),
        50,  # Cap at reasonable maximum
    )

    inferred = {
        "n_factors": n_factors_heuristic,
        "max_iter": 100,  # Default iterations
        "tol": 1e-6,  # Default tolerance
    }

    # Override with any provided kwargs
    inferred.update({k: v for k, v in kwargs.items() if k in inferred})

    return inferred


def _load_nifti_file(
    path: Path,
) -> Tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
    """Load NIfTI file and extract data and coordinates.

    Parameters
    ----------
    path : Path
        Path to NIfTI file.

    Returns
    -------
    tuple
        (data, coords) where data is (n_voxels, n_timepoints) and
        coords is (n_voxels, 3).

    Raises
    ------
    ValueError
        If the NIfTI file is not 4D.
    """
    import nibabel as nib
    
    # Load the NIfTI image
    img = nib.load(str(path))
    data = img.get_fdata()
    
    # Check that it's 4D data (x, y, z, time)
    if data.ndim != 4:
        raise ValueError(
            f"Expected 4D NIfTI file (x, y, z, time), got {data.ndim}D"
        )
    
    # Get dimensions
    nx, ny, nz, n_timepoints = data.shape
    
    # Reshape to (n_voxels, n_timepoints)
    n_voxels = nx * ny * nz
    data_2d = data.reshape(n_voxels, n_timepoints)
    
    # Generate voxel coordinates in MNI space using the affine matrix
    affine = img.affine
    
    # Create voxel indices
    i, j, k = np.meshgrid(
        np.arange(nx), 
        np.arange(ny), 
        np.arange(nz),
        indexing='ij'
    )
    
    # Flatten the indices
    voxel_indices = np.column_stack([
        i.ravel(),
        j.ravel(), 
        k.ravel(),
        np.ones(n_voxels)  # Homogeneous coordinates
    ])
    
    # Transform to MNI coordinates
    mni_coords = voxel_indices @ affine.T
    coords = mni_coords[:, :3]  # Drop the homogeneous coordinate
    
    return data_2d, coords
