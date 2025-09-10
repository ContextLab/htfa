"""NumPy backend for HTFA with standard numerical operations."""

from typing import Any, Optional, Tuple

import numpy as np

from ..backend_base import HTFABackend


class NumPyBackend(HTFABackend):
    """NumPy backend for HTFA operations.
    
    This backend wraps standard NumPy operations to provide a consistent
    interface for the HTFA algorithm. It serves as the reference implementation
    and fallback option when more advanced backends (JAX, PyTorch) are not available.
    """

    def array(self, data: Any) -> np.ndarray:
        """Create array from data.
        
        Parameters
        ----------
        data : array-like
            Input data to convert to numpy array.
            
        Returns
        -------
        np.ndarray
            NumPy array containing the data.
        """
        return np.array(data)

    def zeros(self, shape: Tuple[int, ...], dtype: Any = None) -> np.ndarray:
        """Create zeros array.
        
        Parameters
        ----------
        shape : tuple of int
            Shape of the array.
        dtype : data-type, optional
            Data type of the array. Defaults to float32.
            
        Returns
        -------
        np.ndarray
            Array of zeros with specified shape and dtype.
        """
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape: Tuple[int, ...], dtype: Any = None) -> np.ndarray:
        """Create ones array.
        
        Parameters
        ----------
        shape : tuple of int
            Shape of the array.
        dtype : data-type, optional
            Data type of the array.
            
        Returns
        -------
        np.ndarray
            Array of ones with specified shape and dtype.
        """
        return np.ones(shape, dtype=dtype)

    def random(self, shape: Tuple[int, ...], dtype: Any = None) -> np.ndarray:
        """Create random array.
        
        Parameters
        ----------
        shape : tuple of int
            Shape of the array.
        dtype : data-type, optional
            Data type of the array. Defaults to float32.
            
        Returns
        -------
        np.ndarray
            Array of random values drawn from uniform distribution [0, 1).
        """
        return np.random.random(shape).astype(dtype if dtype else np.float32)

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Matrix multiplication.
        
        Parameters
        ----------
        a : np.ndarray
            First input array.
        b : np.ndarray
            Second input array.
            
        Returns
        -------
        np.ndarray
            Matrix product of a and b.
        """
        return np.matmul(a, b)

    def transpose(
        self, a: np.ndarray, axes: Optional[Tuple[int, ...]] = None
    ) -> np.ndarray:
        """Transpose array.
        
        Parameters
        ----------
        a : np.ndarray
            Input array.
        axes : tuple of ints, optional
            If specified, permute the axes according to this tuple.
            
        Returns
        -------
        np.ndarray
            Transposed array.
        """
        return np.transpose(a, axes)

    def svd(
        self, a: np.ndarray, full_matrices: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Singular Value Decomposition.
        
        Parameters
        ----------
        a : np.ndarray
            Input array to decompose.
        full_matrices : bool, default=True
            If False, return reduced SVD.
            
        Returns
        -------
        u : np.ndarray
            Left singular vectors.
        s : np.ndarray
            Singular values.
        vh : np.ndarray
            Right singular vectors (transposed).
        """
        return np.linalg.svd(a, full_matrices=full_matrices)

    def norm(
        self, a: np.ndarray, axis: Optional[int] = None, keepdims: bool = False
    ) -> np.ndarray:
        """Compute norm.
        
        Parameters
        ----------
        a : np.ndarray
            Input array.
        axis : int, optional
            Axis along which to compute norm. If None, compute over all axes.
        keepdims : bool, default=False
            If True, retain reduced dimensions as size-1 dimensions.
            
        Returns
        -------
        np.ndarray
            Norm of the array.
        """
        return np.linalg.norm(a, axis=axis, keepdims=keepdims)

    def mean(
        self, a: np.ndarray, axis: Optional[int] = None, keepdims: bool = False
    ) -> np.ndarray:
        """Compute mean.
        
        Parameters
        ----------
        a : np.ndarray
            Input array.
        axis : int, optional
            Axis along which to compute mean. If None, compute over all axes.
        keepdims : bool, default=False
            If True, retain reduced dimensions as size-1 dimensions.
            
        Returns
        -------
        np.ndarray
            Mean of the array elements.
        """
        return np.mean(a, axis=axis, keepdims=keepdims)

    def to_numpy(self, a: np.ndarray) -> np.ndarray:
        """Convert to numpy array.
        
        For NumPy backend, this is a no-op since arrays are already numpy arrays.
        
        Parameters
        ----------
        a : np.ndarray
            Input array (already numpy).
            
        Returns
        -------
        np.ndarray
            The same array (no conversion needed).
        """
        return a