"""Abstract backend interface for HTFA numerical operations.

This module defines the abstract base class for backend implementations,
providing a common interface for different numerical computing libraries
(NumPy, JAX, PyTorch, etc.).
"""

from typing import Any, List, Optional, Tuple, Union

from abc import ABC, abstractmethod

import numpy as np


class Backend(ABC):
    """Abstract base class for numerical backend implementations.

    This interface abstracts the core numerical operations needed by TFA and HTFA
    algorithms, allowing for different backend implementations (NumPy, JAX, PyTorch).

    The interface is designed to be minimal but complete, covering:
    - Array creation and manipulation
    - Linear algebra operations
    - Mathematical functions
    - Random number generation
    - Array indexing and slicing

    Backend implementations should handle type conversions between their native
    array types and NumPy arrays as needed for interoperability.

    Extension Points
    ---------------
    To implement a new backend:
    1. Inherit from this Backend class
    2. Implement all abstract methods using your library's operations
    3. Register the backend using the registration mechanism
    4. Handle any library-specific optimizations (JIT, GPU, etc.)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this backend implementation."""
        pass

    @property
    @abstractmethod
    def array_type(self) -> type:
        """Return the native array type for this backend."""
        pass

    # Array creation and manipulation

    @abstractmethod
    def asarray(self, data: Any, dtype: Optional[Any] = None) -> Any:
        """Convert input to backend's array type.

        Parameters
        ----------
        data : Any
            Input data to convert to array.
        dtype : Optional[Any]
            Desired data type.

        Returns
        -------
        array : backend array type
            Input data as backend array.
        """
        pass

    @abstractmethod
    def zeros(
        self, shape: Union[int, Tuple[int, ...]], dtype: Optional[Any] = None
    ) -> Any:
        """Create array filled with zeros.

        Parameters
        ----------
        shape : int or tuple of ints
            Shape of the output array.
        dtype : Optional[Any]
            Data type of the output.

        Returns
        -------
        array : backend array type
            Array filled with zeros.
        """
        pass

    @abstractmethod
    def ones(
        self, shape: Union[int, Tuple[int, ...]], dtype: Optional[Any] = None
    ) -> Any:
        """Create array filled with ones.

        Parameters
        ----------
        shape : int or tuple of ints
            Shape of the output array.
        dtype : Optional[Any]
            Data type of the output.

        Returns
        -------
        array : backend array type
            Array filled with ones.
        """
        pass

    @abstractmethod
    def eye(self, N: int, dtype: Optional[Any] = None) -> Any:
        """Create identity matrix.

        Parameters
        ----------
        N : int
            Size of the identity matrix.
        dtype : Optional[Any]
            Data type of the output.

        Returns
        -------
        array : backend array type
            Identity matrix of shape (N, N).
        """
        pass

    @abstractmethod
    def full(
        self,
        shape: Union[int, Tuple[int, ...]],
        fill_value: Any,
        dtype: Optional[Any] = None,
    ) -> Any:
        """Create array filled with a constant value.

        Parameters
        ----------
        shape : int or tuple of ints
            Shape of the output array.
        fill_value : Any
            Value to fill the array with.
        dtype : Optional[Any]
            Data type of the output.

        Returns
        -------
        array : backend array type
            Array filled with fill_value.
        """
        pass

    # Linear algebra operations

    @abstractmethod
    def matmul(self, a: Any, b: Any) -> Any:
        """Matrix multiplication.

        Parameters
        ----------
        a : backend array type
            First array.
        b : backend array type
            Second array.

        Returns
        -------
        array : backend array type
            Matrix product of a and b.
        """
        pass

    @abstractmethod
    def solve(self, A: Any, b: Any) -> Any:
        """Solve linear system Ax = b.

        Parameters
        ----------
        A : backend array type
            Coefficient matrix.
        b : backend array type
            Right-hand side.

        Returns
        -------
        x : backend array type
            Solution to the linear system.
        """
        pass

    @abstractmethod
    def norm(self, x: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Any:
        """Compute vector/matrix norm.

        Parameters
        ----------
        x : backend array type
            Input array.
        axis : Optional[int or tuple of ints]
            Axis or axes along which to compute the norm.

        Returns
        -------
        norm : backend array type
            Norm of the input.
        """
        pass

    @abstractmethod
    def transpose(self, x: Any, axes: Optional[Tuple[int, ...]] = None) -> Any:
        """Transpose array dimensions.

        Parameters
        ----------
        x : backend array type
            Input array.
        axes : Optional[tuple of ints]
            Permutation of axes. If not specified, reverses all axes.

        Returns
        -------
        array : backend array type
            Transposed array.
        """
        pass

    # Mathematical functions

    @abstractmethod
    def exp(self, x: Any) -> Any:
        """Element-wise exponential function.

        Parameters
        ----------
        x : backend array type
            Input array.

        Returns
        -------
        array : backend array type
            Element-wise exponential of x.
        """
        pass

    @abstractmethod
    def abs(self, x: Any) -> Any:
        """Element-wise absolute value.

        Parameters
        ----------
        x : backend array type
            Input array.

        Returns
        -------
        array : backend array type
            Element-wise absolute value of x.
        """
        pass

    @abstractmethod
    def clip(
        self, x: Any, min_val: Optional[Any] = None, max_val: Optional[Any] = None
    ) -> Any:
        """Clip array values to specified range.

        Parameters
        ----------
        x : backend array type
            Input array.
        min_val : Optional[Any]
            Minimum value.
        max_val : Optional[Any]
            Maximum value.

        Returns
        -------
        array : backend array type
            Clipped array.
        """
        pass

    # Statistical functions

    @abstractmethod
    def mean(self, x: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Any:
        """Compute mean along specified axis.

        Parameters
        ----------
        x : backend array type
            Input array.
        axis : Optional[int or tuple of ints]
            Axis or axes along which to compute the mean.

        Returns
        -------
        mean : backend array type
            Mean of the input.
        """
        pass

    @abstractmethod
    def var(self, x: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Any:
        """Compute variance along specified axis.

        Parameters
        ----------
        x : backend array type
            Input array.
        axis : Optional[int or tuple of ints]
            Axis or axes along which to compute the variance.

        Returns
        -------
        var : backend array type
            Variance of the input.
        """
        pass

    @abstractmethod
    def std(self, x: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Any:
        """Compute standard deviation along specified axis.

        Parameters
        ----------
        x : backend array type
            Input array.
        axis : Optional[int or tuple of ints]
            Axis or axes along which to compute the standard deviation.

        Returns
        -------
        std : backend array type
            Standard deviation of the input.
        """
        pass

    @abstractmethod
    def max(self, x: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Any:
        """Compute maximum along specified axis.

        Parameters
        ----------
        x : backend array type
            Input array.
        axis : Optional[int or tuple of ints]
            Axis or axes along which to compute the maximum.

        Returns
        -------
        max : backend array type
            Maximum of the input.
        """
        pass

    @abstractmethod
    def min(self, x: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Any:
        """Compute minimum along specified axis.

        Parameters
        ----------
        x : backend array type
            Input array.
        axis : Optional[int or tuple of ints]
            Axis or axes along which to compute the minimum.

        Returns
        -------
        min : backend array type
            Minimum of the input.
        """
        pass

    # Random number generation

    @abstractmethod
    def randn(self, *shape: int, seed: Optional[int] = None) -> Any:
        """Generate random numbers from standard normal distribution.

        Parameters
        ----------
        *shape : int
            Shape of the output array.
        seed : Optional[int]
            Random seed.

        Returns
        -------
        array : backend array type
            Random numbers from N(0, 1).
        """
        pass

    @abstractmethod
    def choice(
        self,
        a: Union[int, Any],
        size: Optional[Union[int, Tuple[int, ...]]] = None,
        replace: bool = True,
        p: Optional[Any] = None,
        seed: Optional[int] = None,
    ) -> Any:
        """Generate random sample from array.

        Parameters
        ----------
        a : int or backend array type
            If int, samples from range(a). If array, samples from array.
        size : Optional[int or tuple of ints]
            Output shape.
        replace : bool
            Whether to sample with replacement.
        p : Optional[backend array type]
            Probabilities for each element.
        seed : Optional[int]
            Random seed.

        Returns
        -------
        samples : backend array type
            Random samples.
        """
        pass

    # Utility functions

    @abstractmethod
    def concatenate(self, arrays: List[Any], axis: int = 0) -> Any:
        """Concatenate arrays along specified axis.

        Parameters
        ----------
        arrays : List[backend array type]
            List of arrays to concatenate.
        axis : int
            Axis along which to concatenate.

        Returns
        -------
        array : backend array type
            Concatenated array.
        """
        pass

    @abstractmethod
    def column_stack(self, arrays: List[Any]) -> Any:
        """Stack 1-D arrays as columns into a 2-D array.

        Parameters
        ----------
        arrays : List[backend array type]
            List of 1-D arrays to stack.

        Returns
        -------
        array : backend array type
            2-D array with arrays as columns.
        """
        pass

    @abstractmethod
    def tile(self, x: Any, reps: Union[int, Tuple[int, ...]]) -> Any:
        """Tile array by repeating it.

        Parameters
        ----------
        x : backend array type
            Input array.
        reps : int or tuple of ints
            Number of repetitions along each axis.

        Returns
        -------
        array : backend array type
            Tiled array.
        """
        pass

    @abstractmethod
    def meshgrid(self, *arrays: Any, indexing: str = "xy") -> List[Any]:
        """Create coordinate matrices from coordinate vectors.

        Parameters
        ----------
        *arrays : backend array type
            1-D coordinate vectors.
        indexing : str
            Indexing convention ('xy' or 'ij').

        Returns
        -------
        grids : List[backend array type]
            Coordinate matrices.
        """
        pass

    # Type conversion and interoperability

    @abstractmethod
    def to_numpy(self, x: Any) -> np.ndarray:
        """Convert backend array to NumPy array.

        This is crucial for interoperability with existing NumPy-based code
        and for returning results in a standardized format.

        Parameters
        ----------
        x : backend array type
            Input array in backend format.

        Returns
        -------
        array : np.ndarray
            NumPy array containing the same data.
        """
        pass

    @abstractmethod
    def from_numpy(self, x: np.ndarray) -> Any:
        """Convert NumPy array to backend array.

        Parameters
        ----------
        x : np.ndarray
            Input NumPy array.

        Returns
        -------
        array : backend array type
            Array in backend format.
        """
        pass


# Backend registry for dynamic backend selection
_BACKEND_REGISTRY = {}


def register_backend(name: str, backend_class: type) -> None:
    """Register a backend implementation.

    Parameters
    ----------
    name : str
        Name to identify the backend.
    backend_class : type
        Backend class that implements the Backend interface.
    """
    if not issubclass(backend_class, Backend):
        raise ValueError(f"Backend class {backend_class} must inherit from Backend")

    _BACKEND_REGISTRY[name] = backend_class


def get_available_backends() -> List[str]:
    """Get list of available backend names.

    Returns
    -------
    backends : List[str]
        List of registered backend names.
    """
    return list(_BACKEND_REGISTRY.keys())


def create_backend(name: str, **kwargs) -> Backend:
    """Create backend instance by name.

    Parameters
    ----------
    name : str
        Name of the backend to create.
    **kwargs
        Additional arguments passed to backend constructor.

    Returns
    -------
    backend : Backend
        Backend instance.

    Raises
    ------
    ValueError
        If backend name is not registered.
    """
    if name not in _BACKEND_REGISTRY:
        available = ", ".join(get_available_backends())
        raise ValueError(f"Backend '{name}' not found. Available backends: {available}")

    backend_class = _BACKEND_REGISTRY[name]
    return backend_class(**kwargs)


def get_default_backend() -> str:
    """Get the default backend name.

    Returns 'numpy' if available, otherwise the first registered backend.

    Returns
    -------
    name : str
        Default backend name.

    Raises
    ------
    RuntimeError
        If no backends are registered.
    """
    if not _BACKEND_REGISTRY:
        raise RuntimeError("No backends registered")

    if "numpy" in _BACKEND_REGISTRY:
        return "numpy"

    return next(iter(_BACKEND_REGISTRY.keys()))
