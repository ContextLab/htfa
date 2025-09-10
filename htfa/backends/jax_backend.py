"""JAX backend for HTFA with automatic differentiation support."""

from typing import Any, Optional, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import random

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

    # Create dummy jnp for type hints
    class _DummyJNP:
        ndarray = Any

    jnp = _DummyJNP()

from ..backend_base import HTFABackend


class JAXBackend(HTFABackend):
    """JAX backend with automatic differentiation and JIT compilation."""

    def __init__(self, use_jit: bool = True):
        if not HAS_JAX:
            raise ImportError(
                "JAX is required for JAXBackend. Install with: pip install jax jaxlib"
            )

        self.use_jit = use_jit
        self.key = random.PRNGKey(0)

    def array(
        self, data: Any
    ) -> Any:  # Changed return type to Any to avoid jnp dependency
        """Create JAX array from data."""
        return jnp.array(data)

    def zeros(self, shape: Tuple[int, ...], dtype: Any = None) -> Any:
        """Create zeros array."""
        return jnp.zeros(shape, dtype=dtype)

    def ones(self, shape: Tuple[int, ...], dtype: Any = None) -> Any:
        """Create ones array."""
        return jnp.ones(shape, dtype=dtype)

    def random(self, shape: Tuple[int, ...], dtype: Any = None) -> Any:
        """Create random array."""
        self.key, subkey = random.split(self.key)
        arr = random.normal(subkey, shape)
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    def matmul(self, a: Any, b: Any) -> Any:
        """Matrix multiplication with optional JIT compilation."""
        if self.use_jit:
            return jax.jit(jnp.matmul)(a, b)
        return jnp.matmul(a, b)

    def transpose(self, a: Any, axes: Optional[Tuple[int, ...]] = None) -> Any:
        """Transpose array."""
        return jnp.transpose(a, axes)

    def svd(self, a: Any, full_matrices: bool = True) -> Tuple[Any, Any, Any]:
        """Singular Value Decomposition."""
        return jnp.linalg.svd(a, full_matrices=full_matrices)

    def norm(self, a: Any, axis: Optional[int] = None, keepdims: bool = False) -> Any:
        """Compute norm."""
        return jnp.linalg.norm(a, axis=axis, keepdims=keepdims)

    def mean(self, a: Any, axis: Optional[int] = None, keepdims: bool = False) -> Any:
        """Compute mean."""
        return jnp.mean(a, axis=axis, keepdims=keepdims)

    def to_numpy(self, a: Any) -> np.ndarray:
        """Convert JAX array to numpy."""
        return np.asarray(a)

    def enable_jit(self) -> None:
        """Enable JIT compilation."""
        self.use_jit = True

    def disable_jit(self) -> None:
        """Disable JIT compilation."""
        self.use_jit = False
