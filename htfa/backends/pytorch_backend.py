"""PyTorch backend for HTFA with GPU acceleration support."""

from typing import Any, Optional, Tuple

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

    # Create dummy torch for type hints
    class _DummyTorch:
        Tensor = Any
        device = Any
        dtype = Any
        float32 = Any

    torch = _DummyTorch()

from ..backend_base import HTFABackend


class PyTorchBackend(HTFABackend):
    """PyTorch backend with GPU acceleration support."""

    def __init__(self, device: Optional[str] = None, dtype: Any = None):
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch is required for PyTorchBackend. Install with: pip install torch"
            )

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.dtype = dtype if dtype is not None else torch.float32

    def array(self, data: Any) -> Any:
        """Create PyTorch tensor from data."""
        return torch.tensor(data, device=self.device, dtype=self.dtype)

    def zeros(self, shape: Tuple[int, ...], dtype: Any = None) -> Any:
        """Create zeros tensor."""
        dt = dtype if dtype is not None else self.dtype
        return torch.zeros(shape, device=self.device, dtype=dt)

    def ones(self, shape: Tuple[int, ...], dtype: Any = None) -> Any:
        """Create ones tensor."""
        dt = dtype if dtype is not None else self.dtype
        return torch.ones(shape, device=self.device, dtype=dt)

    def random(self, shape: Tuple[int, ...], dtype: Any = None) -> Any:
        """Create random tensor."""
        dt = dtype if dtype is not None else self.dtype
        return torch.randn(shape, device=self.device, dtype=dt)

    def matmul(self, a: Any, b: Any) -> Any:
        """Matrix multiplication."""
        return torch.matmul(a, b)

    def transpose(self, a: Any, axes: Optional[Tuple[int, ...]] = None) -> Any:
        """Transpose tensor."""
        if axes is None:
            return a.T
        else:
            return a.permute(axes)

    def svd(self, a: Any, full_matrices: bool = True) -> Tuple[Any, Any, Any]:
        """Singular Value Decomposition."""
        U, S, Vh = torch.linalg.svd(a, full_matrices=full_matrices)
        return U, S, Vh

    def norm(self, a: Any, axis: Optional[int] = None, keepdims: bool = False) -> Any:
        """Compute norm."""
        if axis is None:
            return torch.linalg.norm(a)
        else:
            return torch.linalg.norm(a, dim=axis, keepdim=keepdims)

    def mean(self, a: Any, axis: Optional[int] = None, keepdims: bool = False) -> Any:
        """Compute mean."""
        if axis is None:
            return torch.mean(a)
        else:
            return torch.mean(a, dim=axis, keepdim=keepdims)

    def to_numpy(self, a: Any) -> np.ndarray:
        """Convert PyTorch tensor to numpy."""
        return a.detach().cpu().numpy()

    def to(self, device: str) -> "PyTorchBackend":
        """Move backend to different device."""
        self.device = torch.device(device)
        return self

    def is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        return torch.cuda.is_available()

    def get_device(self) -> Any:
        """Get current device."""
        return self.device
