"""Shared validation for real-valued numerical inputs."""

from typing import Any

import numpy as np
from numpy.typing import NDArray


def real_array(values: Any, name: str) -> NDArray[np.float64]:
    """Convert to float64 without discarding an imaginary component."""
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real-valued")
    return np.asarray(values, dtype=np.float64)
