"""Compute engines for the heavy matrix products in the TFA residual and Jacobian.

An engine holds the (possibly compressed) data ``X`` (V, T) and weights ``W`` (K, T)
and provides the three products the residual and Jacobian need. Scaling and signs
are applied by :class:`htfa._kernels.ResidualProblem`.
"""

import numpy as np
from numpy.typing import NDArray


class NumpyEngine:
    """Engine backed by NumPy (BLAS) float64 matrix products."""

    name = "numpy"

    def __init__(self) -> None:
        self._X: NDArray[np.float64] = np.empty((0, 0))
        self._W: NDArray[np.float64] = np.empty((0, 0))

    def set(self, X: NDArray[np.float64], W: NDArray[np.float64]) -> None:
        """Store data ``X`` (V, T) and weights ``W`` (K, T) as float64."""
        self._X = np.ascontiguousarray(X, dtype=np.float64)
        self._W = np.ascontiguousarray(W, dtype=np.float64)

    def residual(self, F: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return ``(X - F @ W).ravel()`` for factors ``F`` (V, K)."""
        out = self._X - F @ self._W
        return out.ravel()

    def times_w(self, A: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return ``(A @ W).ravel()`` for ``A`` (V, K)."""
        return (A @ self._W).ravel()

    def times_wt(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return ``u.reshape(V, T) @ W.T`` (V, K) for a raveled (V*T,) vector."""
        return np.reshape(u, (self._X.shape[0], self._W.shape[1])) @ self._W.T
