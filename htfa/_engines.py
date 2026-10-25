"""Compute engines for the heavy matrix products in the TFA residual and Jacobian.

An engine holds the (possibly compressed) data ``X`` (V, T) and weights ``W`` (K, T)
and provides the three products the residual and Jacobian need. Scaling and signs
are applied by :class:`htfa._kernels.ResidualProblem`.
"""

import importlib.util
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

BACKENDS = ("auto", "numpy", "cpu", "torch", "cuda", "mps")
# Minimum V*T*K for 'auto' to prefer a GPU. Measured on an M2 Max (MPS, under
# load from other jobs; scripts/benchmark.py, 5 TFA iterations): MPS took 2.5x as
# long as NumPy at 2e7 and 1.35x as long at 6e8, so no measured fit size reached
# the 1.2x speedup required. A 1e10 fit did not finish one iteration in 3 h; there
# the engine products alone ran 1.3-1.5x faster on MPS, so 1e10 is provisional.
GPU_MIN_WORK = int(1e10)


class Engine(Protocol):
    """What :class:`htfa._kernels.ResidualProblem` needs from an engine."""

    name: str

    def set(self, X: NDArray[np.float64], W: NDArray[np.float64]) -> None: ...

    def residual(self, F: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def times_w(self, A: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def times_wt(self, u: NDArray[np.float64]) -> NDArray[np.float64]: ...


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


class TorchEngine:
    """Engine backed by torch on ``cpu``, ``cuda`` or ``mps``.

    ``X`` and ``W`` stay resident on the device (float32 on MPS, which has no
    float64; float64 elsewhere). Each call moves in only the small operand and
    returns float64 NumPy.
    """

    def __init__(self, device: str) -> None:
        import torch

        self._torch: Any = torch
        self.device = device
        self.name = f"torch-{device}"
        self._dtype = torch.float32 if device == "mps" else torch.float64
        self._X = torch.empty((0, 0), dtype=self._dtype, device=device)
        self._W = torch.empty((0, 0), dtype=self._dtype, device=device)

    def _put(self, a: NDArray[np.float64]) -> Any:
        t = self._torch.from_numpy(np.ascontiguousarray(a, dtype=np.float64))
        return t.to(device=self.device, dtype=self._dtype)

    @staticmethod
    def _get(t: Any) -> NDArray[np.float64]:
        return np.asarray(t.cpu().numpy(), dtype=np.float64)

    def set(self, X: NDArray[np.float64], W: NDArray[np.float64]) -> None:
        """Store data ``X`` (V, T) and weights ``W`` (K, T) on the device."""
        self._X = self._put(X)
        self._W = self._put(W)

    def residual(self, F: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return ``(X - F @ W).ravel()`` for factors ``F`` (V, K)."""
        return self._get((self._X - self._put(F) @ self._W).reshape(-1))

    def times_w(self, A: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return ``(A @ W).ravel()`` for ``A`` (V, K)."""
        return self._get((self._put(A) @ self._W).reshape(-1))

    def times_wt(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return ``u.reshape(V, T) @ W.T`` (V, K) for a raveled (V*T,) vector."""
        U = self._put(u).reshape(self._X.shape[0], self._W.shape[1])
        return self._get(U @ self._W.T)


def _available_devices(torch: Any) -> list[str]:
    found = []
    if torch.cuda.is_available():
        found.append("cuda")
    if torch.backends.mps.is_available():
        found.append("mps")
    return found


def select_engine(backend: str, V: int, T: int, K: int) -> Engine:
    """Choose an engine for ``backend`` and a (V, T, K)-sized problem.

    ``'auto'`` uses a GPU torch engine only when torch is already installed, a
    CUDA or MPS device exists, and ``V*T*K >= GPU_MIN_WORK``.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if backend in ("numpy", "cpu"):
        return NumpyEngine()
    if backend == "auto":
        if V * T * K < GPU_MIN_WORK or importlib.util.find_spec("torch") is None:
            return NumpyEngine()
        import torch

        gpus = _available_devices(torch)
        return TorchEngine(gpus[0]) if gpus else NumpyEngine()
    from htfa._lazy import ensure_torch

    torch = ensure_torch()
    gpus = _available_devices(torch)
    if backend == "torch":
        return TorchEngine(gpus[0] if gpus else "cpu")
    if backend not in gpus:
        raise RuntimeError(f"backend {backend!r} requested but no {backend} device")
    return TorchEngine(backend)
