"""Compute engines for the heavy matrix products in the TFA residual and Jacobian.

An engine holds the (possibly compressed) data ``X`` (V, T) and weights ``W`` (K, T).
It provides the three products the matrix-free residual and Jacobian need (scipy
path), and the data term's robust-loss cost, gradient and Gauss-Newton matrix for
the structured solver (see `gauss_newton`). Prior terms and the mapping to the
parameter vector are handled by :class:`htfa._kernels.ResidualProblem`.
"""

import importlib.util
import os
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
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
# Elements per voxel chunk (chunk rows * max(T, K**2)) in the structured solver's
# data-term passes: bounds the (chunk, T) and (K, K, chunk) temporaries to 16 MB
# each in float64 per NumPy thread, and to 32 MB each in float32 on a GPU (fewer,
# larger kernels).
CHUNK_ELEMENTS = 2**21
GPU_CHUNK_ELEMENTS = 2**23
_THREADS: ContextVar[int | None] = ContextVar("htfa_engine_threads", default=None)


@contextmanager
def thread_limit(n: int) -> Iterator[None]:
    """Limit the threads of NumPy engines created inside the block to ``n``.

    `htfa.htfa.HTFA` uses this (with ``threadpoolctl.threadpool_limits``) so that
    parallel subject fits do not oversubscribe the CPU.
    """
    token = _THREADS.set(max(1, int(n)))
    try:
        yield
    finally:
        _THREADS.reset(token)


def default_threads() -> int:
    """Threads for a new NumPy engine: the `thread_limit` in effect, else all
    CPUs."""
    limit = _THREADS.get()
    return limit if limit is not None else (os.cpu_count() or 1)


# scipy.optimize.least_squares losses rho(z) of the squared residual z (f_scale=1).
LOSS_NAMES = ("linear", "soft_l1", "huber", "cauchy", "arctan")
# scipy's floor for the Gauss-Newton loss weight rho' + 2 z rho''.
EPS = float(np.finfo(np.float64).eps)


def loss_terms(loss: str, z: Any, xp: Any = np) -> tuple[Any, Any, Any]:
    """Robust loss ``rho(z)`` and its first two derivatives at ``z = r**2``.

    The same formulas as ``scipy.optimize.least_squares`` (``f_scale=1``), written
    with the array namespace ``xp`` (``numpy`` or ``torch``) so that they run on
    NumPy arrays and torch tensors alike.

    Parameters
    ----------
    loss : {'linear', 'soft_l1', 'huber', 'cauchy', 'arctan'}
        Loss name.
    z : array or tensor
        Squared residuals (non-negative).
    xp : module, default=numpy
        ``numpy`` or ``torch``, matching the type of ``z``.

    Returns
    -------
    rho, rho1, rho2 : arrays shaped like ``z``
        ``rho(z)``, ``rho'(z)`` and ``rho''(z)``.

    Raises
    ------
    ValueError
        For an unknown loss name.
    """
    if loss == "linear":
        return z, xp.ones_like(z), xp.zeros_like(z)
    if loss == "soft_l1":
        t = 1 + z
        return 2 * (t**0.5 - 1), t**-0.5, -0.5 * t**-1.5
    if loss == "huber":
        mask = z <= 1
        zs = xp.where(mask, 1.0, z)  # keeps z**-0.5 finite where it is unused
        return (
            xp.where(mask, z, 2 * zs**0.5 - 1),
            xp.where(mask, 1.0, zs**-0.5),
            xp.where(mask, 0.0, -0.5 * zs**-1.5),
        )
    if loss == "cauchy":
        t = 1 + z
        return xp.log1p(z), 1 / t, -1 / t**2
    if loss == "arctan":
        t = 1 + z**2
        return xp.arctan(z), 1 / t, -2 * z / t**2
    raise ValueError(f"loss must be one of {LOSS_NAMES}, got {loss!r}")


class Engine(Protocol):
    """What :class:`htfa._kernels.ResidualProblem` needs from an engine."""

    name: str

    def set(self, X: NDArray[np.float64], W: NDArray[np.float64]) -> None: ...

    def residual(self, F: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def times_w(self, A: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def times_wt(self, u: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def data_cost(self, F: NDArray[np.float64], s: float, loss: str) -> float: ...

    def gauss_newton(
        self, F: NDArray[np.float64], G: NDArray[np.float64], s: float, loss: str
    ) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]: ...


class _ChunkedDataTerm:
    """Data-term cost, ``(rho' r) W^T`` and Gauss-Newton matrix, in voxel chunks.

    Subclasses set ``_xp`` (array namespace), ``_X`` and ``_W`` (device arrays),
    ``_W64`` (float64 NumPy copy of ``W``) and implement ``_put``, ``_get`` and
    ``_permute``. Chunks run on `n_threads` threads (NumPy releases the GIL in
    its array operations); each chunk's temporaries hold about
    ``chunk_elements`` elements.
    """

    _xp: Any
    _X: Any
    _W: Any
    _W64: NDArray[np.float64]
    chunk_elements: int = CHUNK_ELEMENTS
    n_threads: int = 1

    def _put(self, a: NDArray[np.float64]) -> Any:
        raise NotImplementedError

    @staticmethod
    def _get(t: Any) -> NDArray[np.float64]:
        raise NotImplementedError

    @staticmethod
    def _permute(t: Any, dims: tuple[int, ...]) -> Any:
        raise NotImplementedError

    def _reset_cache(self) -> None:
        self._pairs: tuple[Any, Any] | None = None

    def _chunks(self, width: int) -> list[slice]:
        V = self._X.shape[0]
        step = max(1, self.chunk_elements // max(1, width))
        return [slice(i, min(i + step, V)) for i in range(0, V, step)]

    def _map(self, fn: Callable[[slice], Any], slices: list[slice]) -> list[Any]:
        if self.n_threads <= 1 or len(slices) <= 1:
            return [fn(sl) for sl in slices]
        with ThreadPoolExecutor(min(self.n_threads, len(slices))) as pool:
            return list(pool.map(fn, slices))

    def _weight_pairs(self) -> tuple[Any, Any]:
        """``(T, K(K+1)/2)`` products ``W_kt W_lt`` (k <= l) and the (K, K) map
        from ``(k, l)`` to their column, both on the device (cached per `set`)."""
        if self._pairs is None:
            K = self._W64.shape[0]
            iu, ju = np.triu_indices(K)
            index = np.empty((K, K), dtype=np.int64)
            index[iu, ju] = index[ju, iu] = np.arange(iu.size)
            pairs = np.ascontiguousarray((self._W64[iu] * self._W64[ju]).T)
            self._pairs = (self._put(pairs), self._put_index(index))
        return self._pairs

    def _put_index(self, index: NDArray[np.int64]) -> Any:
        return index

    def _residual_chunk(self, Fd: Any, sl: slice, s: float, loss: str) -> Any:
        E = s * (self._X[sl] - Fd[sl] @ self._W)
        return E, loss_terms(loss, E * E, self._xp)

    def data_cost(self, F: NDArray[np.float64], s: float, loss: str) -> float:
        """``0.5 * sum(rho(r**2))`` for the data residual ``r = s (X - F W)``."""
        Fd = self._put(F)

        def chunk(sl: slice) -> float:
            _, (rho, _, _) = self._residual_chunk(Fd, sl, s, loss)
            return float(np.sum(self._get(rho.sum(1))))

        return 0.5 * sum(self._map(chunk, self._chunks(self._X.shape[1])))

    def gauss_newton(
        self, F: NDArray[np.float64], G: NDArray[np.float64], s: float, loss: str
    ) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
        """Data-term cost, ``(rho' r) W^T`` and Gauss-Newton matrix.

        Parameters
        ----------
        F : ndarray, shape (V, K)
            Factors.
        G : ndarray, shape (K, V, A)
            ``G[k, v, a] = dF[v, k] / dtheta[k, a]`` for factor ``k``'s ``A``
            parameters.
        s : float
            Residual scale, ``r = s (X - F W)``.
        loss : str
            Name in `LOSS_NAMES`.

        Returns
        -------
        cost : float
            ``0.5 * sum(rho(r**2))``.
        M : ndarray, shape (V, K)
            ``(rho'(r**2) r) W^T``; the gradient is ``-s sum_v G[k, v, a] M[v, k]``.
        H : ndarray, shape (A, K, A, K)
            ``H[a, k, b, l] = s**2 sum_v G[k, v, a] G[l, v, b] Q_v[k, l]`` with
            ``Q_v = W diag(u_v) W^T`` and scipy's loss weights
            ``u = max(rho' + 2 r**2 rho'', EPS)``. For the linear loss ``u = 1``,
            so ``H = (G^T G) * (W W^T)``: one O(V K^2 A^2) product, independent of
            T. Otherwise ``Q_v`` costs one O(V T K^2 / 2) product and the
            contraction O(V K^2 A^2).
        """
        K, T = self._W64.shape
        V, A = self._X.shape[0], G.shape[2]
        Fd, Gd = self._put(F), self._put(G)
        linear = loss == "linear"
        if not linear:
            pairs, index = self._weight_pairs()

        def chunk(sl: slice) -> tuple[float, slice, Any, Any]:
            E, (rho, rho1, rho2) = self._residual_chunk(Fd, sl, s, loss)
            cost = float(np.sum(self._get(rho.sum(1))))
            M = self._get((rho1 * E) @ self._W.T)
            Gc = Gd[:, sl, :]  # (K, B, A) = [l, v, b]
            if linear:
                Gs = self._permute(Gc, (1, 2, 0)).reshape(-1, A * K)  # [v, (a, k)]
                return cost, sl, M, self._get(Gs.T @ Gs)
            u = (rho1 + 2 * (E * E) * rho2).clip(min=EPS)
            Q = (u @ pairs).T[index]  # (K, K, B) = Q_v[l, k], symmetric
            # acc[a][l, k, b - a] = sum_v Q_v[l, k] G[k, v, a] G[l, v, b], b >= a.
            acc = [self._get((Q * Gc[None, :, :, a]) @ Gc[:, :, a:]) for a in range(A)]
            return cost, sl, M, acc

        width = max(T, A * K) if linear else max(T, K * K)
        cost = 0.0
        M = np.empty((V, K))
        H = np.zeros((A, K, A, K))
        for c, sl, M_c, part in self._map(chunk, self._chunks(width)):
            cost += c
            M[sl] = M_c
            if linear:
                H += part.reshape(A, K, A, K)
            else:
                for a in range(A):
                    H[a, :, a:, :] += np.transpose(part[a], (1, 2, 0))
        if linear:
            H *= (self._W64 @ self._W64.T)[None, :, None, :]
        else:
            for a in range(A):
                for b in range(a + 1, A):
                    H[b, :, a, :] = H[a, :, b, :].T
        return 0.5 * cost, M, s**2 * H


class NumpyEngine(_ChunkedDataTerm):
    """Engine backed by NumPy (BLAS) float64 matrix products."""

    name = "numpy"
    _xp = np

    def __init__(self, n_threads: int | None = None) -> None:
        self.n_threads = default_threads() if n_threads is None else n_threads
        self._X: NDArray[np.float64] = np.empty((0, 0))
        self._W: NDArray[np.float64] = np.empty((0, 0))
        self._W64 = self._W
        self._reset_cache()

    def _put(self, a: NDArray[np.float64]) -> NDArray[np.float64]:
        return a

    @staticmethod
    def _get(t: NDArray[np.float64]) -> NDArray[np.float64]:
        return t

    @staticmethod
    def _permute(t: NDArray[np.float64], dims: tuple[int, ...]) -> Any:
        return np.transpose(t, dims)

    def set(self, X: NDArray[np.float64], W: NDArray[np.float64]) -> None:
        """Store data ``X`` (V, T) and weights ``W`` (K, T) as float64."""
        self._X = np.ascontiguousarray(X, dtype=np.float64)
        self._W = np.ascontiguousarray(W, dtype=np.float64)
        self._W64 = self._W
        self._reset_cache()

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


class TorchEngine(_ChunkedDataTerm):
    """Engine backed by torch on ``cpu``, ``cuda`` or ``mps``.

    ``X`` and ``W`` stay resident on the device (float32 on MPS, which has no
    float64; float64 elsewhere). Each call moves in only the V x K-sized operands
    and returns float64 NumPy; sums over voxels are accumulated in float64.
    """

    def __init__(self, device: str) -> None:
        import torch

        self._torch: Any = torch
        self._xp = torch
        self.chunk_elements = GPU_CHUNK_ELEMENTS
        self.device = device
        self.name = f"torch-{device}"
        self._dtype = torch.float32 if device == "mps" else torch.float64
        self._X = torch.empty((0, 0), dtype=self._dtype, device=device)
        self._W = torch.empty((0, 0), dtype=self._dtype, device=device)
        self._W64 = np.empty((0, 0))
        self._reset_cache()

    def _put_index(self, index: NDArray[np.int64]) -> Any:
        return self._torch.from_numpy(index).to(self.device)

    @staticmethod
    def _permute(t: Any, dims: tuple[int, ...]) -> Any:
        return t.permute(dims)

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
        self._W64 = np.array(W, dtype=np.float64)
        self._reset_cache()

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
