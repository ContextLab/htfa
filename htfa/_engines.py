"""Compute engines for the heavy matrix products in the TFA residual and Jacobian.

An engine holds the (possibly compressed) data ``X`` (V, T) and weights ``W`` (K, T).
It provides the three products the matrix-free residual and Jacobian need (scipy
path), and the data term's robust-loss cost, gradient and Gauss-Newton matrix for
the structured solver (see `gauss_newton`). Prior terms and the mapping to the
parameter vector are handled by :class:`htfa._kernels.ResidualProblem`.
"""

import importlib.util
import os
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

BACKENDS = ("auto", "numpy", "cpu", "torch", "cuda", "mps")
# Minimum V*T*K for 'auto' to prefer a GPU. Measured with the structured solver
# on an M2 Max (MPS float32 vs NumPy, soft_l1, scripts/benchmark.py; other jobs
# kept the CPU load at 60-180 on 12 cores, which handicaps NumPy): 5 iterations at
# V*T*K = 6e8 (K=100) took 31 s on MPS vs 48 s, full fits 880 s vs 1627 s; at
# 3e8 (K=50) MPS was slower (16.9 s vs 13.2 s). Smaller sizes favoured MPS too
# but are dominated by load noise; 5e8 is the smallest size where MPS won with K
# large enough for the T*K^2 Gauss-Newton product to dominate. Provisional until
# measured on an idle machine. Accuracy at that size (20000x300, K=100, init
# 'power', 30 iterations): MPS max matched center error 5.6681 vs CPU 5.6684
# (both unconverged), centers within 3.5e-4, weights r = 0.9999999999.
GPU_MIN_WORK = int(5e8)
# Elements per voxel chunk (chunk rows * max(T, K**2)) in the structured solver's
# data-term passes: bounds the (chunk, T) and (K, K, chunk) temporaries to 16 MB
# each in float64 per NumPy thread, and to 32 MB each in float32 on a GPU (fewer,
# larger kernels).
CHUNK_ELEMENTS = 2**21
# Rows per chunk never fall below this, so that the (rows, T) x (T, K(K+1)/2)
# product stays an efficient GEMM at large K (K=200: 52 rows by the element
# budget alone).
MIN_CHUNK_ROWS = 128
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
    # Square roots and divisions instead of fractional powers: NumPy's pow is
    # about 25x slower than sqrt (76 ms vs 3 ms for the three soft_l1 terms of
    # 4.2e6 residuals); the results agree to rounding.
    if loss == "soft_l1":
        t = 1 + z
        root = xp.sqrt(t)
        rho1 = 1 / root
        return 2 * (root - 1), rho1, -0.5 * rho1 / t
    if loss == "huber":
        mask = z <= 1
        zs = xp.where(mask, 1.0, z)  # keeps 1 / sqrt(z) finite where it is unused
        root = xp.sqrt(zs)
        return (
            xp.where(mask, z, 2 * root - 1),
            xp.where(mask, 1.0, 1 / root),
            xp.where(mask, 0.0, -0.5 / (root * zs)),
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
    eps: float

    def set(self, X: NDArray[np.float64], W: NDArray[np.float64]) -> None: ...

    def set_weights(self, W: NDArray[np.float64]) -> None: ...

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
    ``_W64`` (float64 NumPy copy of ``W``) and implement ``_put`` and ``_get``.
    Chunks run on `n_threads` threads (NumPy releases the GIL in
    its array operations); each chunk's temporaries hold about
    ``chunk_elements`` elements.
    """

    _xp: Any
    _X: Any
    _W: Any
    _W64: NDArray[np.float64]
    chunk_elements: int = CHUNK_ELEMENTS
    min_chunk_rows: int = MIN_CHUNK_ROWS
    n_threads: int = 1

    def _put(self, a: NDArray[np.float64]) -> Any:
        raise NotImplementedError

    @staticmethod
    def _get(t: Any) -> NDArray[np.float64]:
        raise NotImplementedError

    def _reset_cache(self) -> None:
        self._pairs: tuple[Any, Any] | None = None

    def _chunks(self, width: int) -> list[slice]:
        V = self._X.shape[0]
        step = max(self.min_chunk_rows, self.chunk_elements // max(1, width), 1)
        return [slice(i, min(i + step, V)) for i in range(0, V, step)]

    def _imap(self, fn: Callable[[slice], Any], slices: list[slice]) -> Iterator[Any]:
        """``fn`` over ``slices``, yielded in order, with at most ``2 n_threads``
        chunks in flight so that unconsumed partial results stay bounded."""
        if self.n_threads <= 1 or len(slices) <= 1:
            yield from (fn(sl) for sl in slices)
            return
        window = 2 * self.n_threads
        with ThreadPoolExecutor(min(self.n_threads, len(slices))) as pool:
            pending: deque[Any] = deque()
            for sl in slices:
                pending.append(pool.submit(fn, sl))
                if len(pending) >= window:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()

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

        # The chunks of `gauss_newton`'s robust-loss pass: about 10x more than a
        # (chunk, T) element budget alone gives at fMRI sizes, so that the pass
        # spreads over the engine's threads (T=168, K=50, V=25k: 30 chunks, not 2).
        K, T = self._W64.shape
        return 0.5 * sum(self._imap(chunk, self._chunks(max(T, K * K))))

    def gauss_newton(
        self, F: NDArray[np.float64], G: NDArray[np.float64], s: float, loss: str
    ) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
        """Data-term cost, ``(rho' r) W^T`` and Gauss-Newton matrix.

        Parameters
        ----------
        F : ndarray, shape (V, K)
            Factors.
        G : ndarray, shape (K, A, V)
            ``G[k, a, v] = dF[v, k] / dtheta[k, a]`` for factor ``k``'s ``A``
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
            ``(rho'(r**2) r) W^T``; the gradient is ``-s sum_v G[k, a, v] M[v, k]``.
        H : ndarray, shape (K, A, K, A)
            ``H[k, a, l, b] = s**2 sum_v G[k, a, v] G[l, b, v] Q_v[k, l]`` with
            ``Q_v = W diag(u_v) W^T`` and scipy's loss weights
            ``u = max(rho' + 2 r**2 rho'', EPS)``. For the linear loss ``u = 1``,
            so ``H = (G G^T) * (W W^T)``: one O(V K^2 A^2) product, independent of
            T. Otherwise ``Q_v`` costs one O(V T K^2 / 2) product and the
            contraction O(V K^2 A^2).
        """
        K, T = self._W64.shape
        V, A = self._X.shape[0], G.shape[1]
        Fd, Gd = self._put(F), self._put(G)
        linear = loss == "linear"
        if not linear:
            pairs, index = self._weight_pairs()

        def chunk(sl: slice) -> tuple[float, slice, Any, Any]:
            E, (rho, rho1, rho2) = self._residual_chunk(Fd, sl, s, loss)
            cost = float(np.sum(self._get(rho.sum(1))))
            M = self._get((rho1 * E) @ self._W.T)
            Gc = Gd[:, :, sl]  # (K, A, B)
            if linear:
                Gs = Gc.reshape(K * A, -1)  # [(k, a), v]
                return cost, sl, M, self._get(Gs @ Gs.T)
            u = (rho1 + 2 * (E * E) * rho2).clip(min=EPS)
            Q = (u @ pairs).T[index]  # (K, K, B) = Q_v[l, k], symmetric
            # acc[a][l, k, b - a] = sum_v Q_v[l, k] G[k, a, v] G[l, b, v], b >= a.
            GcT = Gc.swapaxes(1, 2)  # (K, B, A) view
            acc = [self._get((Q * Gc[None, :, a, :]) @ GcT[:, :, a:]) for a in range(A)]
            return cost, sl, M, acc

        width = max(T, A * K) if linear else max(T, K * K)
        cost = 0.0
        M = np.empty((V, K))
        H = np.zeros((K, A, K, A))
        for c, sl, M_c, part in self._imap(chunk, self._chunks(width)):
            cost += c
            M[sl] = M_c
            if linear:
                H += part.reshape(K, A, K, A)
            else:
                for a in range(A):
                    H[:, a, :, a:] += np.transpose(part[a], (1, 0, 2))
        if linear:
            H *= (self._W64 @ self._W64.T)[:, None, :, None]
        else:
            for a in range(A):
                for b in range(a + 1, A):
                    H[:, b, :, a] = H[:, a, :, b].T
        return 0.5 * cost, M, s**2 * H


class NumpyEngine(_ChunkedDataTerm):
    """Engine backed by NumPy (BLAS) float64 matrix products."""

    name = "numpy"
    _xp = np
    eps = EPS

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

    def set(self, X: NDArray[np.float64], W: NDArray[np.float64]) -> None:
        """Store data ``X`` (V, T) and weights ``W`` (K, T) as float64."""
        self._X = np.ascontiguousarray(X, dtype=np.float64)
        self.set_weights(W)

    def set_weights(self, W: NDArray[np.float64]) -> None:
        """Replace the weights ``W`` (K, T), keeping the data."""
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
    and returns float64 NumPy. In the structured solver's passes, sums over
    samples and over the voxels of one chunk run in the device dtype (float32 on
    MPS); per-voxel costs and per-chunk partial sums are accumulated in float64.
    `eps` is the device dtype's machine epsilon.
    """

    def __init__(self, device: str) -> None:
        import torch

        self._torch: Any = torch
        self._xp = torch
        self.chunk_elements = GPU_CHUNK_ELEMENTS
        self.device = device
        self.name = f"torch-{device}"
        self._dtype = torch.float32 if device == "mps" else torch.float64
        self.eps = float(torch.finfo(self._dtype).eps)
        self._X = torch.empty((0, 0), dtype=self._dtype, device=device)
        self._W = torch.empty((0, 0), dtype=self._dtype, device=device)
        self._W64 = np.empty((0, 0))
        self._reset_cache()

    def _put_index(self, index: NDArray[np.int64]) -> Any:
        return self._torch.from_numpy(index).to(self.device)

    def _put(self, a: NDArray[np.float64]) -> Any:
        t = self._torch.from_numpy(np.ascontiguousarray(a, dtype=np.float64))
        return t.to(device=self.device, dtype=self._dtype)

    @staticmethod
    def _get(t: Any) -> NDArray[np.float64]:
        return np.asarray(t.cpu().numpy(), dtype=np.float64)

    def set(self, X: NDArray[np.float64], W: NDArray[np.float64]) -> None:
        """Store data ``X`` (V, T) and weights ``W`` (K, T) on the device."""
        self._X = self._put(X)
        self.set_weights(W)

    def set_weights(self, W: NDArray[np.float64]) -> None:
        """Replace the weights ``W`` (K, T) on the device, keeping the data."""
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
