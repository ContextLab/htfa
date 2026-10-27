"""Alignment sensitivity of the operations an HTFA subject fit uses, at the
shapes of the n_jobs test (V=8000, T=60, K=36, P=144), on one BLAS thread.
For each operation: distinct results over 8 float64 offsets from a 64-byte
boundary (1 = alignment-independent). Repeats the whole probe 3 times."""

import hashlib
import os
import platform
import subprocess

import numpy as np
import scipy.linalg
from threadpoolctl import threadpool_info, threadpool_limits


def cpu_name():
    if platform.system() == "Windows":
        cmd = ["powershell", "-NoProfile", "(Get-CimInstance Win32_Processor).Name"]
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    if platform.system() == "Linux":
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def h(a):
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:8]


def at(a, off):
    buf = np.empty(a.size + 16, dtype=a.dtype)
    start = (-(buf.ctypes.data // 8)) % 8 + off
    out = buf[start : start + a.size].reshape(a.shape)
    out[...] = a
    return out


print("CPU:", cpu_name(), "| cpu_count", os.cpu_count(), flush=True)
print("BLAS:", [(p["internal_api"], p.get("architecture")) for p in threadpool_info()
                if p["user_api"] == "blas"], flush=True)
print("numpy SIMD:", np.show_runtime() if False else "", flush=True)
rng = np.random.default_rng(0)
V, T, K, P, CH = 8000, 60, 36, 144, 128
X = rng.standard_normal((V, T))
F = np.exp(-rng.uniform(0, 5, (V, K)))
W = rng.standard_normal((K, T))
G = rng.standard_normal((4 * K, V))
A = G @ G.T / V + np.eye(P)
g = rng.standard_normal(P)
M = rng.standard_normal((K, K))
M = M @ M.T + K * np.eye(K)
Xc, Fc = X[:CH], F[:CH]
ops = {
    "F.T@F (syrk)": lambda o: at(F, o).T @ at(F, o),
    "F.T@X": lambda o: at(F, o).T @ at(X, o),
    "F@W": lambda o: at(F, o) @ at(W, o),
    "G@G.T (syrk)": lambda o: at(G, o) @ at(G, o).T,
    "G@X": lambda o: at(G, o) @ at(X, o),
    "chunk Fc@W": lambda o: at(Fc, o) @ at(W, o),
    "chunk Xc@W.T": lambda o: at(Xc, o) @ at(W, o).T,
    "W@W.T": lambda o: at(W, o) @ at(W, o).T,
    "A@g gemv": lambda o: at(A, o) @ at(g, o),
    "g@A gemv T": lambda o: at(g, o) @ at(A, o),
    "g.g dot": lambda o: at(g, o) @ at(g, (o + 3) % 8),
    "eigh KxK": lambda o: np.linalg.eigh(at(M, o))[1],
    "cho PxP": lambda o: scipy.linalg.cho_solve(scipy.linalg.cho_factor(at(A, o)), at(g, o)),
    "solve KxK": lambda o: np.linalg.solve(at(M, o), at(W, o)),
    "exp": lambda o: np.exp(-at(F, o)),
    "sum": lambda o: np.sum(at(X, o)),
    "sum axis1": lambda o: np.sum(at(X, o) ** 2, axis=1),
    "einsum": lambda o: np.einsum("vk,vt->kt", at(F, o), at(X, o)),
}
with threadpool_limits(1, user_api="blas"):
    for rep in range(3):
        res = {name: len({h(f(o)) for o in range(8)}) for name, f in ops.items()}
        bad = {k: v for k, v in res.items() if v > 1}
        print(f"rep {rep}: alignment-dependent ops: {bad or 'none'}", flush=True)
