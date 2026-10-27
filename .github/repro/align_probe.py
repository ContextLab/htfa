"""Does this machine's BLAS give different bits for the same operands at different
memory alignments? Prints the CPU, the BLAS kernel, and per operation the number
of distinct results over 8 offsets (float64 steps from a 64-byte boundary)."""

import hashlib
import os
import platform
import subprocess

import numpy as np
import scipy.linalg
from threadpoolctl import threadpool_info, threadpool_limits


def cpu_name():
    try:
        if platform.system() == "Windows":
            cmd = ["powershell", "-NoProfile", "(Get-CimInstance Win32_Processor).Name"]
            return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
        if platform.system() == "Linux":
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        cmd = ["sysctl", "-n", "machdep.cpu.brand_string"]
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    except OSError as err:
        return f"unknown ({err})"


def h(a):
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:8]


def at_offset(a, off):
    buf = np.empty(a.size + 16, dtype=np.float64)
    start = (-(buf.ctypes.data // 8)) % 8 + off
    out = buf[start : start + a.size].reshape(a.shape)
    out[...] = a
    assert out.ctypes.data % 64 == 8 * off
    return out


print("CPU:", cpu_name(), "| cpu_count", os.cpu_count(), flush=True)
print("BLAS:", [(p["internal_api"], p.get("architecture"), p["num_threads"])
                for p in threadpool_info() if p["user_api"] == "blas"], flush=True)
rng = np.random.default_rng(0)
with threadpool_limits(1, user_api="blas"):
    for P in (80, 144, 400, 1000):
        B = rng.standard_normal((P, 3 * P))
        A = B @ B.T + np.eye(P)
        g = rng.standard_normal(P)
        X = rng.standard_normal((5 * P, 60))
        c, low = scipy.linalg.cho_factor(A)
        ops = {
            "gemv A@g": lambda o: at_offset(A, o) @ at_offset(g, o),
            "dot g.g": lambda o: np.dot(at_offset(g, o), at_offset(g, (o + 3) % 8)),
            "gemm B@B.T": lambda o: at_offset(B, o) @ at_offset(B, o).T,
            "gemm X.T@X": lambda o: at_offset(X, o).T @ at_offset(X, o),
            "cho_solve": lambda o: scipy.linalg.cho_solve((at_offset(c, o), low),
                                                          at_offset(g, o)),
            "cho_factor": lambda o: scipy.linalg.cho_factor(at_offset(A, o))[0],
            "eigh": lambda o: np.linalg.eigh(at_offset(A, o))[0],
            "sum": lambda o: np.sum(at_offset(X, o)),
        }
        out = {name: len({h(f(o)) for o in range(8)}) for name, f in ops.items()}
        print(f"P={P}: distinct results over 8 alignments:", out, flush=True)
