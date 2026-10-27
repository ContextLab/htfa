"""Does OpenBLAS compute the same bits in every fresh process on this machine?

Driver: starts N fresh processes; each records the CPU number it runs on just
before NumPy (and OpenBLAS) load, then hashes products on one BLAS thread at
several shapes, including long inner dimensions (GEMM blocking along K) and
the shapes of an HTFA subject fit. Prints per-process lines and a summary of
distinct fingerprints.
usage: python blas_fingerprint.py N
"""

import collections
import os
import subprocess
import sys


def cpu_number():
    if sys.platform == "win32":
        import ctypes

        return ctypes.windll.kernel32.GetCurrentProcessorNumber()
    try:
        return os.sched_getcpu()
    except AttributeError:
        return -1


def child():
    cpu_before = cpu_number()
    import hashlib

    import numpy as np
    from threadpoolctl import threadpool_info, threadpool_limits

    def h(a):
        return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:6]

    rng = np.random.default_rng(0)
    shapes = {
        "gemm 64x20000x64": (rng.standard_normal((64, 20000)), rng.standard_normal((20000, 64))),
        "gemm 300x3000x300": (rng.standard_normal((300, 3000)), rng.standard_normal((3000, 300))),
        "gemm 8000x60x36": (rng.standard_normal((8000, 60)), rng.standard_normal((60, 36))),
        "gemm 36x8000x60": (rng.standard_normal((36, 8000)), rng.standard_normal((8000, 60))),
        "gemm 144x8000x60": (rng.standard_normal((144, 8000)), rng.standard_normal((8000, 60))),
        "gemm 128x60x36": (rng.standard_normal((128, 60)), rng.standard_normal((60, 36))),
    }
    S = rng.standard_normal((8000, 144))
    out = {}
    with threadpool_limits(1, user_api="blas"):
        for name, (a, b) in shapes.items():
            out[name] = h(a @ b)
        out["syrk 144x8000"] = h(S.T @ S)
        out["gemv 8000x144"] = h(S @ S[0])
    arch = sorted({p.get("architecture") for p in threadpool_info() if p["user_api"] == "blas"})
    print("FP", cpu_before, cpu_number(), arch, " ".join(f"{k}={v}" for k, v in out.items()))


if len(sys.argv) > 1 and sys.argv[1] == "--child":
    child()
    sys.exit(0)
n = int(sys.argv[1])
counts = collections.Counter()
by_cpu = collections.defaultdict(collections.Counter)
for i in range(n):
    line = subprocess.run([sys.executable, __file__, "--child"], capture_output=True,
                          text=True).stdout.strip()
    print(i, line, flush=True)
    if line.startswith("FP"):
        parts = line.split(" ", 4)
        fp = line.split("] ", 1)[1]
        counts[fp] += 1
        by_cpu[parts[1]][fp[:60]] += 1
print("\nSUMMARY distinct fingerprints:", len(counts))
for fp, c in counts.items():
    print(f"  {c:3d}  {fp}")
print("by CPU number at load:", {k: dict(v) for k, v in sorted(by_cpu.items())})
