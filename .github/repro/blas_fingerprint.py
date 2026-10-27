"""Does every fresh process on this machine pick the same BLAS kernels and
compute the same bits?

Each child records: the CPU number before NumPy loads, NumPy's runtime SIMD
features, every OpenBLAS library's core name and config (numpy's and
scipy's), and hashes of NumPy (numpy's OpenBLAS) and SciPy (scipy's OpenBLAS)
products on one BLAS thread, at the shapes of an HTFA subject fit and at long
inner dimensions. The driver prints the distinct child records.
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
    import ctypes
    import hashlib

    import numpy as np
    import scipy.linalg
    from numpy._core._multiarray_umath import __cpu_features__ as feats
    from threadpoolctl import threadpool_info, threadpool_limits

    def h(a):
        return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:6]

    libs = []
    for p in threadpool_info():
        if p["internal_api"] != "openblas":
            continue
        lib = ctypes.CDLL(p["filepath"])
        names = [n for n in dir(lib)]  # noqa: F841 (ctypes resolves lazily)
        cfg = core = "?"
        for prefix in ("scipy_openblas_", "scipy_openblas64_", "openblas_", ""):
            try:
                f = getattr(lib, f"{prefix}openblas_get_config" if prefix else "openblas_get_config")
                f.restype = ctypes.c_char_p
                cfg = f().decode()
                g = getattr(lib, f"{prefix}openblas_get_corename" if prefix else "openblas_get_corename")
                g.restype = ctypes.c_char_p
                core = g().decode()
                break
            except AttributeError:
                continue
        libs.append(f"{os.path.basename(p['filepath'])[:24]}:{core}:{cfg}")
    simd = ",".join(k for k, v in feats.items() if v and k.startswith("AVX512"))
    rng = np.random.default_rng(0)
    S = rng.standard_normal((8000, 144))
    A = S.T @ S / 8000 + np.eye(144)
    B = rng.standard_normal((1000, 3000))
    C = B @ B.T / 3000 + np.eye(1000)
    M = rng.standard_normal((36, 36))
    M = M @ M.T + 36 * np.eye(36)
    g = rng.standard_normal(144)
    out = {}
    with threadpool_limits(1, user_api="blas"):
        out["np 64x20000x64"] = h(rng.standard_normal((64, 20000)) @ rng.standard_normal((20000, 64)))
        out["np syrk"] = h(S.T @ S)
        out["np 8000x36x36"] = h(S[:, :36] @ M)
        out["sp cho144"] = h(scipy.linalg.cho_solve(scipy.linalg.cho_factor(A), g))
        out["sp cho1000"] = h(scipy.linalg.cho_factor(C)[0])
        out["sp eigh36"] = h(scipy.linalg.eigh(M)[1])
        out["np eigh36"] = h(np.linalg.eigh(M)[1])
        out["sp gemm"] = h(scipy.linalg.blas.dgemm(1.0, B, B.T))
    print("REC", cpu_before, "|", simd, "|", " ; ".join(libs), "|",
          " ".join(f"{k}={v}" for k, v in out.items()), flush=True)


if len(sys.argv) > 1 and sys.argv[1] == "--child":
    child()
    sys.exit(0)
n = int(sys.argv[1])
recs = collections.Counter()
cpus = collections.defaultdict(collections.Counter)
print("OPENBLAS_CORETYPE =", os.environ.get("OPENBLAS_CORETYPE"), flush=True)
for i in range(n):
    res = subprocess.run([sys.executable, __file__, "--child"], capture_output=True, text=True)
    line = next((s for s in res.stdout.splitlines() if s.startswith("REC")), None)
    if line is None:
        print(i, "ERROR", res.stderr[-400:], flush=True)
        continue
    key = line.split("|", 1)[1]
    recs[key] += 1
    cpus[key][line.split()[1]] += 1
    if i < 2:
        print(i, line, flush=True)
print("\nSUMMARY distinct records (ignoring CPU number):", len(recs))
for rec, c in recs.items():
    print(f"  {c:3d} {rec}")
    print(f"      CPU number at load: {dict(cpus[rec])}")
