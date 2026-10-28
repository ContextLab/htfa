"""Repeated-run check of n_jobs independence in fresh processes.

Driver: runs N fresh processes; each fits HTFA on the data of
tests/test_htfa.py::test_htfa_bitwise_independent_of_n_jobs with n_jobs=1 and
n_jobs=n in both modes and prints whether serial and parallel agree bit for
bit, with the hashes. Exits 1 if any process disagrees or fails.
usage: python njobs_loop.py N
"""

import collections
import hashlib
import os
import platform
import subprocess
import sys
import warnings


def child():
    import numpy as np

    from htfa.htfa import HTFA
    from htfa.synthetic import make_htfa_data

    warnings.simplefilter("ignore")
    n = max(2, (os.cpu_count() or 1) // 2 + 1)
    Xs, Rs, _ = make_htfa_data(
        n_subjects=n, grid_shape=(20, 20, 20), n_samples=60, K=20, random_state=0
    )

    def h(m):
        d = hashlib.sha1()
        for a in [
            m.global_centers_,
            m.global_widths_,
            *m.centers_,
            *m.widths_,
            *m.weights_,
        ]:
            d.update(np.ascontiguousarray(a).tobytes())
        return d.hexdigest()[:10]

    for mode in ("fast", "exact"):
        kw = {
            "K": 36,
            "init": "ncorr",
            "max_global_iter": 2,
            "mode": mode,
            "random_state": 0,
        }
        a = h(HTFA(n_jobs=1, **kw).fit(Xs, Rs))
        b = h(HTFA(n_jobs=n, **kw).fit(Xs, Rs))
        print("RESULT", mode, a, b, "EQUAL" if a == b else "DIFFER", flush=True)


def cpu_name():
    if platform.system() == "Windows":
        cmd = ["powershell", "-NoProfile", "(Get-CimInstance Win32_Processor).Name"]
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    if platform.system() == "Linux":
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


if len(sys.argv) > 1 and sys.argv[1] == "--child":
    child()
    sys.exit(0)
N = int(sys.argv[1])
print("CPU:", cpu_name(), flush=True)
counts = collections.Counter()
hashes = collections.defaultdict(collections.Counter)
bad = 0
for i in range(N):
    out = subprocess.run(
        [sys.executable, __file__, "--child"], capture_output=True, text=True
    )
    lines = [s for s in out.stdout.splitlines() if s.startswith("RESULT")]
    if len(lines) != 2:
        bad += 1
        print(i, "ERROR", out.stderr[-600:], flush=True)
        continue
    for line in lines:
        _, mode, a, b, verdict = line.split()
        counts[(mode, verdict)] += 1
        hashes[mode][a] += 1
        if verdict != "EQUAL":
            bad += 1
            print(i, line, flush=True)
print("SUMMARY processes", N, dict(counts), "bad", bad)
for mode, c in hashes.items():
    print(f"  serial hashes, {mode}: {dict(c)}")
sys.exit(1 if bad else 0)
