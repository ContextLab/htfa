"""Which side of the n_jobs test varies between fresh processes?

Runs ROUNDS rounds; each round starts one fresh process per mode, which fits
HTFA as tests/test_htfa.py::test_htfa_bitwise_independent_of_n_jobs does and
prints per-subject and template hashes. Modes:
  serial      HTFA(n_jobs=1)                 (engine chunk threads = cpu_count)
  serial_nt1  HTFA(n_jobs=1, n_threads=1)    (no concurrent BLAS callers)
  parallel    HTFA(n_jobs=n)
  parallel_nt4 HTFA(n_jobs=n, n_threads=4)   (concurrent BLAS callers in workers)
usage: python njobs_modes.py ROUNDS            (driver)
       python njobs_modes.py --mode MODE       (one fit)
"""

import collections
import hashlib
import os
import subprocess
import sys
import warnings


def one(mode):
    import numpy as np

    from htfa.htfa import HTFA
    from htfa.synthetic import make_htfa_data

    warnings.simplefilter("ignore")
    n = max(2, (os.cpu_count() or 1) // 2 + 1)
    Xs, Rs, _ = make_htfa_data(
        n_subjects=n, grid_shape=(20, 20, 20), n_samples=60, K=20, random_state=0
    )
    kw = {"K": 36, "init": "ncorr", "max_global_iter": 2, "random_state": 0}
    extra = {
        "serial": {"n_jobs": 1},
        "serial_nt1": {"n_jobs": 1, "n_threads": 1},
        "parallel": {"n_jobs": n},
        "parallel_nt4": {"n_jobs": n, "n_threads": 4},
        "serial_env1": {"n_jobs": 1, "n_threads": 1},
    }[mode]
    m = HTFA(**kw, **extra).fit(Xs, Rs)

    def h(a):
        return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:8]

    print("RESULT", mode, h(m.global_centers_), *[h(c) for c in m.centers_], flush=True)


if len(sys.argv) > 2 and sys.argv[1] == "--mode":
    one(sys.argv[2])
    sys.exit(0)
rounds = int(sys.argv[1])
modes = sys.argv[2].split(",") if len(sys.argv) > 2 else [
    "serial", "serial_nt1", "parallel", "parallel_nt4"]
seen = collections.defaultdict(collections.Counter)
for r in range(rounds):
    for mode in modes:
        env = dict(os.environ)
        if mode == "serial_env1":  # OpenBLAS loaded with one thread, as in a worker
            env["OPENBLAS_NUM_THREADS"] = env["OMP_NUM_THREADS"] = "1"
        out = subprocess.run(
            [sys.executable, __file__, "--mode", mode],
            capture_output=True, text=True, env=env,
        )
        line = next((s for s in out.stdout.splitlines() if s.startswith("RESULT")), None)
        if line is None:
            print(mode, "ERROR", out.stderr[-500:], flush=True)
            continue
        key = " ".join(line.split()[2:])
        seen[mode][key] += 1
        print(f"round {r} {line}", flush=True)
print("\nSUMMARY (distinct results per mode, with counts):")
allkeys = collections.Counter()
for mode in modes:
    print(f"  {mode}: {dict(seen[mode])}")
    allkeys.update(seen[mode])
print(f"  distinct over all modes: {len(allkeys)}")
