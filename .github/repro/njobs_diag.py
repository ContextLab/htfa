"""Diagnose n_jobs-dependent HTFA results (CI, OpenBLAS).

Same data and settings as tests/test_htfa.py::test_htfa_bitwise_independent_of_n_jobs.
Reports:
  1. BLAS libraries / threads in the main process and in loky workers.
  2. _fit_subject of each subject repeated in the main process with the memory
     layout scrambled (small and large live arrays) -> distinct results?
  3. The same fits in loky workers (fresh and reused) -> distinct results?
  4. Full HTFA n_jobs=1 vs n_jobs=n, REPS times.
usage: python njobs_diag.py REPS
"""

import hashlib
import os
import platform
import sys
import warnings

import numpy as np
from joblib import Parallel, delayed
from threadpoolctl import threadpool_info

from htfa.htfa import HTFA, _fit_subject
from htfa.synthetic import make_htfa_data

warnings.simplefilter("ignore")
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
K = 36


def h(*arrs):
    m = hashlib.sha1()
    for a in arrs:
        m.update(np.ascontiguousarray(a).tobytes())
    return m.hexdigest()[:10]


def pools():
    keys = ("user_api", "internal_api", "num_threads", "version", "architecture")
    return [
        {k: p.get(k) for k in keys} | {"lib": os.path.basename(p["filepath"])}
        for p in threadpool_info()
    ]


def worker_info(_):
    env = {v: os.environ.get(v) for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")}
    return os.getpid(), env, pools()


def scramble(seed):
    rng = np.random.default_rng(seed)
    small = [np.empty(int(n)) for n in rng.integers(1, 2000, size=int(rng.integers(50, 400)))]
    large = [np.empty(int(n)) for n in rng.integers(2000, 200000, size=int(rng.integers(1, 30)))]
    return small + large


def fit_hash(params, X, R, s, seed=None):
    junk = scramble(seed) if seed is not None else None
    f = _fit_subject(params, X, R, None, None, 7 + s, 1, True)
    del junk
    return os.getpid(), h(f.centers_, f.widths_)


print(platform.platform(), platform.machine(), "python", sys.version.split()[0],
      "numpy", np.__version__, "cpu_count", os.cpu_count(), flush=True)
n = max(2, (os.cpu_count() or 1) // 2 + 1)
Xs, Rs, _ = make_htfa_data(n_subjects=n, grid_shape=(20, 20, 20), n_samples=60, K=20,
                           random_state=0)
print("main pools:", pools(), flush=True)
for info in Parallel(n_jobs=n, backend="loky")(delayed(worker_info)(i) for i in range(n)):
    print("worker:", info, flush=True)

params = HTFA(K=K, init="ncorr")._tfa_params(10)
print("\n== 2. subject fits in the main process, scrambled layouts ==", flush=True)
main = {s: {fit_hash(params, Xs[s], Rs[s], s, seed)[1] for seed in range(6)} for s in range(n)}
print({s: sorted(v) for s, v in main.items()}, flush=True)

print("\n== 3. subject fits in loky workers ==", flush=True)
work = {s: set() for s in range(n)}
for rep in range(4):
    out = Parallel(n_jobs=n, backend="loky")(
        delayed(fit_hash)(params, Xs[s], Rs[s], s, rep if rep % 2 else None)
        for s in range(n)
    )
    for s, (pid, hh) in enumerate(out):
        work[s].add(hh)
        print(f"rep {rep} subject {s} pid {pid} {hh}", flush=True)
for s in range(n):
    tag = "SAME" if work[s] == main[s] and len(main[s]) == 1 else "DIFF"
    print(f"subject {s}: main {sorted(main[s])} workers {sorted(work[s])} {tag}", flush=True)

print("\n== 4. HTFA n_jobs=1 vs n_jobs=n ==", flush=True)
kw = {"K": K, "init": "ncorr", "max_global_iter": 2, "random_state": 0}
ser, par = set(), set()
for rep in range(REPS):
    junk = scramble(100 + rep)
    a = HTFA(n_jobs=1, **kw).fit(Xs, Rs)
    b = HTFA(n_jobs=n, **kw).fit(Xs, Rs)
    del junk
    ha = (h(a.global_centers_), tuple(h(c) for c in a.centers_))
    hb = (h(b.global_centers_), tuple(h(c) for c in b.centers_))
    ser.add(ha)
    par.add(hb)
    print(f"rep {rep}: serial {ha} parallel {hb} {'EQUAL' if ha == hb else 'DIFFER'}",
          flush=True)
print(f"distinct serial {len(ser)}, distinct parallel {len(par)}", flush=True)
