"""Time TFA on one synthetic subject of a real workload's shape, by component.

Synthesizes one subject with `htfa.make_tfa_data` (default 69,585 voxels x 1137
samples, K=100: the shape of the 2014 DFFR pilot), runs ``--max-iter`` outer
iterations of `TFA` from one start and prints the wall time of the start, the
mean per-iteration time split into the center/width solve, the ridge weights,
the factor evaluation and the rest (subsampling, matching), the final full-data
objective, the peak resident memory, and the projection
``n_subjects * (start + assume_iter * per-iteration + final)``::

    # the pilot: subsampled, linear loss, 6 subjects in parallel (2 threads each)
    .venv/bin/python scripts/workload_projection.py --max-voxels 3000 \\
        --max-trs 300 --loss linear --init ncorr --threads 2 --max-iter 15
    # full data, 2 iterations, scipy's trf for the "before" numbers
    .venv/bin/python scripts/workload_projection.py --loss linear --max-iter 2 \\
        --method trf

``--threads`` caps BLAS (threadpoolctl) and htfa's engine threads, as `HTFA` does
per parallel subject worker.
"""

import argparse
import resource
import subprocess
import sys
import time
import warnings
from collections import defaultdict

from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

import htfa.tfa as tfa_module
from htfa._engines import thread_limit
from htfa.synthetic import make_tfa_data
from htfa.tfa import TFA

# One-start fits of synthetic data took 13-28 outer iterations
# (scripts/compare_solvers.py, init 'ncorr').
ASSUME_ITER = 20


def _uptime() -> str:
    return subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grid", default="57,37,33", help="voxel grid (V = product)")
    parser.add_argument("--T", type=int, default=1137)
    parser.add_argument("--K", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=2)
    parser.add_argument("--max-voxels", type=int, default=None)
    parser.add_argument("--max-trs", type=int, default=None)
    parser.add_argument("--loss", default="soft_l1")
    parser.add_argument("--method", default="bounded-lm")
    parser.add_argument("--backend", default="numpy")
    parser.add_argument("--init", default="ncorr")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--n-subjects", type=int, default=23)
    parser.add_argument("--assume-iter", type=int, default=ASSUME_ITER)
    args = parser.parse_args()
    grid = tuple(int(g) for g in args.grid.split(","))
    print(f"# before: {_uptime()}", flush=True)
    X, R, _ = make_tfa_data(grid, args.T, K=args.K, min_separation=0.75, random_state=0)
    print(
        f"# V={X.shape[0]} T={X.shape[1]} K={args.K} loss={args.loss} "
        f"method={args.method} max_voxels={args.max_voxels} max_trs={args.max_trs} "
        f"threads={args.threads}",
        flush=True,
    )

    calls: dict[str, list[tuple[float, float]]] = defaultdict(list)

    def timed(name, fn):
        def wrapper(*a, **kw):
            t = time.perf_counter()
            out = fn(*a, **kw)
            calls[name].append((t, time.perf_counter()))
            return out

        return wrapper

    for name in ("solve_centers_widths", "ridge_weights", "factors", "_initial_starts"):
        setattr(tfa_module, name, timed(name, getattr(tfa_module, name)))

    model = TFA(
        K=args.K,
        max_iter=args.max_iter,
        init=args.init,
        nlss_method=args.method,
        nlss_loss=args.loss,
        max_voxels=args.max_voxels,
        max_trs=args.max_trs,
        backend=args.backend,
        random_state=0,
    )
    n_threads = args.threads
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        if n_threads is None:
            model.fit(X, R)
        else:
            with threadpool_limits(n_threads), thread_limit(n_threads):
                model.fit(X, R)
    t_end = time.perf_counter()
    solves = calls["solve_centers_widths"]
    first, last = calls["_initial_starts"][0][1], solves[-1][1]
    n_iter = len(solves)
    per = {
        name: sum(b - a for a, b in calls[name] if first <= a < last) / n_iter
        for name in ("solve_centers_widths", "ridge_weights", "factors")
    }
    iter_total = (last - first) / n_iter
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
    if sys.platform.startswith("linux"):  # kilobytes there, bytes on macOS
        peak *= 1024
    start, final = first - t0, t_end - last
    print(
        f"engine {model.engine_}; {n_iter} iterations; start {start:.1f} s; per "
        f"iteration {iter_total:.2f} s = solve {per['solve_centers_widths']:.2f} + "
        f"weights {per['ridge_weights']:.2f} + factors {per['factors']:.2f} + other "
        f"{iter_total - sum(per.values()):.2f}; final objective {final:.1f} s; "
        f"peak RSS {peak:.2f} GB"
    )
    projected = args.n_subjects * (start + args.assume_iter * iter_total + final)
    print(
        f"projection: {args.n_subjects} subjects x {args.assume_iter} iterations, "
        f"one start, sequential: {projected / 3600:.2f} h"
    )
    print(f"# after: {_uptime()}")


if __name__ == "__main__":
    main()
