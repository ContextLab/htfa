"""Benchmark TFA fits per compute backend, and against BrainIAK.

Prints a markdown table of the wall time of one ``TFA.fit`` on all voxels and
samples (TFA defaults, ``init='auto'`` and ``max_iter=100``, unless ``--init`` or
``--max-iter`` say otherwise) and its largest center error on synthetic data from
`htfa.make_tfa_data`::

    .venv/bin/python scripts/benchmark.py --sizes small,medium --backends numpy,mps

Backends ``numpy``, ``mps`` and ``cuda`` are htfa engines (a GPU row is skipped
when the device is missing). ``brainiak`` fits BrainIAK 0.12's TFA on all voxels
and samples, only for the small and medium sizes; it runs in this interpreter when
brainiak is importable here, and otherwise in ``--brainiak-python`` (default:
``.venv-brainiak/bin/python`` next to this repository) when that exists.
"""

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from htfa._engines import select_engine
from htfa.synthetic import make_tfa_data, match_factors
from htfa.tfa import TFA

# name -> (grid shape, T, K, min_separation). V = prod(grid). Fewer mean sigmas of
# separation for larger K, which could not be placed otherwise.
SIZES = {
    "small": ((25, 20, 10), 200, 20, 2.0),
    "medium": ((40, 25, 20), 300, 100, 1.0),
    "large": ((50, 40, 25), 1000, 200, 0.75),
}
BACKENDS = ("numpy", "mps", "cuda", "brainiak")
BRAINIAK_SIZES = ("small", "medium")
REPO = Path(__file__).resolve().parent.parent

BRAINIAK_FIT = """
import json, sys, time
import numpy as np
from brainiak.factoranalysis.tfa import TFA
data = np.load(sys.argv[1])
X, R, K = data["X"], data["R"], int(data["K"])
t0 = time.perf_counter()
model = TFA(K=K, max_num_voxel=X.shape[0], max_num_tr=X.shape[1]).fit(X, R)
seconds = time.perf_counter() - t0
centers = model.get_centers(model.local_posterior_)
print(json.dumps({"seconds": seconds, "centers": centers.tolist()}))
"""


def max_center_error(est: np.ndarray, true: np.ndarray) -> float:
    """Largest distance between matched estimated and true centers."""
    perm = match_factors(true, est)
    return float(np.max(np.linalg.norm(est[perm] - true, axis=1)))


def brainiak_python(option: str | None) -> str | None:
    """Interpreter that can import brainiak, or None."""
    try:
        import brainiak.factoranalysis.tfa  # noqa: F401
    except ImportError:
        pass
    else:
        return sys.executable
    path = Path(option) if option else REPO / ".venv-brainiak" / "bin" / "python"
    return str(path) if path.exists() else None


def run_htfa(X, R, K, backend, seed, init, max_iter):
    """One htfa TFA fit; returns (seconds, centers, notes).

    A fit that stops on ``max_iter`` says so in the notes instead of warning.
    """
    model = TFA(
        K=K,
        max_iter=max_iter,
        standardize=False,
        init=init,
        backend=backend,
        random_state=seed,
    )
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X, R)
    seconds = time.perf_counter() - t0
    notes = (
        f"{model.engine_}, init {model.init_}, {model.n_iter_} iter "
        f"({model.converged_reason_})"
    )
    return seconds, model.centers_, notes


def run_brainiak(X, R, K, python, timeout):
    """One BrainIAK TFA fit in ``python``; returns (seconds, centers, notes)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "data.npz")
        np.savez(path, X=X, R=R, K=K)
        try:
            out = subprocess.run(
                [python, "-c", BRAINIAK_FIT, path],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True,
            )
        except subprocess.TimeoutExpired:
            return None, None, f"timed out after {timeout:.0f} s"
        except subprocess.CalledProcessError as err:
            raise RuntimeError(f"BrainIAK fit failed:\n{err.stderr}") from err
    result = json.loads(out.stdout.strip().splitlines()[-1])
    return result["seconds"], np.asarray(result["centers"]), "BrainIAK 0.12"


def available(backend: str) -> bool:
    """Whether an htfa GPU backend has its device here."""
    try:
        select_engine(backend, 1, 1, 1)
    except RuntimeError:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", default="small,medium")
    parser.add_argument("--backends", default="numpy,mps,cuda,brainiak")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init", default="auto", help="htfa TFA init (default 'auto')")
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="htfa TFA max_iter (default 100, the TFA default)",
    )
    parser.add_argument("--brainiak-python", default=None)
    parser.add_argument(
        "--brainiak-timeout",
        type=float,
        default=3600.0,
        help="seconds before a BrainIAK fit is abandoned (default 3600)",
    )
    args = parser.parse_args()
    sizes = args.sizes.split(",")
    backends = args.backends.split(",")
    for name, allowed in (("size", SIZES), ("backend", BACKENDS)):
        chosen = sizes if name == "size" else backends
        bad = [c for c in chosen if c not in allowed]
        if bad:
            parser.error(f"unknown {name}(s) {bad}; choose from {tuple(allowed)}")

    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        torch_version = "not installed"
    print(
        f"# {platform.platform()}, {os.cpu_count()} CPUs, "
        f"Python {platform.python_version()}, NumPy {np.__version__}, "
        f"torch {torch_version}; htfa TFA init={args.init!r}, "
        f"max_iter={args.max_iter}\n"
    )
    rows = []
    for size in sizes:
        shape, T, K, sep = SIZES[size]
        X, R, truth = make_tfa_data(
            shape, T, K=K, min_separation=sep, random_state=args.seed
        )
        V = X.shape[0]
        results = {}
        for backend in backends:
            if backend == "brainiak":
                python = brainiak_python(args.brainiak_python)
                if size not in BRAINIAK_SIZES or python is None:
                    continue
                results[backend] = run_brainiak(X, R, K, python, args.brainiak_timeout)
            elif backend in ("mps", "cuda") and not available(backend):
                continue
            else:
                results[backend] = run_htfa(
                    X, R, K, backend, args.seed, args.init, args.max_iter
                )
            print(f"done: {size} {backend} {results[backend][::2]}", file=sys.stderr)
        baseline = results.get("numpy", (None,))[0]
        for backend, (seconds, centers, notes) in results.items():
            wall = speedup = err = "-"
            if seconds is not None:
                wall = f"{seconds:.1f}"
                if baseline is not None:
                    speedup = f"{baseline / seconds:.2f}"
            if centers is not None:
                err = f"{max_center_error(centers, truth['centers']):.4f}"
            rows.append(
                f"| {size} | {V} | {T} | {K} | {V * T * K:.2e} | {backend} | {wall} "
                f"| {speedup} | {err} | {notes} |"
            )
    print(
        "| size | V | T | K | V*T*K | backend | fit (s) | speedup vs numpy "
        "| max center err | notes |"
    )
    print("|-|-|-|-|-|-|-|-|-|-|")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
