"""Regenerate tutorials 09--11 fits locally, retaining source provenance.

Run from a clean source checkout: python scripts/regenerate_tutorial_fits.py OUT.
No uploads are performed. Existing output files are never overwritten.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

import htfa


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(
        ["git", "status", "--porcelain", "htfa"], text=True
    ).strip():
        raise RuntimeError("Commit library changes before generating fits")
    machine = f"{platform.system()} {platform.machine()} {os.cpu_count()} CPUs"
    # Use the tutorial's exact raw-data loader, without its plotting/analysis cells.
    notebook = json.loads(
        Path("docs/tutorials/09_word_decoding_mitchell2008.ipynb").read_text()
    )
    namespace = {}
    for index in (3, 5):
        exec(
            compile(
                "".join(notebook["cells"][index]["source"]),
                f"tutorial09-cell{index}",
                "exec",
            ),
            namespace,
        )
    subjects = namespace["subjects"]
    cases = [
        (
            "mitchell2008_K100",
            [s["X"] for s in subjects],
            [s["coords"] for s in subjects],
            [f"P{i}" for i in range(1, 10)],
            dict(
                K=100,
                max_voxels=10000,
                init="ncorr",
                backend="numpy",
                n_jobs=3,
                random_state=0,
            ),
        )
    ]
    for count, k, extra in [
        (12, 50, dict(max_voxels=3000, max_global_iter=10)),
        (
            6,
            30,
            dict(max_global_iter=6, max_local_iter=5, max_voxels=5000, max_trs=168),
        ),
    ]:
        data = htfa.datasets.fetch_development_fmri(n_subjects=count)
        cases.append(
            (
                f"development_fmri_{count}adults_K{k}",
                data.data,
                data.coords,
                data.subject_ids,
                dict(K=k, init="ncorr", random_state=0, n_jobs=os.cpu_count(), **extra),
            )
        )
    for name, xs, rs, ids, settings in cases:
        target = out / f"htfa_{name}.npz"
        if target.exists():
            raise FileExistsError(target)
        print("Fitting", name, flush=True)
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = htfa.HTFA(**settings).fit(xs, rs)
        seconds = time.perf_counter() - start
        np.savez_compressed(
            target,
            centers=np.stack(model.centers_),
            widths=np.stack(model.widths_),
            weights=np.stack(model.weights_),
            global_centers=model.global_centers_,
            global_widths=model.global_widths_,
            n_iter=model.n_iter_,
            converged=model.converged_,
            converged_reason=model.converged_reason_,
            settings=json.dumps(settings),
            subject_ids=ids,
            n_voxels=[len(x) for x in xs],
            fit_seconds=seconds,
            warnings=[f"{w.category.__name__}: {w.message}" for w in caught],
            htfa_commit=commit,
            htfa_version=htfa.__version__,
            machine=machine,
        )
        print(
            name, seconds, hashlib.sha256(target.read_bytes()).hexdigest(), flush=True
        )


if __name__ == "__main__":
    main()
