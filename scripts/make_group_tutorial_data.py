"""Make the saved fit and permutation null of tutorial 13 (groups).

Fits HTFA with ``groups`` to ten children and ten adults of development_fmri
and runs `htfa.group_permutation_test` on the fit, then writes two ``.npz``
files (no pickled objects) with their provenance:

* ``htfa_development_fmri_10children_10adults_K30.npz``: the fit;
* ``htfa_development_fmri_children_adults_null_100.npz``: the test.

Usage: ``python scripts/make_group_tutorial_data.py OUT_DIR [N_PERMUTATIONS]``.
Run it from a clean checkout: the files record ``git rev-parse HEAD``.
"""

from __future__ import annotations

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
from htfa.datasets import fetch_development_fmri

SETTINGS = dict(
    K=30,
    template_update="fixed",
    max_global_iter=10,
    max_voxels=3000,
    init="ncorr",
    random_state=0,
)
N_CHILDREN = N_ADULTS = 10
PERMUTATION_SEED = 0


def load() -> tuple[list, list, list[str], list[str]]:
    """Ten children (the first of nilearn's selection) then ten adults."""
    both = fetch_development_fmri(n_subjects=13, adults_only=False)
    kids = [i for i, a in enumerate(both.phenotypic["Child_Adult"]) if a == "child"][
        :N_CHILDREN
    ]
    adults = fetch_development_fmri(n_subjects=N_ADULTS)
    Xs = [both.data[i] for i in kids] + list(adults.data)
    Rs = [both.coords[i] for i in kids] + list(adults.coords)
    ids = [both.subject_ids[i] for i in kids] + list(adults.subject_ids)
    labels = ["child"] * len(kids) + ["adult"] * len(adults.data)
    return Xs, Rs, ids, labels


def provenance() -> dict[str, np.ndarray]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "htfa"], capture_output=True, text=True
    ).stdout.strip()
    if dirty:
        raise SystemExit("htfa/ has uncommitted changes; commit before running")
    machine = f"{platform.machine()} {platform.system()} {os.cpu_count()} CPUs"
    return {
        "htfa_commit": np.array(commit),
        "htfa_version": np.array(htfa.__version__),
        "machine": np.array(machine),
    }


def main() -> None:
    out = Path(sys.argv[1])
    n_permutations = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    out.mkdir(parents=True, exist_ok=True)
    prov = provenance()
    Xs, Rs, ids, labels = load()
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = htfa.HTFA(**SETTINGS, n_jobs=os.cpu_count()).fit(
            Xs, Rs, groups=labels, subject_ids=ids
        )
    fit_seconds = time.perf_counter() - t0
    print(f"fit: {fit_seconds:.0f} s, {model.n_iter_} iterations "
          f"({model.converged_reason_})", flush=True)  # fmt: skip
    data = (
        "htfa.datasets.fetch_development_fmri(n_subjects=13, adults_only=False): "
        "its first 10 children; then fetch_development_fmri(n_subjects=10): the "
        "adults"
    )
    fit_file = out / "htfa_development_fmri_10children_10adults_K30.npz"
    np.savez(
        fit_file,
        weights=np.stack(model.weights_),
        centers=np.stack(model.centers_),
        widths=np.stack(model.widths_),
        free_centers=np.stack([c for c, _ in model.free_fits_]),
        free_widths=np.stack([w for _, w in model.free_fits_]),
        global_centers=model.global_centers_,
        global_widths=model.global_widths_,
        global_centers_cov=model.global_centers_cov_,
        global_widths_var=model.global_widths_var_,
        group_keys=np.array(model.group_keys_),
        group_centers=model.group_centers_,
        group_widths=model.group_widths_,
        group_centers_cov=model.group_centers_cov_,
        group_widths_var=model.group_widths_var_,
        prior_centers_cov=model.prior_centers_cov_,
        prior_widths_var=np.array(model.prior_widths_var_),
        labels=np.array(labels),
        subject_ids=np.array(ids),
        n_voxels=np.array([X.shape[0] for X in Xs]),
        n_iter=np.array(model.n_iter_),
        converged=np.array(model.converged_),
        converged_reason=np.array(model.converged_reason_),
        settings=np.array(json.dumps(SETTINGS)),
        data=np.array(data),
        warnings=np.array([f"{w.category.__name__}: {w.message}" for w in caught]),
        fit_seconds=np.array(fit_seconds),
        **prov,
    )
    fit_sha = hashlib.sha256(fit_file.read_bytes()).hexdigest()
    print(fit_file, fit_sha, flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test = htfa.group_permutation_test(
            model, Xs, Rs, n_permutations=n_permutations,
            random_state=PERMUTATION_SEED, verbose=True,
        )  # fmt: skip
    null_file = out / f"htfa_development_fmri_children_adults_null_{n_permutations}.npz"
    np.savez(
        null_file,
        **test.to_arrays(),
        n_permutations=np.array(n_permutations),
        fit_sha256=np.array(fit_sha),
        **prov,
    )
    print(null_file, hashlib.sha256(null_file.read_bytes()).hexdigest())
    for stat in ("center", "width", "connectivity"):
        p = test.pvalues(stat)
        q = test.pvalues(stat, fwe=True)
        print(f"{stat}: {(p <= 0.05).sum()} factors p <= 0.05, "
              f"{(q <= 0.05).sum()} family-wise; min p {p.min():.3f}")  # fmt: skip


if __name__ == "__main__":
    main()
