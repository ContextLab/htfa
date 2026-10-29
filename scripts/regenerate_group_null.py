"""Recompute tutorial 13 permutations against an existing fit (no upload).

Usage: python scripts/regenerate_group_null.py FIT OUTPUT [--permutations 100]
       [--n-jobs 6] [--n-threads 2]
Run from a committed source tree. The fit is preserved; the output records its
SHA-256, the numerical source commit, complete refit parameters and reference.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import warnings
from pathlib import Path

import numpy as np

import htfa
from htfa.groups import group_tree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=6)
    parser.add_argument("--n-threads", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "htfa", "scripts/regenerate_group_null.py"],
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError("Commit the numerical source and this script before running")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    fit_sha = hashlib.sha256(args.fit.read_bytes()).hexdigest()
    with np.load(args.fit, allow_pickle=False) as saved:
        fit = dict(saved)
    both = htfa.datasets.fetch_development_fmri(n_subjects=13, adults_only=False)
    children = [
        i for i, age in enumerate(both.phenotypic["Child_Adult"]) if age == "child"
    ][:10]
    adults = htfa.datasets.fetch_development_fmri(n_subjects=10)
    xs = [both.data[i] for i in children] + list(adults.data)
    rs = [both.coords[i] for i in children] + list(adults.coords)
    ids = [both.subject_ids[i] for i in children] + list(adults.subject_ids)
    labels = ["child"] * len(children) + ["adult"] * len(adults.data)
    if ids != list(fit["subject_ids"]) or labels != list(fit["labels"]):
        raise ValueError("The downloaded participants do not match the saved fit")
    if [x.shape[0] for x in xs] != list(fit["n_voxels"]):
        raise ValueError("The preprocessing mask does not match the saved fit")
    params = json.loads(str(fit["settings"]))
    params.update(n_jobs=args.n_jobs, n_threads=args.n_threads)
    model = htfa.HTFA(**params)
    model.free_fits_ = list(zip(fit["free_centers"], fit["free_widths"], strict=True))
    model.global_centers_ = fit["global_centers"]
    model.group_tree_ = group_tree(labels, len(ids), ids)
    print("Fit SHA-256:", fit_sha, "source:", commit, flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test = htfa.group_permutation_test(
            model,
            xs,
            rs,
            n_permutations=args.permutations,
            random_state=0,
            verbose=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        **test.to_arrays(),
        n_permutations=args.permutations,
        fit_sha256=fit_sha,
        fit_filename=args.fit.name,
        fit_source_commit=fit["htfa_commit"],
        refit_settings=json.dumps(model.get_params(), sort_keys=True),
        htfa_commit=commit,
        htfa_version=htfa.__version__,
        alignment="first-subject stage-0 anchor; one common display permutation",
        machine=f"{platform.machine()} {platform.system()} {os.cpu_count()} CPUs",
    )
    print(args.output, hashlib.sha256(args.output.read_bytes()).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
