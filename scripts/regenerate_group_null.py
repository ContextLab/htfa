"""Recompute tutorial 13 permutations against an existing fit (no upload).

Usage: python scripts/regenerate_group_null.py FIT OUTPUT [--permutations 100]
       [--permutation-jobs 4] [--n-jobs 3] [--n-threads 1]
       python scripts/regenerate_group_null.py --self-test
Run from a committed source tree. The fit is preserved; the output records its
SHA-256, the numerical source commit, complete refit parameters and reference.
"""

import argparse
import hashlib
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
from joblib import Parallel, delayed, parallel_config
from sklearn.base import clone

import htfa
from htfa._matching import match_factors
from htfa.groups import (
    GroupTest,
    _permutation_reference,
    _permuted_labels,
    _statistics,
    group_key,
    group_tree,
)


def array_digest(arrays):
    """Hash numerical values, shapes, and dtypes without a large byte copy."""
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(json.dumps([array.shape, array.dtype.str]).encode())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def source_digest():
    """Identify all numerical source files and this runner for resume checks."""
    digest = hashlib.sha256()
    root = Path(htfa.__file__).parent
    for path in [*sorted(root.glob("*.py")), Path(__file__)]:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def checkpoint_refit(model, xs, rs, labels, level, reference, order, path, signature):
    """One actual fit, or an exact-input validated checkpoint."""
    label_json = json.dumps(labels)
    if path.exists():
        with np.load(path, allow_pickle=False) as cached:
            if (
                str(cached["signature"]) != signature
                or str(cached["labels"]) != label_json
            ):
                raise ValueError(f"Checkpoint inputs differ: {path}")
            statistics = cached["statistics"]
            if statistics.shape != (3, model.K):
                raise ValueError(f"Checkpoint statistic shape differs: {path}")
            return statistics
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = clone(model).fit(xs, rs, starts=model.free_fits_, groups=labels)
    permutation = match_factors(reference, fitted.global_centers_)[order]
    statistics = np.asarray(_statistics(fitted, level, permutation)[1:])
    fd, temporary = tempfile.mkstemp(prefix=path.stem, suffix=".part", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(
                stream,
                statistics=statistics,
                signature=signature,
                labels=label_json,
                seconds=time.perf_counter() - started,
            )
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(f"Saved {path.name}: {time.perf_counter() - started:.1f}s", flush=True)
    return statistics


def checkpointed_group_test(
    model,
    xs,
    rs,
    directory,
    *,
    n_permutations=100,
    permutation_jobs=4,
    provenance=None,
    random_state=0,
    level=0,
):
    """Match the public test with a fixed label stream and parallel cached refits."""
    if n_permutations < 1 or permutation_jobs < 1:
        raise ValueError("permutations and permutation_jobs must be positive")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    labels = model.group_tree_.labels
    reference, order = _permutation_reference(
        model.free_fits_[0], model.global_centers_
    )
    rng = np.random.default_rng(random_state)
    # Generate every permutation before dispatch: worker completion order never
    # affects the RNG stream or the order of rows in the returned null arrays.
    shuffled = [_permuted_labels(labels, level, rng) for _ in range(n_permutations)]
    metadata = {
        "provenance": provenance or {},
        "source_sha256": source_digest(),
        "refit_settings": model.get_params(),
        "data_sha256": array_digest(xs),
        "coords_sha256": array_digest(rs),
        "fit_state_sha256": array_digest(
            [
                *(array for fit in model.free_fits_ for array in fit),
                model.global_centers_,
            ]
        ),
        "labels": labels,
        "level": level,
        "random_state": random_state,
        "n_permutations": n_permutations,
    }
    serialized = json.dumps(metadata, sort_keys=True, indent=2) + "\n"
    metadata_path = directory / "inputs.json"
    if metadata_path.exists() and metadata_path.read_text() != serialized:
        raise ValueError(f"Checkpoint input provenance differs: {metadata_path}")
    metadata_path.write_text(serialized)
    signature = hashlib.sha256(serialized.encode()).hexdigest()
    started = time.perf_counter()
    observed = checkpoint_refit(
        model,
        xs,
        rs,
        labels,
        level,
        reference,
        order,
        directory / "observed.npz",
        signature,
    )
    with parallel_config(backend="loky", inner_max_num_threads=1):
        nulls = np.asarray(
            Parallel(n_jobs=permutation_jobs, pre_dispatch=permutation_jobs)(
                delayed(checkpoint_refit)(
                    model,
                    xs,
                    rs,
                    permuted,
                    level,
                    reference,
                    order,
                    directory / f"permutation-{i:04d}.npz",
                    signature,
                )
                for i, permuted in enumerate(shuffled)
            )
        )
    return GroupTest(
        level=level,
        groups=[
            group_key(path)
            for path in model.group_tree_.paths[1:]
            if len(path) == level + 1
        ],
        center=observed[0],
        width=observed[1],
        connectivity=observed[2],
        center_null=nulls[:, 0],
        width_null=nulls[:, 1],
        connectivity_null=nulls[:, 2],
        permuted_labels=np.asarray([[str(v) for v in p[level]] for p in shuffled]),
        random_state=random_state,
        seconds=time.perf_counter() - started,
        reference_centers=reference,
        reference_indices=order,
    )


def self_test():
    """Tiny real-fit equivalence and checkpoint resume/invalidation regression."""
    xs, rs, _ = htfa.make_htfa_data(4, (6, 6), 12, K=2, random_state=12)
    model = htfa.HTFA(
        K=2,
        template_update="fixed",
        max_global_iter=1,
        max_local_iter=2,
        template_max_iter=2,
        init="power",
        standardize=False,
        random_state=0,
        n_jobs=1,
        n_threads=1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(xs, rs, groups=["a", "a", "b", "b"])
        expected = htfa.group_permutation_test(
            model, xs, rs, n_permutations=3, random_state=0
        )
    with tempfile.TemporaryDirectory(prefix="htfa-group-null-test-") as temporary:
        path = Path(temporary)
        result = checkpointed_group_test(
            model, xs, rs, path, n_permutations=3, permutation_jobs=2
        )
        before = {file.name: file.stat().st_mtime_ns for file in path.glob("*.npz")}
        resumed = checkpointed_group_test(
            model, xs, rs, path, n_permutations=3, permutation_jobs=1
        )
        assert before == {
            file.name: file.stat().st_mtime_ns for file in path.glob("*.npz")
        }
        for key, value in expected.to_arrays().items():
            if key != "seconds":
                np.testing.assert_array_equal(result.to_arrays()[key], value)
                np.testing.assert_array_equal(resumed.to_arrays()[key], value)
        changed = [x.copy() for x in xs]
        changed[0][0, 0] += 0.1
        try:
            checkpointed_group_test(model, changed, rs, path, n_permutations=3)
        except ValueError as exc:
            assert "provenance differs" in str(exc)
        else:
            raise AssertionError("Changed input data reused cached statistics")
    print("Parallel/public exact-array equivalence, resume, and invalidation passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--permutation-jobs", type=int, default=4)
    parser.add_argument("--n-jobs", type=int, default=3)
    parser.add_argument("--n-threads", type=int, default=1)
    parser.add_argument("--checkpoint-dir", type=Path)
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
    checkpoint_dir = args.checkpoint_dir or args.output.with_suffix(".checkpoints")
    test = checkpointed_group_test(
        model,
        xs,
        rs,
        checkpoint_dir,
        n_permutations=args.permutations,
        permutation_jobs=args.permutation_jobs,
        provenance={"fit_sha256": fit_sha, "htfa_commit": commit},
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
        permutation_jobs=args.permutation_jobs,
        source_sha256=source_digest(),
        checkpoint_inputs_sha256=hashlib.sha256(
            (checkpoint_dir / "inputs.json").read_bytes()
        ).hexdigest(),
        timing_kind=(
            "wall time of this invocation; resumed checkpoints exclude previous work"
        ),
        machine=f"{platform.machine()} {platform.system()} {os.cpu_count()} CPUs",
    )
    print(args.output, hashlib.sha256(args.output.read_bytes()).hexdigest(), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        self_test()
    else:
        main()
