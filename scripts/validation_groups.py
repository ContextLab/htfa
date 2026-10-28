"""Validation of grouped HTFA fits (``HTFA.fit(..., groups=...)``).

Subcommands:

``nogroups OUT.npz``
    Fits a fixed set of synthetic HTFA problems without groups (every
    ``template_update``, both modes, with and without ``starts`` and
    ``init_template``, serial and parallel) and saves every fitted array. Run it
    once with the code before groups existed (``PYTHONPATH=<old checkout>``)
    and once with the current code, then ``compare A.npz B.npz``.
``compare A.npz B.npz``
    Bitwise comparison of two ``nogroups`` outputs.
``fpr``
    False-positive rate of `htfa.group_permutation_test` on synthetic groups
    with no true difference, over seeds; writes
    ``docs/_static/validation/groups_fpr.csv``.
``power``
    Recovery of known group shifts (one factor's center moved, another's width
    scaled) and the test's detection rate, over seeds; writes
    ``docs/_static/validation/groups_power.csv``.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np

import htfa
from htfa.synthetic import make_htfa_data, match_factors

STATIC = Path(__file__).resolve().parent.parent / "docs" / "_static" / "validation"
ALPHA = 0.05


def nogroups(out: str) -> None:
    arrays: dict[str, np.ndarray] = {}
    Xs, Rs, _ = make_htfa_data(4, (10, 10, 10), 30, K=5, random_state=0)
    X2, R2, _ = make_htfa_data(3, (14, 14), 25, K=4, noise=0.5, random_state=1)
    cases = []
    for update in ("brainiak", "fixed", "empirical"):
        for mode in ("fast", "exact"):
            cases.append((f"3d_{update}_{mode}", Xs, Rs, update, mode, {}, 1))
            cases.append((f"2d_{update}_{mode}", X2, R2, update, mode, {}, 1))
    cases.append(("3d_fixed_fast_njobs3", Xs, Rs, "fixed", "fast", {}, 3))
    cases.append(("3d_brainiak_fast_njobs3", Xs, Rs, "brainiak", "fast", {}, 3))
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = None
        for name, X, R, update, mode, extra, n_jobs in cases:
            model = htfa.HTFA(
                K=5 if name.startswith("3d") else 4,
                template_update=update,
                mode=mode,
                max_global_iter=4,
                max_local_iter=5,
                standardize=False,
                n_jobs=n_jobs,
                random_state=0,
            ).fit(X, R, **extra)
            if first is None and name.startswith("3d"):
                first = model
            _store(arrays, name, model)
        # starts and init_template (resume) on the first case
        assert first is not None
        resumed = htfa.HTFA(
            K=5, max_global_iter=3, max_local_iter=5, standardize=False,
            random_state=0,
        ).fit(
            Xs, Rs,
            starts=list(zip(first.centers_, first.widths_, strict=True)),
            init_template=(first.global_centers_, first.global_widths_),
        )  # fmt: skip
        _store(arrays, "3d_resumed", resumed)
    print(f"{len(arrays)} arrays in {time.perf_counter() - t0:.1f} s")
    np.savez(out, **arrays)


def _store(arrays: dict[str, np.ndarray], name: str, model: htfa.HTFA) -> None:
    arrays[f"{name}/global_centers"] = model.global_centers_
    arrays[f"{name}/global_widths"] = model.global_widths_
    arrays[f"{name}/global_centers_cov"] = model.global_centers_cov_
    arrays[f"{name}/global_widths_var"] = model.global_widths_var_
    arrays[f"{name}/centers"] = np.stack(model.centers_)
    arrays[f"{name}/widths"] = np.stack(model.widths_)
    arrays[f"{name}/weights"] = np.stack(model.weights_)
    arrays[f"{name}/objective_history"] = np.array(model.objective_history_)
    arrays[f"{name}/n_iter"] = np.array(model.n_iter_)
    arrays[f"{name}/template_history"] = np.stack(
        [np.column_stack([c, w]) for c, w in model.template_history_]
    )


def compare(a: str, b: str) -> None:
    with np.load(a) as za, np.load(b) as zb:
        assert sorted(za.files) == sorted(zb.files), "different keys"
        differ = [k for k in za.files if za[k].tobytes() != zb[k].tobytes()]
        print(f"{len(za.files)} arrays compared, {len(differ)} differ bitwise")
        for k in differ:
            print("  differs:", k)
    sys.exit(1 if differ else 0)


def _fit_and_test(
    sizes: tuple[int, ...],
    seed: int,
    n_permutations: int,
    shift: float = 0.0,
    scale: float = 1.0,
) -> dict[str, float]:
    K = 5
    center_shift = np.zeros((len(sizes), K, 2))
    width_scale = np.ones((len(sizes), K))
    # toward the middle of the grid (the global centers do not depend on it)
    g = htfa.make_group_data(sizes, (16, 16), 2, K=K, random_state=seed)[3]
    toward = -np.sign(g["global_centers"][0, 0] - 7.5)
    center_shift[1, 0, 0] = toward * shift
    width_scale[1, 1] = scale
    Xs, Rs, groups, truth = htfa.make_group_data(
        sizes, grid_shape=(16, 16), n_samples=40, K=K, center_shift=center_shift,
        width_scale=width_scale, center_jitter=0.5, random_state=seed,
    )  # fmt: skip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = htfa.HTFA(
            K=K, template_update="fixed", standardize=False, random_state=seed
        ).fit(Xs, Rs, groups=groups)
        test = htfa.group_permutation_test(
            model, Xs, Rs, n_permutations=n_permutations, random_state=seed
        )
    perm = match_factors(truth["global_centers"], model.global_centers_)
    row: dict[str, float] = {"seed": seed}
    for stat in ("center", "width", "connectivity"):
        p = test.pvalues(stat)[perm]  # in truth order
        row[f"{stat}_p0"], row[f"{stat}_p1"] = float(p[0]), float(p[1])
        row[f"{stat}_rejections"] = float((p <= ALPHA).sum())
        row[f"{stat}_fwe_rejections"] = float((test.pvalues(stat, True) <= ALPHA).sum())
    est = model.group_centers_[:, perm]
    row["center_shift_est"] = float(toward * (est[1, 0, 0] - est[0, 0, 0]))
    row["width_ratio_est"] = float(
        model.group_widths_[1, perm[1]] / model.group_widths_[0, perm[1]]
    )
    row["seconds"] = test.seconds
    return row


def _run(name: str, n_seeds: int, **kw: float) -> None:
    import pandas as pd

    rows = []
    for seed in range(n_seeds):
        rows.append(_fit_and_test((4, 5), seed, 39, **kw))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    STATIC.mkdir(parents=True, exist_ok=True)
    df.to_csv(STATIC / f"groups_{name}.csv", index=False)
    K = 5
    for stat in ("center", "width", "connectivity"):
        rate = df[f"{stat}_rejections"].sum() / (K * len(df))
        fwe = (df[f"{stat}_fwe_rejections"] > 0).mean()
        print(
            f"{stat}: per-factor rejection rate {rate:.3f} "
            f"({int(df[f'{stat}_rejections'].sum())}/{K * len(df)}), "
            f"family-wise {fwe:.3f}; factor 0 p <= {ALPHA}: "
            f"{(df[f'{stat}_p0'] <= ALPHA).mean():.3f}, factor 1: "
            f"{(df[f'{stat}_p1'] <= ALPHA).mean():.3f}"
        )
    print(
        f"center shift estimate: mean {df['center_shift_est'].mean():.3f} "
        f"(sd {df['center_shift_est'].std():.3f}); width ratio estimate: mean "
        f"{df['width_ratio_est'].mean():.3f} (sd {df['width_ratio_est'].std():.3f}); "
        f"{df['seconds'].mean():.1f} s per test"
    )


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "nogroups":
        nogroups(sys.argv[2])
    elif cmd == "compare":
        compare(sys.argv[2], sys.argv[3])
    elif cmd == "fpr":
        _run("fpr", int(sys.argv[2]) if len(sys.argv) > 2 else 100)
    elif cmd == "power":
        _run(
            "power",
            int(sys.argv[2]) if len(sys.argv) > 2 else 50,
            shift=3.0,
            scale=2.0,
        )
    else:
        raise SystemExit(f"unknown command {cmd!r}")
