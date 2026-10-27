"""``mode='fast'`` against ``mode='exact'`` on real fMRI (see `validation_data`).

For each of the twelve adults, K in (50, 100) and both modes, TFA with its
defaults (``init='auto'``, all voxels; ``standardize=False`` on the cleaned,
z-scored data) is fitted

* to the full run: objective and seconds per fit;
* to the training segment of `validation_heldout` (samples 0-79): held-out R^2
  of the test segment with that script's A/B ridge and OLS scoring.

Then validation_decoding's across-group timepoint decoding is run with each
subject's full-run factors in turn as the shared factor set: every subject's
run is ridge-regressed on them, and the accuracy over all 6/6 splits is
averaged over the twelve factor sets.

Writes ``modes.csv`` (one row per subject, K and mode) and ``modes_<K>_<mode>_
<subject>.npz`` to the output directory and ``docs/_static/validation/
modes.csv`` (summary per K and mode).

``--htfa`` instead fits `htfa.HTFA` (defaults, K = 50, ``n_jobs=6``) to the
twelve adults in both modes and prints the wall time, the summed final local
objective, the distance between the two templates and validation_decoding's
accuracy with the local weights and with ridge weights on the template
factors; writes ``modes_htfa.csv`` to ``docs/_static/validation/``.
"""

import itertools
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from validation_data import OUT, STATIC, adult_ids, clean, load_subject
from validation_decoding import accuracy, windowed, zt
from validation_heldout import heldout_r2, split

from htfa._kernels import factors, ridge_weights
from htfa._matching import match_factors
from htfa.htfa import HTFA
from htfa.tfa import TFA, standardize_data

KS = (50, 100)
MODES = ("exact", "fast")
WINDOWS = (1, 6)
SEED = 0


def fit(X: np.ndarray, R: np.ndarray, K: int, mode: str) -> tuple[TFA, float]:
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = TFA(K=K, standardize=False, mode=mode, random_state=SEED).fit(X, R)
    return model, time.perf_counter() - t0


def decoding_accuracy(features: list) -> dict[int, float]:
    """Mean across-group accuracy per window over all splits, for one feature
    set per subject (each z-scored over time)."""
    n = len(features)
    splits = [g for g in itertools.combinations(range(n), n // 2) if 0 in g]
    out = {}
    for w in WINDOWS:
        acc = []
        for g1 in splits:
            g2 = [s for s in range(n) if s not in g1]
            A = windowed(np.mean([features[s] for s in g1], 0), w)
            B = windowed(np.mean([features[s] for s in g2], 0), w)
            acc.append(accuracy(A, B))
        out[w] = float(np.mean(acc))
    return out


def decoding(runs: list, factor_sets: list) -> dict[int, float]:
    """`decoding_accuracy` with ridge weights on each factor set in turn,
    averaged over the factor sets."""
    per_set = [
        decoding_accuracy([zt(ridge_weights(X, factors(R, C, w))) for X, R in runs])
        for C, w in factor_sets
    ]
    return {w: float(np.mean([a[w] for a in per_set])) for w in WINDOWS}


def htfa_comparison(runs: list) -> None:
    rows, templates = [], {}
    for mode in MODES:
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = HTFA(K=50, mode=mode, n_jobs=6, random_state=SEED).fit(
                [X for X, _ in runs], [R for _, R in runs]
            )
        seconds = time.perf_counter() - t0
        C, w = model.global_centers_, model.global_widths_
        templates[mode] = C
        local = decoding_accuracy([zt(W) for W in model.weights_])
        template = decoding_accuracy(
            [zt(ridge_weights(X, factors(R, C, w))) for X, R in runs]
        )
        row = {
            "mode": mode,
            "seconds": seconds,
            "global_iterations": model.n_iter_,
            "objective": model.objective_history_[-1],
            **{f"local_window_{k}": v for k, v in local.items()},
            **{f"template_window_{k}": v for k, v in template.items()},
        }
        rows.append(row)
        print(row, flush=True)
    perm = match_factors(templates["exact"], templates["fast"])
    dist = np.linalg.norm(templates["fast"][perm] - templates["exact"], axis=1)
    print(
        f"template centers, fast vs exact: median {np.median(dist):.2f} mm, "
        f"max {dist.max():.2f} mm"
    )
    STATIC.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        STATIC / "modes_htfa.csv", index=False, float_format="%.6g"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ids = adult_ids()
    runs = []
    for sid in ids:
        X, R, conf = load_subject(sid)
        runs.append((standardize_data(clean(X, conf), "auto"), R))
    if "--htfa" in sys.argv:
        htfa_comparison(runs)
        return
    rows = []
    for K in KS:
        for sid, (X, R) in zip(ids, runs, strict=True):
            Xtr, Xte, R_tr, A = split(sid)
            lam = float(np.var(Xtr))
            for mode in MODES:
                full, t_full = fit(X, R, K, mode)
                train, t_train = fit(Xtr, R_tr, K, mode)
                F = factors(R_tr, train.centers_, train.widths_)
                row = {
                    "subject": sid,
                    "K": K,
                    "mode": mode,
                    "objective": full.cost_,
                    "seconds": t_full,
                    "init": full.init_,
                    "train_objective": train.cost_,
                    "train_seconds": t_train,
                    "heldout_r2": heldout_r2(F, Xte, A, lam),
                    "heldout_r2_ols": heldout_r2(F, Xte, A, 0.0),
                }
                rows.append(row)
                print(row, flush=True)
                np.savez(
                    OUT / f"modes_{K}_{mode}_{sid}.npz",
                    centers=full.centers_,
                    widths=full.widths_,
                )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "modes.csv", index=False)
    summary = []
    for K in KS:
        for mode in MODES:
            sets = []
            for sid in ids:
                z = np.load(OUT / f"modes_{K}_{mode}_{sid}.npz")
                sets.append((z["centers"], z["widths"]))
            acc = decoding(runs, sets)
            d = df[(df.K == K) & (df["mode"] == mode)].set_index("subject")
            e = df[(df.K == K) & (df["mode"] == "exact")].set_index("subject")
            rel = (d.objective - e.objective) / e.objective
            summary.append(
                {
                    "K": K,
                    "mode": mode,
                    "seconds_mean": d.seconds.mean(),
                    "objective_vs_exact_min": rel.min(),
                    "objective_vs_exact_max": rel.max(),
                    "heldout_r2_mean": d.heldout_r2.mean(),
                    "heldout_r2_ols_mean": d.heldout_r2_ols.mean(),
                    "heldout_r2_vs_exact_min": (d.heldout_r2 - e.heldout_r2).min(),
                    "heldout_r2_vs_exact_max": (d.heldout_r2 - e.heldout_r2).max(),
                    **{f"decoding_window_{w}": a for w, a in acc.items()},
                }
            )
    table = pd.DataFrame(summary)
    STATIC.mkdir(parents=True, exist_ok=True)
    table.to_csv(STATIC / "modes.csv", index=False, float_format="%.6g")
    print(table.to_string())


if __name__ == "__main__":
    main()
