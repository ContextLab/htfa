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

``--htfa [MAX_GLOBAL_ITER]`` instead fits `htfa.HTFA` (defaults, K = 50,
``n_jobs=6``, ``max_global_iter`` 10 unless given) to the twelve adults in
each `HTFA_VARIANTS` entry: both modes on the data as is, and on data changed
by a relative 1e-12 (three seeds in exact mode, one in fast mode), the
baseline for how far two fits of one mode end apart. For the full runs it
records the wall time, the summed final local objective and
validation_decoding's accuracy (local weights, and ridge weights on the
template factors); for the training segments the held-out R^2 of each
subject's local factors. Every variant's template and its template subject's
stage-0 fit are compared with the exact fit's (median and max distance of
matched centers). With another ``max_global_iter`` only the full runs of
`HTFA_VARIANTS_LONG` are fitted. Writes ``modes_htfa.csv``
(``modes_htfa_<N>.csv`` for another ``max_global_iter``) to
``docs/_static/validation/``.
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


# HTFA variants: (mode, perturbation seed); seed 0 is the data as is, seeds
# >= 1 multiply every subject's data by (1 + PERTURBATION * N(0, 1)), the
# rounding-level baseline for how far apart two fits of the same mode end.
HTFA_VARIANTS = (
    ("exact", 0),
    ("fast", 0),
    ("exact", 1),
    ("exact", 2),
    ("exact", 3),
    ("fast", 1),
)
# With another max_global_iter (runs to the stopping rule): fewer variants,
# full runs only.
HTFA_VARIANTS_LONG = (("exact", 0), ("exact", 1), ("fast", 0), ("fast", 1))
PERTURBATION = 1e-12


def _perturbed(X: np.ndarray, seed: int, subject: int) -> np.ndarray:
    if seed == 0:
        return X
    rng = np.random.default_rng([seed, subject])
    return X * (1.0 + PERTURBATION * rng.standard_normal(X.shape))


def _matched_distance(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Distances (mm) between the centers of A and the matched centers of B."""
    return np.linalg.norm(B[match_factors(A, B)] - A, axis=1)


def htfa_comparison(runs: list, ids: list, max_global_iter: int) -> None:
    """HTFA in every HTFA_VARIANTS entry on the full runs (objective, template,
    decoding) and on the training segments (held-out R^2 of each subject's
    local factors, scored as in `validation_heldout`)."""
    segments = [split(sid) for sid in ids]
    rows, fits = [], {}
    long = max_global_iter != 10
    for mode, seed in HTFA_VARIANTS_LONG if long else HTFA_VARIANTS:
        for segment in ("full",) if long else ("full", "train"):
            if segment == "full":
                data = list(runs)
            else:
                data = [(Xtr, R) for Xtr, _, R, _ in segments]
            Xs = [_perturbed(X, seed, s) for s, (X, _) in enumerate(data)]
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model = HTFA(
                    K=50,
                    mode=mode,
                    max_global_iter=max_global_iter,
                    n_jobs=6,
                    random_state=SEED,
                ).fit(Xs, [R for _, R in data])
            seconds = time.perf_counter() - t0
            fits[mode, seed, segment] = model
            row = {
                "mode": mode,
                "perturbation_seed": seed,
                "segment": segment,
                "seconds": seconds,
                "global_iterations": model.n_iter_,
                "converged_reason": model.converged_reason_,
                "objective": model.objective_history_[-1],
            }
            if segment == "full":
                C, w = model.global_centers_, model.global_widths_
                local = decoding_accuracy([zt(W) for W in model.weights_])
                template = decoding_accuracy(
                    [zt(ridge_weights(X, factors(R, C, w))) for X, R in runs]
                )
                row.update({f"local_window_{k}": v for k, v in local.items()})
                row.update({f"template_window_{k}": v for k, v in template.items()})
            else:
                r2 = [
                    heldout_r2(
                        factors(R, model.centers_[s], model.widths_[s]),
                        Xte,
                        A,
                        float(np.var(Xtr)),
                    )
                    for s, (Xtr, Xte, R, A) in enumerate(segments)
                ]
                row["heldout_r2_mean"] = float(np.mean(r2))
            rows.append(row)
            print(row, flush=True)
    for row in rows:
        ref = fits["exact", 0, row["segment"]]
        model = fits[row["mode"], row["perturbation_seed"], row["segment"]]
        t = ref.template_subject_
        d = _matched_distance(ref.global_centers_, model.global_centers_)
        free = _matched_distance(ref.free_fits_[t][0], model.free_fits_[t][0])
        ref_obj = ref.objective_history_[-1]
        row.update(
            {
                "template_vs_exact_median_mm": float(np.median(d)),
                "template_vs_exact_max_mm": float(d.max()),
                "stage0_vs_exact_median_mm": float(np.median(free)),
                "objective_vs_exact": (row["objective"] - ref_obj) / ref_obj,
            }
        )
        print(
            f"{row['mode']} seed {row['perturbation_seed']} {row['segment']}: "
            f"template vs exact median {np.median(d):.2f} mm, max {d.max():.2f} mm; "
            f"template subject's stage-0 fit vs exact median {np.median(free):.2f} mm",
            flush=True,
        )
    STATIC.mkdir(parents=True, exist_ok=True)
    name = (
        "modes_htfa.csv"
        if max_global_iter == 10
        else f"modes_htfa_{max_global_iter}.csv"
    )
    pd.DataFrame(rows).to_csv(STATIC / name, index=False, float_format="%.6g")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ids = adult_ids()
    runs = []
    for sid in ids:
        X, R, conf = load_subject(sid)
        runs.append((standardize_data(clean(X, conf), "auto"), R))
    if "--htfa" in sys.argv:
        args = sys.argv[sys.argv.index("--htfa") + 1 :]
        htfa_comparison(runs, ids, int(args[0]) if args else 10)
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
