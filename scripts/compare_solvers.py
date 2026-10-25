"""Compare the structured 'bounded-lm' center/width solver with scipy's 'trf'.

For `htfa.make_tfa_data` in 2D and 3D, several seeds, with and without an HTFA
prior, and the linear and soft_l1 losses, run (a) one center/width solve from the
same start and weights (the 'ncorr' start) and (b) full TFA fits (init 'ncorr')
with each ``nlss_method``. Prints a markdown table with the costs, their relative
difference ``(lm - trf) / trf``, the largest matched center difference and the
wall times::

    .venv/bin/python scripts/compare_solvers.py --seeds 0,1,2,3,4,5

Acceptance (Task 11b): lm cost <= trf cost * (1 + 1e-6) and matched centers
within 1e-3 of trf's, or a lower lm cost.
"""

import argparse
import dataclasses
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from htfa._engines import NumpyEngine
from htfa._kernels import (
    Prior,
    ResidualProblem,
    bounds,
    factors,
    max_sigma,
    ridge_weights,
    solve_centers_widths,
)
from htfa.synthetic import make_tfa_data, match_factors
from htfa.tfa import TFA, _initial_starts, standardize_data

DATA = {"2d": ((40, 40), 6), "3d": ((20, 20, 20), 8)}
COST_RTOL = 1e-6
CENTER_ATOL = 1e-3


def make_prior(R, truth, rng):
    """A template near the truth with HTFA's initial (co)variances."""
    K, D = truth["centers"].shape
    length = np.sqrt(max_sigma(R) / 2.0)
    centers = truth["centers"] + rng.normal(0.0, 0.3 * length, size=(K, D))
    cov = np.cov(R.T) * K ** (-2.0 / 3.0)
    return Prior(
        centers=centers,
        widths=truth["widths"] * rng.uniform(0.8, 1.2, size=K),
        centers_cov=np.tile(cov, (K, 1, 1)),
        widths_var=np.full(K, max_sigma(R)),
        scale=0.5,
    )


def matched_diff(C_ref, C):
    perm = match_factors(C_ref, C)
    return float(np.max(np.abs(C[perm] - C_ref)))


def one_solve(X, R, K, prior, loss, seed):
    lb, ub = bounds(R, K, 0.02, 1.8)
    start = _initial_starts(
        X, R, K, ("ncorr",), np.random.default_rng(seed), lb[-1], ub[-1]
    )["ncorr"]
    if prior is not None:
        start = (prior.centers, prior.widths)
    x0 = np.clip(np.concatenate([start[0].ravel(), start[1]]), lb, ub)
    W = ridge_weights(X, factors(R, *start))
    out = {}
    for method in ("trf", "bounded-lm"):
        prob = ResidualProblem(X, W, R, NumpyEngine(), prior, compress=loss == "linear")
        t0 = time.perf_counter()
        x, cost = solve_centers_widths(prob, x0, (lb, ub), method=method, loss=loss)
        out[method] = (
            x[: K * R.shape[1]].reshape(K, -1),
            cost,
            time.perf_counter() - t0,
        )
    return out


def full_fit(X, R, K, prior, loss, seed):
    out = {}
    for method in ("trf", "bounded-lm"):
        model = TFA(
            K=K, nlss_method=method, nlss_loss=loss, init="ncorr", random_state=seed
        )
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(X, R, prior=prior)
        out[method] = (
            model.centers_,
            model.cost_,
            time.perf_counter() - t0,
            model.n_iter_,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", default="0,1,2,3,4,5")
    parser.add_argument("--data", default="2d,3d")
    parser.add_argument("--modes", default="solve,fit")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    print(
        "| mode | data | seed | prior | loss | cost trf | cost lm | rel diff "
        "| max dC | time trf (s) | time lm (s) | iters trf/lm | ok |"
    )
    print("|-|-|-|-|-|-|-|-|-|-|-|-|-|")
    n_bad = 0
    for mode in args.modes.split(","):
        for name in args.data.split(","):
            shape, K = DATA[name]
            for seed in seeds:
                X, R, truth = make_tfa_data(shape, 100, K=K, random_state=seed)
                Xz = standardize_data(X, "auto")
                rng = np.random.default_rng(1000 + seed)
                base_prior = make_prior(R, truth, rng)
                for use_prior in (False, True):
                    for loss in ("linear", "soft_l1"):
                        if mode == "solve":
                            prior = (
                                dataclasses.replace(base_prior) if use_prior else None
                            )
                            res = one_solve(Xz, R, K, prior, loss, seed)
                        else:
                            prior = base_prior if use_prior else None
                            res = full_fit(X, R, K, prior, loss, seed)
                        (C_t, c_t, t_t, *it_t), (C_l, c_l, t_l, *it_l) = (
                            res["trf"],
                            res["bounded-lm"],
                        )
                        rel = (c_l - c_t) / c_t
                        dC = matched_diff(C_t, C_l)
                        ok = c_l <= c_t * (1 + COST_RTOL) and (
                            dC < CENTER_ATOL or c_l < c_t
                        )
                        n_bad += not ok
                        iters = f"{it_t[0]}/{it_l[0]}" if it_t else "-"
                        verdict = "yes" if ok else "NO"
                        print(
                            f"| {mode} | {name} | {seed} | {use_prior} | {loss} "
                            f"| {c_t:.10g} | {c_l:.10g} | {rel:+.2e} | {dC:.2e} "
                            f"| {t_t:.2f} | {t_l:.2f} | {iters} | {verdict} |",
                            flush=True,
                        )
    print(f"\n{n_bad} case(s) failed the acceptance criterion")


if __name__ == "__main__":
    main()
