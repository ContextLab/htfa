"""Audit of the HTFA template (MAP) update on real fMRI.

Fits `htfa.HTFA` unchanged to the downloaded adults of development_fmri (cleaned
and z-scored, see `validation_data`) and records, per global iteration, through
recording wrappers around ``htfa.htfa.map_update`` and ``HTFA._fit_subjects``
(the fit itself is not modified):

* the trace of the template centers' posterior covariance (identical for every
  factor) and the widths' posterior variance;
* template movement: mean and max distance between consecutive template centers;
* subject dispersion: mean distance of the subjects' local centers to the prior
  template they were fitted against, and the across-subject spread of the local
  centers around their mean;
* a check of every MAP center update against an independent precision-form
  Gaussian posterior ``P = S0^-1 + n Sigma^-1``, ``m = P^-1 (S0^-1 m0 + Sigma^-1
  sum(obs))`` on the recorded real inputs, plus the closed form
  ``S_i^-1 = S_0^-1 + i * n * Sigma^-1`` of the covariance after i iterations.

With the argument ``--fixed-cov`` (a control, not HTFA's behaviour), the
wrapper instead computes every center update from the *initial* prior
covariance, so the template covariance stays at its one-pass value instead of
shrinking (widths are unaffected); outputs then carry the suffix ``_fixedcov``.

Writes ``template_audit.csv`` and ``template_<...>.npz`` to the output directory,
and ``docs/_static/validation/template_audit.{csv,png}``.
"""

import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from validation_data import OUT, STATIC, adult_ids, clean, load_subject

import htfa.htfa as htfa_mod
from htfa._matching import match_factors

K = 30
MAX_GLOBAL_ITER = 10
MAX_LOCAL_ITER = 3
MAX_VOXELS = 1000
MAX_TRS = 40
SEED = 0

MAP_CALLS: list[tuple] = []
FIT_CALLS: list[list[tuple]] = []
_orig_map_update = htfa_mod.map_update
FIXED = "--fixed-cov" in sys.argv
SUFFIX = "_fixedcov" if FIXED else ""


def recording_map_update(prior_mean, prior_cov, global_cov_scaled, observations):
    if FIXED and MAP_CALLS:
        prior_cov = MAP_CALLS[0][1]
    out = _orig_map_update(prior_mean, prior_cov, global_cov_scaled, observations)
    MAP_CALLS.append(
        tuple(
            np.array(a, dtype=float)
            for a in (prior_mean, prior_cov, global_cov_scaled, observations)
        )
        + tuple(np.array(o) for o in out)
    )
    return out


class RecordingHTFA(htfa_mod.HTFA):
    def _fit_subjects(self, *args, **kwargs):
        fits = super()._fit_subjects(*args, **kwargs)
        FIT_CALLS.append([(f.centers_.copy(), f.widths_.copy(), f.cost_) for f in fits])
        return fits


def precision_form(m0, S0, Sigma, obs):
    n = len(obs)
    Si = np.linalg.inv(Sigma)
    P = np.linalg.inv(S0) + n * Si
    cov = np.linalg.inv(P)
    return cov @ (np.linalg.solve(S0, m0) + Si @ obs.sum(0)), cov


def main() -> None:
    htfa_mod.map_update = recording_map_update
    ids = adult_ids()
    Xs, Rs = [], []
    for s in ids:
        X, R, conf = load_subject(s)
        Xs.append(clean(X, conf))
        Rs.append(R)
    n = len(ids)
    model = RecordingHTFA(
        K=K,
        max_global_iter=MAX_GLOBAL_ITER,
        max_local_iter=MAX_LOCAL_ITER,
        max_voxels=MAX_VOXELS,
        max_trs=MAX_TRS,
        init="ncorr",
        backend="numpy",
        n_jobs=3,
        random_state=SEED,
        verbose=True,
    )
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(Xs, Rs)
    print(f"fit {time.time() - t0:.0f} s, n_iter {model.n_iter_}", flush=True)
    n_iter = model.n_iter_
    assert len(MAP_CALLS) == K * n_iter, len(MAP_CALLS)

    rows = []
    max_mean_err = max_cov_err = max_closed_err = 0.0
    S0 = MAP_CALLS[0][1]
    Sigma = MAP_CALLS[0][2] * n
    for i in range(n_iter):
        calls = MAP_CALLS[i * K : (i + 1) * K]
        prior_C = np.array([c[0] for c in calls])
        obs = np.stack([c[3] for c in calls], axis=1)  # (n, K, D)
        post_C = np.array([c[4] for c in calls])
        post_cov = calls[0][5]
        for m0, S, G, o, pm, pc in calls:
            em, ec = precision_form(m0, S, G * n, o)
            max_mean_err = max(max_mean_err, float(np.max(np.abs(em - pm))))
            max_cov_err = max(
                max_cov_err, float(np.max(np.abs(ec - pc)) / np.max(np.abs(pc)))
            )
        closed = np.linalg.inv(np.linalg.inv(S0) + (i + 1) * n * np.linalg.inv(Sigma))
        max_closed_err = max(
            max_closed_err,
            float(np.max(np.abs(closed - post_cov)) / np.max(np.abs(post_cov))),
        )
        same_cov = max(float(np.max(np.abs(c[5] - post_cov))) for c in calls)
        move = np.linalg.norm(post_C - prior_C, axis=1)
        to_prior = np.linalg.norm(obs - prior_C[None], axis=2)
        spread = np.linalg.norm(obs - obs.mean(0, keepdims=True), axis=2)
        rows.append(
            {
                "iteration": i + 1,
                "prior_cov_trace": float(np.trace(calls[0][1])),
                "post_cov_trace": float(np.trace(post_cov)),
                "post_sd_mm": float(np.sqrt(np.trace(post_cov) / 3)),
                "max_cov_diff_across_factors": same_cov,
                "template_move_mean_mm": float(move.mean()),
                "template_move_max_mm": float(move.max()),
                "subject_to_prior_mean_mm": float(to_prior.mean()),
                "subject_spread_mean_mm": float(spread.mean()),
            }
        )
    # Widths: posterior variance recursion is data independent as well.
    widths_var = model.global_widths_var_
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"template_audit{SUFFIX}.csv", index=False)
    STATIC.mkdir(parents=True, exist_ok=True)
    df.round(6).to_csv(STATIC / f"template_audit{SUFFIX}.csv", index=False)
    print(df.round(4).to_string())
    sigma_trace = float(np.trace(Sigma))
    print(
        f"subjects {n}; K {K}; S0 trace {np.trace(S0):.4f} mm^2; "
        f"Sigma trace {sigma_trace:.4f} mm^2"
    )
    print(
        "final widths posterior variance (all factors equal: "
        f"{np.ptp(widths_var) == 0}): {widths_var[0]:.6g}"
    )
    w_var0 = 2.0 * float(np.max(np.std(Rs[0], axis=0))) ** 2
    print(f"initial widths variance (template subject max_sigma, approx): {w_var0:.6g}")
    print(f"MAP mean vs precision form: max abs diff {max_mean_err:.3g} mm")
    print(f"MAP cov vs precision form: max rel diff {max_cov_err:.3g}")
    print(
        "posterior cov vs closed form S0^-1 + i n Sigma^-1: "
        f"max rel diff {max_closed_err:.3g}"
    )
    # Subject fits in the last iteration vs the final template.
    final = [
        np.linalg.norm(
            c - model.global_centers_[match_factors(c, model.global_centers_)], axis=1
        ).mean()
        for c in model.centers_
    ]
    print(
        f"final local-to-template mean distance per subject (mm): {np.round(final, 2)}"
    )
    np.savez(
        OUT / f"template_final{SUFFIX}.npz",
        global_centers=model.global_centers_,
        global_widths=model.global_widths_,
        centers=np.array(model.centers_),
        ids=np.array(ids),
    )
    np.save(OUT / f"template_weights{SUFFIX}.npy", np.stack(model.weights_))
    plot(df)


def plot(df: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0), dpi=110)
    it = df["iteration"]
    axes[0].semilogy(it, df["post_sd_mm"], marker="o")
    axes[0].set_title("template center posterior SD (mm)")
    axes[1].plot(it, df["template_move_mean_mm"], marker="o", label="mean")
    axes[1].plot(it, df["template_move_max_mm"], marker="o", label="max")
    axes[1].set_title("template movement (mm)")
    axes[1].legend(frameon=False)
    axes[2].plot(
        it, df["subject_to_prior_mean_mm"], marker="o", label="to prior template"
    )
    axes[2].plot(
        it, df["subject_spread_mean_mm"], marker="o", label="around subject mean"
    )
    axes[2].set_title("local centers: mean distance (mm)")
    axes[2].legend(frameon=False)
    for ax in axes:
        ax.set_xlabel("global iteration")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(STATIC / f"template_audit{SUFFIX}.png")


if __name__ == "__main__":
    main()
