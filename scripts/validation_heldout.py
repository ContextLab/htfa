"""Leak-free held-out reconstruction on real fMRI with a frozen-start control.

For each adult of nilearn's development_fmri (see `validation_data`):

* Train samples 0-79, a gap of 8 samples, test samples 88-167. Each segment is
  detrended and confound-regressed on its own samples; z-scoring parameters come
  from the training segment and are applied unchanged to the test segment.
* TFA (K factors) is fitted on the training segment only (``standardize=False``),
  from the 'ncorr' and the 'kmeans' (BrainIAK) start, with ``max_iter=0`` (the
  start itself, "frozen") and with ``max_iter=N_ITER`` (inference), using the same
  start, K, preprocessing and weight estimator.
* Baselines: matched-K PCA spatial maps of the training segment; random voxel
  centers with the median fitted width; and, as a model-free reference, the mean
  of the other voxel set within 8 mm.
* Test: voxels are split into two sets of 16 mm spatial blocks (A/B, checkerboard).
  Test weights are ridge-estimated from set A (penalty frozen at the training
  variance) and R^2 is scored on set B against the training-mean predictor (0 after
  training z-scoring); then A and B swap, and the two R^2 are averaged. The same
  score with OLS test weights is also reported.

``--score-only`` rescores saved fits without refitting.

Writes ``heldout.csv`` and ``heldout_<subject>_<cond>.npz`` to the output
directory and ``docs/_static/validation/heldout.{csv,png}``.
"""

import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from validation_data import OUT, STATIC, adult_ids, clean, load_subject, zscore_with

from htfa._kernels import factors
from htfa.tfa import TFA

K = 50
N_ITER = 10
MAX_VOXELS = 5000
MAX_TRS = 84
TRAIN, GAP = 80, 8
BLOCK_MM = 16.0
SEED = 0


def split(sid: str):
    X, R, conf = load_subject(sid)
    Xtr = clean(X[:, :TRAIN], conf[:TRAIN])
    Xte = clean(X[:, TRAIN + GAP :], conf[TRAIN + GAP :])
    mu, sd = Xtr.mean(1), Xtr.std(1)
    Xtr, Xte = zscore_with(Xtr, mu, sd), zscore_with(Xte, mu, sd)
    blocks = np.floor((R - R.min(0)) / BLOCK_MM).astype(int).sum(1) % 2 == 0
    return Xtr, Xte, R, blocks


def heldout_r2(F: np.ndarray, Xte: np.ndarray, A: np.ndarray, lam: float) -> float:
    """Mean of the two A->B and B->A held-out R^2 values (``lam=0``: OLS)."""
    r2 = []
    for src, dst in ((A, ~A), (~A, A)):
        Fs = F[src]
        W = np.linalg.solve(Fs.T @ Fs + lam * np.eye(F.shape[1]), Fs.T @ Xte[src])
        resid = Xte[dst] - F[dst] @ W
        r2.append(1.0 - np.sum(resid**2) / np.sum(Xte[dst] ** 2))
    return float(np.mean(r2))


def fit_one(sid: str, init: str, max_iter: int) -> dict:
    Xtr, Xte, R, A = split(sid)
    t0 = time.time()
    with threadpool_limits(2), warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m = TFA(
            K=K,
            max_iter=max_iter,
            max_voxels=MAX_VOXELS,
            max_trs=MAX_TRS,
            standardize=False,
            init=init,
            backend="numpy",
            random_state=SEED,
        ).fit(Xtr, R)
    F = factors(R, m.centers_, m.widths_)
    resid = Xtr - F @ m.weights_
    out = {
        "subject": sid,
        "cond": f"{init}_{'start' if max_iter == 0 else 'fit'}",
        "train_objective": m.cost_,
        "train_r2": 1.0 - np.sum(resid**2) / np.sum(Xtr**2),
        "n_iter": m.n_iter_,
        "reason": m.converged_reason_,
        "seconds": time.time() - t0,
    }
    np.savez(
        OUT / f"heldout_{sid}_{out['cond']}.npz", centers=m.centers_, widths=m.widths_
    )
    print(out, flush=True)
    return out


def neighbour_r2(Xte: np.ndarray, R: np.ndarray, A: np.ndarray, radius: float) -> float:
    """Held-out R^2 of predicting each voxel by the mean of the other set's voxels
    within ``radius`` mm (a model-free spatial-smoothness reference)."""
    r2 = []
    for src, dst in ((A, ~A), (~A, A)):
        nb = cKDTree(R[src]).query_ball_point(R[dst], radius)
        P = np.array([Xte[src][n].mean(0) if n else np.zeros(Xte.shape[1]) for n in nb])
        r2.append(1.0 - np.sum((Xte[dst] - P) ** 2) / np.sum(Xte[dst] ** 2))
    return float(np.mean(r2))


def score(sid: str) -> list[dict]:
    """Held-out R^2 (ridge with the frozen training penalty, and OLS) of every
    condition of one subject, from the saved fits."""
    Xtr, Xte, R, A = split(sid)
    lam = float(np.var(Xtr))
    maps = {}
    for c in CONDS:
        z = np.load(OUT / f"heldout_{sid}_{c}.npz")
        maps[c] = factors(R, z["centers"], z["widths"])
    maps["pca"] = np.linalg.svd(Xtr, full_matrices=False)[0][:, :K]
    rng = np.random.default_rng(SEED)
    w = np.median(np.load(OUT / f"heldout_{sid}_ncorr_fit.npz")["widths"])
    maps["random"] = factors(R, R[rng.choice(len(R), K, replace=False)], np.full(K, w))
    rows = [
        {
            "subject": sid,
            "cond": c,
            "heldout_r2": heldout_r2(F, Xte, A, lam),
            "heldout_r2_ols": heldout_r2(F, Xte, A, 0.0),
        }
        for c, F in maps.items()
    ]
    rows.append(
        {
            "subject": sid,
            "cond": "neighbour_8mm",
            "heldout_r2": neighbour_r2(Xte, R, A, 8.0),
        }
    )
    print(rows, flush=True)
    return rows


CONDS = ("ncorr_start", "ncorr_fit", "kmeans_start", "kmeans_fit")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ids = adult_ids()
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    if "--score-only" not in sys.argv:
        jobs = [
            (s, i, it) for i in ("ncorr", "kmeans") for it in (0, N_ITER) for s in ids
        ]
        fits = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(fit_one)(*j) for j in jobs
        )
        pd.DataFrame(fits).to_csv(OUT / "heldout_fits.csv", index=False)
    fits = pd.read_csv(OUT / "heldout_fits.csv")
    rows = Parallel(n_jobs=n_jobs, backend="loky")(delayed(score)(s) for s in ids)
    df = pd.DataFrame([r for rr in rows for r in rr])
    df.to_csv(OUT / "heldout.csv", index=False)
    wide = df.pivot(index="subject", columns="cond", values="heldout_r2")
    ols = df.pivot(index="subject", columns="cond", values="heldout_r2_ols")
    for c in (*CONDS, "pca"):
        wide[f"ols_{c}"] = ols[c]
    dist = {}
    for s in ids:
        c = {k: np.load(OUT / f"heldout_{s}_{k}.npz")["centers"] for k in CONDS}
        d1, d0 = (
            cdist(c["ncorr_fit"], c["kmeans_fit"]),
            cdist(c["ncorr_start"], c["kmeans_start"]),
        )
        dist[s] = (
            d1[linear_sum_assignment(d1)].mean(),
            d0[linear_sum_assignment(d0)].mean(),
        )
    wide["match_mm_fit"] = [dist[s][0] for s in wide.index]
    wide["match_mm_start"] = [dist[s][1] for s in wide.index]
    for col in ("train_objective", "train_r2"):
        p = fits.pivot(index="subject", columns="cond", values=col)
        for c in CONDS:
            wide[f"{col}_{c}"] = p[c]
    STATIC.mkdir(parents=True, exist_ok=True)
    wide.round(5).to_csv(STATIC / "heldout.csv")
    print(fits[["subject", "cond", "n_iter", "reason", "seconds"]].to_string())
    print(wide.round(4).T.to_string())
    for a, b in (
        ("ncorr_fit", "ncorr_start"),
        ("kmeans_fit", "kmeans_start"),
        ("ols_ncorr_fit", "ols_ncorr_start"),
        ("ols_kmeans_fit", "ols_kmeans_start"),
        ("train_objective_ncorr_start", "train_objective_ncorr_fit"),
        ("train_objective_kmeans_start", "train_objective_kmeans_fit"),
    ):
        d = wide[a] - wide[b]
        print(
            f"{a} - {b}: mean {d.mean():.5f}, min {d.min():.5f}, "
            f"max {d.max():.5f}, n>0 {(d > 0).sum()}/{len(d)}"
        )
    plot(wide)


def plot(wide: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = [
        "random",
        "kmeans_start",
        "kmeans_fit",
        "ncorr_start",
        "ncorr_fit",
        "pca",
        "neighbour_8mm",
    ]
    labels = [
        "random\ncenters",
        "kmeans\nstart",
        "kmeans\nfitted",
        "ncorr\nstart",
        "ncorr\nfitted",
        "PCA",
        "8 mm\nneighbours",
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=110)
    x = np.arange(len(conds))
    for s in wide.index:
        ax.plot(x, wide.loc[s, conds], color="0.7", lw=0.8, marker="o", ms=3, zorder=1)
    ax.plot(x, wide[conds].mean(), color="C0", lw=2, marker="o", zorder=2, label="mean")
    ax.set_xticks(x, labels)
    ax.set_ylabel("held-out R$^2$ (test half, other voxels)")
    ax.set_title(f"Held-out reconstruction, K={K}, {len(wide)} adults")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(STATIC / "heldout.png")


if __name__ == "__main__":
    main()
