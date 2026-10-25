"""Across-group timepoint decoding on real fMRI (after Manning et al., 2018).

All participants watched the same movie, so the same time window should look
alike across subjects. For every bipartition of the subjects into two equal
groups, each group's features are averaged over subjects (each feature z-scored
over time within subject first), averaged within non-overlapping windows of
``w`` samples, and every window of one group is matched to the window of the
other group with the highest Pearson correlation. Accuracy is the fraction of
windows matched to themselves, averaged over both directions; chance is
1 / n_windows. A circular-shift null shifts the second group's time course by a
random non-zero number of windows.

Features (per subject, all from the cleaned, z-scored data of `validation_data`):

* ``htfa_local``: the per-subject HTFA weights from `validation_template`
  (factor order matched to the template);
* ``htfa_template``: ridge weights of each subject's data on the fitted global
  template factors;
* ``frozen_start``: ridge weights on the 'ncorr' start (``max_iter=0``) of the
  template subject, with the same K;
* ``atlas``: mean signal in each Harvard-Oxford cortical parcel (max-prob, 25%).

Writes ``docs/_static/validation/decoding.{csv,png}``.
"""

import itertools

import numpy as np
import pandas as pd
from validation_data import OUT, STATIC, adult_ids, clean, load_subject

from htfa._kernels import factors, ridge_weights
from htfa.tfa import TFA, standardize_data

WINDOWS = (1, 6)
SEED = 0


def zt(W: np.ndarray) -> np.ndarray:
    """z-score each feature (row) over time."""
    s = W.std(1, keepdims=True)
    return (W - W.mean(1, keepdims=True)) / np.where(s > 0, s, 1.0)


def windowed(G: np.ndarray, w: int) -> np.ndarray:
    n = G.shape[1] // w
    return G[:, : n * w].reshape(G.shape[0], n, w).mean(2).T  # (n_windows, F)


def accuracy(A: np.ndarray, B: np.ndarray) -> float:
    a = (A - A.mean(1, keepdims=True)) / A.std(1, keepdims=True)
    b = (B - B.mean(1, keepdims=True)) / B.std(1, keepdims=True)
    C = a @ b.T / a.shape[1]
    idx = np.arange(len(C))
    return 0.5 * (np.mean(C.argmax(1) == idx) + np.mean(C.argmax(0) == idx))


def atlas_features(sid: str, Xz: np.ndarray, R: np.ndarray) -> np.ndarray:
    import nibabel as nib
    from nilearn.datasets import fetch_atlas_harvard_oxford

    atlas = fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    img = atlas.maps if hasattr(atlas.maps, "affine") else nib.load(atlas.maps)
    ijk = np.rint(nib.affines.apply_affine(np.linalg.inv(img.affine), R)).astype(int)
    lab = np.asarray(img.dataobj)[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    # The same parcels for every subject; a parcel with no voxels is all zeros.
    zero = np.zeros(Xz.shape[1])
    return np.stack(
        [Xz[lab == v].mean(0) if np.any(lab == v) else zero for v in range(1, 49)]
    )


def main() -> None:
    ids = adult_ids()
    fin = np.load(OUT / "template_final.npz")
    assert list(fin["ids"]) == ids
    local = np.load(OUT / "template_weights.npy")
    Cg, wg = fin["global_centers"], fin["global_widths"]
    K = Cg.shape[0]
    data = []
    for s in ids:
        X, R, conf = load_subject(s)
        data.append((standardize_data(clean(X, conf), "auto"), R))
    X0, R0 = data[0]
    frozen = TFA(
        K=K, max_iter=0, init="ncorr", standardize=False, random_state=SEED
    ).fit(X0, R0)
    feats = {
        "htfa_local": [zt(W) for W in local],
        "htfa_template": [zt(ridge_weights(X, factors(R, Cg, wg))) for X, R in data],
        "frozen_start": [
            zt(ridge_weights(X, factors(R, frozen.centers_, frozen.widths_)))
            for X, R in data
        ],
        "atlas": [
            zt(atlas_features(s, X, R)) for s, (X, R) in zip(ids, data, strict=True)
        ],
    }
    n = len(ids)
    splits = [g for g in itertools.combinations(range(n), n // 2) if 0 in g]
    rng = np.random.default_rng(SEED)
    rows = []
    for name, F in feats.items():
        for w in WINDOWS:
            for g1 in splits:
                g2 = [s for s in range(n) if s not in g1]
                A = windowed(np.mean([F[s] for s in g1], 0), w)
                B = windowed(np.mean([F[s] for s in g2], 0), w)
                shift = int(rng.integers(1, len(B)))
                rows.append(
                    {
                        "features": name,
                        "window": w,
                        "n_windows": len(A),
                        "split": "".join(map(str, g1)),
                        "accuracy": accuracy(A, B),
                        "null_accuracy": accuracy(A, np.roll(B, shift, axis=0)),
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "decoding.csv", index=False)
    summary = (
        df.groupby(["features", "window"])
        .agg(
            n_windows=("n_windows", "first"),
            n_splits=("accuracy", "size"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_min=("accuracy", "min"),
            null_mean=("null_accuracy", "mean"),
        )
        .reset_index()
    )
    summary["chance"] = 1.0 / summary["n_windows"]
    STATIC.mkdir(parents=True, exist_ok=True)
    summary.round(4).to_csv(STATIC / "decoding.csv", index=False)
    print(f"{n} subjects, K={K}")
    print(summary.round(4).to_string())
    plot(summary)


def plot(summary: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["frozen_start", "htfa_template", "htfa_local", "atlas"]
    fig, axes = plt.subplots(1, len(WINDOWS), figsize=(8.0, 3.0), dpi=110)
    for ax, w in zip(axes, WINDOWS, strict=True):
        s = summary[summary["window"] == w].set_index("features").loc[names]
        ax.bar(range(len(names)), s["accuracy_mean"], color="C0", label="decoding")
        ax.scatter(
            range(len(names)),
            s["null_mean"],
            color="k",
            marker="_",
            s=200,
            label="shift null",
        )
        ax.axhline(s["chance"].iloc[0], color="C3", ls="--", lw=1, label="chance")
        ax.set_xticks(range(len(names)), [n.replace("_", "\n") for n in names])
        ax.set_title(f"window {w} TR ({int(s['n_windows'].iloc[0])} windows)")
        ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("timepoint decoding accuracy")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(STATIC / "decoding.png")


if __name__ == "__main__":
    main()
