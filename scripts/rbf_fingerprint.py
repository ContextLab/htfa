"""Bitwise fingerprint of the 'rbf' kernels and fits, for comparing two versions.

Runs a fixed battery on the installed ``htfa`` (whichever version is first on
``sys.path``) and writes every resulting array to an ``.npz``: factors, factor
derivatives, costs, gradients, Gauss-Newton matrices and Hessians of the
fixed-weight and variable-projection problems (with and without a prior, both
modes, four losses), `bounded_lm` solves, TFA fits (every solver, both modes,
subsampling, a prior), HTFA fits ('brainiak', 'fixed', 'empirical'; serial and
``n_jobs=2``), synthetic data and effective nodes. Only the API that existed
before the kernel shapes is used, so the same script runs on both versions.

Usage::

    python scripts/rbf_fingerprint.py OUT.npz            # write
    python scripts/rbf_fingerprint.py --compare A.npz B.npz

``--compare`` reports every key whose arrays are not bit for bit equal (and
keys present in only one file) and exits 1 if there is any.
"""

import sys
import warnings

import numpy as np


def battery() -> dict[str, np.ndarray]:
    from htfa import _kernels as kn
    from htfa._engines import NumpyEngine
    from htfa.brainnet import effective_nodes
    from htfa.htfa import HTFA
    from htfa.synthetic import make_htfa_data, make_tfa_data
    from htfa.tfa import TFA

    out: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(0)
    for D in (2, 3):
        grid = (14, 13) if D == 2 else (8, 7, 6)
        X, R, truth = make_tfa_data(grid, 30, K=4, noise=0.3, random_state=D)
        out[f"data{D}"] = X
        C, w = truth["centers"], truth["widths"]
        out[f"factors{D}"] = kn.factors(R, C, w)
        out[f"factors_grid{D}"] = kn.factors(R, C, w, kn.axis_grid(R))
        for j, a in enumerate(kn.factor_derivatives(R, C, w, kn.axis_grid(R))):
            out[f"derivs{D}_{j}"] = a
        x = np.concatenate([C.ravel(), w]) + 0.05 * rng.normal(size=C.size + 4)
        W = kn.ridge_weights(X, kn.factors(R, C, w))
        prior = kn.Prior(
            C + 0.3, w * 1.1, np.tile(np.eye(D), (4, 1, 1)), np.full(4, 2.0), 0.7
        )
        for use_prior in (False, True):
            p = prior if use_prior else None
            for loss in ("linear", "soft_l1", "huber", "cauchy"):
                rp = kn.ResidualProblem(X, W, R, NumpyEngine(1), p, loss == "linear")
                for mode in ("fast", "exact"):
                    for j, a in enumerate(rp.cost_grad_hess(x, loss, mode=mode)):
                        out[f"res{D}_{use_prior}_{loss}_{mode}_{j}"] = np.asarray(a)
                out[f"rescost{D}_{use_prior}_{loss}"] = np.asarray(rp.cost(x, loss))
                out[f"resid{D}_{use_prior}_{loss}"] = rp.residual(x)
                out[f"colnorm{D}_{use_prior}_{loss}"] = rp.column_norms(x)
                J = rp.jacobian(x)
                out[f"jmv{D}_{use_prior}_{loss}"] = J.matvec(np.arange(x.size) + 1.0)
                out[f"jrmv{D}_{use_prior}_{loss}"] = J.rmatvec(
                    np.linspace(-1, 1, rp.n_residuals)
                )
            for method in ("rr", "ols"):
                pp = kn.ProjectedProblem(X, R, 4, NumpyEngine(1), p, method)
                for mode in ("fast", "exact"):
                    for j, a in enumerate(pp.cost_grad_hess(x, mode=mode)):
                        out[f"proj{D}_{use_prior}_{method}_{mode}_{j}"] = np.asarray(a)
                lb, ub = kn.bounds(R, 4, 0.02, 1.8)
                for mode in ("fast", "exact"):
                    pp = kn.ProjectedProblem(X, R, 4, NumpyEngine(1), p, method)
                    xs, cost, status, nfev = kn.bounded_lm(
                        pp, x, (lb, ub), "linear", np.ones(x.size), mode=mode
                    )
                    out[f"blm{D}_{use_prior}_{method}_{mode}"] = np.concatenate(
                        [xs, [cost, status, nfev]]
                    )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for method, loss in (
                ("auto", "linear"),
                ("bounded-lm", "linear"),
                ("auto", "soft_l1"),
                ("auto", "linear+soft_l1"),
                ("trf", "linear"),
            ):
                for mode in ("fast", "exact"):
                    m = TFA(
                        K=4, nlss_method=method, nlss_loss=loss, mode=mode,
                        random_state=1, max_iter=30,
                    ).fit(X, R)  # fmt: skip
                    key = f"tfa{D}_{method}_{loss}_{mode}"
                    out[key] = np.concatenate(
                        [m.centers_.ravel(), m.widths_, m.weights_.ravel(), [m.cost_]]
                    )
            m = TFA(K=4, voxel_ratio=0.6, tr_ratio=0.7, random_state=2).fit(X, R)
            out[f"tfa{D}_sub"] = np.concatenate(
                [m.centers_.ravel(), m.widths_, m.objective_history_]
            )
            m = TFA(K=4, random_state=3, mode="exact").fit(X, R, prior=prior)
            out[f"tfa{D}_prior"] = np.concatenate([m.centers_.ravel(), m.widths_])
        mask = R[np.linalg.norm(R - R.mean(0), axis=1) < 5]
        for j, a in enumerate(effective_nodes(C, w, mask)[:2]):
            out[f"eff{D}_{j}"] = a
    Xs, Rs, _ = make_htfa_data(3, (11, 10), 25, K=3, noise=0.3, random_state=5)
    out["htfa_data"] = np.stack(Xs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for update in ("brainiak", "fixed", "empirical"):
            for mode in ("fast", "exact"):
                for n_jobs in (None, 2):
                    h = HTFA(
                        K=3, max_global_iter=4, max_local_iter=5,
                        template_update=update, mode=mode, n_jobs=n_jobs,
                        random_state=0,
                    ).fit(Xs, Rs)  # fmt: skip
                    key = f"htfa_{update}_{mode}_{n_jobs}"
                    out[key] = np.concatenate(
                        [
                            h.global_centers_.ravel(),
                            h.global_widths_,
                            h.global_centers_cov_.ravel(),
                            h.global_widths_var_,
                            np.concatenate([c.ravel() for c in h.centers_]),
                            np.concatenate(h.widths_),
                            np.concatenate([a.ravel() for a in h.weights_]),
                            h.objective_history_,
                        ]
                    )
    return out


def compare(a_path: str, b_path: str) -> int:
    a, b = np.load(a_path), np.load(b_path)
    bad = sorted(set(a.files) ^ set(b.files))
    for key in sorted(set(a.files) & set(b.files)):
        if a[key].shape != b[key].shape or not np.array_equal(
            a[key], b[key], equal_nan=True
        ):
            bad.append(key)
    print(f"{len(set(a.files) & set(b.files))} arrays compared; {len(bad)} differ")
    for key in bad:
        print("  differs:", key)
    return 1 if bad else 0


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    np.savez(sys.argv[1], **battery())
    print("wrote", sys.argv[1])
