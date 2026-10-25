"""Tests for the single-subject TFA estimator and data standardization."""

import time
import warnings

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning

from htfa._kernels import Prior, bounds, factors, max_sigma
from htfa.synthetic import make_tfa_data, match_factors
from htfa.tfa import TFA, standardize_data

# Recovery tolerances: about 2-3x the error measured on seed 0 (standardize=False,
# init='auto'; synthetic data stand for already z-scored patterns).
CENTER_TOL_3D = 0.04  # measured: 0.0164 (seed 0)
WIDTH_RTOL_3D = 0.008  # measured: 0.00312 (seed 0)
CENTER_TOL_2D = 0.03  # measured: 0.0120 (seed 0)
WIDTH_RTOL_2D = 0.025  # measured: 0.0101 (seed 0)
CENTER_TOL_SUB = 0.25  # measured: 0.100 (seed 0; stops on plateau after 20 iter)
WIDTH_RTOL_SUB = 0.025  # measured: 0.00995 (seed 0)
MIN_WEIGHT_R = 0.95  # brief: every matched weight row correlates at r > 0.95
EXACT_ATOL = 1e-6  # layout/dtype invariance (controller ruling)
IMPROVEMENT = (
    5.0  # final error <= init error / 5 (measured ratios: 64 centers, 30 widths)
)
TRUTH_OBJ_RTOL = 1e-3  # final vs truth-started objective; measured: 1.2e-5 (seed 0)
NEGLIGIBLE_COV = 1e12  # prior (co)variance that makes the penalty negligible
# (30,30) K=2 helper data, seed 0:
VARIANT_CENTER_TOL = 0.1  # linear loss / ols weights; measured: 0.0452 (seed 0)
VARIANT_WIDTH_RTOL = 0.03  # linear loss / ols weights; measured: 0.0133 (seed 0)
RECON_STD_MAX = 1.0  # z-scored data have std 1; measured reconstruction: 0.395
PRIOR_PULL_RATIO = 1e-5  # tight-prior / free distance to template; measured: 1.7e-6


def _errors(centers, widths, truth):
    """Max center error and median relative width error after matching."""
    perm = match_factors(truth["centers"], centers)
    c_err = np.linalg.norm(centers[perm] - truth["centers"], axis=1)
    w_rel = np.abs(widths[perm] - truth["widths"]) / truth["widths"]
    return float(c_err.max()), float(np.median(w_rel))


def _recovery_errors(model, truth):
    """Max center error, median relative width error, min weight correlation."""
    c_err, w_err = _errors(model.centers_, model.widths_, truth)
    perm = match_factors(truth["centers"], model.centers_)
    r = [
        np.corrcoef(est, true)[0, 1]
        for est, true in zip(model.weights_[perm], truth["weights"], strict=True)
    ]
    print(
        f"max center err {c_err:.4g}, median width rel err {w_err:.4g}, min weight "
        f"r {min(r):.6f}, n_iter {model.n_iter_}, converged {model.converged_}, "
        f"init {model.init_}, objectives {model.init_objectives_}, "
        f"engine {model.engine_}"
    )
    return c_err, w_err, min(r)


def _fit_and_score(X, R, truth, **kwargs):
    start = time.perf_counter()
    model = TFA(
        K=truth["centers"].shape[0], standardize=False, random_state=0, **kwargs
    ).fit(X, R)
    print(f"fit time {time.perf_counter() - start:.2f} s")
    assert model.converged_reason_ in ("threshold", "plateau")
    assert model.converged_
    return _recovery_errors(model, truth)


def _negligible_prior(centers, widths):
    """A prior whose penalty is negligible: a private way to choose the start."""
    K, D = centers.shape
    return Prior(
        centers,
        widths,
        np.tile(NEGLIGIBLE_COV * np.eye(D), (K, 1, 1)),
        np.full(K, NEGLIGIBLE_COV),
        1.0,
    )


def test_tfa_recovers_known_parameters_3d():
    # Measured 42.7 s with init='auto' (three candidate fits) on an M2 Max.
    X, R, truth = make_tfa_data((20, 20, 20), 100, K=8, noise=0.1, random_state=0)
    c_err, w_err, r = _fit_and_score(X, R, truth)
    assert c_err < CENTER_TOL_3D
    assert w_err < WIDTH_RTOL_3D
    assert r > MIN_WEIGHT_R


def _data_2d():
    return make_tfa_data((40, 40), 100, K=6, noise=0.1, random_state=0)


def test_tfa_recovers_known_parameters_2d():
    X, R, truth = _data_2d()
    c_err, w_err, r = _fit_and_score(X, R, truth)
    assert c_err < CENTER_TOL_2D
    assert w_err < WIDTH_RTOL_2D
    assert r > MIN_WEIGHT_R


def test_tfa_subsampling_still_recovers():
    X, R, truth = make_tfa_data((20, 20, 20), 100, K=8, noise=0.1, random_state=0)
    # A fresh subsample every iteration keeps the scaled change above 1e-3, so the
    # fit stops on a plateau instead; it must not warn.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        c_err, w_err, r = _fit_and_score(X, R, truth, voxel_ratio=0.25, tr_ratio=0.5)
    assert c_err < CENTER_TOL_SUB
    assert w_err < WIDTH_RTOL_SUB
    assert r > MIN_WEIGHT_R


def test_inference_improves_on_initialization():
    """The start alone is evaluated with ``init=<winner>, max_iter=0``."""
    X, R, truth = _data_2d()
    kw = {"K": 6, "standardize": False, "random_state": 0}
    final = TFA(**kw).fit(X, R)
    start = TFA(init=final.init_, max_iter=0, **kw).fit(X, R)
    at_truth = TFA(**kw).fit(
        X, R, prior=_negligible_prior(truth["centers"], truth["widths"])
    )
    c0, w0 = _errors(start.centers_, start.widths_, truth)
    c1, w1 = _errors(final.centers_, final.widths_, truth)
    rel = (final.cost_ - at_truth.cost_) / at_truth.cost_
    print(
        f"init {final.init_}: center err {c0:.4g} -> {c1:.4g}, width err "
        f"{w0:.4g} -> {w1:.4g}, objective {start.cost_:.6g} -> {final.cost_:.6g}, "
        f"truth-started {at_truth.cost_:.6g} (rel diff {rel:.3g})"
    )
    assert start.n_iter_ == 0
    assert c1 <= c0 / IMPROVEMENT
    assert w1 <= w0 / IMPROVEMENT
    assert final.cost_ < start.cost_
    assert abs(rel) <= TRUTH_OBJ_RTOL


@pytest.mark.parametrize("width_factor", [0.5, 2.0])
def test_basin_of_attraction(width_factor):
    X, R, truth = _data_2d()
    rng = np.random.default_rng(0)
    direction = rng.standard_normal(truth["centers"].shape)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    sigma = np.sqrt(truth["widths"] / 2.0)
    prior = _negligible_prior(
        truth["centers"] + sigma[:, None] * direction, width_factor * truth["widths"]
    )
    model = TFA(K=6, standardize=False, random_state=0).fit(X, R, prior=prior)
    c_err, w_err, r = _recovery_errors(model, truth)
    assert model.init_ == "prior"
    assert c_err < CENTER_TOL_2D
    assert w_err < WIDTH_RTOL_2D
    assert r > MIN_WEIGHT_R


def _small_data(random_state=0, T=20):
    return make_tfa_data((30, 30), T, K=2, noise=0.1, random_state=random_state)


def test_init_kmeans_is_brainiak_start():
    X, R, _ = _small_data()
    kw = {"K": 2, "init": "kmeans", "max_iter": 0, "random_state": 0}
    a = TFA(**kw).fit(X, R)
    b = TFA(**kw).fit(X, R)
    assert a.init_ == "kmeans" and list(a.init_objectives_) == ["kmeans"]
    assert a.centers_.shape == (2, 2) and a.weights_.shape == (2, 20)
    np.testing.assert_array_equal(a.widths_, max_sigma(R))
    np.testing.assert_array_equal(a.centers_, b.centers_)
    assert a.n_iter_ == 0 and not a.converged_


def test_init_auto_records_objectives():
    X, R, _ = _small_data()
    model = TFA(K=2, standardize=False, random_state=0).fit(X, R)
    print("objectives", model.init_objectives_, "winner", model.init_)
    assert list(model.init_objectives_) == ["ncorr", "power", "kmeans"]
    assert model.init_ == min(model.init_objectives_, key=model.init_objectives_.get)
    assert model.cost_ == model.init_objectives_[model.init_]
    sub = TFA(K=2, init=["kmeans", "ncorr"], random_state=0).fit(X, R)
    assert list(sub.init_objectives_) == ["kmeans", "ncorr"]

    # Uniform power (every voxel has the same sum of squares): 'power' is skipped.
    signs = np.where(np.random.default_rng(0).random(X.shape) < 0.5, -1.0, 1.0)
    flat = TFA(K=2, standardize=False, random_state=0, max_iter=1, threshold=1e9)
    flat.fit(signs, R)
    assert list(flat.init_objectives_) == ["ncorr", "kmeans"]
    # One sample: 'ncorr' falls back to 'power', which runs anyway.
    single = TFA(K=2, standardize=False, random_state=0, max_iter=1, threshold=1e9)
    single.fit(X[:, :1], R)
    assert list(single.init_objectives_) == ["power", "kmeans"]
    # Per-voxel z-scoring makes every sum_t X**2 equal T: 'power' is skipped.
    zscored = TFA(K=2, random_state=0, max_iter=1, threshold=1e9).fit(X, R)
    assert list(zscored.init_objectives_) == ["ncorr", "kmeans"]
    alone = TFA(K=2, init="ncorr", standardize=False, max_iter=0, random_state=0)
    assert list(alone.fit(X[:, :1], R).init_objectives_) == ["ncorr"]
    for bad in ("random", ["kmeans", "kmeans"], [], 3):
        with pytest.raises(ValueError, match="init must be"):
            TFA(K=2, init=bad).fit(X, R)


def test_tfa_rejects_k_not_less_than_voxels():
    X, R, _ = make_tfa_data((3, 3), 5, K=1, random_state=0)
    with pytest.raises(ValueError, match=r"K=9 must be smaller than the number of "):
        TFA(K=9).fit(X, R)
    with pytest.raises(ValueError, match=r"K=10 must be smaller .* voxels \(9\)"):
        TFA(K=10).fit(X, R)


def test_tfa_validation_messages():
    X, R, _ = _small_data()
    with pytest.raises(ValueError, match="X must be 2D"):
        TFA(K=2).fit(X[:, 0], R)
    with pytest.raises(ValueError, match="coords must have shape"):
        TFA(K=2).fit(X, R[:-1])
    with pytest.raises(ValueError, match="coords must have shape"):
        TFA(K=2).fit(X, R[:, 0])
    with pytest.raises(ValueError, match="2 or 3 dimensions"):
        TFA(K=2).fit(X, np.column_stack([R, R]))
    bad = X.copy()
    bad[3, 4] = np.nan
    with pytest.raises(ValueError, match="X contains NaN or Inf"):
        TFA(K=2).fit(bad, R)
    bad = X.copy()
    bad[0, 0] = np.inf
    with pytest.raises(ValueError, match="X contains NaN or Inf"):
        TFA(K=2).fit(bad, R)
    bad_R = R.copy()
    bad_R[2, 1] = np.nan
    with pytest.raises(ValueError, match="coords contains NaN or Inf"):
        TFA(K=2).fit(X, bad_R)
    with pytest.raises(ValueError, match="K must be a positive integer"):
        TFA(K=0).fit(X, R)
    with pytest.raises(ValueError, match="max_iter must be >= 0"):
        TFA(K=2, max_iter=-1).fit(X, R)
    with pytest.raises(ValueError, match="weight method"):
        TFA(K=2, weight_method="lasso").fit(X, R)
    with pytest.raises(ValueError, match="nlss_loss must be one of"):
        TFA(K=2, nlss_loss="l2").fit(X, R)
    with pytest.raises(ValueError, match="standardize must be"):
        TFA(K=2, standardize="yes").fit(X, R)
    prior = Prior(
        np.zeros((3, 2)), np.ones(3), np.tile(np.eye(2), (3, 1, 1)), np.ones(3), 1.0
    )
    with pytest.raises(ValueError, match=r"prior.centers must have shape \(2, 2\)"):
        TFA(K=2).fit(X, R, prior=prior)


def test_tfa_accepts_float32_and_strided_views():
    X, R, _ = _small_data()
    kw = {"K": 2, "random_state": 0, "backend": "numpy"}
    ref = TFA(**kw).fit(X, R)

    big = np.zeros((X.shape[0], 2 * X.shape[1]))
    big[:, ::2] = X
    strided = big[:, ::2]
    assert not strided.flags.c_contiguous and not strided.flags.f_contiguous
    fortran = np.asfortranarray(X)
    assert fortran.flags.f_contiguous and not fortran.flags.c_contiguous
    for variant in (strided, fortran):
        fit = TFA(**kw).fit(variant, R)
        for attr in ("centers_", "widths_", "weights_"):
            np.testing.assert_allclose(
                getattr(fit, attr), getattr(ref, attr), rtol=0, atol=EXACT_ATOL
            )

    X32 = X.astype(np.float32)
    fit32 = TFA(**kw).fit(X32, R)
    fit64 = TFA(**kw).fit(X32.astype(np.float64), R)
    for attr in ("centers_", "widths_", "weights_"):
        assert getattr(fit32, attr).dtype == np.float64
        np.testing.assert_allclose(
            getattr(fit32, attr), getattr(fit64, attr), rtol=0, atol=EXACT_ATOL
        )


def test_tfa_deterministic_with_random_state():
    X, R, _ = _small_data()
    kw = {"K": 2, "voxel_ratio": 0.5, "tr_ratio": 0.5, "max_iter": 4}
    with pytest.warns(ConvergenceWarning):
        a = TFA(random_state=0, **kw).fit(X, R)
        b = TFA(random_state=0, **kw).fit(X, R)
        gen = TFA(random_state=np.random.default_rng(0), **kw).fit(X, R)
        other = TFA(random_state=1, **kw).fit(X, R)
    for attr in ("centers_", "widths_", "weights_"):
        np.testing.assert_array_equal(getattr(a, attr), getattr(b, attr))
    assert a.cost_ == b.cost_ and a.n_iter_ == b.n_iter_
    assert a.init_objectives_ == b.init_objectives_
    np.testing.assert_array_equal(gen.centers_, a.centers_)
    assert not np.array_equal(other.centers_, a.centers_)


def test_standardize_modes():
    X = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 5.0], [0.0, 0.0, 2.0, 2.0]])
    for mode in ("auto", True):
        Z = standardize_data(X, mode)
        np.testing.assert_allclose(Z.mean(axis=1), 0, atol=1e-15)
        np.testing.assert_allclose(Z.std(axis=1), [1, 0, 1], atol=1e-15)
        np.testing.assert_array_equal(Z[1], 0)
        np.testing.assert_allclose(Z[2], [-1, -1, 1, 1])
    np.testing.assert_array_equal(standardize_data(X, False), X)

    col = np.array([[1.0], [3.0], [5.0]])
    np.testing.assert_array_equal(standardize_data(col, "auto"), col)
    np.testing.assert_array_equal(standardize_data(col, False), col)
    z = standardize_data(col, True)
    np.testing.assert_allclose(z.ravel(), (col.ravel() - 3.0) / np.std(col))
    np.testing.assert_array_equal(standardize_data(np.full((4, 1), 2.0), True), 0)

    # Near-constant rows (std from rounding only) are exactly constant: become 0.
    const = np.full((2, 5), 0.1)
    np.testing.assert_array_equal(standardize_data(const, "auto"), 0)
    X32 = X.astype(np.float32)
    assert standardize_data(X32, "auto").dtype == np.float64
    assert standardize_data(X32, False).dtype == np.float64
    with pytest.raises(ValueError, match="standardize must be"):
        standardize_data(X, "zscore")


@pytest.mark.parametrize("name", ["ncorr", "power", "kmeans"])
def test_init_alone_matches_its_auto_candidate(name):
    X, R, _ = _small_data()
    kw = {"K": 2, "standardize": False, "random_state": 0}
    auto = TFA(n_screen_iter=0, **kw).fit(X, R)
    alone = TFA(init=name, **kw).fit(X, R)
    print(name, "alone", alone.cost_, "in auto", auto.init_objectives_[name])
    assert alone.cost_ == auto.init_objectives_[name]
    if name == auto.init_:
        np.testing.assert_array_equal(alone.centers_, auto.centers_)


def test_screening_keeps_the_full_multistart_winner():
    X, R, _ = _small_data()
    kw = {"K": 2, "standardize": False, "random_state": 0}
    full = TFA(n_screen_iter=0, **kw).fit(X, R)
    screened = TFA(n_screen_iter=3, **kw).fit(X, R)
    print("full", full.init_objectives_, "screened", screened.init_objectives_)
    assert screened.init_ == full.init_
    # The winner continues from its screened state: the same trajectory, bit for bit.
    np.testing.assert_array_equal(screened.centers_, full.centers_)
    np.testing.assert_array_equal(screened.weights_, full.weights_)
    assert screened.cost_ == full.cost_ == screened.init_objectives_[screened.init_]
    assert screened.n_iter_ == full.n_iter_
    for name, objective in screened.init_objectives_.items():
        if name == screened.init_:
            continue
        # A screened-out candidate's objective is the one after 3 iterations.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            short = TFA(init=name, max_iter=3, **kw).fit(X, R)
        assert objective == short.cost_
        assert objective > screened.cost_
    with pytest.raises(ValueError, match="n_screen_iter must be an integer >= 0"):
        TFA(K=2, n_screen_iter=-1).fit(X, R)


def test_tfa_standardized_path_runs():
    X, R, _ = _small_data()
    model = TFA(K=2, random_state=0).fit(5.0 + 3.0 * X, R)
    lb, ub = bounds(R, 2, model.lower_ratio, model.upper_ratio)
    x = np.concatenate([model.centers_.ravel(), model.widths_])
    print("z-scored fit", model.centers_, model.widths_, model.init_objectives_)
    for attr in ("centers_", "widths_", "weights_"):
        assert np.all(np.isfinite(getattr(model, attr)))
    assert np.all(x >= lb) and np.all(x <= ub)
    # reconstruct() is in the standardized space: same scale as the z-scored data.
    rec = model.reconstruct()
    assert rec.shape == X.shape and np.std(rec) < RECON_STD_MAX


def test_brainiak_bug_sampling_clip():
    X, R, _ = make_tfa_data((10, 10, 5), 10, K=3, random_state=0)
    assert X.shape[0] == 500
    model = TFA(
        K=3, max_voxels=10**6, max_trs=10**6, random_state=0, max_iter=2, threshold=1e9
    )
    model.fit(X, R)
    assert np.all(np.isfinite(model.centers_))
    assert model.weights_.shape == (3, 10)


def test_tfa_fitted_attributes_and_factors():
    X, R, _ = _small_data()
    model = TFA(K=2, random_state=0, backend="numpy").fit(X, R)
    assert model.centers_.shape == (2, 2) and model.widths_.shape == (2,)
    assert model.weights_.shape == (2, X.shape[1])
    assert isinstance(model.n_iter_, int) and 1 <= model.n_iter_ <= model.max_iter
    assert model.converged_ is True and model.converged_reason_ == "threshold"
    assert isinstance(model.cost_, float) and model.cost_ > 0
    assert model.init_ in ("ncorr", "power", "kmeans")
    assert model.engine_ == "numpy"
    F = model.factors()
    np.testing.assert_array_equal(F, factors(R, model.centers_, model.widths_))
    np.testing.assert_array_equal(model.reconstruct(), F @ model.weights_)
    F2 = model.factors(R[:10])
    np.testing.assert_array_equal(F2, F[:10])
    with pytest.raises(ValueError, match="coords must have shape"):
        model.factors(np.column_stack([R, R[:, 0]])[:, :3])
    with pytest.raises(ValueError, match="coords contains NaN or Inf"):
        model.factors(np.full((2, 2), np.nan))


def test_tfa_unfitted_raises():
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        TFA().factors()
    with pytest.raises(NotFittedError):
        TFA().reconstruct()


def test_tfa_sklearn_params():
    model = TFA(K=7, voxel_ratio=0.3, random_state=4)
    params = model.get_params()
    assert params["K"] == 7 and params["voxel_ratio"] == 0.3
    assert params["random_state"] == 4 and params["standardize"] == "auto"
    assert params["init"] == "auto" and params["n_screen_iter"] == 0
    assert params["max_iter"] == 100 and params["threshold"] == 1e-3
    assert set(params) == {
        "K", "max_iter", "threshold", "nlss_method", "nlss_loss", "x_scale",
        "weight_method", "upper_ratio", "lower_ratio", "voxel_ratio", "tr_ratio",
        "max_voxels", "max_trs", "standardize", "init", "n_screen_iter", "backend",
        "random_state", "verbose",
    }  # fmt: skip
    assert clone(model).get_params() == params


def test_tfa_threshold_and_max_iter_control_stopping():
    X, R, _ = _small_data()
    loose = TFA(K=2, threshold=1e6, random_state=0).fit(X, R)
    assert loose.n_iter_ == 1 and loose.converged_
    with pytest.warns(ConvergenceWarning, match="did not converge in max_iter=2"):
        strict = TFA(K=2, threshold=0.0, max_iter=2, random_state=0).fit(X, R)
    assert strict.n_iter_ == 2 and not strict.converged_
    assert strict.converged_reason_ == "max_iter"


def test_tfa_linear_loss_and_ols():
    X, R, truth = _small_data()
    for kw in ({"nlss_loss": "linear"}, {"weight_method": "ols"}):
        model = TFA(K=2, standardize=False, random_state=0, **kw).fit(X, R)
        c_err, w_err = _errors(model.centers_, model.widths_, truth)
        print(kw, "max center err", c_err, "median width rel err", w_err)
        assert c_err < VARIANT_CENTER_TOL and w_err < VARIANT_WIDTH_RTOL


def test_tfa_with_prior_initializes_from_and_shrinks_to_template():
    X, R, truth = _small_data()
    K, D = truth["centers"].shape
    sigma = max_sigma(R)
    prior = Prior(
        centers=truth["centers"] + 2.0,
        widths=truth["widths"],
        centers_cov=np.tile(np.eye(D), (K, 1, 1)),
        widths_var=np.full(K, sigma),
        scale=123.0,
    )
    kw = {"K": K, "standardize": False, "random_state": 0}
    one = TFA(max_iter=1, threshold=1e9, **kw).fit(X, R, prior=prior)
    assert prior.scale == 123.0  # the caller's prior is not modified
    assert one.init_ == "prior" and list(one.init_objectives_) == ["prior"]
    # Initialized at the template: factors stay in template order.
    np.testing.assert_array_equal(match_factors(prior.centers, one.centers_), [0, 1])
    free = TFA(**kw).fit(X, R)
    tight = Prior(
        prior.centers, prior.widths, 1e-8 * prior.centers_cov, prior.widths_var, 1.0
    )
    pulled = TFA(**kw).fit(X, R, prior=tight)
    perm = match_factors(prior.centers, free.centers_)
    d_free = np.linalg.norm(free.centers_[perm] - prior.centers, axis=1)
    d_pulled = np.linalg.norm(pulled.centers_ - prior.centers, axis=1)
    print("distance to template: free", d_free, "with tight prior", d_pulled)
    assert np.all(d_pulled < PRIOR_PULL_RATIO * d_free)


def test_tfa_verbose_reports_iterations(capsys):
    X, R, _ = _small_data()
    TFA(K=2, init="kmeans", max_iter=2, threshold=1e9, verbose=True).fit(X, R)
    out = capsys.readouterr().out
    assert "TFA kmeans iteration 1: scaled change" in out


def test_tfa_explicit_start_overrides_init_and_prior_start():
    X, R, truth = _small_data()
    K, D = truth["centers"].shape
    start = (truth["centers"][::-1] + 0.5, truth["widths"][::-1])
    prior = Prior(
        centers=truth["centers"],
        widths=truth["widths"],
        centers_cov=np.tile(np.eye(D), (K, 1, 1)),
        widths_var=np.full(K, max_sigma(R)),
        scale=1.0,
    )
    kw = {"K": K, "standardize": False, "random_state": 0, "threshold": 1e9}
    for p in (None, prior):
        one = TFA(max_iter=1, **kw).fit(X, R, prior=p, start=start)
        print("start fit:", one.init_, one.centers_)
        assert one.init_ == "start" and list(one.init_objectives_) == ["start"]
        # Started from the reversed order, factors stay in that order.
        np.testing.assert_array_equal(match_factors(start[0], one.centers_), [0, 1])
    with pytest.raises(ValueError, match="start must be"):
        TFA(**kw).fit(X, R, start=(truth["centers"][:1], truth["widths"]))


def test_tfa_rejects_unknown_nlss_method():
    X, R, _ = _small_data()
    with pytest.raises(ValueError, match="nlss_method must be one of"):
        TFA(K=2, nlss_method="lm").fit(X, R)


def _photo_patch():
    """A real photograph (sklearn's china.jpg), block-averaged to 22 x 33 gray."""
    from sklearn.datasets import load_sample_image

    photo = load_sample_image("china.jpg").astype(np.float64)
    b = 19
    h, w = photo.shape[0] // b, photo.shape[1] // b
    small = photo[: h * b, : w * b].reshape(h, b, w, b, 3).mean(axis=(1, 3))
    gray = small @ np.array([0.299, 0.587, 0.114])
    R = np.stack(np.meshgrid(np.arange(h), np.arange(w), indexing="ij"), -1)
    return gray.reshape(-1, 1), R.reshape(-1, 2).astype(np.float64)


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize("standardize", ["auto", False])
def test_tfa_warns_on_raw_single_sample_image(standardize):
    X, R = _photo_patch()
    print(f"photo: V={X.shape[0]}, mean {X.mean():.1f}, var {X.var():.1f}")
    with pytest.warns(UserWarning, match="standardize=True"):
        TFA(K=3, max_iter=1, standardize=standardize, init="power").fit(X, R)


@pytest.mark.parametrize("standardize", ["auto", True])
def test_tfa_no_raw_scale_warning_on_zscored_image(standardize):
    X, R = _photo_patch()
    if standardize == "auto":
        X = (X - X.mean()) / X.std()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        warnings.simplefilter("ignore", ConvergenceWarning)  # a UserWarning too
        TFA(K=3, max_iter=1, standardize=standardize, init="power").fit(X, R)
