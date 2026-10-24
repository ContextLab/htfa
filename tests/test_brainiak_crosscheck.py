"""Cross-checks against the reference BrainIAK TFA/HTFA implementation.

Run in an environment with brainiak installed (see .github/workflows/brainiak.yml):
``pytest -m brainiak -v -s``. BrainIAK 0.12's HTFA fails on NumPy >= 2.4 (it assigns
a size-1 array to a scalar slot in ``_map_update_posterior``), so that environment
pins ``numpy<2.4``.
"""

import time

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from htfa._engines import NumpyEngine
from htfa._kernels import Prior, ResidualProblem, factors, ridge_weights
from htfa.htfa import HTFA
from htfa.synthetic import make_htfa_data, make_tfa_data, match_factors
from htfa.tfa import TFA

pytestmark = pytest.mark.brainiak

brainiak_tfa = pytest.importorskip(
    "brainiak.factoranalysis.tfa", reason="the BrainIAK cross-check needs brainiak"
)
brainiak_utils = pytest.importorskip(
    "brainiak.utils.utils", reason="the BrainIAK cross-check needs brainiak"
)

# BrainIAK's C++ ``recon`` accumulates X - F W in a single-precision ``float``, so
# the data residual agrees with ours only to about float32 precision.
RESIDUAL_RTOL = 1e-5
FACTORS_RTOL = 1e-12
# Ours may be at most 10% worse than BrainIAK on the same data (spec).
NOT_WORSE = 1.1
# Absolute bound on our max center error (mm); measured ~0.02 (TFA), ~0.06 (HTFA).
MAX_CENTER_ERR = 0.1
# Small problems keep each BrainIAK fit to seconds (a 20^3 grid, K=8 took ~280 s).
GRID, K, T = (12, 12, 12), 4, 50


def _brainiak_htfa():
    """BrainIAK's HTFA module; skips when mpi4py or its MPI library is missing.

    Importing mpi4py without an MPI library raises RuntimeError, not ImportError,
    which ``pytest.importorskip`` would not turn into a skip.
    """
    try:
        return pytest.importorskip(
            "brainiak.factoranalysis.htfa",
            reason="BrainIAK's HTFA needs brainiak with mpi4py",
        )
    except RuntimeError as err:
        pytest.skip(f"BrainIAK's HTFA needs an MPI library for mpi4py: {err}")


def _max_center_error(est: np.ndarray, true: np.ndarray) -> float:
    perm = match_factors(true, est)
    return float(np.max(np.linalg.norm(est[perm] - true, axis=1)))


def _brainiak_tfa(X: np.ndarray, R: np.ndarray, K: int):
    """A BrainIAK TFA whose layout fields are set as ``fit`` sets them."""
    model = brainiak_tfa.TFA(K=K, max_num_voxel=X.shape[0], max_num_tr=X.shape[1])
    model.n_dim = R.shape[1]
    model.cov_vec_size = int(np.sum(np.arange(model.n_dim) + 1))
    model.map_offset = model.get_map_offset()
    return model


def _zscored_problem(seed: int, full_cov: bool):
    """Globally z-scored data plus perturbed parameters, weights and a template.

    The template's center covariances are diagonal unless ``full_cov`` is set.
    """
    X, R, truth = make_tfa_data(GRID, T, K=K, noise=0.1, random_state=seed)
    X = (X - X.mean()) / X.std()
    rng = np.random.default_rng(seed)
    centers = truth["centers"] + rng.normal(scale=0.5, size=truth["centers"].shape)
    widths = truth["widths"] * rng.uniform(0.8, 1.2, size=K)
    W = ridge_weights(X, factors(R, centers, widths))
    if full_cov:
        A = rng.normal(size=(K, 3, 3))
        cov = A @ np.swapaxes(A, 1, 2) + np.eye(3)
    else:
        cov = np.stack([np.diag(d) for d in rng.uniform(0.5, 2.0, size=(K, 3))])
    template = Prior(
        centers=truth["centers"],
        widths=truth["widths"],
        centers_cov=cov,
        widths_var=rng.uniform(0.5, 2.0, size=K),
        scale=0.5,  # BrainIAK's sample_scaling with full data
    )
    return X, R, centers, widths, W, template


def _residuals(seed: int, full_cov: bool):
    """Our and BrainIAK's residual vectors at the same point, and the problem."""
    X, R, centers, widths, W, template = _zscored_problem(seed, full_cov)
    x = np.concatenate([centers.ravel(), widths])
    ours = ResidualProblem(X, W, R, NumpyEngine(), prior=template).residual(x)
    bk = _brainiak_tfa(X, R, K)
    bk.sample_scaling = template.scale
    unique_R, inds = bk.get_unique_R(R)
    tri = np.stack([brainiak_utils.from_sym_2_tri(c) for c in template.centers_cov])
    theirs = bk._residual_multivariate(
        x,
        unique_R,
        inds,
        X,
        W,
        template.centers,
        tri,
        template.widths.reshape(K, 1),
        1.0 / template.widths_var.reshape(K, 1),
        np.std(X) / np.sqrt(2.0),
    )
    assert theirs.shape == ours.shape == (X.size + 2 * K,)
    return ours, theirs, X, centers, template


def test_residual_matches_brainiak_on_zscored_data() -> None:
    # BrainIAK multiplies the data residual by std(X)/sqrt(2); ours divides by
    # sqrt(2) std(X). They coincide only when std(X) == 1, hence the z-scoring.
    ours, theirs, X, _, _ = _residuals(seed=0, full_cov=False)
    print(f"std(X) = {np.std(X)!r}")
    n = X.size
    scale = np.max(np.abs(ours[:n]))
    data_diff = np.max(np.abs(ours[:n] - theirs[:n])) / scale
    prior_diff = np.max(np.abs(np.abs(ours[n:]) - theirs[n:]) / theirs[n:])
    print(f"data residual: max |diff| / max |r| = {data_diff:.3e} (V*T = {n})")
    print(f"prior residuals (diagonal cov): max relative diff = {prior_diff:.3e}")
    np.testing.assert_allclose(ours[:n], theirs[:n], rtol=0, atol=RESIDUAL_RTOL * scale)
    # Our width residuals are signed (a smooth Jacobian); BrainIAK's are sqrt(d^2).
    np.testing.assert_allclose(np.abs(ours[n:]), theirs[n:], rtol=RESIDUAL_RTOL)


def test_brainiak_center_prior_reads_only_upper_triangle() -> None:
    # BrainIAK bug: utils.from_tri_2_sym fills only the upper triangle, so its
    # center prior is d^T triu(Sigma)^-1 d rather than d^T Sigma^-1 d. With a full
    # covariance the center residuals therefore differ; data and width parts agree.
    ours, theirs, X, centers, template = _residuals(seed=0, full_cov=True)
    n = X.size
    d = centers - template.centers
    s = template.scale
    sym = [
        np.sqrt(s * dk @ np.linalg.solve(c, dk))
        for dk, c in zip(d, template.centers_cov, strict=True)
    ]
    upper = [
        np.sqrt(s * dk @ np.linalg.solve(np.triu(c), dk))
        for dk, c in zip(d, template.centers_cov, strict=True)
    ]
    print(f"center residuals ours {ours[n : n + K]}, BrainIAK {theirs[n : n + K]}")
    print(
        f"d^T Sigma^-1 d form {np.array(sym)}, "
        f"d^T triu(Sigma)^-1 d form {np.array(upper)}"
    )
    np.testing.assert_allclose(ours[n : n + K], sym, rtol=1e-12)
    np.testing.assert_allclose(theirs[n : n + K], upper, rtol=1e-12)
    assert np.max(np.abs(ours[n : n + K] - theirs[n : n + K])) > 1e-3
    np.testing.assert_allclose(
        np.abs(ours[n + K :]), theirs[n + K :], rtol=RESIDUAL_RTOL
    )


def test_factors_match_brainiak() -> None:
    X, R, centers, widths, _, _ = _zscored_problem(seed=1, full_cov=False)
    bk = _brainiak_tfa(X, R, K)
    unique_R, inds = bk.get_unique_R(R)
    theirs = bk.get_factors(unique_R, inds, centers, widths.reshape(K, 1))
    ours = factors(R, centers, widths)
    print(f"factors: max relative diff = {np.max(np.abs(ours - theirs) / theirs):.3e}")
    np.testing.assert_allclose(ours, theirs, rtol=FACTORS_RTOL)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tfa_recovery_not_worse_than_brainiak(seed: int) -> None:
    X, R, truth = make_tfa_data(GRID, T, K=K, noise=0.1, random_state=seed)
    t0 = time.perf_counter()
    # BrainIAK defaults (2-point Jacobian, K-means start), on all voxels and TRs.
    theirs = brainiak_tfa.TFA(K=K, max_num_voxel=X.shape[0], max_num_tr=T).fit(X, R)
    t_theirs = time.perf_counter() - t0
    t0 = time.perf_counter()
    ours = TFA(K=K, standardize=False, random_state=0).fit(X, R)
    t_ours = time.perf_counter() - t0
    err_theirs = _max_center_error(
        theirs.get_centers(theirs.local_posterior_), truth["centers"]
    )
    err_ours = _max_center_error(ours.centers_, truth["centers"])
    print(
        f"seed {seed}: max center error ours {err_ours:.4f} ({t_ours:.1f} s, init "
        f"{ours.init_}), BrainIAK {err_theirs:.4f} ({t_theirs:.1f} s)"
    )
    assert err_ours <= NOT_WORSE * err_theirs
    assert err_ours < MAX_CENTER_ERR


# BrainIAK's own NumPy 1.25 deprecation (size-1 array to scalar), not ours.
@pytest.mark.filterwarnings(
    "ignore:Conversion of an array with ndim > 0 to a scalar:DeprecationWarning"
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_htfa_global_recovery_not_worse_than_brainiak(seed: int) -> None:
    brainiak_htfa = _brainiak_htfa()
    Xs, Rs, truth = make_htfa_data(3, GRID, T, K=K, noise=0.1, random_state=seed)
    target = np.mean(truth["centers"], axis=0)
    V = Xs[0].shape[0]
    # BrainIAK draws its template subject from the global NumPy RNG unseeded.
    np.random.seed(seed)  # noqa: NPY002 - BrainIAK reads the legacy global RNG
    t0 = time.perf_counter()
    theirs = brainiak_htfa.HTFA(
        K=K, n_subj=3, voxel_ratio=1.0, tr_ratio=1.0, max_voxel=V, max_tr=T
    ).fit(Xs, Rs)
    t_theirs = time.perf_counter() - t0
    t0 = time.perf_counter()
    # Measured (seeds 0-2): our template is still moving after the default 10 global
    # iterations, so exactly one ConvergenceWarning is expected; recovery is checked.
    with pytest.warns(ConvergenceWarning, match="HTFA did not converge"):
        ours = HTFA(K=K, standardize=False, random_state=0).fit(Xs, Rs)
    t_ours = time.perf_counter() - t0
    err_theirs = _max_center_error(theirs.get_centers(theirs.global_posterior_), target)
    err_ours = _max_center_error(ours.global_centers_, target)
    print(
        f"seed {seed}: max global center error ours {err_ours:.4f} ({t_ours:.1f} s, "
        f"{ours.n_iter_} global iter), BrainIAK {err_theirs:.4f} ({t_theirs:.1f} s)"
    )
    assert err_ours <= NOT_WORSE * err_theirs
    assert err_ours < MAX_CENTER_ERR
