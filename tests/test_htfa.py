"""Tests for the HTFA estimator."""

import time

import numpy as np

from htfa._matching import match_factors
from htfa.htfa import HTFA, assign_posterior, map_update
from htfa.synthetic import make_htfa_data

# Tolerances: measured values (seed 0) are recorded with ~2-3x headroom.
# Global error is measured against the mean of the true subject centers.
GLOBAL_CENTER_TOL_MM = 0.3  # measured: 0.119 (seed 0, mixed T; 3D 0.050, 2D 0.054)
LOCAL_CENTER_TOL_MM = 0.1  # measured: 0.034 (seed 0, 2D subject 0)
LOCAL_CENTER_TOL_T1_MM = 1.0  # measured: 0.369 (seed 0, the T=1 subject)
WEIGHT_MIN_R = 0.9


def test_map_update_matches_closed_form() -> None:
    # Scalar Gaussian conjugate update: prior N(m0, s0), observations whose mean
    # has variance g (= global_var / n_subj). Posterior precision 1/s0 + 1/g.
    m0, s0, g = 2.0, 4.0, 1.0
    obs = np.array([[1.0], [3.0], [5.0], [-1.0]])  # (n_subj, D=1), mean 2.0
    obs = obs + 1.0  # mean 3.0
    post_mean, post_cov = map_update(
        np.array([m0]), np.array([[s0]]), np.array([[g]]), obs
    )
    expected_var = 1.0 / (1.0 / s0 + 1.0 / g)
    expected_mean = expected_var * (m0 / s0 + 3.0 / g)
    print("posterior", post_mean, post_cov, "expected", expected_mean, expected_var)
    np.testing.assert_allclose(post_mean, [expected_mean], rtol=1e-12)
    np.testing.assert_allclose(post_cov, [[expected_var]], rtol=1e-12)


def test_map_update_multivariate_matches_precision_form() -> None:
    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 3))
    S0 = A @ A.T + np.eye(3)
    B = rng.normal(size=(3, 3))
    G = B @ B.T + np.eye(3)
    m0 = rng.normal(size=3)
    obs = rng.normal(size=(5, 3))
    mean, cov = map_update(m0, S0, G, obs)
    P = np.linalg.inv(S0) + np.linalg.inv(G)
    exp_cov = np.linalg.inv(P)
    exp_mean = exp_cov @ (np.linalg.solve(S0, m0) + np.linalg.solve(G, obs.mean(0)))
    np.testing.assert_allclose(mean, exp_mean, rtol=1e-10)
    np.testing.assert_allclose(cov, exp_cov, rtol=1e-10)


def test_global_assignment_permutes_all_fields() -> None:
    rng = np.random.default_rng(0)
    K, D = 5, 3
    centers = rng.uniform(0, 50, size=(K, D))
    widths = rng.uniform(1, 10, size=K)
    cov = np.stack([np.eye(D) * (k + 1) for k in range(K)])
    var = np.arange(1.0, K + 1)
    perm = np.array([3, 0, 4, 1, 2])
    noisy = centers[perm] + rng.normal(scale=0.01, size=(K, D))
    out = assign_posterior(centers, noisy, widths[perm], cov[perm], var[perm])
    print("assigned centers error", np.max(np.abs(out[0] - centers)))
    np.testing.assert_allclose(out[0], centers, atol=0.05)
    np.testing.assert_array_equal(out[1], widths)
    np.testing.assert_array_equal(out[2], cov)
    np.testing.assert_array_equal(out[3], var)


def _check_recovery(model: HTFA, Rs: list, truth: dict, Xs: list) -> None:
    # Target: mean of the true subject centers (truth keeps the global order), the
    # best any estimator can do from these subjects; also report the generator's.
    target = np.mean(truth["centers"], axis=0)
    g_perm = match_factors(target, model.global_centers_)
    est = model.global_centers_[g_perm]
    g_err = np.max(np.linalg.norm(est - target, axis=1))
    gen_err = np.max(np.linalg.norm(est - truth["global_centers"], axis=1))
    print(f"global center max error: {g_err:.4f} (to generator: {gen_err:.4f})")
    assert g_err < GLOBAL_CENTER_TOL_MM
    for s in range(len(Xs)):
        perm = match_factors(truth["centers"][s], model.centers_[s])
        err = np.max(
            np.linalg.norm(model.centers_[s][perm] - truth["centers"][s], axis=1)
        )
        W = model.weights_[s][perm]
        if Xs[s].shape[1] > 1:
            rs = [
                np.corrcoef(W[k], truth["weights"][s][k])[0, 1] for k in range(model.K)
            ]
            print(f"subject {s}: center err {err:.4f}, min weight r {min(rs):.4f}")
            assert min(rs) > WEIGHT_MIN_R
            assert err < LOCAL_CENTER_TOL_MM
        else:
            print(f"subject {s} (T=1): center err {err:.4f}")
            assert err < LOCAL_CENTER_TOL_T1_MM


def test_htfa_recovers_global_template() -> None:
    Xs, Rs, truth = make_htfa_data(
        n_subjects=4, grid_shape=(15, 15, 15), n_samples=80, K=6, random_state=0
    )
    t0 = time.perf_counter()
    model = HTFA(K=6, standardize=False, random_state=0).fit(Xs, Rs)
    print(f"fit time {time.perf_counter() - t0:.2f}s, n_iter {model.n_iter_}")
    assert model.global_centers_.shape == (6, 3)
    assert model.global_widths_.shape == (6,)
    assert model.global_centers_cov_.shape == (6, 3, 3)
    assert model.global_widths_var_.shape == (6,)
    assert len(model.centers_) == 4 and model.weights_[0].shape == (6, 80)
    _check_recovery(model, Rs, truth, Xs)


def test_htfa_2d() -> None:
    Xs, Rs, truth = make_htfa_data(
        n_subjects=4, grid_shape=(30, 30), n_samples=80, K=6, random_state=0
    )
    model = HTFA(K=6, standardize=False, random_state=0).fit(Xs, Rs)
    _check_recovery(model, Rs, truth, Xs)


def test_htfa_mixed_sample_counts() -> None:
    Xs, Rs, truth = make_htfa_data(
        n_subjects=3, grid_shape=(15, 15, 15), n_samples=80, K=6, random_state=0
    )
    Xs[2] = Xs[2][:, :1]
    truth["weights"][2] = truth["weights"][2][:, :1]
    model = HTFA(K=6, standardize=False, random_state=0).fit(Xs, Rs)
    assert model.weights_[2].shape == (6, 1)
    _check_recovery(model, Rs, truth, Xs)


def test_htfa_parallel_equals_serial() -> None:
    Xs, Rs, _ = make_htfa_data(
        n_subjects=3, grid_shape=(12, 12, 12), n_samples=40, K=4, random_state=1
    )
    a = HTFA(K=4, standardize=False, n_jobs=1, random_state=3).fit(Xs, Rs)
    b = HTFA(K=4, standardize=False, n_jobs=2, random_state=3).fit(Xs, Rs)
    np.testing.assert_array_equal(a.global_centers_, b.global_centers_)
    np.testing.assert_array_equal(a.global_widths_, b.global_widths_)
    for s in range(3):
        np.testing.assert_array_equal(a.centers_[s], b.centers_[s])
        np.testing.assert_array_equal(a.weights_[s], b.weights_[s])


def test_htfa_different_voxel_counts_per_subject() -> None:
    Xs, Rs, truth = make_htfa_data(
        n_subjects=3, grid_shape=(15, 15, 15), n_samples=80, K=6, random_state=0
    )
    # Drop one edge slab from subject 1 and two from subject 2 (different sides).
    for s, (axis, keep) in {1: (0, lambda r: r > 0), 2: (1, lambda r: r < 13)}.items():
        mask = keep(Rs[s][:, axis])
        Xs[s], Rs[s] = Xs[s][mask], Rs[s][mask]
    assert len({X.shape[0] for X in Xs}) == 3
    model = HTFA(K=6, standardize=False, random_state=0).fit(Xs, Rs)
    _check_recovery(model, Rs, truth, Xs)
