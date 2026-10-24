"""Tests for the synthetic ground-truth generators."""

import numpy as np
import pytest
from scipy.spatial.distance import pdist

from htfa._kernels import factors
from htfa.synthetic import make_htfa_data, make_tfa_data, match_factors

GRID = (10, 10, 10)


def test_tfa_shapes_and_truth_keys():
    X, R, truth = make_tfa_data(GRID, n_samples=30, K=5, random_state=0)
    assert X.shape == (1000, 30) and R.shape == (1000, 3)
    assert truth["centers"].shape == (5, 3)
    assert truth["widths"].shape == (5,)
    assert truth["weights"].shape == (5, 30)
    assert X.dtype == R.dtype == np.float64


def test_tfa_2d_and_spacing():
    X, R, truth = make_tfa_data((8, 6), n_samples=10, K=3, spacing=2.0, random_state=1)
    assert R.shape == (48, 2) and X.shape == (48, 10)
    assert R.max(axis=0).tolist() == [14.0, 10.0]
    assert truth["centers"].min() >= 0.1 * 10.0 - 1e-12


def test_tfa_deterministic():
    a = make_tfa_data(GRID, n_samples=20, K=4, random_state=7)
    b = make_tfa_data(GRID, n_samples=20, K=4, random_state=7)
    c = make_tfa_data(GRID, n_samples=20, K=4, random_state=8)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[2]["centers"], b[2]["centers"])
    assert not np.array_equal(a[0], c[0])


def test_tfa_centers_separated_and_central():
    _, _, truth = make_tfa_data(GRID, n_samples=5, K=10, random_state=3)
    c, w = truth["centers"], truth["widths"]
    assert pdist(c).min() >= 0.5 * np.sqrt(w.mean()) - 1e-12
    assert c.min() >= 0.1 * 9 - 1e-12 and c.max() <= 0.9 * 9 + 1e-12
    sigma = np.sqrt(w / 2) / 9
    assert sigma.min() >= 0.06 - 1e-12 and sigma.max() <= 0.12 + 1e-12


def test_tfa_noise_std():
    noise = 0.3
    X, R, truth = make_tfa_data(GRID, n_samples=100, K=6, noise=noise, random_state=2)
    resid = X - factors(R, truth["centers"], truth["widths"]) @ truth["weights"]
    assert abs(resid.std() / noise - 1) < 0.05


def test_tfa_impossible_separation_raises():
    with pytest.raises(ValueError, match="separat"):
        make_tfa_data((2, 2), n_samples=5, K=1000, random_state=0)


def test_htfa_shapes_determinism_and_clipping():
    kw = dict(n_subjects=3, grid_shape=GRID, n_samples=20, K=4, center_jitter=3.0)
    Xs, Rs, truth = make_htfa_data(**kw, random_state=5)
    Xs2, _, truth2 = make_htfa_data(**kw, random_state=5)
    assert len(Xs) == len(Rs) == 3
    assert all(x.shape == (1000, 20) for x in Xs)
    assert truth["global_centers"].shape == (4, 3)
    assert truth["global_widths"].shape == (4,)
    assert len(truth["centers"]) == len(truth["widths"]) == len(truth["weights"]) == 3
    assert truth["weights"][0].shape == (4, 20)
    np.testing.assert_array_equal(Xs[1], Xs2[1])
    np.testing.assert_array_equal(truth["centers"][2], truth2["centers"][2])
    for c in truth["centers"]:
        assert c.min() >= 0.0 and c.max() <= 9.0
    assert not np.allclose(truth["centers"][0], truth["global_centers"])


def test_htfa_zero_jitter_matches_global():
    _, _, truth = make_htfa_data(
        n_subjects=2, grid_shape=GRID, n_samples=5, K=3,
        center_jitter=0.0, width_jitter=0.0, random_state=4,
    )  # fmt: skip
    for s in range(2):
        np.testing.assert_allclose(truth["centers"][s], truth["global_centers"])
        np.testing.assert_allclose(truth["widths"][s], truth["global_widths"])


def test_match_factors_recovers_shuffle():
    _, _, truth = make_tfa_data(GRID, n_samples=5, K=8, random_state=6)
    true_c = truth["centers"]
    shuffle = np.random.default_rng(0).permutation(8)
    est = true_c[shuffle] + 1e-3
    perm = match_factors(true_c, est)
    np.testing.assert_allclose(est[perm], true_c, atol=1e-2)
    assert sorted(perm.tolist()) == list(range(8))
