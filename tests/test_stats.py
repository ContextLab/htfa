"""Independent formula checks and actual estimator parameter counts."""

import warnings

import numpy as np
import pytest

import htfa
from htfa.stats import (
    gaussian_fit_statistics,
    parameter_count,
    reconstruction_statistics,
)


def test_gaussian_known_likelihood_and_penalty():
    x = np.array([[0.0, 2.0, 0.0, 2.0], [0.0, 2.0, 0.0, 2.0]])
    s = gaussian_fit_statistics(x, np.ones_like(x), n_parameters=2)
    assert s.rss == s.tss == 8
    assert s.r2 == 0
    assert s.log_likelihood == pytest.approx(-4 * (np.log(2 * np.pi) + 1))
    assert s.aic == pytest.approx(4 - 2 * s.log_likelihood)
    assert s.aicc == pytest.approx(s.aic + 12 / 5)
    assert s.bic == pytest.approx(2 * np.log(8) - 2 * s.log_likelihood)
    bigger = gaussian_fit_statistics(x, np.ones_like(x), n_parameters=3)
    assert bigger.aic - s.aic == pytest.approx(2)


def test_edge_cases():
    x = np.ones((2, 3))
    s = gaussian_fit_statistics(x, x, n_parameters=6)
    assert np.isnan(s.r2)
    assert s.aic == -np.inf
    assert s.aicc == np.inf
    assert gaussian_fit_statistics(x, x * 0, n_parameters=1).rss == 6
    for p in [0, -1, True, 1.5]:
        with pytest.raises(ValueError):
            gaussian_fit_statistics(x, x, n_parameters=p)
    for y in [np.zeros((3, 2)), x * np.nan]:
        with pytest.raises(ValueError):
            gaussian_fit_statistics(x, y, n_parameters=1)
    with pytest.raises(ValueError):
        gaussian_fit_statistics([], [], n_parameters=1)


@pytest.mark.parametrize("kernel", ["rbf", "laplace", "mexican_hat"])
@pytest.mark.parametrize("m", [1, 2])
def test_fitted_count_and_scoring(kernel, m):
    """Parameter counts and scores of a fit stopped after one global iteration.

    The fit is cut short on purpose (only counts and scores are checked), so
    HTFA warns once that it did not converge; that warning is required. A
    stage-0 (unpenalized) subject fit may also reach TFA's max_iter=100 and
    warn (here: with 2 Mexican-hat components), since HTFA does not suppress
    those; whether it does depends on the numerical libraries, so up to one
    such TFA ConvergenceWarning per subject is allowed, not required, as is
    one numpy build's spurious matmul RuntimeWarning (see below). Any other
    warning fails the test.
    """
    xs, rs, _ = htfa.make_htfa_data(
        n_subjects=2, grid_shape=(8, 8), K=2, n_samples=15, random_state=1
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = htfa.HTFA(
            K=2, kernel=kernel, n_components=m, max_global_iter=1, random_state=0
        ).fit(xs, rs)
    got = sorted(
        (w.category.__name__, str(w.message).split(" did not converge")[0])
        for w in caught
    )
    print("warnings:", [str(w.message) for w in caught])
    induced, optional = ("ConvergenceWarning", "HTFA"), ("ConvergenceWarning", "TFA")
    # numpy 2.2.6's Accelerate build on Apple silicon (Python 3.10 on macOS)
    # raises this from the float32 G @ G.T in the engine's Hessian on finite
    # inputs, with a finite result equal to einsum's; the same product
    # repeated does not warn. A spurious floating-point flag, allowed (not
    # required); the finite rss below shows that the fit has no NaN.
    spurious = ("RuntimeWarning", "invalid value encountered in matmul")
    assert got.count(induced) == 1
    assert got.count(optional) <= len(xs)
    assert [g for g in got if g not in (induced, optional, spurious)] == []
    res = htfa.HTFAResults(
        model, [htfa.Subject(x, r) for x, r in zip(xs, rs, strict=True)]
    )
    count = parameter_count(res)
    assert count["spatial"] == 4 * (3 * m + m - 1)
    assert count["temporal"] == 60
    assert count["total"] == count["spatial"] + 61
    score = reconstruction_statistics(res)
    standardized = [htfa.tfa.standardize_data(x, model.standardize) for x in xs]
    rss = sum(
        np.linalg.norm(x - res.reconstruct(i)) ** 2 for i, x in enumerate(standardized)
    )
    print("rss", rss, "score.rss", score.rss)
    assert np.isfinite(rss) and score.rss == pytest.approx(rss)
    assert score.n_observations == 2 * 64 * 15


def test_tfa_unstandardized_and_subject_pooling():
    x, r, _ = htfa.make_tfa_data(grid_shape=(8, 8), K=2, n_samples=10, random_state=2)
    model = htfa.TFA(K=2, standardize=False, random_state=0).fit(x, r)
    res = htfa.HTFAResults(model, [htfa.Subject(x, r)])
    score = reconstruction_statistics(res)
    assert score.n_parameters == 2 * 3 + 2 * 10 + 1
    assert score.rss == pytest.approx(np.sum((x - res.reconstruct()) ** 2))
    repeated = gaussian_fit_statistics(
        [x, x], [res.reconstruct(), res.reconstruct()], n_parameters=score.n_parameters
    )
    assert repeated.r2 == pytest.approx(score.r2)
    assert repeated.rss == pytest.approx(2 * score.rss)


def test_pooling_uses_voxel_baselines_and_unequal_subject_lengths():
    xs = [np.array([[10.0, 12.0]]), np.array([[100.0, 102.0, 104.0]])]
    ys = [np.array([[9.0, 9.0]]), np.array([[100.0, 100.0, 100.0]])]
    score = gaussian_fit_statistics(xs, ys, n_parameters=2)
    assert score.n_observations == 5
    assert score.rss == 30
    assert score.tss == 10
    assert score.r2 == -2


@pytest.mark.parametrize("scale", [1e200, 1e-200])
def test_unrepresentable_error_scales_are_not_misreported(scale):
    x = np.array([[1.0, -1.0]]) * scale
    with pytest.raises(ValueError, match="rescale"):
        gaussian_fit_statistics(x, np.zeros_like(x), n_parameters=1)


def test_complex_inputs_are_not_silently_truncated():
    x = np.ones((2, 3))
    with pytest.raises(ValueError, match="real-valued"):
        gaussian_fit_statistics(x, x + 1j, n_parameters=1)


def test_log_variance_does_not_underflow_after_dividing_by_n():
    x = np.zeros((1, 10000))
    x[0, 0] = 1e-160
    score = gaussian_fit_statistics(x, np.zeros_like(x), n_parameters=1)
    assert score.rss > 0
    assert np.isfinite(score.log_likelihood)
