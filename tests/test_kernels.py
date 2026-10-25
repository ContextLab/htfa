"""Tests for the TFA numerical kernels (factors, weights, residual, Jacobian)."""

import time

import numpy as np
import pytest
import scipy.linalg

from htfa._engines import NumpyEngine
from htfa._kernels import (
    Prior,
    ResidualProblem,
    bounds,
    factors,
    init_centers_widths,
    max_sigma,
    ridge_weights,
    solve_centers_widths,
    sq_distances,
)

# Factors vs. a scalar double loop. Measured max abs error: 0 (D=2), 1.4e-21 (D=3);
# 1e-12 leaves room for libm differences only.
FACTOR_ATOL = 1e-12
# Ridge/OLS weights vs. an independent np.linalg.solve of the normal equations.
# Measured max relative error: 3.3e-16 (rr), 5.1e-16 (ols).
WEIGHTS_RTOL = 1e-10
# Brief: central differences with step 1e-6, max abs err < 1e-6 * max|J|.
# Measured relative error: 3.1e-10 to 5.4e-10 over D in {2,3}, with/without prior.
FD_STEP = 1e-6
FD_REL_TOL = 1e-6
# rmatvec vs. dense J^T u, and column norms vs. dense column norms: both are the
# same sums in a different order. Measured relative error: <= 7.1e-15 (rmatvec),
# 5.7e-16 (column norms).
ADJOINT_RTOL = 1e-12
# Compressed vs. uncompressed linear-loss solve. Brief: x within 1e-6.
# Measured max |dx|: 2.0e-8 (T>K), 6.0e-9 (T<K), with LSMR_TOL = 1e-8 in _kernels;
# measured relative cost difference: <= 4.5e-15.
COMPRESS_X_ATOL = 1e-6
COMPRESS_COST_RTOL = 1e-9
# Recovery with the true weights (brief): centers within 0.5 mm, widths within 10%.
# Measured: max center error 0.016 mm, max width relative error 0.004.
RECOVERY_CENTER_ATOL_MM = 0.5
RECOVERY_WIDTH_RTOL = 0.10


def _random_problem_inputs(rng, D, V=60, T=7, K=3):
    R = rng.uniform(0.0, 10.0, size=(V, D))
    centers = rng.uniform(2.0, 8.0, size=(K, D))
    widths = rng.uniform(4.0, 12.0, size=K)
    W = rng.normal(size=(K, T))
    X = factors(R, centers, widths) @ W + 0.1 * rng.normal(size=(V, T))
    return R, centers, widths, W, X


def _random_prior(rng, centers, widths, scale=0.37):
    K, D = centers.shape
    covs = np.empty((K, D, D))
    for k in range(K):
        A = rng.normal(size=(D, D))
        covs[k] = A @ A.T + D * np.eye(D)
    return Prior(
        centers=centers + rng.normal(0.0, 1.0, size=centers.shape),
        widths=widths * rng.uniform(0.7, 1.3, size=widths.shape),
        centers_cov=covs,
        widths_var=rng.uniform(1.0, 5.0, size=widths.shape),
        scale=scale,
    )


def _dense(op):
    return op.matmat(np.eye(op.shape[1]))


@pytest.mark.parametrize("D", [2, 3])
def test_factors_match_bruteforce(D):
    rng = np.random.default_rng(0)
    R = rng.uniform(-50.0, 50.0, size=(40, D))
    centers = rng.uniform(-40.0, 40.0, size=(4, D))
    widths = rng.uniform(50.0, 500.0, size=4)
    F = factors(R, centers, widths)
    expected = np.empty((40, 4))
    for v in range(40):
        for k in range(4):
            d2 = sum((R[v, j] - centers[k, j]) ** 2 for j in range(D))
            expected[v, k] = np.exp(-d2 / widths[k])
    err = np.abs(F - expected).max()
    print(f"D={D}: max |F - bruteforce| = {err:.3e}")
    assert F.dtype == np.float64
    assert F.shape == (40, 4)
    assert err < FACTOR_ATOL


def test_sq_distances_no_cancellation():
    base = 1.0e6
    R = base + np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 3.0, 1.0]])
    C = base + np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    d = sq_distances(R, C)
    expected = np.array([[0.0, 3.0], [1.0, 2.0], [14.0, 5.0]])
    print("sq_distances near 1e6:\n", d)
    np.testing.assert_array_equal(d, expected)


@pytest.mark.parametrize("method", ["rr", "ols"])
def test_ridge_weights_matches_normal_equations(method):
    rng = np.random.default_rng(1)
    R, centers, widths, _, X = _random_problem_inputs(rng, 3, V=80, T=11, K=4)
    F = factors(R, centers, widths)
    beta = np.var(X) if method == "rr" else 0.0
    expected = np.linalg.solve(F.T @ F + beta * np.eye(4), F.T @ X)
    W = ridge_weights(X, F, method=method)
    rel = np.abs(W - expected).max() / np.abs(expected).max()
    print(f"{method}: max relative error vs normal equations = {rel:.3e}")
    assert W.shape == (4, 11)
    assert rel < WEIGHTS_RTOL


def test_ridge_weights_rejects_unknown_method():
    F = np.eye(3)
    X = np.ones((3, 2))
    with pytest.raises(ValueError, match="xx"):
        ridge_weights(X, F, method="xx")


def test_max_sigma_value():
    R = np.array([[0.0, 0.0], [2.0, 6.0], [4.0, 0.0]])
    # std per dim: sqrt(8/3), sqrt(8); max**2 = 8 -> 2 * 8 = 16
    assert max_sigma(R) == pytest.approx(16.0, rel=1e-15)


def test_bounds_values():
    R = np.array([[0.0, 0.0, 0.0], [2.0, 6.0, -3.0], [4.0, 0.0, 3.0]])
    # std per dim: sqrt(8/3), sqrt(8), sqrt(6); max**2 = 8 -> max_sigma = 16
    lb, ub = bounds(R, 2, 0.5, 4.0)
    np.testing.assert_allclose(lb, [0, 0, -3, 0, 0, -3, 8.0, 8.0], rtol=1e-15)
    np.testing.assert_allclose(ub, [4, 6, 3, 4, 6, 3, 64.0, 64.0], rtol=1e-15)


def test_bounds_ignore_nan():
    R = np.array([[0.0, np.nan], [2.0, 6.0], [4.0, 1.0]])
    lb, ub = bounds(R, 1, 0.1, 1.0)
    assert lb[:2].tolist() == [0.0, 1.0]
    assert ub[:2].tolist() == [4.0, 6.0]


def test_init_centers_widths():
    rng = np.random.default_rng(2)
    blobs = np.array([[0.0, 0.0, 0.0], [30.0, 0.0, 0.0], [0.0, 30.0, 0.0]])
    R = np.concatenate([b + rng.normal(0.0, 0.5, size=(50, 3)) for b in blobs])
    centers, widths = init_centers_widths(R, 3, random_state=0)
    assert centers.shape == (3, 3)
    assert widths.shape == (3,)
    np.testing.assert_allclose(widths, max_sigma(R))
    for b in blobs:
        dist = np.linalg.norm(centers - b, axis=1).min()
        print(f"blob {b}: nearest k-means center at distance {dist:.3f}")
        assert dist < 0.5
    again, _ = init_centers_widths(R, 3, random_state=0)
    np.testing.assert_array_equal(centers, again)


def test_numpy_engine_ops():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(5, 4))
    W = rng.normal(size=(2, 4))
    F = rng.normal(size=(5, 2))
    u = rng.normal(size=20)
    eng = NumpyEngine()
    assert eng.name == "numpy"
    eng.set(X, W)
    np.testing.assert_allclose(eng.residual(F), (X - F @ W).ravel(), rtol=1e-15)
    np.testing.assert_allclose(eng.times_w(F), (F @ W).ravel(), rtol=1e-15)
    np.testing.assert_allclose(eng.times_wt(u), u.reshape(5, 4) @ W.T, rtol=1e-15)


def test_residual_values():
    rng = np.random.default_rng(4)
    R, centers, widths, W, X = _random_problem_inputs(rng, 3)
    prior = _random_prior(rng, centers, widths)
    x = np.concatenate([centers.ravel(), widths])
    prob = ResidualProblem(X, W, R, NumpyEngine(), prior=prior)
    r = prob.residual(x)
    V, T = X.shape
    K = 3
    s = 1.0 / (np.sqrt(2.0) * np.std(X))
    data = s * (X - factors(R, centers, widths) @ W)
    dc = centers - prior.centers
    rc = np.array(
        [
            np.sqrt(prior.scale * dc[k] @ np.linalg.solve(prior.centers_cov[k], dc[k]))
            for k in range(K)
        ]
    )
    rw = np.sqrt(prior.scale / prior.widths_var) * (widths - prior.widths)
    assert r.shape == (V * T + 2 * K,)
    np.testing.assert_allclose(r[: V * T], data.ravel(), rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(r[V * T : V * T + K], rc, rtol=1e-13)
    np.testing.assert_allclose(r[V * T + K :], rw, rtol=1e-13)
    no_prior = ResidualProblem(X, W, R, NumpyEngine())
    assert no_prior.residual(x).shape == (V * T,)


def test_constant_data_uses_unit_scale():
    R = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    X = np.full((3, 2), 5.0)
    W = np.ones((1, 2))
    prob = ResidualProblem(X, W, R, NumpyEngine())
    x = np.array([0.0, 0.0, 1.0])
    expected = (X - factors(R, np.zeros((1, 2)), np.ones(1)) @ W).ravel()
    np.testing.assert_allclose(prob.residual(x), expected, rtol=1e-15)


@pytest.mark.parametrize("D", [2, 3])
@pytest.mark.parametrize("use_prior", [False, True], ids=["prior=None", "prior=Prior"])
def test_jacobian_matches_finite_differences(D, use_prior):
    rng = np.random.default_rng(10 + D)
    R, centers, widths, W, X = _random_problem_inputs(rng, D)
    prior = _random_prior(rng, centers, widths) if use_prior else None
    prob = ResidualProblem(X, W, R, NumpyEngine(), prior=prior)
    x = np.concatenate(
        [(centers + rng.normal(0, 0.5, centers.shape)).ravel(), widths * 1.1]
    )
    J = prob.jacobian(x)
    n_res = prob.residual(x).size
    assert J.shape == (n_res, x.size)
    J_dense = _dense(J)
    J_fd = np.empty_like(J_dense)
    for j in range(x.size):
        e = np.zeros_like(x)
        e[j] = FD_STEP
        J_fd[:, j] = (prob.residual(x + e) - prob.residual(x - e)) / (2 * FD_STEP)
    rel = np.abs(J_dense - J_fd).max() / np.abs(J_dense).max()
    print(f"D={D} prior={use_prior}: max|J - J_fd| / max|J| = {rel:.3e}")
    assert rel < FD_REL_TOL

    for _ in range(3):
        p = rng.normal(size=x.size)
        np.testing.assert_allclose(
            J.matvec(p), J_dense @ p, rtol=ADJOINT_RTOL, atol=1e-14
        )
        u = rng.normal(size=n_res)
        expected = J_dense.T @ u
        got = J.rmatvec(u)
        adj = np.abs(got - expected).max() / np.abs(expected).max()
        print(f"  rmatvec relative error = {adj:.3e}")
        assert adj < ADJOINT_RTOL
    U = rng.normal(size=(n_res, 2))
    np.testing.assert_allclose(
        J.rmatmat(U), J_dense.T @ U, rtol=ADJOINT_RTOL, atol=1e-12
    )


def test_center_prior_jacobian_zero_at_prior_mean():
    rng = np.random.default_rng(20)
    R, centers, widths, W, X = _random_problem_inputs(rng, 3)
    prior = _random_prior(rng, centers, widths)
    prior.centers = centers.copy()
    prob = ResidualProblem(X, W, R, NumpyEngine(), prior=prior)
    x = np.concatenate([centers.ravel(), widths])
    V, T = X.shape
    J_dense = _dense(prob.jacobian(x))
    rows = J_dense[V * T : V * T + 3]
    assert np.all(np.isfinite(J_dense))
    np.testing.assert_array_equal(rows, 0.0)


@pytest.mark.parametrize("use_prior", [False, True], ids=["prior=None", "prior=Prior"])
def test_column_norms_exact(use_prior):
    rng = np.random.default_rng(30)
    R, centers, widths, W, X = _random_problem_inputs(rng, 3)
    prior = _random_prior(rng, centers, widths) if use_prior else None
    prob = ResidualProblem(X, W, R, NumpyEngine(), prior=prior)
    x = np.concatenate([(centers + 0.3).ravel(), widths * 0.9])
    expected = np.linalg.norm(_dense(prob.jacobian(x)), axis=0)
    got = prob.column_norms(x)
    rel = np.abs(got - expected).max() / expected.max()
    print(f"prior={use_prior}: column norm relative error = {rel:.3e}")
    assert got.shape == (12,)
    assert rel < ADJOINT_RTOL


@pytest.mark.parametrize("T", [40, 2], ids=["T>K", "T<K"])
def test_compressed_problem_same_solution(T):
    rng = np.random.default_rng(40)
    R, centers, widths, W, X = _random_problem_inputs(rng, 3, V=150, T=T, K=3)
    lb, ub = bounds(R, 3, 0.02, 4.0)
    x0 = np.concatenate(
        [(centers + rng.normal(0, 0.5, centers.shape)).ravel(), widths * 1.2]
    )
    full = ResidualProblem(X, W, R, NumpyEngine())
    comp = ResidualProblem(X, W, R, NumpyEngine(), compress=True)
    assert full.cost_offset == 0.0
    s = 1.0 / (np.sqrt(2.0) * np.std(X))
    # At any x the compressed linear cost + offset equals the uncompressed one.
    for x in (x0, np.concatenate([centers.ravel(), widths])):
        c_full = 0.5 * np.sum(full.residual(x) ** 2)
        c_comp = 0.5 * np.sum(comp.residual(x) ** 2) + comp.cost_offset
        assert c_comp == pytest.approx(c_full, rel=COMPRESS_COST_RTOL)
    Q, _ = np.linalg.qr(W.T)
    expected_offset = 0.5 * s**2 * np.sum((X - X @ Q @ Q.T) ** 2)
    assert comp.cost_offset == pytest.approx(expected_offset, rel=1e-12)

    x_full, cost_full = solve_centers_widths(full, x0, (lb, ub), loss="linear")
    x_comp, cost_comp = solve_centers_widths(comp, x0, (lb, ub), loss="linear")
    dx = np.abs(x_full - x_comp).max()
    dc = abs(cost_full - cost_comp) / cost_full
    print(f"T={T}: max|dx| = {dx:.3e}, relative cost diff = {dc:.3e}")
    assert dx < COMPRESS_X_ATOL
    assert dc < COMPRESS_COST_RTOL


def test_solve_reports_uncompressed_cost():
    rng = np.random.default_rng(41)
    R, centers, widths, W, X = _random_problem_inputs(rng, 2, V=80, T=5, K=2)
    lb, ub = bounds(R, 2, 0.02, 4.0)
    x0 = np.concatenate([centers.ravel(), widths])
    prob = ResidualProblem(X, W, R, NumpyEngine())
    x, cost = solve_centers_widths(prob, x0, (lb, ub), loss="linear")
    assert cost == pytest.approx(0.5 * np.sum(prob.residual(x) ** 2), rel=1e-12)
    assert np.all(x >= lb) and np.all(x <= ub)


def test_solve_recovers_centers_widths_given_true_weights():
    rng = np.random.default_rng(50)
    # V = 20 * 10 * 10 = 2000 voxels on a 2 mm grid.
    axes = [np.arange(n) * 2.0 for n in (20, 10, 10)]
    R = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    K, T = 5, 100
    true_centers = np.array(
        [
            [8.0, 6.0, 6.0],
            [30.0, 12.0, 8.0],
            [18.0, 9.0, 13.0],
            [6.0, 14.0, 12.0],
            [32.0, 5.0, 14.0],
        ]
    )
    true_widths = np.array([20.0, 30.0, 25.0, 18.0, 35.0])
    W = rng.normal(size=(K, T))
    X = factors(R, true_centers, true_widths) @ W + 0.1 * rng.normal(
        size=(R.shape[0], T)
    )
    lb, ub = bounds(R, K, 0.02, 4.0)
    x0 = np.concatenate(
        [(true_centers + rng.normal(0.0, 2.0, size=(K, 3))).ravel(), true_widths * 1.2]
    )
    x0 = np.clip(x0, lb, ub)
    prob = ResidualProblem(X, W, R, NumpyEngine())
    t0 = time.perf_counter()
    x, cost = solve_centers_widths(prob, x0, (lb, ub))
    elapsed = time.perf_counter() - t0
    centers, widths = x[: K * 3].reshape(K, 3), x[K * 3 :]
    c_err = np.abs(centers - true_centers).max()
    w_err = (np.abs(widths - true_widths) / true_widths).max()
    print(
        f"recovery: {elapsed:.2f}s, cost={cost:.3f}, max center err={c_err:.4f} mm, "
        f"max width rel err={w_err:.4f}"
    )
    assert c_err < RECOVERY_CENTER_ATOL_MM
    assert w_err < RECOVERY_WIDTH_RTOL


def test_solve_x_scale_jac_matches_column_norms():
    """x_scale='jac' must give the same answer as passing 1/column_norms(x0)."""
    rng = np.random.default_rng(60)
    R, centers, widths, W, X = _random_problem_inputs(rng, 3)
    lb, ub = bounds(R, 3, 0.02, 4.0)
    x0 = np.concatenate([centers.ravel(), widths])
    prob = ResidualProblem(X, W, R, NumpyEngine())
    x_a, c_a = solve_centers_widths(prob, x0, (lb, ub))
    x_b, c_b = solve_centers_widths(
        prob, x0, (lb, ub), x_scale=1 / prob.column_norms(x0)
    )
    np.testing.assert_array_equal(x_a, x_b)
    assert c_a == c_b


def test_ridge_weights_uses_scipy_pos_solver():
    """The rr system is SPD; check against scipy's Cholesky-based solve directly."""
    rng = np.random.default_rng(70)
    F = rng.uniform(size=(30, 3))
    X = rng.normal(size=(30, 4))
    expected = scipy.linalg.solve(
        F.T @ F + np.var(X) * np.eye(3), F.T @ X, assume_a="pos"
    )
    np.testing.assert_array_equal(ridge_weights(X, F), expected)


def test_solve_rejects_unknown_x_scale_string():
    rng = np.random.default_rng(80)
    R, centers, widths, W, X = _random_problem_inputs(rng, 2, K=2)
    prob = ResidualProblem(X, W, R, NumpyEngine())
    x0 = np.concatenate([centers.ravel(), widths])
    with pytest.raises(ValueError, match="x_scale"):
        solve_centers_widths(prob, x0, bounds(R, 2, 0.02, 4.0), x_scale="auto")


# --- Structured Gauss-Newton solver (bounded-lm) -----------------------------------

LOSS_NAMES = ("linear", "soft_l1", "huber", "cauchy", "arctan")
# loss_terms vs scipy's own loss functions: the same formulas; measured max
# relative difference 0 (identical expressions), 1e-14 allows libm differences.
LOSS_RTOL = 1e-14
# Central differences of rho and rho' with step 1e-4 * z: measured relative error
# <= 5.7e-10 (soft_l1); a 1e-6 step is rounding-limited at small z (1.3e-7).
LOSS_FD_STEP = 1e-4
LOSS_FD_RTOL = 1e-7
# Gradient (and Hessian) vs central differences, relative to the largest entry.
GRAD_FD_RTOL = 1e-6
# Structured gradient and Gauss-Newton matrix vs dense J^T (rho' r) and J^T u J:
# same sums, different order.
GN_RTOL = 1e-11
# Acceptance (brief T11b): the structured solver's cost may exceed trf's by at most
# 1e-6 relative, and its centers must match trf's within 1e-3 (unless its cost is
# lower, then it found a better point).
SAME_OPTIMUM_COST_RTOL = 1e-6
SAME_OPTIMUM_CENTER_ATOL = 1e-3


def _scipy_loss(name, z):
    from scipy.optimize._lsq.least_squares import IMPLEMENTED_LOSSES

    rho = np.empty((3, z.size))
    if name == "linear":
        return z, np.ones_like(z), np.zeros_like(z)
    IMPLEMENTED_LOSSES[name](z, rho, cost_only=False)
    return rho


@pytest.mark.parametrize("name", LOSS_NAMES)
def test_loss_terms_match_scipy_and_finite_differences(name):
    from htfa._engines import loss_terms

    z = np.concatenate([[0.0, 0.5, 1.0], np.geomspace(1e-4, 50.0, 40)])
    got = loss_terms(name, z)
    expected = _scipy_loss(name, z)
    for i in range(3):
        err = np.abs(got[i] - expected[i]).max() / max(np.abs(expected[i]).max(), 1)
        print(f"{name}: rho^({i}) max relative difference vs scipy = {err:.3e}")
        assert err <= LOSS_RTOL
    zz = z[z > 1e-3]
    zz = zz[np.abs(zz - 1.0) > 1e-3]  # huber's kink at z = 1
    h = LOSS_FD_STEP * zz
    for i in (1, 2):
        fd = (loss_terms(name, zz + h)[i - 1] - loss_terms(name, zz - h)[i - 1]) / (
            2 * h
        )
        err = np.abs(fd - loss_terms(name, zz)[i]).max() / max(np.abs(fd).max(), 1)
        print(f"{name}: rho^({i}) vs central differences, relative error {err:.3e}")
        assert err < LOSS_FD_RTOL


def test_loss_terms_rejects_unknown():
    from htfa._engines import loss_terms

    with pytest.raises(ValueError, match="loss must be one of"):
        loss_terms("l2", np.ones(2))


def _gn_problem(D, use_prior, seed, V=60, T=7, K=3):
    rng = np.random.default_rng(seed)
    R, centers, widths, W, X = _random_problem_inputs(rng, D, V=V, T=T, K=K)
    # Heavier tails so that the robust weights differ visibly from 1.
    X = X + rng.standard_t(2, size=X.shape) * 0.3
    prior = _random_prior(rng, centers, widths) if use_prior else None
    prob = ResidualProblem(X, W, R, NumpyEngine(), prior=prior)
    x = np.concatenate(
        [(centers + rng.normal(0, 0.5, centers.shape)).ravel(), widths * 1.1]
    )
    return prob, x, rng


@pytest.mark.parametrize("loss", LOSS_NAMES)
@pytest.mark.parametrize("D", [2, 3])
@pytest.mark.parametrize("use_prior", [False, True], ids=["prior=None", "prior=Prior"])
def test_cost_matches_residual(loss, D, use_prior):
    from htfa._engines import loss_terms

    prob, x, _ = _gn_problem(D, use_prior, 100 + D)
    r = prob.residual(x)
    expected = 0.5 * np.sum(loss_terms(loss, r * r)[0])
    got = prob.cost(x, loss)
    gn_cost = prob.cost_grad_hess(x, loss)[0]
    print(f"{loss} D={D} prior={use_prior}: cost {got:.12g} vs {expected:.12g}")
    assert got == pytest.approx(expected, rel=1e-13)
    assert gn_cost == pytest.approx(expected, rel=1e-13)


@pytest.mark.parametrize("loss", LOSS_NAMES)
@pytest.mark.parametrize("D", [2, 3])
@pytest.mark.parametrize("use_prior", [False, True], ids=["prior=None", "prior=Prior"])
def test_gradient_matches_finite_differences(loss, D, use_prior):
    prob, x, _ = _gn_problem(D, use_prior, 200 + D)
    _, g, _, _ = prob.cost_grad_hess(x, loss)
    g_fd = np.empty_like(x)
    for j in range(x.size):
        e = np.zeros_like(x)
        e[j] = FD_STEP
        g_fd[j] = (prob.cost(x + e, loss) - prob.cost(x - e, loss)) / (2 * FD_STEP)
    rel = np.abs(g - g_fd).max() / np.abs(g_fd).max()
    print(f"{loss} D={D} prior={use_prior}: max|g - g_fd| / max|g_fd| = {rel:.3e}")
    assert rel < GRAD_FD_RTOL


@pytest.mark.parametrize("loss", LOSS_NAMES)
@pytest.mark.parametrize("D", [2, 3])
def test_hessian_matches_finite_differences(loss, D):
    """Data-term Hessian = central differences of the (FD-checked) gradient, with
    scipy's EPS floor on the loss weight u = rho' + 2 r^2 rho'' in the J^T u J part
    (it binds only for cauchy and arctan, where u < 0 for large residuals); the
    Gauss-Newton matrix = dense J^T diag(max(u, EPS)) J."""
    from htfa._engines import EPS, loss_terms

    prob, x, _ = _gn_problem(D, False, 300 + D)
    J = _dense(prob.jacobian(x))
    r = prob.residual(x)
    _, rho1, rho2 = loss_terms(loss, r * r)
    u = rho1 + 2 * r * r * rho2
    floor = J.T @ ((np.maximum(u, EPS) - u)[:, None] * J)
    _, g, gn, H = prob.cost_grad_hess(x, loss)
    H_fd = np.empty_like(H)
    for j in range(x.size):
        e = np.zeros_like(x)
        e[j] = FD_STEP
        H_fd[:, j] = (
            prob.cost_grad_hess(x + e, loss)[1] - prob.cost_grad_hess(x - e, loss)[1]
        ) / (2 * FD_STEP)
    expected = H_fd + floor
    rel = np.abs(H - expected).max() / np.abs(expected).max()
    print(
        f"{loss} D={D}: max|H - (H_fd + floor)| / max = {rel:.3e}; "
        f"floor share {np.abs(floor).max() / np.abs(expected).max():.2e}; "
        f"min eig(H) {np.linalg.eigvalsh(H).min():.3g}"
    )
    assert rel < GRAD_FD_RTOL
    np.testing.assert_allclose(g, J.T @ (rho1 * r), rtol=GN_RTOL, atol=1e-13)
    np.testing.assert_array_equal(H, H.T)
    gn_dense = J.T @ (np.maximum(u, EPS)[:, None] * J)
    gn_rel = np.abs(gn - gn_dense).max() / np.abs(gn_dense).max()
    print(f"{loss} D={D}: max|GN - J^T u J| / max = {gn_rel:.3e}")
    assert gn_rel < GN_RTOL
    np.testing.assert_array_equal(gn, gn.T)


@pytest.mark.parametrize("loss", ["linear", "soft_l1"])
@pytest.mark.parametrize("D", [2, 3])
def test_prior_hessian_terms_match_finite_differences(loss, D):
    """Prior part: exact Hessian of the center term (a function of q = r_c^2) and
    of the linear width residuals, against central differences of the prior
    gradient (the EPS floor is inactive for these losses)."""
    prob, x, _ = _gn_problem(D, True, 400 + D)
    K = prob.K

    def prior_grad(y):
        C, w = prob._unpack(y)
        return prob._prior_terms(C, w, loss, derivatives=True)[1]

    def prior_cost(y):
        C, w = prob._unpack(y)
        return prob._prior_terms(C, w, loss, derivatives=False)[0]

    C, w = prob._unpack(x)
    _, g, H = prob._prior_terms(C, w, loss, derivatives=True)
    H_fd = np.empty_like(H)
    g_fd = np.empty_like(g)
    for j in range(x.size):
        e = np.zeros_like(x)
        e[j] = FD_STEP
        H_fd[:, j] = (prior_grad(x + e) - prior_grad(x - e)) / (2 * FD_STEP)
        g_fd[j] = (prior_cost(x + e) - prior_cost(x - e)) / (2 * FD_STEP)
    g_rel = np.abs(g - g_fd).max() / np.abs(g_fd).max()
    nc = K * D
    for rows, cols in (
        (slice(0, nc), slice(0, nc)),
        (slice(nc, None), slice(nc, None)),
    ):
        rel = np.abs(H[rows, cols] - H_fd[rows, cols]).max() / np.abs(H_fd).max()
        print(f"{loss} D={D}: prior Hessian block rel err {rel:.3e}")
        assert rel < GRAD_FD_RTOL
    print(f"{loss} D={D}: prior gradient rel err {g_rel:.3e}")
    assert g_rel < GRAD_FD_RTOL
    np.testing.assert_array_equal(H[:nc, nc:], 0.0)
    assert np.all(np.linalg.eigvalsh(H) >= 0)


@pytest.mark.parametrize("loss", ["linear", "soft_l1"])
def test_cost_grad_hess_independent_of_chunk_size(loss):
    prob, x, _ = _gn_problem(3, True, 500, V=97, T=11, K=4)
    whole = prob.cost_grad_hess(x, loss)
    prob.engine.chunk_elements = 3 * 16  # 3 voxels per chunk (K**2 = A*K = 16 > T)
    prob.engine.min_chunk_rows = 1
    chunked = prob.cost_grad_hess(x, loss)
    assert len(prob.engine._chunks(16)) == 33
    assert chunked[0] == pytest.approx(whole[0], rel=1e-13)
    np.testing.assert_allclose(chunked[1], whole[1], rtol=1e-12, atol=1e-14)
    for i in (2, 3):
        np.testing.assert_allclose(chunked[i], whole[i], rtol=1e-12, atol=1e-14)
    assert prob.cost(x, loss) == pytest.approx(whole[0], rel=1e-13)


def _trf_and_lm(prob, x0, bnds, loss):
    x_trf, c_trf = solve_centers_widths(prob, x0, bnds, method="trf", loss=loss)
    t0 = time.perf_counter()
    x_lm, c_lm = solve_centers_widths(prob, x0, bnds, method="bounded-lm", loss=loss)
    return x_trf, c_trf, x_lm, c_lm, time.perf_counter() - t0


@pytest.mark.parametrize("loss", ["linear", "soft_l1"])
@pytest.mark.parametrize("D", [2, 3])
@pytest.mark.parametrize("use_prior", [False, True], ids=["prior=None", "prior=Prior"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_bounded_lm_reaches_trf_optimum(loss, D, use_prior, seed):
    prob, x0, _ = _gn_problem(D, use_prior, 600 + seed, V=150, T=20, K=3)
    lb, ub = bounds(prob.R, prob.K, 0.02, 4.0)
    x0 = np.clip(x0, lb, ub)
    x_trf, c_trf, x_lm, c_lm, secs = _trf_and_lm(prob, x0, (lb, ub), loss)
    nc = prob.K * D
    dc = np.abs(x_lm[:nc] - x_trf[:nc]).max()
    print(
        f"{loss} D={D} prior={use_prior} seed={seed}: cost lm {c_lm:.12g} trf "
        f"{c_trf:.12g} (rel {(c_lm - c_trf) / c_trf:+.2e}), max|dc| {dc:.2e}, "
        f"lm {secs:.3f}s"
    )
    assert c_lm <= c_trf * (1 + SAME_OPTIMUM_COST_RTOL)
    assert dc < SAME_OPTIMUM_CENTER_ATOL or c_lm < c_trf
    assert np.all(x_lm >= lb) and np.all(x_lm <= ub)
    assert c_lm == pytest.approx(prob.cost(x_lm, loss), rel=1e-15)


@pytest.mark.parametrize("loss", ["linear", "soft_l1"])
def test_bounded_lm_active_bounds_kkt(loss):
    """Optimum with the widths pressed against a low upper bound: the solution
    sits on the bound with the gradient pointing out of the box (KKT), and
    matches trf. Both stop on ftol, so the interior gradient is compared with
    trf's (measured: 2.1e-5 vs 1.5e-3 linear, 1.1e-5 vs 1.4e-3 soft_l1)."""
    from htfa._kernels import bounded_lm

    prob, x0, _ = _gn_problem(3, False, 700, V=150, T=20, K=3)
    lb, ub = bounds(prob.R, prob.K, 0.02, 4.0)
    nc = prob.K * 3
    ub[nc:] = 0.5 * x0[nc:]  # well below the widths that fit the data
    x0 = np.clip(x0, lb, ub)
    scale = 1.0 / prob.column_norms(x0)
    x, cost, status, nfev = bounded_lm(prob, x0, (lb, ub), loss, scale)
    _, g, _, _ = prob.cost_grad_hess(x, loss)
    at_ub = x >= ub
    print(f"{loss}: status {status}, nfev {nfev}, at ub {at_ub.nonzero()[0]}")
    assert np.all(at_ub[nc:]) and np.all(g[nc:] < 0)
    x_trf, c_trf = solve_centers_widths(prob, x0, (lb, ub), method="trf", loss=loss)
    g_trf = prob.cost_grad_hess(x_trf, loss)[1]
    interior = (x > lb) & (x < ub)
    g_in, g_in_trf = np.abs(g[interior]).max(), np.abs(g_trf[interior]).max()
    print(
        f"{loss}: cost lm {cost:.12g} trf {c_trf:.12g}; interior max|g| lm "
        f"{g_in:.3e} trf {g_in_trf:.3e}"
    )
    assert g_in < 10 * g_in_trf
    assert cost <= c_trf * (1 + SAME_OPTIMUM_COST_RTOL)
    assert status in (1, 2, 3, 4)


def test_bounded_lm_respects_max_nfev():
    from htfa._kernels import bounded_lm

    prob, x0, _ = _gn_problem(2, False, 800)
    lb, ub = bounds(prob.R, prob.K, 0.02, 4.0)
    x0 = np.clip(x0, lb, ub)
    scale = 1.0 / prob.column_norms(x0)
    x, cost, status, nfev = bounded_lm(prob, x0, (lb, ub), "soft_l1", scale, 1)
    assert status == 0 and nfev == 1
    np.testing.assert_array_equal(x, x0)
    assert cost == prob.cost(x0, "soft_l1")
    x, cost3, status, nfev = bounded_lm(prob, x0, (lb, ub), "soft_l1", scale, 3)
    assert nfev <= 3 and cost3 <= cost


def test_solve_rejects_unknown_method():
    prob, x0, _ = _gn_problem(2, False, 900)
    with pytest.raises(ValueError, match="method must be one of"):
        solve_centers_widths(prob, x0, bounds(prob.R, 3, 0.02, 4.0), method="lm")
