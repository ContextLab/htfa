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


@pytest.mark.parametrize("D", [2, 3])
def test_factor_derivatives_bitwise_equal_to_their_formulas(D):
    """E, F and G equal the formulas of their docstring evaluated directly (from
    the strided view of R), bit for bit."""
    from htfa._kernels import factor_derivatives

    rng = np.random.default_rng(D)
    R = rng.uniform(-50.0, 50.0, size=(500, D))
    C = rng.uniform(-40.0, 40.0, size=(6, D))
    w = rng.uniform(50.0, 500.0, size=6)
    E, F, G = factor_derivatives(R, C, w)
    # einsum's rounding depends on the strides of its operands: the reference
    # runs on the same (K, D + 2, V) layout.
    ref = np.empty_like(E)
    delta = ref[:, :D]
    np.subtract(R.T[None, :, :], C[:, :, None], out=delta)
    d = np.einsum("kjv,kjv->kv", delta, delta, out=ref[:, D])
    F_ref = np.exp(-d / w[:, None])
    Fw = F_ref / w[:, None]
    n_diff = [
        int(np.sum(E[:, :D] != delta)),
        int(np.sum(E[:, D] != d)),
        int(np.sum(E[:, D + 1] != 1.0)),
        int(np.sum(F != F_ref)),
        int(np.sum(G[:, :D] != 2.0 * Fw[:, None, :] * delta)),
        int(np.sum(G[:, D] != Fw * (d / w[:, None]))),
    ]
    print(f"D={D}: entries differing (delta, d, ones, F, G_c, G_w): {n_diff}")
    assert n_diff == [0] * 6


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


def _tfa_solve_case(name, seed, use_prior, loss):
    """One center/width solve on make_tfa_data, as in scripts/compare_solvers.py:
    z-scored data, the 'ncorr' start (or the prior's), its ridge weights, and a
    prior near the truth with HTFA's initial (co)variances."""
    from htfa.synthetic import make_tfa_data
    from htfa.tfa import _initial_starts, standardize_data

    shape, K = {"2d": ((40, 40), 6), "3d": ((20, 20, 20), 8)}[name]
    X, R, truth = make_tfa_data(shape, 100, K=K, random_state=seed)
    X = standardize_data(X, "auto")
    lb, ub = bounds(R, K, 0.02, 1.8)
    if use_prior:
        rng = np.random.default_rng(1000 + seed)
        length = np.sqrt(max_sigma(R) / 2.0)
        prior = Prior(
            centers=truth["centers"]
            + rng.normal(0.0, 0.3 * length, size=(K, len(shape))),
            widths=truth["widths"] * rng.uniform(0.8, 1.2, size=K),
            centers_cov=np.tile(np.cov(R.T) * K ** (-2.0 / 3.0), (K, 1, 1)),
            widths_var=np.full(K, max_sigma(R)),
            scale=0.5,
        )
        start = (prior.centers, prior.widths)
    else:
        prior = None
        start = _initial_starts(
            X, R, K, ("ncorr",), np.random.default_rng(seed), lb[-1], ub[-1]
        )["ncorr"]
    x0 = np.clip(np.concatenate([np.ravel(start[0]), start[1]]), lb, ub)
    W = ridge_weights(X, factors(R, *start))
    prob = ResidualProblem(X, W, R, NumpyEngine(), prior, compress=loss == "linear")
    return prob, x0, (lb, ub)


@pytest.mark.parametrize("loss", ["linear", "soft_l1"])
@pytest.mark.parametrize("case", [("2d", 2), ("3d", 2)])
def test_bounded_lm_does_not_jump_basins_on_indefinite_hessian(case, loss):
    """With a prior the Hessian is indefinite at the start, and a damped Newton
    step along negative curvature jumps to a worse basin: 2D seed 2 (+9.5% vs
    trf, a center moved 22.7 mm, before the derivative-layout change) and 3D
    seed 2 (+4e-4, centers 6.8 mm off, with the current code). With the
    Gauss-Newton fallback both reach trf's optimum; with the Hessian always
    used, the 3D case fails here (checked by disabling the switch)."""
    prob, x0, bnds = _tfa_solve_case(*case, True, loss)
    nc = prob.K * prob.D
    x_trf, c_trf = solve_centers_widths(prob, x0, bnds, method="trf", loss=loss)
    x_lm, c_lm = solve_centers_widths(prob, x0, bnds, loss=loss)
    dc = np.abs(x_lm[:nc] - x_trf[:nc]).max()
    print(f"{case} {loss}: cost lm {c_lm:.10g} trf {c_trf:.10g}, max|dc| {dc:.3g}")
    assert c_lm <= c_trf * (1 + SAME_OPTIMUM_COST_RTOL)


@pytest.mark.slow
@pytest.mark.parametrize("loss", ["linear", "soft_l1"])
@pytest.mark.parametrize("use_prior", [False, True], ids=["prior=None", "prior=Prior"])
@pytest.mark.parametrize("case", [("2d", 0), ("2d", 4), ("3d", 1)])
def test_bounded_lm_matches_trf_on_tfa_data(case, use_prior, loss):
    """A subset of scripts/compare_solvers.py's single-solve sweep."""
    prob, x0, bnds = _tfa_solve_case(*case, use_prior, loss)
    _, c_trf = solve_centers_widths(prob, x0, bnds, method="trf", loss=loss)
    t0 = time.perf_counter()
    _, c_lm = solve_centers_widths(prob, x0, bnds, loss=loss)
    print(
        f"{case} prior={use_prior} {loss}: cost lm {c_lm:.10g} trf {c_trf:.10g} "
        f"(rel {(c_lm - c_trf) / c_trf:+.2e}), lm {time.perf_counter() - t0:.2f} s"
    )
    assert c_lm <= c_trf * (1 + SAME_OPTIMUM_COST_RTOL)


def test_ftol_for_engine_precision():
    from htfa._kernels import FTOL, ftol_for

    assert NumpyEngine().eps == np.finfo(np.float64).eps
    assert ftol_for(NumpyEngine().eps) == FTOL == 1e-8
    assert ftol_for(float(np.finfo(np.float32).eps)) == pytest.approx(1.19e-7, rel=1e-2)


def test_bounded_lm_raises_clearly_on_non_finite_values():
    """A NaN in the data makes cost, gradient and Hessian NaN (scipy's
    cho_factor would raise an unrelated ValueError): a clear error instead."""
    from htfa._kernels import bounded_lm

    rng = np.random.default_rng(910)
    R, centers, widths, W, X = _random_problem_inputs(rng, 2, K=2)
    X[3, 2] = np.nan
    prob = ResidualProblem(X, W, R, NumpyEngine())
    x0 = np.concatenate([centers.ravel(), widths])
    lb, ub = bounds(R, 2, 0.02, 4.0)
    with pytest.raises(FloatingPointError, match="non-finite"):
        bounded_lm(prob, x0, (lb, ub), "soft_l1", np.ones_like(x0))
    with pytest.raises(FloatingPointError, match="non-finite"):
        solve_centers_widths(prob, x0, (lb, ub), x_scale=np.ones_like(x0))


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_bounded_lm_falls_back_to_gauss_newton_on_non_finite_hessian():
    """A width of 1e-110 makes that factor exactly 0 (exp underflow), so cost,
    gradient and Gauss-Newton matrix are finite, but the second-order term divides
    0 by w**3 = 0 (underflow) and is NaN. The solver must take Gauss-Newton steps
    (scipy's cho_factor raises ValueError on the NaN Hessian) and still descend."""
    from htfa._kernels import bounded_lm

    rng = np.random.default_rng(920)
    R = rng.uniform(0.0, 10.0, size=(60, 2))
    C = np.array([[3.3, 4.1], [6.2, 7.7]])
    W = rng.normal(size=(2, 7))
    X = factors(R, C, np.array([8.0, 5.0])) @ W + 0.1 * rng.normal(size=(60, 7))
    prob = ResidualProblem(X, W, R, NumpyEngine())
    x0 = np.concatenate([C.ravel(), [8.0, 1e-110]])
    cost0, g, gn, hess = prob.cost_grad_hess(x0, "soft_l1")
    assert np.isfinite(cost0) and np.all(np.isfinite(g)) and np.all(np.isfinite(gn))
    assert not np.all(np.isfinite(hess))
    lb = np.array([0.0, 0.0, 0.0, 0.0, 1e-120, 1e-120])
    ub = np.array([10.0, 10.0, 10.0, 10.0, 50.0, 50.0])
    x, cost, status, nfev = bounded_lm(prob, x0, (lb, ub), "soft_l1", np.ones(6))
    print(f"cost {cost0:.6f} -> {cost:.6f}, status {status}, nfev {nfev}, x {x}")
    assert np.all(np.isfinite(x)) and cost < cost0
    assert status in (1, 2, 3, 4)


def _projected_problem(D, use_prior, seed, method="rr", V=60, T=7, K=3):
    from htfa._kernels import ProjectedProblem

    rng = np.random.default_rng(seed)
    R, centers, widths, _, X = _random_problem_inputs(rng, D, V=V, T=T, K=K)
    prior = _random_prior(rng, centers, widths) if use_prior else None
    prob = ProjectedProblem(X, R, K, NumpyEngine(), prior, method)
    x = np.concatenate(
        [(centers + rng.normal(0, 0.5, centers.shape)).ravel(), widths * 1.1]
    )
    return prob, x, X, R, prior


@pytest.mark.parametrize("method", ["rr", "ols"])
@pytest.mark.parametrize("D", [2, 3])
@pytest.mark.parametrize("use_prior", [False, True], ids=["prior=None", "prior=Prior"])
def test_projected_cost_is_ridge_objective(method, D, use_prior):
    """phi(x) = 0.5 s^2 (||X - F W||^2 + beta ||W||^2) + prior, W = ridge(x), both
    from cost and from cost_grad_hess."""
    prob, x, X, R, prior = _projected_problem(D, use_prior, 400 + D, method)
    K = prob.K
    C, w = x[: K * D].reshape(K, D), x[K * D :]
    W = ridge_weights(X, factors(R, C, w), method)
    beta = np.var(X) if method == "rr" else 0.0
    s2 = 1.0 / (2.0 * np.var(X))
    fixed = ResidualProblem(X, W, R, NumpyEngine(), prior)
    expected = 0.5 * s2 * beta * np.sum(W * W) + fixed.cost(x, "linear")
    got, got_gn = prob.cost(x), prob.cost_grad_hess(x)[0]
    print(f"{method} D={D} prior={use_prior}: {got:.14g} {got_gn:.14g} {expected:.14g}")
    assert got == pytest.approx(expected, rel=1e-12)
    assert got_gn == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("method", ["rr", "ols"])
@pytest.mark.parametrize("D", [2, 3])
@pytest.mark.parametrize("use_prior", [False, True], ids=["prior=None", "prior=Prior"])
def test_projected_gradient_and_hessian_match_finite_differences(method, D, use_prior):
    """Envelope gradient and the exact projected Hessian H_xx - H_xW H_WW^-1 H_Wx
    against central differences of phi and of the gradient; the Kaufman
    Gauss-Newton matrix is symmetric positive semidefinite."""
    prob, x, *_ = _projected_problem(D, use_prior, 500 + D, method)
    _, g, gn, H = prob.cost_grad_hess(x)
    g_fd, H_fd = np.empty_like(x), np.empty_like(H)
    for j in range(x.size):
        e = np.zeros_like(x)
        e[j] = FD_STEP
        g_fd[j] = (prob.cost(x + e) - prob.cost(x - e)) / (2 * FD_STEP)
        H_fd[:, j] = (prob.cost_grad_hess(x + e)[1] - prob.cost_grad_hess(x - e)[1]) / (
            2 * FD_STEP
        )
    g_rel = np.abs(g - g_fd).max() / np.abs(g_fd).max()
    H_rel = np.abs(H - H_fd).max() / np.abs(H_fd).max()
    eig = np.linalg.eigvalsh(gn)
    print(
        f"{method} D={D} prior={use_prior}: grad rel {g_rel:.3e}, hess rel "
        f"{H_rel:.3e}, GN eig min {eig.min():.3g} max {eig.max():.3g}"
    )
    assert g_rel < GRAD_FD_RTOL
    assert H_rel < GRAD_FD_RTOL
    np.testing.assert_array_equal(H, H.T)
    np.testing.assert_array_equal(gn, gn.T)
    assert eig.min() > -1e-10 * eig.max()


def test_projected_solve_is_a_fixed_point_of_the_alternation():
    """bounded_lm on phi stops where one more linear-loss alternation step (ridge
    weights, then the fixed-weight solve) does not move: the same stationary
    points; and 300 alternations from the same start approach the same point
    without getting lower."""
    from htfa._kernels import ProjectedProblem, bounded_lm
    from htfa.synthetic import make_tfa_data

    X, R, truth = make_tfa_data((30, 30), 40, K=4, noise=0.3, random_state=1)
    K, D = truth["centers"].shape
    rng = np.random.default_rng(3)
    x0 = np.concatenate(
        [(truth["centers"] + rng.normal(0, 1.5, (K, D))).ravel(), truth["widths"] * 1.5]
    )
    bnds = bounds(R, K, 0.02, 1.8)
    prob = ProjectedProblem(X, R, K, NumpyEngine())
    x, cost, status, nfev = bounded_lm(
        prob, x0, bnds, "linear", 1.0 / prob.column_norms(x0)
    )
    fixed = ResidualProblem(X, prob.weights(x), R, NumpyEngine())
    x_alt, _ = solve_centers_widths(fixed, x, bnds, loss="linear")
    step = np.abs(x_alt - x).max()
    xa = x0
    for _ in range(300):
        problem = ResidualProblem(X, prob.weights(xa), R, NumpyEngine())
        xa, _ = solve_centers_widths(problem, xa, bnds, loss="linear")
    alt_cost = prob.cost(xa)
    print(
        f"projected: status {status}, nfev {nfev}, cost {cost:.12g}; one "
        f"alternation step from it moves {step:.3e}; 300 alternations: cost "
        f"{alt_cost:.12g}, max |x_alt - x| {np.abs(xa - x).max():.3e}"
    )
    assert status in (1, 2, 4)
    assert step < 5e-5  # measured 1.1e-5 (ftol-limited; grid units, extent 30)
    assert cost <= alt_cost * (1 + 1e-10)
    assert np.abs(xa - x).max() < 1e-4  # measured 2.5e-5


class _CountingEngine(NumpyEngine):
    """A NumPy engine that counts how often the data are sent."""

    def __init__(self) -> None:
        super().__init__()
        self.n_set = 0

    def set(self, X, W):
        self.n_set += 1
        super().set(X, W)


def test_projected_cost_matches_cost_grad_hess_and_sends_data_once():
    """cost and cost_grad_hess evaluate phi the same way (one engine residual),
    also where the OLS gram matrix is nearly singular, and the data go to the
    engine once per problem."""
    for method, scale in (("rr", 1.1), ("ols", 1.1), ("ols", 1e-6)):
        prob, x, *_ = _projected_problem(3, False, 600, method)
        x = x.copy()
        x[prob.K * prob.D] *= scale  # 1e-6: factor 0 nearly vanishes
        engine = _CountingEngine()
        prob = type(prob)(prob.X, prob.R, prob.K, engine, None, method)
        c1, c2 = prob.cost(x), prob.cost_grad_hess(x)[0]
        prob.column_norms(x)
        print(f"{method} x{scale}: cost {c1:.15g} vs {c2:.15g}, sets {engine.n_set}")
        assert c1 == pytest.approx(c2, rel=1e-13)
        assert engine.n_set == 1


def test_projected_ols_with_a_vanishing_factor_uses_the_pseudo_inverse():
    """With 'ols' and a factor that is 0 at every voxel, F^T F is singular: the
    objective is that of the minimum-norm least-squares weights, it is finite
    with finite derivatives, and the solver runs through it."""
    from htfa._kernels import bounded_lm

    prob, x, X, R, _ = _projected_problem(2, False, 700, "ols")
    K, D = prob.K, prob.D
    x = x.copy()
    x[:D] = R.max(axis=0) + 1e3  # factor 0 far outside the grid: F[:, 0] == 0
    F = factors(R, x[: K * D].reshape(K, D), x[K * D :])
    assert np.all(F[:, 0] == 0)
    W = np.linalg.lstsq(F, X, rcond=None)[0]
    expected = 0.5 * prob.s**2 * np.sum((X - F @ W) ** 2)
    cost, g, gn, H = prob.cost_grad_hess(x)
    print(f"cost {prob.cost(x):.12g} (cgh {cost:.12g}), lstsq {expected:.12g}")
    assert prob.cost(x) == pytest.approx(expected, rel=1e-10)
    assert all(np.all(np.isfinite(a)) for a in (g, gn, H))
    lb = np.concatenate([np.full(K * D, -1e4), np.full(K, 0.1)])
    ub = np.concatenate([np.full(K * D, 1e4), np.full(K, 1e3)])
    x_new, c_new, status, _ = bounded_lm(prob, x, (lb, ub), "linear", np.ones(x.size))
    print(f"solved: cost {c_new:.12g}, status {status}")
    assert np.isfinite(c_new) and c_new <= expected


@pytest.mark.parametrize("gap", [1e-3, 1e-6, 1e-9])
def test_projected_ols_with_near_duplicate_factors_stays_consistent(gap):
    """Two factors whose centers are ``gap`` apart make the OLS gram matrix
    nearly singular (Cholesky still succeeds); the weights must stay bounded so
    that cost and cost_grad_hess agree to the solver's ftol and better."""
    prob, x, *_ = _projected_problem(3, False, 800, "ols")
    K, D = prob.K, prob.D
    x = x.copy()
    x[D : 2 * D] = x[:D] + gap
    x[K * D + 1] = x[K * D]
    c1, c2 = prob.cost(x), prob.cost_grad_hess(x)[0]
    W = prob.weights(x)
    print(
        f"gap {gap}: cost {c1:.15g} vs {c2:.15g}, rel {abs(c1 - c2) / c1:.2e}, "
        f"max|W| {np.abs(W).max():.3g}"
    )
    assert abs(c1 - c2) <= 1e-12 * c1
