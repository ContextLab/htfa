"""Kernel shapes and mixtures (`htfa._kernels.Kernel`): factors, derivatives,
problems and solves, against brute-force formulas and finite differences."""

import itertools

import numpy as np
import pytest

from htfa._engines import EPS, NumpyEngine, loss_terms
from htfa._kernels import (
    KERNELS,
    LAPLACE_EPS,
    MIXING_LOGIT_BOUND,
    RBF,
    Kernel,
    Prior,
    ProjectedProblem,
    ResidualProblem,
    axis_grid,
    bounded_lm,
    bounds,
    factor_derivatives,
    mixture_derivatives,
    param_positions,
    x_factors,
)

# Every kernel with one, two and three components (rbf with one is the plain
# path, covered by tests/test_kernels.py and tests/test_equivalence.py).
SHAPES = [
    (name, M)
    for name, M in itertools.product(KERNELS, (1, 2, 3))
    if (name, M) != ("rbf", 1)
]
SHAPE_IDS = [f"{n}-M{m}" for n, m in SHAPES]
# Central differences with step 1e-6 (as tests/test_kernels.py). Measured
# relative errors: Jacobians <= 3.5e-9, gradients <= 1.1e-7, Hessians <= 1.6e-8.
FD_STEP = 1e-6
FD_RTOL = 1e-6
# Gauss-Newton matrix and gradient vs. the dense Jacobian (the same sums in
# another order): measured <= 4e-15.
DENSE_RTOL = 1e-11


def _phi(name, d, w, D):
    """The profiles written out from their definitions (``d`` the squared
    distance); the mexican hat as -Laplacian(exp(-d / w)) / its value at 0,
    with Laplacian(exp(-d / w)) = (4 d / w**2 - 2 D / w) exp(-d / w)."""
    if name == "rbf":
        return np.exp(-d / w)
    if name == "laplace":
        return np.exp(LAPLACE_EPS - np.sqrt(d / w + LAPLACE_EPS**2))
    lap = (4 * d / w**2 - 2 * D / w) * np.exp(-d / w)
    return -lap / (2 * D / w)


def _brute_factors(R, C, w, logits, name, M):
    """Factors by a double loop over voxels and components."""
    K = len(logits)
    F = np.zeros((len(R), K))
    for k in range(K):
        a = np.concatenate([[0.0], logits[k]])
        pi = np.exp(a) / np.sum(np.exp(a))
        for v in range(len(R)):
            for m in range(M):
                c = k * M + m
                d = float(np.sum((R[v] - C[c]) ** 2))
                F[v, k] += pi[m] * _phi(name, d, w[c], R.shape[1])
    return F


def _shape_inputs(name, M, D, seed, V=60, T=7, K=2):
    kernel = Kernel(name, M)
    rng = np.random.default_rng(seed)
    R = rng.uniform(0.0, 10.0, size=(V, D))
    n = K * M
    C = rng.uniform(2.0, 8.0, size=(n, D))
    w = rng.uniform(4.0, 12.0, size=n)
    logits = rng.normal(0.0, 0.7, size=(K, M - 1))
    W = rng.normal(size=(K, T))
    x_true = Kernel.join(C, w, logits)
    F = x_factors(R, C, x_true[n * D :], K, kernel, None)
    X = F @ W + 0.1 * rng.normal(size=(V, T)) + 0.3 * rng.standard_t(2, (V, T))
    x = x_true + np.concatenate(
        [rng.normal(0, 0.4, n * D), w * rng.uniform(0.0, 0.2, n),
         rng.normal(0, 0.3, K * (M - 1))]
    )  # fmt: skip
    return kernel, R, X, W, x, rng


def _shape_prior(rng, kernel, x, K, D):
    C, w, logits = kernel.split(x, K, D)
    n = len(C)
    covs = np.empty((n, D, D))
    for c in range(n):
        A = rng.normal(size=(D, D))
        covs[c] = A @ A.T + D * np.eye(D)
    M = kernel.n_components
    return Prior(
        centers=C + rng.normal(0.0, 1.0, C.shape),
        widths=w * rng.uniform(0.7, 1.3, n),
        centers_cov=covs,
        widths_var=rng.uniform(1.0, 5.0, n),
        scale=0.37,
        logits=None if M == 1 else logits + rng.normal(0, 0.5, logits.shape),
        logits_var=None if M == 1 else rng.uniform(0.5, 2.0, logits.shape),
    )


def _fd(f, x, step=FD_STEP):
    """Central differences of f (scalar or vector valued), one column per x_j."""
    cols = []
    for j in range(x.size):
        e = np.zeros_like(x)
        e[j] = step
        cols.append((np.asarray(f(x + e)) - np.asarray(f(x - e))) / (2 * step))
    return np.stack(cols, axis=-1)


def test_kernel_validation_and_layout():
    with pytest.raises(ValueError, match="kernel must be one of"):
        Kernel("mixture")
    for bad in (0, 1.5, True, "2"):
        with pytest.raises(ValueError, match="n_components must be an integer"):
            Kernel("rbf", bad)
    k = Kernel("laplace", 3)
    assert k.n_params(4, 2) == 4 * 3 * 3 + 4 * 2
    x = np.arange(k.n_params(4, 2), dtype=float)
    C, w, a = k.split(x, 4, 2)
    assert C.shape == (12, 2) and w.shape == (12,) and a.shape == (4, 2)
    np.testing.assert_array_equal(Kernel.join(C, w, a), x)
    # Positions: each factor's components (c, w) in turn, then its logits.
    pos = param_positions(4, 2, 3).reshape(4, -1)
    assert sorted(pos.ravel()) == list(range(x.size))
    np.testing.assert_array_equal(pos[1, :3], [6, 7, 24 + 3])  # comp 3: c, w
    np.testing.assert_array_equal(pos[1, 9:], [36 + 2, 36 + 3])
    np.testing.assert_array_equal(param_positions(5, 3, 1), param_positions(5, 3))
    pi = Kernel.mixing(np.array([[0.0, np.log(3.0)]]))
    np.testing.assert_allclose(pi, [[0.2, 0.2, 0.6]], rtol=1e-15)
    np.testing.assert_allclose(Kernel.logits(pi), [[0.0, np.log(3.0)]], atol=1e-15)
    with pytest.raises(ValueError, match="positive"):
        Kernel.logits(np.array([[0.5, 0.0]]))


def test_start_and_bounds_per_kernel():
    R = np.stack(np.meshgrid(np.arange(6.0), np.arange(5.0)), -1).reshape(-1, 2)
    C = np.array([[1.0, 2.0], [4.0, 3.0]])
    w = np.array([3.0, 5.0])
    np.testing.assert_array_equal(Kernel("laplace").start(C, w), [1, 2, 4, 3, 3, 5])
    x = Kernel("rbf", 3).start(C, w)
    Cc, wc, a = Kernel("rbf", 3).split(x, 2, 2)
    np.testing.assert_array_equal(Cc, np.repeat(C, 3, axis=0))
    np.testing.assert_allclose(wc, [3, 0.75, 0.1875, 5, 1.25, 0.3125])
    np.testing.assert_array_equal(a, 0.0)
    lb, ub = bounds(R, 2, 0.1, 2.0, Kernel("mexican_hat", 2))
    lb1, ub1 = bounds(R, 4, 0.1, 2.0)
    np.testing.assert_array_equal(lb[:12], lb1)
    np.testing.assert_array_equal(ub[:12], ub1)
    np.testing.assert_array_equal(lb[12:], -MIXING_LOGIT_BOUND)
    np.testing.assert_array_equal(ub[12:], MIXING_LOGIT_BOUND)


@pytest.mark.parametrize("name,M", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("D", [2, 3])
def test_kernel_factors_match_bruteforce(name, M, D):
    """x_factors (with and without the axis grid) against a scalar loop over
    the profile definitions; profiles are 1 at their centers."""
    kernel, *_ = _shape_inputs(name, M, D, 10 + D)
    rng = np.random.default_rng(D)
    axes = [np.arange(n, dtype=float) for n in (7, 6, 5)[:D]]
    R = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, D)
    K = 3
    C = rng.uniform(0, 5, (K * M, D))
    w = rng.uniform(1, 9, K * M)
    a = rng.normal(0, 1, (K, M - 1))
    tail = np.concatenate([w, a.ravel()])
    expected = _brute_factors(R, C, w, a, name, M)
    for grid in (None, axis_grid(R)):
        got = x_factors(R, C, tail, K, kernel, grid)
        err = np.abs(got - expected).max()
        print(f"{name} M={M} D={D} grid={grid is not None}: max err {err:.2e}")
        assert err < 1e-14
    one = Kernel(name).profile(np.zeros(1), D)
    assert one[0] == 1.0


@pytest.mark.parametrize("D", [2, 3])
def test_one_rbf_component_general_derivatives_equal_rbf_formulas(D):
    """mixture_derivatives on one RBF component (the general path) gives the
    factors and first derivatives of factor_derivatives, and its curvature
    equals the second derivatives of exp(-d / w) by central differences of G."""
    rng = np.random.default_rng(D)
    R = rng.uniform(0, 10, (50, D))
    C = rng.uniform(2, 8, (3, D))
    w = rng.uniform(4, 12, 3)
    der = mixture_derivatives(R, C, w, np.zeros((3, 0)), RBF)
    _, F, G = factor_derivatives(R, C, w)
    np.testing.assert_allclose(der.F, F, rtol=1e-14, atol=0)
    np.testing.assert_allclose(der.G, G, rtol=1e-12, atol=1e-15)
    MT = rng.normal(size=(3, 50))
    S = der.curvature(MT)
    x = np.concatenate([C.ravel(), w])
    pos = param_positions(3, D).reshape(3, D + 1)

    def g_of(y):
        Cy, wy = y[: 3 * D].reshape(3, D), y[3 * D :]
        return factor_derivatives(R, Cy, wy)[2]

    dG = _fd(g_of, x)  # (K, A, V, P)
    for k in range(3):
        expected = np.einsum("v,avb->ab", MT[k], dG[k][:, :, pos[k]])
        rel = np.abs(S[k] - expected).max() / np.abs(expected).max()
        assert rel < FD_RTOL, rel


@pytest.mark.parametrize("name,M", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("D", [2, 3])
@pytest.mark.parametrize("loss", ["linear", "soft_l1", "cauchy"])
@pytest.mark.parametrize("use_prior", [False, True], ids=["noprior", "prior"])
def test_shape_derivatives_match_finite_differences(name, M, D, loss, use_prior):
    """ResidualProblem for every kernel and mixture: the gradient against
    central differences of the cost, the Hessian against central differences
    of the gradient (plus scipy's EPS floor on the loss weights, as in
    tests/test_kernels.py), and gradient, Gauss-Newton matrix and column norms
    against the dense matrix-free Jacobian (itself checked against central
    differences of the residual)."""
    kernel, R, X, W, x, rng = _shape_inputs(name, M, D, 100 * M + D)
    K = W.shape[0]
    prior = _shape_prior(rng, kernel, x, K, D) if use_prior else None
    prob = ResidualProblem(X, W, R, NumpyEngine(), prior, kernel=kernel)
    assert prob.n_params == x.size
    cost, g, gn, H = prob.cost_grad_hess(x, loss)
    r = prob.residual(x)
    assert cost == pytest.approx(0.5 * np.sum(loss_terms(loss, r * r)[0]), rel=1e-13)
    assert prob.cost(x, loss) == pytest.approx(cost, rel=1e-13)
    J = prob.jacobian(x).matmat(np.eye(x.size))
    J_fd = _fd(prob.residual, x)
    j_rel = np.abs(J - J_fd).max() / np.abs(J_fd).max()
    g_fd = _fd(lambda y: prob.cost(y, loss), x)
    g_rel = np.abs(g - g_fd).max() / np.abs(g_fd).max()
    _, rho1, rho2 = loss_terms(loss, r * r)
    u = rho1 + 2 * r * r * rho2
    floor = J.T @ ((np.maximum(u, EPS) - u)[:, None] * J)
    H_fd = _fd(lambda y: prob.cost_grad_hess(y, loss)[1], x) + floor
    h_rel = np.abs(H - H_fd).max() / np.abs(H_fd).max()
    # The Gauss-Newton matrix holds the prior's exact (floored) Hessian, not
    # J^T J of its square-root center residuals: compare the data part.
    nd = prob.n_data
    gn_dense = J[:nd].T @ (np.maximum(u[:nd], EPS)[:, None] * J[:nd])
    if prior is not None:
        gn = gn - prob._prior_terms(*prob._unpack(x), loss, derivatives=True)[2]
    gn_rel = np.abs(gn - gn_dense).max() / np.abs(gn_dense).max()
    norms = prob.column_norms(x)
    n_rel = np.abs(norms - np.linalg.norm(J, axis=0)).max() / norms.max()
    rt = prob.jacobian(x).rmatvec(r)
    print(
        f"{name} M={M} D={D} {loss} prior={use_prior}: J {j_rel:.2e} grad "
        f"{g_rel:.2e} hess {h_rel:.2e} GN {gn_rel:.2e} norms {n_rel:.2e}"
    )
    assert j_rel < FD_RTOL
    assert g_rel < FD_RTOL
    assert h_rel < FD_RTOL
    assert gn_rel < DENSE_RTOL
    assert n_rel < DENSE_RTOL
    np.testing.assert_allclose(g, J.T @ (rho1 * r), rtol=DENSE_RTOL, atol=1e-13)
    np.testing.assert_allclose(rt, J.T @ r, rtol=DENSE_RTOL, atol=1e-12)
    np.testing.assert_array_equal(H, H.T)
    np.testing.assert_array_equal(gn, gn.T)


@pytest.mark.parametrize("name,M", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("D", [2, 3])
@pytest.mark.parametrize("method", ["rr", "ols"])
@pytest.mark.parametrize("use_prior", [False, True], ids=["noprior", "prior"])
def test_projected_shape_derivatives_match_finite_differences(
    name, M, D, method, use_prior
):
    """ProjectedProblem (varpro) for every kernel and mixture: envelope
    gradient and projected Hessian against central differences; the Kaufman
    Gauss-Newton matrix is symmetric positive semidefinite; fast mode is close
    to exact."""
    kernel, R, X, W, x, rng = _shape_inputs(name, M, D, 200 * M + D)
    K = W.shape[0]
    prior = _shape_prior(rng, kernel, x, K, D) if use_prior else None
    prob = ProjectedProblem(X, R, K, NumpyEngine(), prior, method, kernel=kernel)
    cost, g, gn, H = prob.cost_grad_hess(x)
    assert prob.cost(x) == pytest.approx(cost, rel=1e-12)
    g_fd = _fd(prob.cost, x)
    H_fd = _fd(lambda y: prob.cost_grad_hess(y)[1], x)
    g_rel = np.abs(g - g_fd).max() / np.abs(g_fd).max()
    H_rel = np.abs(H - H_fd).max() / np.abs(H_fd).max()
    eig = np.linalg.eigvalsh(gn)
    fast = prob.cost_grad_hess(x, mode="fast")
    f_rel = np.abs(fast[3] - H).max() / np.abs(H).max()
    print(
        f"{name} M={M} D={D} {method} prior={use_prior}: grad {g_rel:.2e} "
        f"hess {H_rel:.2e} GN eig min {eig.min():.3g}, fast vs exact {f_rel:.2e}"
    )
    assert g_rel < FD_RTOL
    assert H_rel < FD_RTOL
    assert eig.min() > -1e-10 * eig.max()
    np.testing.assert_array_equal(H, H.T)
    assert fast[0] == cost
    np.testing.assert_array_equal(fast[1], g)
    assert f_rel < 1e-4


@pytest.mark.parametrize("name,M", SHAPES, ids=SHAPE_IDS)
def test_bounded_lm_recovers_true_parameters_given_true_weights(name, M):
    """Noiseless data with the true weights: bounded_lm from a perturbed start
    returns the true centers, widths and logits (same component order)."""
    D, K, T = 2, 2, 30
    kernel = Kernel(name, M)
    rng = np.random.default_rng(M)
    axes = np.arange(16.0)
    R = np.stack(np.meshgrid(axes, axes, indexing="ij"), -1).reshape(-1, 2)
    n = K * M
    C = np.repeat(np.array([[4.0, 4.0], [11.0, 10.0]]), M, axis=0)
    C = C + rng.normal(0, 1.5, C.shape)
    w = np.tile(8.0 * 0.45 ** np.arange(M), K) * rng.uniform(0.8, 1.2, n)
    a = rng.normal(0, 0.5, (K, M - 1))
    x_true = Kernel.join(C, w, a)
    W = rng.normal(size=(K, T))
    X = x_factors(R, C, x_true[n * D :], K, kernel, None) @ W
    prob = ResidualProblem(X, W, R, NumpyEngine(), kernel=kernel)
    x0 = x_true + np.concatenate(
        [rng.normal(0, 0.3, n * D), w * 0.1, rng.normal(0, 0.2, K * (M - 1))]
    )
    lb, ub = bounds(R, K, 0.01, 2.0, kernel)
    x, cost, status, nfev = bounded_lm(prob, x0, (lb, ub), "linear", np.ones(x0.size))
    err = np.abs(x - x_true)
    print(f"{name} M={M}: status {status}, nfev {nfev}, cost {cost:.3e}, max err "
          f"{err.max():.2e}")  # fmt: skip
    assert err.max() < 1e-5
