"""Dense oracle for the kernel shapes and mixtures (tests/test_shape_equivalence.py).

Independent of ``htfa._kernels``: each factor ``F(r; theta) = sum_m pi_m
phi(||r - c_m||**2 / w_m)`` is written symbolically from the definitions of
the profiles (the mexican hat as the normalized negative Laplacian of a
Gaussian, computed by sympy), and its gradient and Hessian in ``theta`` are
sympy's derivatives, evaluated per voxel. The objectives are then assembled
densely: the full Jacobian (V T x P), the residual Hessians
``-s sum_k W_kt d2F_vk``, the prior terms from their definitions, and, for
variable projection, the Hessian of the joint objective in (theta, W) reduced
by the Schur complement, and Kaufman's Gauss-Newton matrix from the dense
weight Jacobian.

Parameter order: ``x = [component centers (K M, D), component widths (K M,),
logits (K, M - 1)]``, components factor-major; a factor's local parameters
are its components' (c_1..c_D, w) in turn, then its logits.
"""

from functools import cache

import numpy as np
import sympy as sp

EPS_LAPLACE = sp.Rational(1, 10)
LOSS_EPS = np.finfo(float).eps  # scipy's floor on the loss weights


def _profile(name: str, delta, w):
    """Profile at the offset ``delta = r - c`` (a list of D expressions)."""
    d = sum(e**2 for e in delta)
    if name == "rbf":
        return sp.exp(-d / w)
    if name == "laplace":
        return sp.exp(EPS_LAPLACE - sp.sqrt(d / w + EPS_LAPLACE**2))
    # mexican hat: -Laplacian(exp(-|q|^2 / w)), divided by its value at q = 0.
    q = sp.symbols(f"q0:{len(delta)}", real=True)
    g = sp.exp(-sum(qi**2 for qi in q) / w)
    neg_lap = -sum(sp.diff(g, qi, 2) for qi in q)
    at0 = neg_lap.subs({qi: 0 for qi in q})
    return (neg_lap / at0).subs(dict(zip(q, delta, strict=True)), simultaneous=True)


@cache
def factor_functions(name: str, M: int, D: int):
    """Numerical F, dF/dtheta (A,) and d2F/dtheta2 (A, A) of one factor, as
    functions of (coordinates (V, D), theta (A,)) returning arrays over V."""
    r = sp.symbols(f"r0:{D}", real=True)
    params = []
    F = 0
    logits = [sp.Integer(0)] + [sp.Symbol(f"a{m}", real=True) for m in range(1, M)]
    denom = sum(sp.exp(a) for a in logits)
    for m in range(M):
        c = sp.symbols(f"c{m}_0:{D}", real=True)
        w = sp.Symbol(f"w{m}", positive=True)
        params += [*c, w]
        delta = [ri - ci for ri, ci in zip(r, c, strict=True)]
        F += sp.exp(logits[m]) / denom * _profile(name, delta, w)
    params += logits[1:]
    grad = [sp.diff(F, p) for p in params]
    hess = [[sp.diff(g, q) for q in params] for g in grad]
    args = (*r, *params)
    f_F = sp.lambdify(args, F, "numpy", cse=True)
    f_G = sp.lambdify(args, grad, "numpy", cse=True)
    f_H = sp.lambdify(args, hess, "numpy", cse=True)
    A = len(params)

    def flat(out):
        if isinstance(out, (list, tuple)):
            return [leaf for o in out for leaf in flat(o)]
        return [out]

    def call(fn, R, theta, shape):
        V = R.shape[0]
        leaves = flat(fn(*R.T, *theta))
        rows = [np.broadcast_to(np.asarray(o, dtype=float), (V,)) for o in leaves]
        return np.array(rows).reshape(*shape, V)

    return (
        lambda R, t: np.broadcast_to(np.asarray(f_F(*R.T, *t), float), (len(R),)),
        lambda R, t: call(f_G, R, t, (A,)),
        lambda R, t: call(f_H, R, t, (A, A)),
        A,
    )


def positions(K: int, M: int, D: int) -> np.ndarray:
    """(K, A) positions in x of each factor's local parameters."""
    n = K * M
    pos = []
    for k in range(K):
        row = []
        for m in range(M):
            c = k * M + m
            row += [c * D + j for j in range(D)] + [n * D + c]
        row += [n * (D + 1) + k * (M - 1) + j for j in range(M - 1)]
        pos.append(row)
    return np.array(pos)


def dense_factors(name, M, R, x, K):
    """F (V, K), G (K, A, V) and H (K, A, A, V) at x."""
    D = R.shape[1]
    f_F, f_G, f_H, A = factor_functions(name, M, D)
    pos = positions(K, M, D)
    F = np.stack([f_F(R, x[pos[k]]) for k in range(K)], axis=1)
    G = np.stack([f_G(R, x[pos[k]]) for k in range(K)])
    H = np.stack([f_H(R, x[pos[k]]) for k in range(K)])
    return F, G, H, pos


def loss_terms(loss, z):
    """rho, rho', rho'' of scipy's losses at z = r**2."""
    if loss == "linear":
        return z, np.ones_like(z), np.zeros_like(z)
    if loss == "soft_l1":
        t = 1 + z
        return 2 * (t**0.5 - 1), t**-0.5, -0.5 * t**-1.5
    if loss == "huber":
        mask = z <= 1
        rho = np.where(mask, z, 2 * np.sqrt(np.maximum(z, 1)) - 1)
        rho1 = np.where(mask, 1, np.maximum(z, 1) ** -0.5)
        rho2 = np.where(mask, 0, -0.5 * np.maximum(z, 1) ** -1.5)
        return rho, rho1, rho2
    if loss == "cauchy":
        return np.log1p(z), 1 / (1 + z), -1 / (1 + z) ** 2
    raise ValueError(loss)


def prior_terms(x, K, M, D, prior, loss):
    """Cost, gradient and Hessian (floored as scipy floors loss weights) of
    the prior: center residuals sqrt(scale dc^T S^-1 dc) per component, and
    linear width and logit residuals sqrt(scale / var) (p - mean)."""
    n = K * M
    P = x.size
    C = x[: n * D].reshape(n, D)
    rest = x[n * D :]
    mean = np.concatenate([prior["widths"], np.ravel(prior.get("logits", []))])
    var = np.concatenate([prior["widths_var"], np.ravel(prior.get("logits_var", []))])
    scale = prior["scale"]
    cost, grad, H = 0.0, np.zeros(P), np.zeros((P, P))
    for c in range(n):
        S = np.linalg.inv(prior["centers_cov"][c])
        S = 0.5 * (S + S.T)
        dc = C[c] - prior["centers"][c]
        q = scale * dc @ S @ dc
        rho, rho1, rho2 = (float(v) for v in loss_terms(loss, np.array(q)))
        cost += 0.5 * rho
        sl = slice(c * D, (c + 1) * D)
        grad[sl] = rho1 * scale * S @ dc
        along = max(rho1 + 2 * q * rho2, LOSS_EPS)
        beta = (along - rho1) / q if q > 0 else 0.0
        H[sl, sl] = scale * (rho1 * S + beta * scale * np.outer(S @ dc, S @ dc))
    coef = np.sqrt(scale / var)
    r = coef * (rest - mean)
    rho, rho1, rho2 = loss_terms(loss, r * r)
    cost += 0.5 * float(np.sum(rho))
    idx = np.arange(n * D, P)
    grad[idx] = rho1 * r * coef
    H[idx, idx] = coef**2 * np.maximum(rho1 + 2 * r * r * rho2, LOSS_EPS)
    return cost, grad, H


def fixed_weight(name, M, R, X, W, x, prior, loss):
    """Cost, gradient, Gauss-Newton matrix and Hessian of the fixed-weight
    problem 0.5 sum rho((s (X - F W))**2) + prior, from the dense Jacobian."""
    K, T = W.shape
    V, D = R.shape
    std = float(np.std(X))
    s = 1 / (np.sqrt(2) * std) if std > 0 else 1.0
    F, G, H, pos = dense_factors(name, M, R, x, K)
    P = x.size
    r = s * (X - F @ W)  # (V, T)
    rho, rho1, rho2 = loss_terms(loss, r * r)
    J = np.zeros((V, T, P))
    for k in range(K):
        for a, p in enumerate(pos[k]):
            J[:, :, p] = -s * np.outer(G[k, a], W[k])
    J = J.reshape(V * T, P)
    rv, r1, r2 = r.ravel(), rho1.ravel(), rho2.ravel()
    u = np.maximum(r1 + 2 * rv * rv * r2, LOSS_EPS)
    cost = 0.5 * float(np.sum(rho))
    grad = J.T @ (r1 * rv)
    gn = J.T @ (u[:, None] * J)
    second = np.zeros((P, P))
    Mvk = (rho1 * r) @ W.T  # (V, K): sum_t rho' r W_kt
    for k in range(K):
        block = -s * np.einsum("v,abv->ab", Mvk[:, k], H[k])
        second[np.ix_(pos[k], pos[k])] += block
    if prior is not None:
        c_p, g_p, H_p = prior_terms(x, K, M, D, prior, loss)
        cost, grad, gn = cost + c_p, grad + g_p, gn + H_p
    return cost, grad, gn, gn + second


def projected(name, M, R, X, x, K, prior, method):
    """Variable projection: cost, gradient, Kaufman Gauss-Newton matrix and
    Hessian, with W = argmin 0.5 s**2 (||X - F W||**2 + beta ||W||**2) from a
    dense pseudo-inverse with the code's cutoffs, and the Hessian as the
    Schur complement of the joint Hessian in (theta, W)."""
    V, D = R.shape
    T = X.shape[1]
    beta = float(np.var(X)) if method == "rr" else 0.0
    std = float(np.std(X))
    s = 1 / (np.sqrt(2) * std) if std > 0 else 1.0
    F, G, H, pos = dense_factors(name, M, R, x, K)
    gram = F.T @ F + beta * np.eye(K)
    vals, vecs = np.linalg.eigh(gram)
    keep = vals > max(np.sqrt(np.finfo(float).eps) * vals[-1],
                      np.finfo(float).tiny / np.finfo(float).eps)  # fmt: skip
    inv = (vecs[:, keep] / vals[keep]) @ vecs[:, keep].T
    W = inv @ F.T @ X
    E = X - F @ W
    P = x.size
    cost = 0.5 * s**2 * (float(np.sum(E * E)) + beta * float(np.sum(W * W)))
    J = np.zeros((V, T, P))
    for k in range(K):
        for a, p in enumerate(pos[k]):
            J[:, :, p] = -s * np.outer(G[k, a], W[k])
    J = J.reshape(V * T, P)
    grad = J.T @ (s * E).ravel()
    # Joint Hessian blocks; W flattened as (l, t).
    H_tt = J.T @ J
    for k in range(K):
        Mk = -s * (s * E) @ W[k]  # sum_t r_vt (-s W_kt), r = s E
        H_tt[np.ix_(pos[k], pos[k])] += np.einsum("v,abv->ab", Mk, H[k])
    H_tW = np.zeros((P, K * T))
    for k in range(K):
        for a, p in enumerate(pos[k]):
            for j in range(K):
                # d2/dtheta_ka dW_jt of 0.5 s^2 ||X - F W||^2
                term = s**2 * (G[k, a] @ F[:, j]) * W[k]
                if j == k:
                    term = term - s**2 * (G[k, a] @ E)
                H_tW[p, j * T : (j + 1) * T] = term
    H_WW = np.kron(s**2 * gram, np.eye(T))
    H_WW_inv = np.kron(inv / s**2, np.eye(T))
    hess = H_tt - H_tW @ H_WW_inv @ H_tW.T
    JW = -s * np.kron(F, np.eye(T))  # (V T, K T)
    gn = J.T @ J - J.T @ JW @ H_WW_inv @ JW.T @ J
    del H_WW
    if prior is not None:
        c_p, g_p, H_p = prior_terms(x, K, M, D, prior, "linear")
        cost, grad, gn, hess = cost + c_p, grad + g_p, gn + H_p, hess + H_p
    return cost, grad, gn, hess, W
