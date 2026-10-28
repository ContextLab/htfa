"""Kernel shapes and mixtures against a dense sympy oracle, over generated scenarios.

The oracle (``tests/reference_shapes.py``) writes every factor symbolically
from the profile definitions and takes sympy's derivatives; it assembles the
objectives densely (full Jacobian, Schur complement of the joint Hessian for
variable projection). Each scenario draws a kernel, a number of components,
coordinates, sizes, widths, centers, mixing logits, weights, loss, prior and
data from `FAMILIES`, and compares factors, cost, gradient, Gauss-Newton
matrix, Hessian (and the varpro weights) at three points. Failures print the
scenario (``scenario(i)`` rebuilds it).

Scales as in tests/test_equivalence.py: factors relative to max|F|, costs
relative to the zero model's cost plus the prior's, gradients relative to
``sqrt(2 cost_scale lambda_max)`` of the fixed-weight Gauss-Newton matrix,
matrices relative to their largest entry (and, for varpro, that of the
fixed-weight matrices and of their difference). The tolerance is the larger
of `RTOL` and `AMP_FACTOR` times machine epsilon times the amplification
(the gram's condition number for varpro, max(|F| |W|) / max|X| otherwise);
stages whose tolerance would exceed `TOL_CAP` compare the cost and ``F W``
only (`ILL_TOL`) and are recorded in `SKIPS`.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
import reference_shapes as ref
from test_equivalence import _coords

import htfa._kernels as new
from htfa._engines import NumpyEngine

RTOL = 1e-9
AMP_FACTOR = 1e3
TOL_CAP = 1e-6
ILL_TOL = 1e-8
N_SCENARIOS = 240
SKIPS: list[tuple[int, str]] = []
RECORDS: list[tuple[int, str, float, float]] = []

FAMILIES: dict[str, tuple[Any, ...]] = {
    "kernel": new.KERNELS,
    "M": (1, 2, 3),
    "D": (2, 3),
    "coords": ("grid", "mni2mm", "rotated", "jittered", "large_offset", "random"),
    "V": ("tiny", "small"),
    "T": (1, 2, 7),
    "K": (1, 2, 4),
    "widths": ("small", "typical", "huge", "mixed"),
    "centers": ("inside", "edge", "outside", "mixed"),
    "logits": ("zero", "moderate", "extreme"),
    "weights": ("rr", "ols"),
    "loss": ("varpro", "linear", "soft_l1", "huber", "cauchy"),
    "prior": (False, True),
    "data": ("noiseless", "exact_fit", "realistic", "heavy_tailed", "offsets"),
}


@dataclass
class Scenario:
    index: int
    choice: dict[str, Any]
    R: np.ndarray = field(repr=False)
    X: np.ndarray = field(repr=False)
    x: np.ndarray = field(repr=False)
    prior: dict[str, Any] | None = field(repr=False)

    @property
    def kernel(self) -> new.Kernel:
        return new.Kernel(self.choice["kernel"], self.choice["M"])

    def __str__(self) -> str:
        return (
            f"scenario({self.index}): {self.choice} -> V={len(self.R)}, "
            f"T={self.X.shape[1]}, P={self.x.size}"
        )


def scenario(i: int) -> Scenario:
    rng = np.random.default_rng([2026, 10, 28, i])
    choice = {n: v[int(rng.integers(len(v)))] for n, v in FAMILIES.items()}
    D, K, M, T = choice["D"], choice["K"], choice["M"], choice["T"]
    V = {"tiny": int(rng.integers(8, 30)), "small": int(rng.integers(40, 200))}
    R = _coords(rng, D, choice["coords"], V[choice["V"]])
    V = len(R)
    lo, hi = R.min(0), R.max(0)
    extent = float(np.max(hi - lo)) + 1.0
    sig2 = 2.0 * float(np.max(np.std(R, axis=0))) ** 2 + 1.0
    n = K * M
    w = {
        "small": lambda: sig2 * rng.uniform(0.01, 0.05, n),
        "typical": lambda: sig2 * rng.uniform(0.05, 0.5, n),
        "huge": lambda: sig2 * rng.uniform(1e2, 1e4, n),
        "mixed": lambda: sig2 * 10 ** rng.uniform(-2, 2, n),
    }[choice["widths"]]()
    inside = R[rng.integers(0, V, n)] + rng.normal(0, 0.5, (n, D))
    edge = np.where(rng.uniform(size=(n, D)) < 0.5, lo, hi) + rng.normal(0, 0.1, (n, D))
    outside = hi + extent * rng.uniform(0.5, 1.5, (n, D))
    C = {
        "inside": inside,
        "edge": edge,
        "outside": outside,
        "mixed": np.where(rng.uniform(size=(n, 1)) < 0.5, inside, outside),
    }[choice["centers"]]
    a = {
        "zero": np.zeros((K, M - 1)),
        "moderate": rng.normal(0, 1, (K, M - 1)),
        "extreme": rng.choice([-6.0, 6.0], (K, M - 1)) + rng.normal(0, 0.3, (K, M - 1)),
    }[choice["logits"]]
    x = new.Kernel.join(C, w, a)
    kernel = new.Kernel(choice["kernel"], M)
    C_true = R[rng.integers(0, V, n)]
    w_true = sig2 * rng.uniform(0.05, 0.3, n)
    x_true = new.Kernel.join(C_true, w_true, rng.normal(0, 1, (K, M - 1)))
    W_true = rng.normal(size=(K, T))
    signal = new.x_factors(R, C_true, x_true[n * D :], K, kernel, None) @ W_true
    sd = float(np.std(signal)) or 1.0
    data = choice["data"]
    if data == "noiseless":
        X = signal
    elif data == "exact_fit":
        X = new.x_factors(R, C, x[n * D :], K, kernel, None) @ W_true
        X = X / (np.max(np.abs(X)) or 1.0)
    elif data == "realistic":
        X = signal + 4.9 * sd * rng.normal(size=signal.shape)
    elif data == "heavy_tailed":
        X = signal + 0.5 * sd * rng.standard_t(2, size=signal.shape)
    else:
        X = signal + 0.3 * sd * rng.normal(size=signal.shape)
        X += rng.normal(100.0, 10.0, (V, 1))
    prior = None
    if choice["prior"]:
        covs = np.empty((n, D, D))
        for c in range(n):
            A = rng.normal(size=(D, D))
            covs[c] = (A @ A.T + D * np.eye(D)) * extent
        prior = dict(
            centers=C + rng.normal(0, 2.0, C.shape),
            widths=w * rng.uniform(0.7, 1.3, n),
            centers_cov=covs,
            widths_var=(w * rng.uniform(0.1, 0.5, n)) ** 2 + 1e-12,
            scale=float(rng.uniform(0.1, 3.0)),
        )
        if M > 1:
            prior["logits"] = a + rng.normal(0, 0.5, a.shape)
            prior["logits_var"] = rng.uniform(0.5, 4.0, a.shape)
    return Scenario(i, choice, R, np.ascontiguousarray(X), x, prior)


def _points(sc: Scenario, rng) -> list[np.ndarray]:
    K, M, D = sc.choice["K"], sc.choice["M"], sc.choice["D"]
    n = K * M
    scale = np.concatenate(
        [np.ones(n * D), sc.x[n * D : n * (D + 1)], np.ones(K * (M - 1))]
    )
    x = sc.x
    lb = np.concatenate([np.full(n * (D + 1), -np.inf), np.full(K * (M - 1), -6.9)])
    ub = np.concatenate([np.full(n * (D + 1), np.inf), np.full(K * (M - 1), 6.9)])
    far = x + 0.2 * scale * np.abs(rng.normal(size=x.size))
    return [x, x + 1e-3 * scale * rng.normal(size=x.size), np.clip(far, lb, ub)]


def _new_prior(sc: Scenario):
    return None if sc.prior is None else new.Prior(**sc.prior)


def compare(sc: Scenario) -> dict[str, float]:
    rng = np.random.default_rng(sc.index)
    R, X, kernel = sc.R, sc.X, sc.kernel
    K, M, D = sc.choice["K"], sc.choice["M"], sc.choice["D"]
    name, method, loss = sc.choice["kernel"], sc.choice["weights"], sc.choice["loss"]
    n = K * M
    x_max = float(np.max(np.abs(X)))
    eps = np.finfo(float).eps
    out: dict[str, float] = {}

    def check(what, diff, scale, tol):
        r = diff / scale if scale > 0 else diff
        out[what] = max(out.get(what, 0.0), r)
        RECORDS.append((sc.index, what, r, tol))
        assert r <= tol, f"{what}: rel diff {r:.3e} > {tol:.1e}\n{sc}"

    def mdiff(a, b):
        return (
            float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) if np.size(a) else 0.0
        )

    def mabs(a):
        return float(np.max(np.abs(a))) if np.size(a) else 0.0

    for x in _points(sc, rng):
        C, rest = x[: n * D].reshape(n, D), x[n * D :]
        F_new = new.x_factors(R, C, rest, K, kernel, new.axis_grid(R))
        F_ref, *_ = ref.dense_factors(name, M, R, x, K)
        check("factors", mdiff(F_new, F_ref), max(mabs(F_ref), 1e-300), RTOL)
        if mabs(F_ref) < 1e-150:
            SKIPS.append((sc.index, "every factor below 1e-150"))
            continue
        prior = _new_prior(sc)
        if loss == "varpro":
            beta = float(np.var(X)) if method == "rr" else 0.0
            vals = np.linalg.eigvalsh(F_ref.T @ F_ref + beta * np.eye(K))
            if vals[0] <= max(np.sqrt(eps) * vals[-1], 1e-290):
                SKIPS.append((sc.index, "varpro: a gram eigenvalue is dropped"))
                continue
            amp = vals[-1] / vals[0]
            tol = max(RTOL, AMP_FACTOR * eps * amp)
            p = new.ProjectedProblem(X, R, K, NumpyEngine(1), prior, method, kernel)
            c_r, g_r, gn_r, H_r, W_r = ref.projected(
                name, M, R, X, x, K, sc.prior, method
            )
            Wf = p.weights(x)
            fx = ref.fixed_weight(name, M, R, X, W_r, x, sc.prior, "linear")
            zero = 0.5 * float(np.sum((X / (np.sqrt(2) * np.std(X))) ** 2))
            cost_scale = zero + abs(fx[0] - zero) if np.std(X) > 0 else 1.0
            if tol > TOL_CAP:
                check("varpro cost (ill)", abs(p.cost(x) - c_r), cost_scale, ILL_TOL)
                check(
                    "varpro F W (ill)", mdiff(F_new @ Wf, F_ref @ W_r), x_max, ILL_TOL
                )
                SKIPS.append((sc.index, f"varpro: amplification {amp:.1e}"))
                continue
            c, g, gn, H = p.cost_grad_hess(x)
            check("varpro cost", abs(c - c_r), cost_scale, tol)
            check("varpro weights", mdiff(Wf, W_r), mabs(W_r), tol)
            lam = float(np.max(np.sum(np.abs(fx[2]), axis=1)))
            check("varpro gradient", mdiff(g, g_r),
                  max(mabs(g_r), np.sqrt(2 * cost_scale * lam)), tol)  # fmt: skip
            for what, a, b, f in (("varpro Gauss-Newton", gn, gn_r, fx[2]),
                                  ("varpro Hessian", H, H_r, fx[3])):  # fmt: skip
                check(what, mdiff(a, b), max(mabs(b), mabs(f), mabs(f - b)), tol)
            continue
        Wd = new.ProjectedProblem(X, R, K, NumpyEngine(1), None, method,
                                  kernel).weights(x)  # fmt: skip
        amp = max(mabs(np.abs(F_ref) @ np.abs(Wd)) / x_max if x_max > 0 else 1.0, 1.0)
        tol = max(RTOL, AMP_FACTOR * eps * amp)
        c_r, g_r, gn_r, H_r = ref.fixed_weight(name, M, R, X, Wd, x, sc.prior, loss)
        p = new.ResidualProblem(X, Wd, R, NumpyEngine(1), prior, loss == "linear",
                                kernel=kernel)  # fmt: skip
        s = p.s
        zero = 0.5 * float(np.sum(ref.loss_terms(loss, (s * X) ** 2)[0]))
        prior_cost = 0.0 if sc.prior is None else ref.prior_terms(
            x, K, M, D, sc.prior, loss)[0]  # fmt: skip
        cost_scale = zero + abs(prior_cost)
        if tol > TOL_CAP:
            c = p.cost(x, loss) + p.cost_offset
            check(f"{loss} cost (ill)", abs(c - c_r), cost_scale, ILL_TOL)
            SKIPS.append((sc.index, f"{loss}: amplification {amp:.1e}"))
            continue
        c, g, gn, H = p.cost_grad_hess(x, loss)
        # The compressed linear-loss problem keeps the projected-out part apart.
        check(f"{loss} cost", abs(c + p.cost_offset - c_r), cost_scale, tol)
        check(f"{loss} cost()", abs(p.cost(x, loss) - c), cost_scale, tol)
        lam = float(np.max(np.sum(np.abs(gn_r), axis=1)))
        check(f"{loss} gradient", mdiff(g, g_r),
              max(mabs(g_r), np.sqrt(2 * cost_scale * lam)), tol)  # fmt: skip
        check(f"{loss} Gauss-Newton", mdiff(gn, gn_r), mabs(gn_r), tol)
        check(f"{loss} Hessian", mdiff(H, H_r), max(mabs(H_r), mabs(gn_r)), tol)
    return out


def test_scenarios_cover_every_family_value():
    seen = {n: set() for n in FAMILIES}
    for i in range(N_SCENARIOS):
        for n, v in scenario(i).choice.items():
            seen[n].add(v)
    missing = {n: set(v) - seen[n] for n, v in FAMILIES.items() if set(v) - seen[n]}
    assert not missing


@pytest.mark.parametrize("block", range(8))
def test_shapes_equal_dense_oracle_on_generated_scenarios(block):
    size = N_SCENARIOS // 8
    worst: dict[str, float] = {}
    for i in range(block * size, (block + 1) * size):
        for k, v in compare(scenario(i)).items():
            worst[k] = max(worst.get(k, 0.0), v)
    print(
        f"block {block}: " + ", ".join(f"{k} {v:.2e}" for k, v in sorted(worst.items()))
    )
    print(f"skipped stages so far: {len(SKIPS)}")
