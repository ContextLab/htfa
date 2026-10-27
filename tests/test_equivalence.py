"""The current kernels against the formulas of commit 3298278, over generated scenarios.

The oracle is ``tests/reference_kernels_3298278.py``: the kernels as they were
before the per-axis factors, the reuse of a cost evaluation in cost_grad_hess
and the ``||X||**2 - <F^T X, W>`` objective. Each scenario draws coordinates,
sizes, widths, centers, weights, loss, prior and data from the families listed
in `FAMILIES`, and compares factors, factor derivatives, cost, gradient,
Gauss-Newton matrix, Hessian and weights at three points. Failures print the
scenario (``scenario(i)`` rebuilds it).

Tolerances (relative to the largest entry of each array): `RTOL` for
factors and their derivatives. For cost, gradient, Gauss-Newton matrix,
Hessian and weights (the gradient relative to its scale ``sqrt(2 cost
lambda_max(GN))``, the projected matrices relative to the fixed-weight ones),
the larger of `RTOL` and `COND_FACTOR` x amplification x
the measured factor difference, where the amplification is the gram's
condition number (varpro) or the cancellation ratio max(|F| |W|) / max|X - F W|,
whichever is larger: the first-order bound for a linear solve, or a residual,
whose inputs change by that much. The sweep prints how often this exceeds
`RTOL`.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
import reference_kernels_3298278 as ref

import htfa._kernels as new
from htfa._engines import NumpyEngine

# Measured maxima over the sweep are printed by test_sweep_summary.
RTOL = 1e-10
# Factors per entry (normal numbers): per-axis products vs exp of the sum.
FACTOR_RTOL = 1e-12
COND_FACTOR = 100.0
N_FAST = 480
N_SLOW = 12
# (scenario, quantity, relative difference, tolerance) of every comparison,
# for summaries of the sweep (scripts/equivalence_summary.py).
RECORDS: list[tuple[int, str, float, float]] = []

FAMILIES: dict[str, tuple[Any, ...]] = {
    "D": (2, 3),
    "coords": ("grid", "mni2mm", "rotated", "jittered", "large_offset", "random"),
    "V": ("tiny", "small", "medium"),
    "T": (1, 2, 7, 40),
    "K": (1, 2, 5, 12, 30),
    "widths": ("tiny", "small", "typical", "huge", "mixed"),
    "centers": ("inside", "edge", "outside", "mixed"),
    "weights": ("rr", "ols"),
    "special": ("none", "near_duplicate", "vanishing", "near_singular"),
    "loss": ("varpro", "linear", "soft_l1", "huber", "linear+soft_l1"),
    "prior": (False, True),
    "data": ("noiseless", "realistic", "heavy_tailed", "constant_voxels", "offsets"),
}


@dataclass
class Scenario:
    index: int
    choice: dict[str, Any]
    R: np.ndarray = field(repr=False)
    X: np.ndarray = field(repr=False)
    C: np.ndarray = field(repr=False)
    w: np.ndarray = field(repr=False)
    prior: dict[str, Any] | None = field(repr=False)

    def __str__(self) -> str:
        K, D = self.C.shape
        return (
            f"scenario({self.index}): {self.choice} -> V={len(self.R)}, "
            f"T={self.X.shape[1]}, K={K}, D={D}, "
            f"grid={new.axis_grid(self.R) is not None}"
        )


def _grid_axes(rng, D, n_per_axis, spacing):
    origin = rng.uniform(-50, 50, D)
    return [origin[j] + spacing[j] * np.arange(n_per_axis[j]) for j in range(D)]


def _coords(rng, D, kind, V):
    """Coordinates (about V voxels) of one family, in mm."""
    if kind == "mni2mm":
        # A 2 mm grid in MNI space masked by an ellipsoid, sized for about V voxels.
        n = max(2, int(np.ceil((V / 0.52) ** (1 / D))))
        axes = [np.arange(n) * 2.0 + o for o in (-90.0, -126.0, -72.0)[:D]]
        G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, D)
        mid, rad = G.mean(0), (G.max(0) - G.min(0)) / 2 + 1e-9
        R = G[np.sum(((G - mid) / rad) ** 2, axis=1) <= 1.0]
        R = R if len(R) else G
    else:
        n = max(2, int(np.ceil((V / 0.6) ** (1 / D))))
        spacing = rng.choice([1.0, 2.0, 3.5, 4.0], D)  # anisotropic voxels
        axes = _grid_axes(rng, D, [n] * D, spacing)
        G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, D)
        R = G[rng.uniform(size=len(G)) < 0.6]
        R = R if len(R) else G
        if kind == "rotated":
            Q, _ = np.linalg.qr(rng.normal(size=(D, D)))
            R = R @ Q.T
        elif kind == "jittered":
            R = R + rng.normal(0, 0.1 * spacing.min(), R.shape)
        elif kind == "large_offset":
            R = R + 1e6
        elif kind == "random":
            R = rng.uniform(R.min(0), R.max(0), (len(R), D))
    if len(R) < V:
        R = np.concatenate([R, R[: V - len(R)]]) if len(R) else R
    R = R[rng.permutation(len(R))[:V]]
    return np.ascontiguousarray(R, dtype=np.float64)


def scenario(i: int) -> Scenario:
    """Scenario ``i``: one value of each family and the rest of the draw,
    all seeded by ``i``."""
    rng = np.random.default_rng([2026, 10, 26, i])
    choice = {
        name: values[int(rng.integers(len(values)))]
        for name, values in FAMILIES.items()
    }
    D, K = choice["D"], choice["K"]
    V = {"tiny": max(1, K // 2), "small": int(rng.integers(20, 300)), "medium": 2000}[
        choice["V"]
    ]
    T = choice["T"]
    R = _coords(rng, D, choice["coords"], V)
    V = len(R)
    lo, hi = R.min(0), R.max(0)
    extent = float(np.max(hi - lo)) + 1.0
    sig2 = 2.0 * float(np.max(np.std(R, axis=0))) ** 2 + 1.0  # max_sigma scale
    wk = {
        "tiny": lambda: rng.uniform(1e-4, 1e-2, K),
        "small": lambda: sig2 * rng.uniform(0.005, 0.05, K),
        "typical": lambda: sig2 * rng.uniform(0.05, 0.5, K),
        "huge": lambda: sig2 * rng.uniform(1e3, 1e5, K),
        "mixed": lambda: sig2 * 10 ** rng.uniform(-6, 4, K),
    }[choice["widths"]]
    w = wk()
    inside = R[rng.integers(0, V, K)] + rng.normal(0, 0.5, (K, D))
    edge = np.where(rng.uniform(size=(K, D)) < 0.5, lo, hi) + rng.normal(0, 0.1, (K, D))
    outside = hi + extent * rng.uniform(2, 5, (K, D))
    C = {
        "inside": inside,
        "edge": edge,
        "outside": outside,
        "mixed": np.where(rng.uniform(size=(K, 1)) < 0.5, inside, outside),
    }[choice["centers"]]
    if K >= 2 and choice["special"] == "near_duplicate":
        C[1], w[1] = C[0] + 1e-6, w[0]
    elif K >= 2 and choice["special"] == "near_singular":
        C[1], w[1] = C[0] + 1e-3 * np.sqrt(w[0]), w[0] * (1 + 1e-4)
    elif choice["special"] == "vanishing":
        C[0], w[0] = hi + 50 * extent, 1e-3
    # Data from different true factors than the evaluation points.
    C_true = R[rng.integers(0, V, K)]
    W_true = rng.normal(size=(K, T))
    signal = ref.factors(R, C_true, sig2 * rng.uniform(0.05, 0.3, K)) @ W_true
    sd = float(np.std(signal)) or 1.0
    data = choice["data"]
    if data == "noiseless":
        X = signal
    elif data == "realistic":  # R^2 about 0.04
        X = signal + 4.9 * sd * rng.normal(size=signal.shape)
    elif data == "heavy_tailed":
        X = signal + 0.5 * sd * rng.standard_t(2, size=signal.shape)
    elif data == "constant_voxels":
        X = signal + 0.3 * sd * rng.normal(size=signal.shape)
        X[rng.uniform(size=V) < 0.3] = 2.5
    else:  # mean offsets
        X = signal + 0.3 * sd * rng.normal(size=signal.shape)
        X += rng.normal(100.0, 10.0, (V, 1))
    prior = None
    if choice["prior"]:
        covs = np.empty((K, D, D))
        for k in range(K):
            A = rng.normal(size=(D, D))
            covs[k] = A @ A.T + D * np.eye(D)
        prior = dict(
            centers=C + rng.normal(0, 2.0, C.shape),
            widths=w * rng.uniform(0.7, 1.3, K),
            centers_cov=covs * extent,
            widths_var=(w * rng.uniform(0.1, 0.5, K)) ** 2 + 1e-12,
            scale=float(rng.uniform(0.1, 3.0)),
        )
    return Scenario(i, choice, R, np.ascontiguousarray(X), C, w, prior)


def _points(sc: Scenario, rng) -> list[np.ndarray]:
    x = np.concatenate([sc.C.ravel(), sc.w])
    K, D = sc.C.shape
    scale = np.concatenate([np.full(K * D, 1.0), sc.w])
    return [x, x + 1e-3 * scale * rng.normal(size=x.size), x + 0.2 * scale * np.abs(
        rng.normal(size=x.size))]  # fmt: skip


def _rel(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if np.array_equal(a, b, equal_nan=True):
        return 0.0  # also NaN where the reference is NaN
    top = float(np.max(np.abs(a))) if a.size else 0.0
    diff = float(np.max(np.abs(a - b))) if a.size else 0.0
    return diff / top if top > 0 else diff


def _priors(sc: Scenario):
    if sc.prior is None:
        return None, None
    return new.Prior(**sc.prior), ref.Prior(**sc.prior)


def _fixed_weights(X, R, K, x, method):
    """Weights for the fixed-weight ResidualProblem: the reference's ridge or
    minimum-norm OLS solution at ``x`` (pseudo-inverse of the gram: it can be
    singular here, e.g. 'rr' on constant data, where `ridge_weights` raises)."""
    return ref.ProjectedProblem(X, R, K, NumpyEngine(1), None, method).weights(x)


def _gram_condition(F: np.ndarray, beta: float) -> tuple[float, float]:
    """Condition number of the gram on its kept eigenvalues, and the smallest
    eigenvalue ratio's distance (in factors) from the truncation cutoff."""
    vals = np.linalg.eigvalsh(F.T @ F + beta * np.eye(F.shape[1]))
    if vals[-1] <= 0:
        return 1.0, np.inf
    ratios = vals / vals[-1]
    kept = ratios[ratios > new.GRAM_RCOND]
    near = np.min(np.abs(np.log10(np.maximum(ratios, 1e-300) / new.GRAM_RCOND)))
    return float(1.0 / kept.min()), float(10**near)


def compare(sc: Scenario) -> dict[str, float]:
    """Max relative differences of every compared quantity; asserts tolerances."""
    rng = np.random.default_rng(sc.index)
    R, X = sc.R, sc.X
    K, D = sc.C.shape
    grid = new.axis_grid(R)
    loss = sc.choice["loss"]
    method = sc.choice["weights"]
    out: dict[str, float] = {}

    def check(name, a, b, tol):
        r = _rel(a, b)
        out[name] = max(out.get(name, 0.0), r)
        RECORDS.append((sc.index, name, r, tol))
        assert r <= tol, f"{name}: rel diff {r:.3e} > {tol:.1e}\n{sc}"

    for x in _points(sc, rng):
        C, w = x[: K * D].reshape(K, D), x[K * D :]
        F_ref = ref.factors(R, C, w)
        F_new = new.factors(R, C, w, grid)
        normal = F_ref > np.finfo(np.float64).tiny
        per_entry = np.abs(F_new - F_ref)
        rel_f = (
            float(np.max(per_entry[normal] / F_ref[normal])) if normal.any() else 0.0
        )
        out["factors per entry"] = max(out.get("factors per entry", 0.0), rel_f)
        assert rel_f <= FACTOR_RTOL, f"factors: {rel_f:.3e}\n{sc}"
        assert np.all(per_entry[~normal] <= np.finfo(np.float64).tiny), str(sc)
        np.testing.assert_array_equal(
            new.sq_distances(R, C, grid), ref.sq_distances(R, C)
        )
        dF = _rel(F_ref, F_new)
        E_n, Fk_n, G_n = new.factor_derivatives(R, C, w, grid)
        E_r, Fk_r, G_r = ref.factor_derivatives(R, C, w)
        check("factor_derivatives E", E_r, E_n, RTOL)
        check("factor_derivatives F", Fk_r, Fk_n, RTOL)
        check("factor_derivatives G", G_r, G_n, RTOL)
        p_new, p_ref = _priors(sc)
        beta = float(np.var(X)) if method == "rr" else 0.0
        try:
            _fixed_weights(X, R, K, x, method)
        except np.linalg.LinAlgError:
            # LAPACK fails on a gram of subnormal numbers (all factors near
            # underflow); the current code must fail the same way.
            with pytest.raises(np.linalg.LinAlgError):
                new.ProjectedProblem(X, R, K, NumpyEngine(1), None, method).weights(x)
            out["both raised LinAlgError"] = out.get("both raised LinAlgError", 0) + 1
            continue
        cond, near_cutoff = _gram_condition(F_ref, beta)
        if near_cutoff < 100:
            # An eigenvalue ratio within a factor 100 of GRAM_RCOND: whether it
            # is kept depends on rounding, so the weights may differ by design.
            out["skipped near cutoff"] = out.get("skipped near cutoff", 0) + 1
            continue
        dF = max(dF, _rel(Fk_r, Fk_n), _rel(E_r, E_n), np.finfo(np.float64).eps)
        stages = ["varpro", "soft_l1"] if loss == "linear+soft_l1" else [loss]
        for stage in stages:
            if stage == "varpro":
                a = new.ProjectedProblem(X, R, K, NumpyEngine(1), p_new, method)
                b = ref.ProjectedProblem(X, R, K, NumpyEngine(1), p_ref, method)
                lname = "linear"
                W = b.weights(x)
            else:
                W = _fixed_weights(X, R, K, x, method)
                a = new.ResidualProblem(
                    X, W, R, NumpyEngine(1), p_new, stage == "linear"
                )
                b = ref.ResidualProblem(
                    X, W, R, NumpyEngine(1), p_ref, stage == "linear"
                )
                lname = stage
            # Cancellation in X - F W (large weights of opposite sign, e.g. near-
            # duplicate factors under 'ols') amplifies factor rounding too.
            resid = float(np.max(np.abs(X - F_ref @ W)))
            amp = float(np.max(np.abs(F_ref) @ np.abs(W))) / resid if resid > 0 else 1.0
            amp = max(amp, cond if stage == "varpro" else 1.0, 1.0)
            tol = max(RTOL, COND_FACTOR * amp * dF)
            out["amplification"] = max(out.get("amplification", 0.0), amp)
            if tol > RTOL:
                out["evaluations with tol > RTOL"] = (
                    out.get("evaluations with tol > RTOL", 0) + 1
                )
            if stage == "varpro":
                check("varpro weights", W, a.weights(x), tol)
            ca, cb = a.cost(x, lname), b.cost(x, lname)
            check(f"{stage} cost", cb, ca, tol)
            ra, rb = a.cost_grad_hess(x, lname), b.cost_grad_hess(x, lname)
            check(f"{stage} cgh cost", rb[0], ra[0], tol)
            # The projected matrices are the fixed-weight ones minus the
            # weight-coupling terms, which can cancel them: compare at the
            # fixed-weight matrices' scale.
            if stage == "varpro":
                fx = ref.ResidualProblem(X, W, R, NumpyEngine(1), p_ref)
                _, _, gn_f, H_f = fx.cost_grad_hess(x, "linear")
            else:
                gn_f, H_f = rb[2], rb[3]
            for name, ma, mb, mf in (
                ("Gauss-Newton", ra[2], rb[2], gn_f),
                ("Hessian", ra[3], rb[3], H_f),
            ):
                top = max(float(np.max(np.abs(mb))), float(np.max(np.abs(mf))))
                r_m = float(np.max(np.abs(ma - mb))) / top if top > 0 else 0.0
                key = f"{stage} {name}"
                out[key] = max(out.get(key, 0.0), r_m)
                RECORDS.append((sc.index, key, r_m, tol))
                assert r_m <= tol, f"{key}: {r_m:.3e} > {tol:.1e}\n{sc}"
            # The gradient relative to its natural scale sqrt(2 cost lambda_max)
            # (|J^T r| <= ||J|| ||r||): near an optimum, or with vanishing
            # factors, the gradient is a cancellation of terms of that size.
            # lambda_max from Gershgorin's bound (no LAPACK on overflowed values).
            lam = float(np.max(np.sum(np.abs(gn_f), axis=1)))
            g_scale = max(float(np.max(np.abs(rb[1]))), np.sqrt(2 * abs(rb[0]) * lam))
            r_g = float(np.max(np.abs(ra[1] - rb[1]))) / g_scale if g_scale > 0 else 0.0
            out[f"{stage} gradient"] = max(out.get(f"{stage} gradient", 0.0), r_g)
            RECORDS.append((sc.index, f"{stage} gradient", r_g, tol))
            assert r_g <= tol, f"{stage} gradient: {r_g:.3e} > {tol:.1e}\n{sc}"
            out["cond"] = max(out.get("cond", 0.0), cond)
    return out


def _sweep(indices):
    worst: dict[str, float] = {}
    for i in indices:
        sc = scenario(i)
        for k, v in compare(sc).items():
            worst[k] = max(worst.get(k, 0.0), v)
    return worst


def test_scenarios_cover_every_family_value():
    """Each value of each family occurs in the fast sweep (and coordinates
    of both kinds, grid and not)."""
    seen = {name: set() for name in FAMILIES}
    grid_kinds = set()
    for i in range(N_FAST):
        sc = scenario(i)
        for name, value in sc.choice.items():
            seen[name].add(value)
        grid_kinds.add(new.axis_grid(sc.R) is not None)
    missing = {n: set(v) - seen[n] for n, v in FAMILIES.items() if set(v) - seen[n]}
    print("coverage:", {n: len(v) for n, v in seen.items()}, "grid kinds", grid_kinds)
    assert not missing
    assert grid_kinds == {True, False}


@pytest.mark.parametrize("block", range(8))
def test_kernels_equal_reference_on_generated_scenarios(block):
    """N_FAST scenarios in 8 blocks: factors, derivatives, cost, gradient,
    Gauss-Newton matrix, Hessian and weights of the current kernels vs the
    3298278 formulas (see the module docstring for the tolerances)."""
    size = N_FAST // 8
    worst = _sweep(range(block * size, (block + 1) * size))
    print(
        f"block {block}: " + ", ".join(f"{k} {v:.2e}" for k, v in sorted(worst.items()))
    )


def _large_scenario(i: int) -> Scenario:
    rng = np.random.default_rng([7, i])
    K = (50, 100)[i % 2]
    D = (2, 3)[i % 2] if i < 4 else 3
    kind = ("mni2mm", "grid", "jittered")[i % 3]
    R = _coords(rng, D, kind, 20000)
    T = (1, 300, 168)[i % 3]
    sig2 = 2.0 * float(np.max(np.std(R, axis=0))) ** 2
    C = R[rng.integers(0, len(R), K)] + rng.normal(0, 1.0, (K, D))
    w = sig2 * rng.uniform(0.01, 0.2, K)
    W = rng.normal(size=(K, T))
    X = ref.factors(R, R[rng.integers(0, len(R), K)], w) @ W
    X = X + 4.9 * np.std(X) * rng.normal(size=X.shape)
    choice = dict(D=D, coords=kind, V="large", T=T, K=K, widths="typical")
    choice.update(centers="inside", special="none", prior=False, data="realistic")
    choice["loss"] = ("varpro", "soft_l1", "linear+soft_l1")[i % 3]
    choice["weights"] = "rr"
    return Scenario(10_000 + i, choice, R, np.ascontiguousarray(X), C, w, None)


@pytest.mark.slow
@pytest.mark.parametrize("i", range(N_SLOW))
def test_kernels_equal_reference_on_large_scenarios(i):
    """V = 20000, K = 50 or 100, T up to 300, grid and jittered coordinates."""
    sc = _large_scenario(i)
    worst = compare(sc)
    print(f"{sc}: " + ", ".join(f"{k} {v:.2e}" for k, v in sorted(worst.items())))


@pytest.mark.parametrize("i", range(0, N_FAST, 12))
def test_reuse_is_exact_under_interleaved_calls(i):
    """Interleaved cost and cost_grad_hess calls at changing points, repeated
    points after another point, and a prior changed in place: every result is
    bit for bit that of a fresh problem with the same prior (and equal to the
    reference within the sweep's tolerances, checked there)."""
    sc = scenario(i)
    rng = np.random.default_rng(i)
    K, D = sc.C.shape
    xs = _points(sc, rng)
    loss = sc.choice["loss"]
    method = sc.choice["weights"]
    stage = "varpro" if loss in ("varpro", "linear+soft_l1") else loss
    p_new, _ = _priors(sc)

    def make(prior):
        if stage == "varpro":
            return new.ProjectedProblem(sc.X, sc.R, K, NumpyEngine(1), prior, method)
        x0 = np.concatenate([sc.C.ravel(), sc.w])
        W = _fixed_weights(sc.X, sc.R, K, x0, method)
        return new.ResidualProblem(
            sc.X, W, sc.R, NumpyEngine(1), prior, stage == "linear"
        )

    lname = "linear" if stage == "varpro" else stage
    order = [0, 0, 1, 0, 2, 2, 1, 1, 0, 2]
    kinds = ["cost", "cgh", "cost", "cgh", "cost", "cgh", "cgh", "cost", "cgh", "cgh"]
    prob = make(p_new)
    for j, kind in zip(order, kinds, strict=True):
        got = (
            prob.cost(xs[j], lname)
            if kind == "cost"
            else prob.cost_grad_hess(xs[j], lname)
        )
        fresh = make(p_new)
        if kind == "cost":
            # A fresh problem picks its objective form at its first point; give
            # it the same first point as `prob`.
            fresh.cost(xs[0], lname)
            expected = fresh.cost(xs[j], lname)
            assert got == expected, f"{kind} at x{j}\n{sc}"
        else:
            fresh.cost(xs[0], lname)
            expected = fresh.cost_grad_hess(xs[j], lname)
            for g, e in zip(got, expected, strict=True):
                np.testing.assert_array_equal(g, e, err_msg=f"{kind} at x{j}\n{sc}")
    if p_new is not None:
        # The prior's centers changed in place after an evaluation at xs[1].
        prob.cost_grad_hess(xs[1], lname)
        p_new.centers += 0.5
        fresh = make(p_new)
        fresh.cost(xs[0], lname)
        got, expected = (
            prob.cost_grad_hess(xs[1], lname),
            fresh.cost_grad_hess(xs[1], lname),
        )
        for g, e in zip(got, expected, strict=True):
            np.testing.assert_array_equal(g, e, err_msg=f"after prior change\n{sc}")
