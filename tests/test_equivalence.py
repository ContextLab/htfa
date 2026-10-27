"""The current kernels against the formulas of commit 3298278, over generated scenarios.

The oracle is ``tests/reference_kernels_3298278.py``: the kernels as they were
before the per-axis factors, the reuse of a cost evaluation in cost_grad_hess
and the ``||X||**2 - <F^T X, W>`` objective. Each scenario draws coordinates,
sizes, widths, centers, weights, loss, prior and data from the families listed
in `FAMILIES`, and compares factors, factor derivatives, cost, gradient,
Gauss-Newton matrix, Hessian and weights at three points. Failures print the
scenario (``scenario(i)`` rebuilds it).

Every quantity is compared on a natural scale that does not shrink with the
fit: factors per entry (`FACTOR_RTOL`); costs relative to the cost of the zero
model, ``0.5 sum rho(s**2 X**2)``, plus the prior terms; ``F W`` relative to
max|X|; weights relative to their largest entry; the gradient relative to
``sqrt(2 cost_scale lambda_max)`` of the fixed-weight Gauss-Newton matrix; the
Gauss-Newton matrix and Hessian relative to the largest entry of their own,
the fixed-weight matrices and the difference of the two (the coupling terms,
which can cancel most of the fixed-weight matrices). The tolerance is the
larger of `RTOL`
and `AMP_FACTOR` x amplification x the measured factor difference, where the
amplification is the larger of the gram's condition number on its kept
eigenvalues (varpro) and max(|F| |W|) / max|X| (weights that cancel). It never
exceeds `TOL_CAP`: a stage whose amplification would need more is compared on
its cost and ``F W`` only (`ILL_TOL`), and recorded in `SKIPS`. So are points
where the two versions keep different gram eigenvalues, which are checked on
the current code's own properties (`_check_own`) instead.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
import reference_kernels_3298278 as ref

import htfa._kernels as new
from htfa._engines import NumpyEngine, loss_terms

RTOL = 1e-10
# Factors per entry (normal numbers): per-axis products vs exp of the sum.
FACTOR_RTOL = 1e-12
# Measured: observed difference / (amplification x factor difference) at most
# 23.7 (varpro Hessian, whose coupling terms use the gram inverse twice),
# at most 1.3 for the other quantities.
AMP_FACTOR = 100.0
TOL_CAP = 1e-6
# Cost and F W where the amplification needs more than TOL_CAP (ill-conditioned
# 'ols' grams): measured at most 3.9e-10 and 1.8e-9.
ILL_TOL = 1e-8
N_FAST = 480
N_SLOW = 12
# (scenario, quantity, relative difference, tolerance, amplification x factor
# difference) of every comparison and
# (scenario, reason) of every stage not compared with the oracle, for summaries
# of the sweep (scripts/equivalence_summary.py).
RECORDS: list[tuple[int, str, float, float, float]] = []
SKIPS: list[tuple[int, str]] = []

FAMILIES: dict[str, tuple[Any, ...]] = {
    "D": (2, 3),
    "coords": ("grid", "mni2mm", "rotated", "jittered", "large_offset", "random"),
    "V": ("at_most_K", "small", "medium"),
    "T": (1, 2, 7, 40),
    "K": (1, 2, 5, 12, 30),
    "widths": ("tiny", "small", "typical", "huge", "mixed"),
    "centers": ("inside", "edge", "outside", "mixed"),
    "weights": ("rr", "ols"),
    "special": ("none", "near_duplicate", "vanishing", "near_singular", "all_vanish"),
    "loss": ("varpro", "linear", "soft_l1", "huber", "linear+soft_l1"),
    "prior": (False, True),
    "data": (
        "noiseless",
        "exact_fit",
        "realistic",
        "heavy_tailed",
        "constant_voxels",
        "offsets",
    ),
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
    V = {
        "at_most_K": max(1, K // 2),  # fewer voxels than factors when K >= 2
        "small": int(rng.integers(20, 300)),
        "medium": 2000,
    }[choice["V"]]
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
    elif choice["special"] == "all_vanish":
        # Every factor below about 1e-154 at every voxel: a gram of subnormal
        # numbers (exp(-355) per factor at the nearest voxel).
        C = hi + extent * rng.uniform(1.5, 2.5, (K, D))
        d_min = np.min(ref.sq_distances(R, C), axis=0)
        w = d_min / rng.uniform(352.0, 358.0, K)
    # Data from different true factors than the evaluation points.
    C_true = R[rng.integers(0, V, K)]
    W_true = rng.normal(size=(K, T))
    signal = ref.factors(R, C_true, sig2 * rng.uniform(0.05, 0.3, K)) @ W_true
    sd = float(np.std(signal)) or 1.0
    data = choice["data"]
    if data == "noiseless":
        X = signal
    elif data == "exact_fit":  # X = F W at the scenario's own centers and widths
        X = ref.factors(R, C, w) @ W_true
        # Unit scale (still in the span of F): factors that all but vanish
        # would otherwise give data of 1e-154, whose 1 / std overflows.
        X = X / (np.max(np.abs(X)) or 1.0)
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


def _priors(sc: Scenario):
    if sc.prior is None:
        return None, None
    return new.Prior(**sc.prior), ref.Prior(**sc.prior)


def _kept(vals: np.ndarray, floor: float) -> np.ndarray:
    """Kept gram eigenvalues: `GRAM_RCOND` relative cutoff, and ``floor``
    (`new.GRAM_FLOOR` for the current code, 0 for the reference)."""
    return vals > max(new.GRAM_RCOND * vals[-1], floor)


def _check_own(sc: Scenario, x, prior, method) -> None:
    """The current code's own properties where the oracle does not apply:
    finite weights and cost; where no gram eigenvalue is kept (every factor
    vanishes), weights 0, cost ``0.5 s**2 ||X||**2`` plus the prior terms, and
    the ValueError of cost_grad_hess for a degenerate start."""
    K = sc.C.shape[0]
    p = new.ProjectedProblem(sc.X, sc.R, K, NumpyEngine(1), prior, method)
    W, c = p.weights(x), p.cost(x)
    assert np.all(np.isfinite(W)) and np.isfinite(c), str(sc)
    F = new.factors(sc.R, x[: K * p.D].reshape(K, p.D), x[K * p.D :])
    vals = np.linalg.eigvalsh(F.T @ F + p.beta * np.eye(K))
    if not _kept(vals, new.GRAM_FLOOR).any():
        expected = 0.5 * p.s**2 * float(np.sum(sc.X * sc.X))
        expected += p._prior_cost(*p._unpack(x))
        assert not W.any(), str(sc)
        assert c == pytest.approx(expected, rel=1e-12), str(sc)
        with pytest.raises(ValueError, match="factors are 0 at every voxel"):
            p.cost_grad_hess(x)


def _residual_weights(sc: Scenario, x, method):
    """Weights for the fixed-weight ResidualProblem at ``x``: the current
    code's ridge or minimum-norm OLS weights (finite also where the reference's
    gram inverse overflows)."""
    K = sc.C.shape[0]
    return new.ProjectedProblem(sc.X, sc.R, K, NumpyEngine(1), None, method).weights(x)


def compare(sc: Scenario) -> dict[str, float]:
    """Max relative differences of every compared quantity; asserts tolerances."""
    rng = np.random.default_rng(sc.index)
    R, X = sc.R, sc.X
    K, D = sc.C.shape
    grid = new.axis_grid(R)
    method = sc.choice["weights"]
    loss = sc.choice["loss"]
    stages = ["varpro", "soft_l1"] if loss == "linear+soft_l1" else [loss]
    x_max = float(np.max(np.abs(X)))
    out: dict[str, float] = {}

    base = 0.0  # amplification x factor difference of the current stage

    def check(name, diff, scale, tol):
        r = diff / scale if scale > 0 else diff
        out[name] = max(out.get(name, 0.0), r)
        RECORDS.append((sc.index, name, r, tol, base))
        assert r <= tol, f"{name}: rel diff {r:.3e} > {tol:.1e}\n{sc}"

    def mdiff(a, b):
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        if np.array_equal(a, b):
            return 0.0
        return float(np.max(np.abs(a - b)))

    def mabs(a):
        return float(np.max(np.abs(a))) if np.size(a) else 0.0

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
        RECORDS.append((sc.index, "factors per entry", rel_f, FACTOR_RTOL, 0.0))
        assert rel_f <= FACTOR_RTOL, f"factors: {rel_f:.3e}\n{sc}"
        assert np.all(per_entry[~normal] <= np.finfo(np.float64).tiny), str(sc)
        np.testing.assert_array_equal(
            new.sq_distances(R, C, grid), ref.sq_distances(R, C)
        )
        E_n, Fk_n, G_n = new.factor_derivatives(R, C, w, grid)
        E_r, Fk_r, G_r = ref.factor_derivatives(R, C, w)
        for name, a, b in (("E", E_r, E_n), ("F", Fk_r, Fk_n), ("G", G_r, G_n)):
            check(f"factor_derivatives {name}", mdiff(a, b), mabs(a), RTOL)
        dF = max(rel_f, mdiff(Fk_r, Fk_n) / max(mabs(Fk_r), 1e-300), 2.3e-16)
        p_new, p_ref = _priors(sc)
        beta = float(np.var(X)) if method == "rr" else 0.0
        vals_r = np.linalg.eigvalsh(F_ref.T @ F_ref + beta * np.eye(K))
        vals_n = np.linalg.eigvalsh(F_new.T @ F_new + beta * np.eye(K))
        keep_r, keep_n = _kept(vals_r, 0.0), _kept(vals_n, new.GRAM_FLOOR)
        for stage in stages:
            if stage == "varpro":
                if not keep_n.any():
                    _check_own(sc, x, p_new, method)
                    SKIPS.append(
                        (sc.index, "varpro: every factor vanishes (own checks)")
                    )
                    continue
                if keep_n.sum() != keep_r.sum():
                    _check_own(sc, x, p_new, method)
                    SKIPS.append(
                        (sc.index, "varpro: different kept eigenvalues (own checks)")
                    )
                    continue
                a = new.ProjectedProblem(X, R, K, NumpyEngine(1), p_new, method)
                b = ref.ProjectedProblem(X, R, K, NumpyEngine(1), p_ref, method)
                lname, W = "linear", b.weights(x)
                cond = float(vals_n[-1] / vals_n[keep_n].min())
            else:
                lname, W = stage, _residual_weights(sc, x, method)
                a = new.ResidualProblem(
                    X, W, R, NumpyEngine(1), p_new, stage == "linear"
                )
                b = ref.ResidualProblem(
                    X, W, R, NumpyEngine(1), p_ref, stage == "linear"
                )
                cond = 1.0
            cancel = mabs(np.abs(F_ref) @ np.abs(W)) / x_max if x_max > 0 else 1.0
            amp = max(cond, cancel, 1.0)
            base = amp * dF
            tol = max(RTOL, AMP_FACTOR * base)
            # Natural scales: the zero model's cost (plus the prior's), and the
            # fixed-weight Gauss-Newton matrix and Hessian.
            s = b.s
            rho = loss_terms(lname, (s * X) ** 2)[0]
            fx = ref.ResidualProblem(X, W, R, NumpyEngine(1), p_ref)
            c_fx, _, gn_f, H_f = fx.cost_grad_hess(x, lname)
            prior_cost = c_fx - fx.engine.data_cost(F_ref, s, lname)
            cost_scale = 0.5 * float(np.sum(rho)) + abs(prior_cost)
            ca, cb = a.cost(x, lname), b.cost(x, lname)
            if tol > TOL_CAP:
                # The weights and derivatives are not determined to TOL_CAP, but
                # the cost and F W (a projection) still are.
                check(
                    f"{stage} cost (ill-conditioned)", abs(ca - cb), cost_scale, ILL_TOL
                )
                fw = mdiff(F_new @ a.weights(x), F_ref @ W)
                check(f"{stage} F W (ill-conditioned)", fw, x_max, ILL_TOL)
                SKIPS.append(
                    (
                        sc.index,
                        f"{stage}: amplification {amp:.1e} needs tol {tol:.1e} "
                        "(cost and F W compared only)",
                    )
                )
                continue
            check(f"{stage} cost", abs(ca - cb), cost_scale, tol)
            ra, rb = a.cost_grad_hess(x, lname), b.cost_grad_hess(x, lname)
            check(f"{stage} cgh cost", abs(ra[0] - rb[0]), cost_scale, tol)
            for name, ma, mb, mf in (
                ("Gauss-Newton", ra[2], rb[2], gn_f),
                ("Hessian", ra[3], rb[3], H_f),
            ):
                # The projected matrix is the fixed-weight one minus the coupling
                # terms (mf - mb), which can be the larger: their scale too.
                scale = max(mabs(mb), mabs(mf), mabs(mf - mb))
                check(f"{stage} {name}", mdiff(ma, mb), scale, tol)
            # lambda_max from Gershgorin's bound.
            lam = float(np.max(np.sum(np.abs(gn_f), axis=1)))
            g_scale = max(mabs(rb[1]), np.sqrt(2 * cost_scale * lam))
            check(f"{stage} gradient", mdiff(ra[1], rb[1]), g_scale, tol)
            if stage == "varpro":
                Wa = a.weights(x)
                check("varpro weights", mdiff(Wa, W), mabs(W), tol)
                check("varpro F W", mdiff(F_new @ Wa, F_ref @ W), x_max, tol)
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
    3298278 formulas (see the module docstring for the scales and tolerances)."""
    size = N_FAST // 8
    worst = _sweep(range(block * size, (block + 1) * size))
    print(
        f"block {block}: " + ", ".join(f"{k} {v:.2e}" for k, v in sorted(worst.items()))
    )


def _large_scenario(i: int) -> Scenario:
    rng = np.random.default_rng([7, i])
    K = (50, 100)[i % 2]
    D = (2, 3)[(i // 2) % 2]
    kind = ("mni2mm", "grid", "jittered")[i % 3]
    R = _coords(rng, D, kind, 6000)
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
    """V = 6000, D = 2 and 3, K = 50 and 100, T up to 300; MNI-like, grid and
    jittered coordinates."""
    sc = _large_scenario(i)
    worst = compare(sc)
    print(f"{sc}: " + ", ".join(f"{k} {v:.2e}" for k, v in sorted(worst.items())))


@pytest.mark.parametrize("i", range(0, N_FAST, 12))
def test_reuse_is_exact_under_interleaved_calls(i):
    """Interleaved cost and cost_grad_hess calls where only the widths, only the
    centers or only the prior change between calls, and repeated points after
    another point: every result is bit for bit that of a fresh problem with the
    same prior."""
    sc = scenario(i)
    K, D = sc.C.shape
    x0 = np.concatenate([sc.C.ravel(), sc.w])
    x_w = x0.copy()
    x_w[K * D :] *= 1.01  # widths only
    x_c = x0.copy()
    x_c[: K * D] += 0.3  # centers only
    xs = [x0, x_w, x_c]
    loss = sc.choice["loss"]
    method = sc.choice["weights"]
    stage = "varpro" if loss in ("varpro", "linear+soft_l1") else loss
    p_new, _ = _priors(sc)
    W0 = _residual_weights(sc, x0, method)

    def make(prior):
        if stage == "varpro":
            return new.ProjectedProblem(sc.X, sc.R, K, NumpyEngine(1), prior, method)
        return new.ResidualProblem(
            sc.X, W0, sc.R, NumpyEngine(1), prior, stage == "linear"
        )

    def same(got, expected, what):
        got = got if isinstance(got, tuple) else (got,)
        expected = expected if isinstance(expected, tuple) else (expected,)
        for g, e in zip(got, expected, strict=True):
            np.testing.assert_array_equal(g, e, err_msg=f"{what}\n{sc}")

    def call(prob, kind, x):
        if kind == "cost":
            return prob.cost(x, lname)
        try:
            return prob.cost_grad_hess(x, lname)
        except ValueError:  # every factor vanishes at x: no Hessian there
            return "ValueError"

    def fresh_value(kind, x, prior):
        # A fresh, unprimed problem (its cache empty), with the objective form
        # that `prob` chose at its first point copied over.
        fresh = make(prior)
        if hasattr(prob, "_identity"):
            fresh._identity = prob._identity
        return call(fresh, kind, x)

    lname = "linear" if stage == "varpro" else stage
    order = [0, 0, 1, 1, 0, 2, 2, 1, 0, 2, 1]
    kinds = ["cost", "cgh", "cost", "cgh", "cgh", "cost", "cgh", "cgh", "cost"]
    kinds += ["cgh", "cost"]
    prob = make(p_new)
    prob.cost(xs[0], lname)
    for j, kind in zip(order, kinds, strict=True):
        got = call(prob, kind, xs[j])
        expected = fresh_value(kind, xs[j], p_new)
        if isinstance(got, str) or isinstance(expected, str):
            assert got == expected, f"{kind} at x{j}\n{sc}"
        else:
            same(got, expected, f"{kind} at x{j}")
    if p_new is not None:
        # The prior's centers, then its widths, changed in place after an
        # evaluation at the same point.
        for field_name, delta in (("centers", 0.5), ("widths", 0.1)):
            call(prob, "cgh", xs[1])
            getattr(p_new, field_name)[...] += delta
            for kind in ("cost", "cgh"):
                got, expected = call(prob, kind, xs[1]), fresh_value(kind, xs[1], p_new)
                if isinstance(got, str) or isinstance(expected, str):
                    assert got == expected, f"after prior {field_name}\n{sc}"
                else:
                    same(got, expected, f"{kind} after prior {field_name} change")
