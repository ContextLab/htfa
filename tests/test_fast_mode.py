"""``mode='fast'`` against ``mode='exact'``, both computed here (same machine,
same BLAS), over the scenario families of the equivalence suite.

The two modes minimize the same objective, with float64 costs and gradients
and the same stopping rules. They differ in the step `htfa._kernels.bounded_lm`
takes where the Hessian is not positive definite on the free variables (exact:
Gauss-Newton; fast: the Hessian with more damping), and in the precision of
the curvature: fast mode forms the largest curvature products of the linear
loss in float32. Where the Hessian stays positive definite the two runs take
the same kind of step from slightly different (float32-rounded) curvature, so
they end within the stopping tolerances of each other; where a solve meets
negative curvature they can end at different local minima.

* Single solves (`FAST_SOLVE_COST_RTOL`): the solver families of
  ``test_kernels.py`` (Gauss-Newton test problems; ``make_tfa_data`` solves
  with and without a prior), linear and soft_l1 losses. Every case agrees with
  the exact solve to rounding except the solves in `BASIN_JUMPS`, where the
  first step from the prior meets negative curvature (the case
  ``test_bounded_lm_does_not_jump_basins_on_indefinite_hessian`` documents for
  the exact mode).
* Whole TFA fits: the 36 problems of ``test_fit_equivalence.py`` plus soft_l1,
  two-stage, 'ols', prior and 'bounded-lm' variants; on every well-conditioned
  one (the criterion of ``test_fit_equivalence.py``, computed from the exact
  fit) the fast fit is within that file's tolerances of the exact fit.
* Whole HTFA fits: 2D and 3D, K = 3 and 6, the three template updates, linear
  and soft_l1 losses; on every well-conditioned case the global template and
  the summed objective are within `HTFA_GLOBAL_LENGTHS` and
  `HTFA_OBJECTIVE_RTOL` of the exact fit.
* Indefinite families (`INDEFINITE_FAMILIES`): more factors fitted than
  generated, low SNR, large K and noise-dominated data, where the exact
  solver meets an indefinite Hessian at most accepted steps (as on real
  fMRI). There the modes reach different local minima, so the fast fit is
  compared with the spread of exact fits from other starts
  (`test_fast_fits_on_indefinite_families`).

Real development_fmri fits (K = 50 and 100, where the modes do differ) are
checked by the ``network`` test at the end and documented in
docs/validation.rst.
"""

import warnings

import numpy as np
import pytest
from test_fit_equivalence import (
    COST_RTOL,
    PERTURBATION,
    RECOVERY_LENGTHS,
    THRESHOLD,
)
from test_kernels import _gn_problem, _tfa_solve_case

from htfa._engines import EPS
from htfa._kernels import (
    FAILED_FACTOR_GROWTH,
    LM_TAU,
    Prior,
    bounded_lm,
    bounds,
    max_sigma,
    solve_centers_widths,
)
from htfa._matching import match_factors
from htfa.htfa import HTFA
from htfa.synthetic import make_htfa_data, make_tfa_data
from htfa.tfa import TFA

# Single solves: fast cost at most this much above the exact one. Measured over
# the 44 solves below: +4.15e-4 (linear) and +3.82e-4 (soft_l1) in BASIN_JUMPS,
# within 1e-11 elsewhere.
FAST_SOLVE_COST_RTOL = 1e-3
# Same answer to rounding: cost and centers (in data units).
SAME_COST_RTOL = 1e-9
SAME_CENTER_ATOL = 1e-3
BASIN_JUMPS = {("tfa", "linear", ("3d", 2), True), ("tfa", "soft_l1", ("3d", 2), True)}
# HTFA: template (lengths sqrt(max_sigma / 2)) and summed final local objective
# of the fast fit against the exact one. Measured over the 18 well-conditioned
# cases below: 2.9e-5 lengths and 7.0e-8 relative (macOS), 2.9e-5 and 7.6e-8 (Linux,
# OpenBLAS).
HTFA_GLOBAL_LENGTHS = 1e-3
HTFA_OBJECTIVE_RTOL = 1e-6
TRUTH_START_LENGTHS = 0.1
MIN_WELL_CONDITIONED_TFA = 50
MIN_WELL_CONDITIONED_HTFA = 15


def _solve_cases():
    cases = []
    for loss in ("linear", "soft_l1"):
        for D in (2, 3):
            for use_prior in (False, True):
                for seed in range(3):
                    cases.append(("gn", loss, (D, seed), use_prior))
        for case in (("2d", 0), ("2d", 2), ("2d", 4), ("3d", 1), ("3d", 2)):
            for use_prior in (False, True):
                cases.append(("tfa", loss, case, use_prior))
    return cases


@pytest.mark.parametrize("case", _solve_cases(), ids=str)
def test_fast_solves_match_exact_except_at_negative_curvature(case):
    kind, loss, spec, use_prior = case
    if kind == "gn":
        D, seed = spec
        prob, x0, _ = _gn_problem(D, use_prior, 600 + seed, V=150, T=20, K=3)
        bnds = bounds(prob.R, prob.K, 0.02, 4.0)
        x0 = np.clip(x0, *bnds)
    else:
        prob, x0, bnds = _tfa_solve_case(*spec, use_prior, loss)
    nc = prob.K * prob.D
    x_e, c_e = solve_centers_widths(prob, x0, bnds, loss=loss, mode="exact")
    x_f, c_f = solve_centers_widths(prob, x0, bnds, loss=loss, mode="fast")
    rel = (c_f - c_e) / c_e
    dc = float(np.abs(x_f[:nc] - x_e[:nc]).max())
    print(f"{case}: cost fast - exact {rel:+.3e} (relative), max |dc| {dc:.3g}")
    assert np.all(x_f >= bnds[0]) and np.all(x_f <= bnds[1])
    assert rel <= FAST_SOLVE_COST_RTOL
    same = abs(rel) <= SAME_COST_RTOL and dc <= SAME_CENTER_ATOL
    assert same or case in BASIN_JUMPS, f"{case} differs: {rel:+.3e}, {dc:.3g}"


def test_bounded_lm_fast_uses_the_hessian_where_it_is_indefinite():
    """At the start of a BASIN_JUMPS solve the Hessian is indefinite on the free
    variables: the exact mode's first step is the Gauss-Newton step, the fast
    mode's the (more damped) Newton step, so the first accepted points differ;
    both lower the cost."""
    prob, x0, bnds = _tfa_solve_case("3d", 2, True, "linear")
    _, _, gn, hess = prob.cost_grad_hess(x0, "linear")
    assert np.linalg.eigvalsh(hess).min() < 0 <= np.linalg.eigvalsh(gn).min() + 1e-9
    scale = 1.0 / prob.column_norms(x0)
    cost0 = prob.cost(x0, "linear")
    x_e, c_e, _, nfev_e = bounded_lm(prob, x0, bnds, "linear", scale, 2, "exact")
    x_f, c_f, _, nfev_f = bounded_lm(prob, x0, bnds, "linear", scale, 2, "fast")
    print(f"first step: cost {cost0:.8g} -> exact {c_e:.8g}, fast {c_f:.8g}")
    assert c_e < cost0 and c_f < cost0
    assert not np.array_equal(x_e, x_f)
    with pytest.raises(ValueError, match="mode must be one of"):
        bounded_lm(prob, x0, bnds, "linear", scale, 2, "approximate")


def test_failed_factorizations_grow_the_damping_per_sequence():
    """Where the damped Hessian is not positive definite, the k-th consecutive
    failed factorization of a step multiplies the damping by
    FAILED_FACTOR_GROWTH**k, so after n failures it is 2**(n (n + 1) / 2)
    times the start, and the previous value (one factor less) still fails.
    Checked on the first step of fast solves that meet negative curvature,
    where the starting damping is known (LM_TAU times the largest scaled
    Gauss-Newton diagonal entry)."""
    prob, x0, bnds = _tfa_solve_case("3d", 2, True, "linear")
    scale = 1.0 / prob.column_norms(x0)
    x, checked = x0, 0
    for _ in range(15):
        _, g, gn, hess = prob.cost_grad_hess(x, "linear", mode="fast")
        free = ~(((x <= bnds[0]) & (g > 0)) | ((x >= bnds[1]) & (g < 0)))
        d = scale[free]
        H_s = d[:, None] * hess[np.ix_(free, free)] * d[None, :]
        lam0 = LM_TAU * max(float(np.max(scale**2 * np.diag(gn))), EPS)
        trace: list = []
        x, *_ = bounded_lm(prob, x, bnds, "linear", scale, 2, "fast", trace)
        if not trace:
            break
        n, lam = trace[0]["cholesky_failures"], trace[0]["lam"]
        assert trace[0]["model"] == "newton"
        lowest = float(np.linalg.eigvalsh(H_s).min())
        print(
            f"{n} failures: lam {lam:.4g} = {lam / lam0:g} x start, lowest {lowest:.4g}"
        )
        assert lam == lam0 * FAILED_FACTOR_GROWTH ** (n * (n + 1) // 2)
        if n:
            previous = lam / FAILED_FACTOR_GROWTH**n
            assert lowest + lam > 0 >= lowest + previous - 1e-12 * abs(lowest)
            checked += 1
    assert checked >= 3


# (grid, n_samples, K generated, K fitted, noise) of each family.
INDEFINITE_FAMILIES = {
    "overfit": ((15, 15, 15), 100, 5, 15, 0.3),
    "low_snr": ((15, 15, 15), 100, 8, 8, 3.0),
    "large_k": ((16, 16, 16), 100, 24, 24, 0.5),
    "noise_dominated": ((16, 16, 16), 100, 16, 24, 4.0),
    "overfit_2d": ((40, 40), 100, 6, 18, 0.5),
}
INDEFINITE_SEEDS = (0, 1, 2)
# Exact fits from these other random_states (K-means seeds) give the spread of
# objectives and recoveries that the exact mode itself reaches.
OTHER_STARTS = (1, 2, 3)
# Fast objective at most max(SPREAD_FACTOR x the exact start spread,
# SPREAD_FLOOR) above the exact fit's (relative); fast recovery error of the
# noiseless signal at most RECOVERY_FACTOR times the worst exact start's.
# Measured: fast - exact at most 0.99 times the spread on macOS (Accelerate)
# and 1.0 times on Linux (OpenBLAS; noise_dominated seed 0: +3.96e-5 against
# 3.96e-5); fast recovery at most 1.03 times the worst exact start's (both;
# overfit_2d seed 1: 0.0217 against 0.0212).
SPREAD_FACTOR = 2.0
SPREAD_FLOOR = 1e-6
RECOVERY_FACTOR = 1.1
# The families must exercise the fast mode's step: the exact solver met an
# indefinite Hessian at 47% to 93% (macOS) and 44% to 93% (Linux) of its
# accepted steps per fit.
MIN_INDEFINITE_FRACTION = 0.4
MIN_MEAN_INDEFINITE_FRACTION = 0.7


@pytest.mark.slow
def test_fast_fits_on_indefinite_families(monkeypatch):
    """In the INDEFINITE_FAMILIES the exact solver meets negative curvature at
    most steps, and the two modes reach different local minima. The fast fit
    must not be worse than the exact mode's own start-to-start variation: its
    objective at most max(SPREAD_FACTOR x spread, SPREAD_FLOOR) above the exact
    fit's, where spread is the largest relative objective difference between
    the exact fit and exact fits from OTHER_STARTS; and its error in
    reconstructing the noiseless signal at most RECOVERY_FACTOR times the
    worst of those exact fits'. Lower objectives and better recoveries are
    allowed."""
    import htfa._kernels as kernels

    traces: list = []
    solve = kernels.bounded_lm

    def traced(*args, **kwargs):
        return solve(*args, **kwargs, trace=traces[-1])

    monkeypatch.setattr(kernels, "bounded_lm", traced)

    def fit(X, R, K, mode, random_state, signal):
        traces.append([])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = TFA(K=K, mode=mode, standardize=False, random_state=random_state)
            m.fit(X, R)
        F = factors_of(R, m)
        error = float(np.sum((F @ m.weights_ - signal) ** 2) / np.sum(signal**2))
        steps = traces[-1]
        indefinite = float(np.mean([s["indefinite"] for s in steps]))
        return m.cost_, error, indefinite

    fractions = []
    for name, (grid, T, K_true, K_fit, noise) in INDEFINITE_FAMILIES.items():
        for seed in INDEFINITE_SEEDS:
            X, R, truth = make_tfa_data(
                grid, T, K=K_true, noise=noise, random_state=seed
            )
            signal = (
                kernels.factors(R, truth["centers"], truth["widths"]) @ truth["weights"]
            )
            exact = fit(X, R, K_fit, "exact", 0, signal)
            others = [fit(X, R, K_fit, "exact", r, signal) for r in OTHER_STARTS]
            fast = fit(X, R, K_fit, "fast", 0, signal)
            spread = max(abs(c - exact[0]) / exact[0] for c, _, _ in others)
            gap = (fast[0] - exact[0]) / exact[0]
            worst = max(e for _, e, _ in (exact, *others))
            print(
                f"{name} seed {seed}: indefinite at {exact[2]:.2f} of exact steps; "
                f"fast - exact {gap:+.2e} (exact start spread {spread:.2e}); "
                f"signal error exact {exact[1]:.4f}, worst exact start {worst:.4f}, "
                f"fast {fast[1]:.4f}"
            )
            fractions.append(exact[2])
            assert exact[2] >= MIN_INDEFINITE_FRACTION, (name, seed)
            assert gap <= max(SPREAD_FACTOR * spread, SPREAD_FLOOR), (name, seed)
            assert fast[1] <= RECOVERY_FACTOR * worst, (name, seed)
    assert np.mean(fractions) >= MIN_MEAN_INDEFINITE_FRACTION


def factors_of(R, model):
    """The fitted factors (V, K) of ``model`` at ``R``."""
    from htfa._kernels import factors

    return factors(R, model.centers_, model.widths_)


def _tfa_cases():
    grids = ((20, 20), (30, 30), (25, 15), (10, 10, 10), (12, 12, 12), (14, 10, 8))
    cases = [
        (
            "default",
            dict(grid_shape=g, n_samples=40, K=K, noise=n, random_state=100 + i),
        )
        for i, (g, K, n) in enumerate(
            (g, K, n) for g in grids for K in (2, 4, 6) for n in (0.1, 0.5)
        )
    ]
    j = 0
    for g in ((30, 30), (12, 12, 12)):
        for K in (4, 8):
            for family in ("soft_l1", "two_stage", "ols", "prior", "bounded_lm"):
                data = dict(grid_shape=g, n_samples=40, K=K, noise=0.3)
                cases.append((family, {**data, "random_state": 300 + j}))
                j += 1
    return cases


FAMILY_PARAMS = {
    "default": {},
    "soft_l1": {"nlss_loss": "soft_l1"},
    "two_stage": {"nlss_loss": "linear+soft_l1"},
    "ols": {"weight_method": "ols"},
    "prior": {},
    "bounded_lm": {"nlss_method": "bounded-lm"},
}


def _tfa_fit(family, data, mode, perturbation=0.0):
    X, R, truth = make_tfa_data(**data)
    if perturbation:
        rng = np.random.default_rng(data["random_state"] + 1)
        X = X * (1 + perturbation * rng.standard_normal(X.shape))
    K, D = truth["centers"].shape
    length = float(np.sqrt(max_sigma(R) / 2.0))
    prior = None
    if family == "prior":
        rng = np.random.default_rng(data["random_state"] + 2)
        prior = Prior(
            truth["centers"] + rng.normal(0.0, 0.1 * length, (K, D)),
            truth["widths"] * 1.1,
            np.tile(np.eye(D) * (0.3 * length) ** 2, (K, 1, 1)),
            np.full(K, max_sigma(R)),
            1.0,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = TFA(
            K=K, mode=mode, random_state=data["random_state"], **FAMILY_PARAMS[family]
        ).fit(X, R, prior=prior)
    return dict(
        centers=m.centers_,
        widths=m.widths_,
        weights=m.weights_,
        cost=np.array(m.cost_),
        true_centers=truth["centers"],
        length=np.array(length),
    )


def _differences(a, b):
    perm = match_factors(a["centers"], b["centers"])
    C, w, W = b["centers"][perm], b["widths"][perm], b["weights"][perm]
    return dict(
        centers=float(np.max(np.abs(C - a["centers"]))) / float(a["length"]),
        widths=float(np.max(np.abs(w / a["widths"] - 1))),
        weights=float(np.max(np.abs(W - a["weights"])) / np.max(np.abs(a["weights"]))),
        cost=float(abs(b["cost"] - a["cost"]) / a["cost"]),
    )


def _within(d):
    return (
        max(d["centers"], d["widths"], d["weights"]) <= THRESHOLD
        and d["cost"] <= COST_RTOL
    )


def _recovery(fit):
    perm = match_factors(fit["true_centers"], fit["centers"])
    err = np.max(np.abs(fit["centers"][perm] - fit["true_centers"]))
    return float(err) / float(fit["length"])


def test_fast_tfa_fits_match_exact_on_well_conditioned_problems():
    """56 synthetic problems (see the module docstring): on every one whose
    exact fit recovers the truth to RECOVERY_LENGTHS and is stable under a
    PERTURBATION of the data, the fast fit recovers the truth as well and is
    within THRESHOLD (centers, widths, weights) and COST_RTOL of the exact fit.
    Measured: 54 cases qualified on macOS (Accelerate) and 55 on Linux
    (OpenBLAS), with max differences of 5.1e-8 and 3.5e-8 lengths (centers),
    1.7e-7 and 1.9e-7 (widths), 9.5e-8 and 1.2e-7 (weights), 1.4e-10 and
    2.3e-10 (cost)."""
    rows, excluded = [], []
    for i, (family, data) in enumerate(_tfa_cases()):
        exact = _tfa_fit(family, data, "exact")
        stability = _differences(exact, _tfa_fit(family, data, "exact", PERTURBATION))
        if _recovery(exact) >= RECOVERY_LENGTHS or not _within(stability):
            excluded.append((i, family, data))
            continue
        fast = _tfa_fit(family, data, "fast")
        rows.append((i, family, data, _differences(exact, fast), _recovery(fast)))
    for case in excluded:
        print("excluded (not well-conditioned):", case)
    print(f"{len(rows)} of {len(rows) + len(excluded)} cases well-conditioned")
    for name in ("centers", "widths", "weights", "cost"):
        v = np.array([d[name] for *_, d, _ in rows])
        print(f"fast vs exact {name:8s} median {np.median(v):.2e} max {v.max():.2e}")
    assert len(rows) >= MIN_WELL_CONDITIONED_TFA
    for i, family, data, d, recovery in rows:
        assert _within(d), f"case {i} {family} {data}: {d}"
        assert recovery < RECOVERY_LENGTHS, f"case {i} {family} {data}: {recovery}"


def _htfa_cases():
    cases = []
    for grid in ((20, 20), (12, 12, 12)):
        for K in (3, 6):
            for update in ("brainiak", "fixed", "empirical"):
                for loss in ("linear", "soft_l1"):
                    cases.append((grid, K, update, loss, 500 + len(cases)))
    return cases


def _htfa_fit(case, mode, perturbation=0.0, from_truth=False):
    grid, K, update, loss, seed = case
    Xs, Rs, truth = make_htfa_data(3, grid, 40, K=K, noise=0.3, random_state=seed)
    if perturbation:
        rng = np.random.default_rng(seed + 1)
        Xs = [X * (1 + perturbation * rng.standard_normal(X.shape)) for X in Xs]
    starts = init_template = None
    if from_truth:
        starts = list(zip(truth["centers"], truth["widths"], strict=True))
        init_template = (truth["global_centers"], truth["global_widths"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = HTFA(
            K=K, template_update=update, nlss_loss=loss, mode=mode, random_state=seed
        ).fit(Xs, Rs, starts=starts, init_template=init_template)
    length = float(np.sqrt(max_sigma(Rs[0]) / 2.0))
    return model, np.mean(truth["centers"], axis=0), length


def _template_difference(a, b, length):
    perm = match_factors(a.global_centers_, b.global_centers_)
    diff = np.max(np.abs(b.global_centers_[perm] - a.global_centers_)) / length
    obj_a, obj_b = a.objective_history_[-1], b.objective_history_[-1]
    return float(diff), float(abs(obj_b - obj_a) / abs(obj_a))


def _template_error(model, target, length):
    perm = match_factors(target, model.global_centers_)
    return float(np.max(np.abs(model.global_centers_[perm] - target))) / length


def test_fast_htfa_fits_match_exact_on_well_conditioned_problems():
    """24 synthetic HTFA problems (3 subjects; see the module docstring). A
    case is well-conditioned when the exact fit recovers the mean true centers
    to RECOVERY_LENGTHS, is stable under a PERTURBATION of the data (template
    and local centers within THRESHOLD lengths), and ends near the template of
    the exact fit started from the truth (subject starts and template):
    within TRUTH_START_LENGTHS. The global loop stops on loose rules, so that
    distance is not rounding-sized: measured 1.2e-6 to 2.5e-2 lengths on the
    cases that pass, 0.17 to 1.85 on those where the exact fit stops at
    another optimum (case 518: 0.17 lengths from the truth against 0.035 from
    the truth start; on Linux the fast fit reached 0.028 there). On every
    well-conditioned case the fast template is within HTFA_GLOBAL_LENGTHS of
    the exact one, its summed final local objective within
    HTFA_OBJECTIVE_RTOL, and it recovers the truth as well."""
    rows, excluded = [], []
    for case in _htfa_cases():
        exact, target, length = _htfa_fit(case, "exact")
        perturbed = _htfa_fit(case, "exact", PERTURBATION)[0]
        from_truth = _htfa_fit(case, "exact", from_truth=True)[0]
        stable = (
            _template_difference(exact, perturbed, length)[0] <= THRESHOLD
            and all(
                np.max(np.abs(a - b)) / length <= THRESHOLD
                for a, b in zip(exact.centers_, perturbed.centers_, strict=True)
            )
            and _template_difference(exact, from_truth, length)[0]
            <= TRUTH_START_LENGTHS
        )
        error = _template_error(exact, target, length)
        if error >= RECOVERY_LENGTHS or not stable:
            excluded.append(case)
            continue
        fast, _, _ = _htfa_fit(case, "fast")
        diff, obj = _template_difference(exact, fast, length)
        fast_error = _template_error(fast, target, length)
        print(
            f"{case}: template fast - exact {diff:.2e} lengths, objective {obj:.2e}, "
            f"error to truth exact {error:.4f} fast {fast_error:.4f}"
        )
        rows.append((case, diff, obj, error, fast_error))
    for case in excluded:
        print("excluded (not well-conditioned):", case)
    assert len(rows) >= MIN_WELL_CONDITIONED_HTFA
    for case, diff, obj, error, fast_error in rows:
        assert diff <= HTFA_GLOBAL_LENGTHS, case
        assert obj <= HTFA_OBJECTIVE_RTOL, case
        assert fast_error <= error + HTFA_GLOBAL_LENGTHS, case


@pytest.mark.parametrize("mode", ["fast", "exact"])
def test_repeated_fits_are_bitwise_identical(mode):
    """Run to run, in one process, both modes give the same bits (K = 36: every
    parameter-sized vector is above the small-allocation sizes whose varying
    alignment changes Accelerate's rounding; see test_htfa.N_JOBS_TEST_K)."""
    X, R, _ = make_tfa_data((20, 20, 20), 60, K=20, random_state=0)
    fits = [
        TFA(K=36, init="auto", max_iter=20, mode=mode, random_state=0).fit(X, R)
        for _ in range(3)
    ]
    for f in fits[1:]:
        np.testing.assert_array_equal(f.centers_, fits[0].centers_)
        np.testing.assert_array_equal(f.widths_, fits[0].widths_)
        np.testing.assert_array_equal(f.weights_, fits[0].weights_)
        assert f.cost_ == fits[0].cost_


def test_tfa_rejects_unknown_mode():
    X, R, _ = make_tfa_data((10, 10), 5, K=2, random_state=0)
    with pytest.raises(ValueError, match="mode must be one of"):
        TFA(K=2, mode="approximate").fit(X, R)


# Real data: full-run fit of the first adult, defaults otherwise. Measured over
# the 12 adults (docs/validation.rst): fast - exact objective from -4.5e-4 to
# +2.7e-4 at K = 50 and from -4.7e-4 to +3.4e-4 at K = 100.
REAL_COST_RTOL = 1e-3


@pytest.mark.network
@pytest.mark.slow
@pytest.mark.parametrize("K", [50, 100])
def test_fast_mode_on_development_fmri(K):
    """On real fMRI the two modes end at different local minima of the same
    objective: their final objectives agree within REAL_COST_RTOL (either can
    be lower), and the fast fit takes less time (printed, not asserted: it
    depends on the machine's load)."""
    import time

    from htfa.datasets import fetch_development_fmri

    ds = fetch_development_fmri(n_subjects=1)
    X, R = ds.data[0], ds.coords[0]
    fits, seconds = {}, {}
    for mode in ("exact", "fast"):
        t0 = time.perf_counter()
        fits[mode] = TFA(K=K, mode=mode, random_state=0).fit(X, R)
        seconds[mode] = time.perf_counter() - t0
    rel = (fits["fast"].cost_ - fits["exact"].cost_) / fits["exact"].cost_
    print(
        f"K={K}: objective exact {fits['exact'].cost_:.6f} ({seconds['exact']:.1f} s, "
        f"init {fits['exact'].init_}), fast {fits['fast'].cost_:.6f} "
        f"({seconds['fast']:.1f} s, init {fits['fast'].init_}), relative {rel:+.2e}"
    )
    assert abs(rel) <= REAL_COST_RTOL


def _old_rule_solver(traces: list):
    """``bounded_lm`` wrapper for ``htfa._kernels``: runs the solver in mode
    'exact' and the solver of reference_kernels_3298278 (the damping rule of
    main before mode 'fast': lam *= nu, nu *= 2 at a failed factorization) on
    the same inputs, requires equal outputs bit for bit, and records the
    exact solve's trace."""
    import reference_kernels_3298278 as ref

    import htfa._kernels as kernels

    solve = kernels.bounded_lm

    def both(problem, x0, bnds, loss, scale, max_nfev=None, mode="exact"):
        assert mode == "exact"
        trace: list = []
        new = solve(problem, x0, bnds, loss, scale, max_nfev, mode, trace)
        old = ref.bounded_lm(problem, x0, bnds, loss, scale, max_nfev)
        np.testing.assert_array_equal(new[0], old[0])
        assert new[1:] == old[1:]
        traces.append(trace)
        return new

    return both


@pytest.mark.parametrize("case", _solve_cases(), ids=str)
def test_exact_solves_equal_the_earlier_solver(case):
    """Mode 'exact' is the solver as it was before mode 'fast' (the bounded_lm
    of reference_kernels_3298278), bit for bit, on the solver families."""
    kind, loss, spec, use_prior = case
    if kind == "gn":
        D, seed = spec
        prob, x0, _ = _gn_problem(D, use_prior, 600 + seed, V=150, T=20, K=3)
        bnds = bounds(prob.R, prob.K, 0.02, 4.0)
        x0 = np.clip(x0, *bnds)
    else:
        prob, x0, bnds = _tfa_solve_case(*spec, use_prior, loss)
    scale = 1.0 / prob.column_norms(x0)
    _old_rule_solver([])(prob, x0, bnds, loss, scale)


@pytest.mark.network
@pytest.mark.slow
def test_exact_mode_keeps_the_earlier_damping_rule_on_failed_factorizations(
    monkeypatch,
):
    """Exact fits of development_fmri at K = 100 meet failed damped
    factorizations (sub-pixar125: 6 in the fit, measured);
    every solve of the fit must equal the earlier solver's bit for bit, so
    the fast mode's damping rule cannot leak into mode 'exact'."""
    import htfa._kernels as kernels
    from htfa.datasets import fetch_development_fmri

    traces: list = []
    monkeypatch.setattr(kernels, "bounded_lm", _old_rule_solver(traces))
    ds = fetch_development_fmri(n_subjects=3)
    TFA(K=100, mode="exact", random_state=0).fit(ds.data[2], ds.coords[2])
    failures = sum(s["cholesky_failures"] for t in traces for s in t)
    print(f"{len(traces)} solves, {failures} failed factorizations, all equal")
    assert failures > 0
