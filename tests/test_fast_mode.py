"""``mode='fast'`` against ``mode='exact'``, both computed here (same machine,
same BLAS), over the scenario families of the equivalence suite.

The two modes minimize the same objective with the same arithmetic and stopping
rules; they differ only in the step `htfa._kernels.bounded_lm` takes where the
Hessian is not positive definite on the free variables (exact: Gauss-Newton;
fast: the Hessian with more damping). Where the Hessian stays positive definite
the two runs are identical, so the families below differ only where a solve
meets negative curvature.

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

from htfa._kernels import Prior, bounded_lm, bounds, max_sigma, solve_centers_widths
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
# of the fast fit against the exact one. Measured over the 19 well-conditioned
# cases below: 1.3e-4 lengths and 7e-8 relative.
HTFA_GLOBAL_LENGTHS = 1e-3
HTFA_OBJECTIVE_RTOL = 1e-6
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
    Measured: max differences 5.2e-8 lengths (centers), 1.7e-7 (widths),
    1.4e-10 (cost)."""
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


def _htfa_fit(case, mode, perturbation=0.0):
    grid, K, update, loss, seed = case
    Xs, Rs, truth = make_htfa_data(3, grid, 40, K=K, noise=0.3, random_state=seed)
    if perturbation:
        rng = np.random.default_rng(seed + 1)
        Xs = [X * (1 + perturbation * rng.standard_normal(X.shape)) for X in Xs]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = HTFA(
            K=K, template_update=update, nlss_loss=loss, mode=mode, random_state=seed
        ).fit(Xs, Rs)
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
    """24 synthetic HTFA problems (3 subjects; see the module docstring): on
    every one whose exact template recovers the mean true centers to
    RECOVERY_LENGTHS and is stable under a PERTURBATION of the data (template
    and local centers within THRESHOLD lengths), the fast template is within
    HTFA_GLOBAL_LENGTHS of the exact one, its summed final local objective
    within HTFA_OBJECTIVE_RTOL, and it recovers the truth as well."""
    rows, excluded = [], []
    for case in _htfa_cases():
        exact, target, length = _htfa_fit(case, "exact")
        perturbed, _, _ = _htfa_fit(case, "exact", PERTURBATION)
        stable = _template_difference(exact, perturbed, length)[0] <= THRESHOLD and all(
            np.max(np.abs(a - b)) / length <= THRESHOLD
            for a, b in zip(exact.centers_, perturbed.centers_, strict=True)
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
