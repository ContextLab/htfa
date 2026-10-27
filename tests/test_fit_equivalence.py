"""Whole TFA fits by the current kernels against the 3298278 kernels, on the same
machine, on problems that are demonstrably well-conditioned.

The oracle fit is ``htfa.tfa.TFA`` running on ``tests/reference_kernels_3298278.py``
(`oracle_kernels`): ``htfa/tfa.py`` is unchanged since commit 3298278, so this
is the 3298278 fit, computed with this machine's BLAS. Both fits therefore see
the same platform rounding, and only the kernels differ.

A case counts as well-conditioned when the oracle fit (a) recovers the true
centers to `RECOVERY_LENGTHS` lengths ``sqrt(max_sigma / 2)`` and (b) is stable:
the oracle fit on the data changed by a relative 1e-12 (`PERTURBATION`) agrees
with it within the stopping rule's tolerances below. Only such cases are
compared; the others (local optima, or optima that rounding moves) are counted
and printed.

A fit stops when no center moved by more than ``threshold`` (1e-3) lengths and
no width by more than ``threshold`` relatively in the last iteration, so two
fits whose arithmetic differs by rounding can end that far apart. Those are the
tolerances for centers, widths and weights; the cost's is 1e-8.
"""

from contextlib import contextmanager

import numpy as np
import reference_kernels_3298278 as ref

import htfa.tfa as tfa_module
from htfa._kernels import max_sigma
from htfa._matching import match_factors
from htfa.synthetic import make_tfa_data
from htfa.tfa import TFA

THRESHOLD = 1e-3  # TFA's default threshold
COST_RTOL = 1e-8
RECOVERY_LENGTHS = 0.2
PERTURBATION = 1e-12
MIN_WELL_CONDITIONED = 30
GRIDS = ((20, 20), (30, 30), (25, 15), (10, 10, 10), (12, 12, 12), (14, 10, 8))
FIT_CASES = [
    dict(grid_shape=g, n_samples=40, K=K, noise=noise, random_state=100 + i)
    for i, (g, K, noise) in enumerate(
        (g, K, noise) for g in GRIDS for K in (2, 4, 6) for noise in (0.1, 0.5)
    )
]
# The kernel names htfa.tfa imports.
KERNEL_NAMES = (
    "Prior",
    "ProjectedProblem",
    "ResidualProblem",
    "bounds",
    "factors",
    "init_centers_widths",
    "max_sigma",
    "ridge_weights",
    "solve_centers_widths",
)


def _oracle_solve(*args, mode: str, **kwargs):
    """The 3298278 solver, which has only the 'exact' step model."""
    assert mode == "exact", mode
    return ref.solve_centers_widths(*args, **kwargs)


@contextmanager
def oracle_kernels():
    """``htfa.tfa`` with the 3298278 kernels in place of the current ones."""
    saved = {name: getattr(tfa_module, name) for name in KERNEL_NAMES}
    try:
        for name in KERNEL_NAMES:
            setattr(tfa_module, name, getattr(ref, name))
        tfa_module.solve_centers_widths = _oracle_solve
        yield
    finally:
        for name, value in saved.items():
            setattr(tfa_module, name, value)


def fit_case(case: dict, perturbation: float = 0.0) -> dict[str, np.ndarray]:
    """TFA with its defaults, in mode 'exact' (the 3298278 solver's), on the
    synthetic problem of ``case``."""
    X, R, truth = make_tfa_data(**case)
    if perturbation:
        rng = np.random.default_rng(case["random_state"] + 1)
        X = X * (1 + perturbation * rng.standard_normal(X.shape))
    m = TFA(K=case["K"], mode="exact", random_state=case["random_state"]).fit(X, R)
    return dict(
        centers=m.centers_,
        widths=m.widths_,
        weights=m.weights_,
        cost=np.array(m.cost_),
        true_centers=truth["centers"],
        length=np.array(np.sqrt(max_sigma(R) / 2.0)),
    )


def _differences(a: dict, b: dict) -> dict[str, float]:
    """Differences of fit ``b`` from fit ``a`` (factors matched by center)."""
    perm = match_factors(a["centers"], b["centers"])
    C, w, W = b["centers"][perm], b["widths"][perm], b["weights"][perm]
    return dict(
        centers=float(np.max(np.abs(C - a["centers"]))) / float(a["length"]),
        widths=float(np.max(np.abs(w / a["widths"] - 1))),
        weights=float(np.max(np.abs(W - a["weights"])) / np.max(np.abs(a["weights"]))),
        cost=float(abs(b["cost"] - a["cost"]) / a["cost"]),
    )


def _within(d: dict[str, float]) -> bool:
    return (
        d["centers"] <= THRESHOLD
        and d["widths"] <= THRESHOLD
        and d["weights"] <= THRESHOLD
        and d["cost"] <= COST_RTOL
    )


def _recovery(fit: dict) -> float:
    perm = match_factors(fit["true_centers"], fit["centers"])
    err = np.max(np.abs(fit["centers"][perm] - fit["true_centers"]))
    return float(err) / float(fit["length"])


def test_fits_match_the_3298278_kernels_on_well_conditioned_problems():
    """36 candidate problems (2D and 3D grids, K = 2, 4, 6, noise 0.1 and 0.5):
    on every well-conditioned one (see the module docstring), centers within
    THRESHOLD lengths, widths and weights within THRESHOLD (relative), cost
    within COST_RTOL of the oracle fit. At least MIN_WELL_CONDITIONED of the
    candidates must qualify."""
    rows, excluded = [], []
    for i, case in enumerate(FIT_CASES):
        with oracle_kernels():
            old = fit_case(case)
            old_perturbed = fit_case(case, PERTURBATION)
        recovery = _recovery(old)
        stability = _differences(old, old_perturbed)
        if recovery >= RECOVERY_LENGTHS or not _within(stability):
            excluded.append((i, case, f"recovery {recovery:.3f}", stability))
            continue
        rows.append((i, case, _differences(old, fit_case(case))))
    for i, case, rec, stab in excluded:
        print(f"excluded case {i} {case}: {rec}, perturbed oracle {stab}")
    print(f"{len(rows)} of {len(FIT_CASES)} cases well-conditioned")
    for name in ("centers", "widths", "weights", "cost"):
        v = np.array([d[name] for *_, d in rows])
        print(f"{name:8s} median {np.median(v):.2e} max {v.max():.2e}")
    assert len(rows) >= MIN_WELL_CONDITIONED
    for i, case, d in rows:
        assert _within(d), f"case {i}: {case}: {d}"
