"""Whole TFA fits against saved fits of commit 3298278 on well-conditioned problems.

``tests/data/fits_3298278.npz`` holds the centers, widths, weights and cost of
`FIT_CASES` fitted by the code of commit 3298278 (written by
``scripts/make_fit_references.py``; provenance in the file). Each case is a
synthetic problem whose reference fit recovers the true centers (checked here),
so its optimum is well determined (one case, `LOCAL_OPTIMA`, ends at a local
optimum in both versions and is compared all the same).

A fit stops when no center moved by more than ``threshold`` (1e-3) times the
length ``sqrt(max_sigma / 2)`` and no width by more than ``threshold``
relatively in the last iteration, so two fits whose solver arithmetic differs by
rounding can end that far apart. Those are the tolerances; the measured
differences are printed (and recorded in docs/validation.rst).
"""

import os

import numpy as np
import pytest

from htfa._kernels import max_sigma
from htfa._matching import match_factors
from htfa.synthetic import make_tfa_data
from htfa.tfa import TFA

THRESHOLD = 1e-3  # TFA's default threshold
REFERENCE = os.path.join(os.path.dirname(__file__), "data", "fits_3298278.npz")
GRIDS = ((20, 20), (30, 30), (25, 15), (10, 10, 10), (12, 12, 12), (14, 10, 8))
FIT_CASES = [
    dict(grid_shape=g, n_samples=40, K=K, noise=noise, random_state=100 + i)
    for i, (g, K, noise) in enumerate(
        (g, K, noise) for g in GRIDS for K in (2, 4, 6) for noise in (0.1, 0.5)
    )
]


# Reference fits that ended at a local optimum (max center error 1.75
# lengths); they are compared all the same, but are not well-conditioned.
LOCAL_OPTIMA = {23}  # (10, 10, 10), K = 6, noise 0.5


def fit_case(case: dict) -> dict[str, np.ndarray]:
    """TFA with its defaults on the synthetic problem of ``case``."""
    X, R, truth = make_tfa_data(**case)
    m = TFA(K=case["K"], random_state=case["random_state"]).fit(X, R)
    return dict(
        centers=m.centers_,
        widths=m.widths_,
        weights=m.weights_,
        cost=np.array(m.cost_),
        true_centers=truth["centers"],
        length=np.array(np.sqrt(max_sigma(R) / 2.0)),
    )


@pytest.fixture(scope="module")
def reference():
    return np.load(REFERENCE)


def _differences(case_index: int, ref) -> dict[str, float]:
    got = fit_case(FIT_CASES[case_index])
    key = f"case{case_index}_"
    C_ref, w_ref = ref[key + "centers"], ref[key + "widths"]
    perm = match_factors(C_ref, got["centers"])
    C, w, W = got["centers"][perm], got["widths"][perm], got["weights"][perm]
    W_ref = ref[key + "weights"]
    length = float(got["length"])
    return dict(
        centers=float(np.max(np.abs(C - C_ref))) / length,
        widths=float(np.max(np.abs(w / w_ref - 1))),
        weights=float(np.max(np.abs(W - W_ref)) / np.max(np.abs(W_ref))),
        cost=float(abs(got["cost"] - ref[key + "cost"]) / ref[key + "cost"]),
        recovery=float(
            np.max(
                np.abs(
                    C_ref[match_factors(got["true_centers"], C_ref)]
                    - got["true_centers"]
                )
            )
        )
        / length,
    )


def test_reference_provenance(reference):
    """The reference file says where it came from."""
    meta = str(reference["provenance"])
    print(meta)
    assert "3298278" in meta
    assert len([k for k in reference.files if k.endswith("_centers")]) == len(FIT_CASES)


def test_fits_match_3298278_on_well_conditioned_problems(reference):
    """36 fits (2D and 3D grids, K = 2, 4, 6, noise 0.1 and 0.5): centers within
    THRESHOLD lengths, widths and weights within THRESHOLD (relative), cost
    within 1e-8 (relative) of the 3298278 fits. All reference fits but those in
    LOCAL_OPTIMA recover the true centers to 0.2 lengths (a well-determined
    optimum)."""
    rows = [_differences(i, reference) for i in range(len(FIT_CASES))]
    for name in ("centers", "widths", "weights", "cost", "recovery"):
        v = np.array([r[name] for r in rows])
        print(f"{name:9s} median {np.median(v):.2e} max {v.max():.2e}")
    for i, r in enumerate(rows):
        msg = f"case {i}: {FIT_CASES[i]}: {r}"
        assert (r["recovery"] < 0.2) != (i in LOCAL_OPTIMA), msg
        assert r["centers"] <= THRESHOLD, msg
        assert r["widths"] <= THRESHOLD, msg
        assert r["weights"] <= THRESHOLD, msg
        assert r["cost"] <= 1e-8, msg
