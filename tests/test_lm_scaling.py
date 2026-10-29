"""Finite local optimization metrics for nearly vanished mixture components."""

import numpy as np
import pytest

from htfa._engines import NumpyEngine
from htfa._kernels import (
    Kernel,
    Prior,
    ProjectedProblem,
    ResidualProblem,
    _finite_lm_scale,
    _lm_initial_damping,
    bounds,
    solve_centers_widths,
    x_factors,
)


def test_representable_scaled_curvature_keeps_tiny_column_scale():
    scale = np.array([1e158, 2.0])
    gn = np.diag([1e-316, 3.0])
    gradient = np.array([1e-158, 1.0])
    assert _finite_lm_scale(scale, gradient, gn, gn) is scale
    damping = _lm_initial_damping(scale, gn)
    assert np.isfinite(damping) and damping > 0
    assert damping == pytest.approx(0.012)


@pytest.mark.parametrize("mode", ["exact", "fast"])
@pytest.mark.parametrize("projected", [False, True])
@pytest.mark.parametrize("use_prior", [False, True])
def test_solver_handles_a_nearly_vanished_mixture_component(mode, projected, use_prior):
    kernel = Kernel("rbf", 2)
    coords = np.stack(
        np.meshgrid(np.arange(8.0), np.arange(8.0), indexing="ij"), -1
    ).reshape(-1, 2)
    centers = np.array([[2.0, 2.0], [5.0, 5.0], [2.0, 5.0], [5.0, 2.0]])
    widths = np.full(4, 4.0)
    logits = np.full((2, 1), -365.0)
    start = kernel.join(centers, widths, logits)
    truth = kernel.join(centers + 0.5, widths, np.zeros((2, 1)))
    weights = np.random.default_rng(0).normal(size=(2, 10))
    data = x_factors(coords, centers + 0.5, truth[8:], 2, kernel, None) @ weights
    prior = (
        Prior(
            centers.copy(),
            widths.copy(),
            np.tile(np.eye(2), (4, 1, 1)),
            np.ones(4),
            1.0,
            logits.copy(),
            np.ones((2, 1)),
        )
        if use_prior
        else None
    )
    problem = (
        ProjectedProblem(data, coords, 2, NumpyEngine(), prior, kernel=kernel)
        if projected
        else ResidualProblem(data, weights, coords, NumpyEngine(), prior, kernel=kernel)
    )
    norms = problem.column_norms(start)
    assert np.any((norms > 0) & (norms < 1 / np.sqrt(np.finfo(float).max)))
    before = problem.cost(start, "linear")
    with np.errstate(over="raise", invalid="raise"):
        solution, cost = solve_centers_widths(
            problem,
            start,
            bounds(coords, 2, 0.02, 1.8, kernel),
            method="varpro" if projected else "bounded-lm",
            loss="linear",
            mode=mode,
            max_nfev=30,
        )
    assert np.isfinite(solution).all() and np.isfinite(cost)
    assert cost == pytest.approx(problem.cost(solution, "linear") + problem.cost_offset)
    assert cost <= before + problem.cost_offset + 1e-10
    if use_prior:
        # This is the HTFA failure: finite prior curvature must permit a real
        # improving step despite a nearly absent data-Jacobian column.
        assert cost < before + problem.cost_offset - 1e-10
        assert np.linalg.norm(solution - start) > 1e-10


def test_scaling_guard_checks_both_ends_of_overflowing_cross_terms():
    scale = np.array([1e158, 1e-158])
    hess = np.array([[0.0, 1e200], [1e200, 0.0]])
    gn = np.diag([1e-316, 1.0])
    adjusted = _finite_lm_scale(scale, np.zeros(2), gn, hess)
    np.testing.assert_array_equal(adjusted, [1.0, 1e-158])
    with np.errstate(over="raise", invalid="raise"):
        assert np.isfinite(adjusted[:, None] * hess * adjusted[None, :]).all()
    assert _finite_lm_scale(adjusted, np.zeros(2), gn, hess) is adjusted
