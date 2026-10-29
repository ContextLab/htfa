"""Real connectome renders keep threshold labels separate from scale endpoints."""

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from htfa import TFA, HTFAResults, load
from htfa.results import _space_connectome_ticks


@pytest.fixture(scope="module")
def result():
    coords = np.stack(
        np.meshgrid(np.arange(-20, 21, 10), np.arange(-20, 21, 10), [-10, 0, 10]),
        axis=-1,
    ).reshape(-1, 3)
    centers = np.array([[-12, 0, 0], [12, 0, 0], [0, 12, 0]], dtype=float)
    widths = np.full(3, 100.0)
    factors = np.exp(-np.sum((coords[:, None] - centers) ** 2, axis=2) / widths)
    correlation = np.array([[1, 0.27, -0.26], [0.27, 1, 0.1], [-0.26, 0.1, 1]])
    scores = np.random.default_rng(0).normal(size=(20, 3))
    scores -= scores.mean(axis=0)
    orthogonal, _ = np.linalg.qr(scores)
    weights = np.linalg.cholesky(correlation) @ orthogonal.T
    data = factors @ weights
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = TFA(K=3, max_iter=0, standardize=False, weight_method="ols").fit(
            data, coords, start=(centers, widths)
        )
    return HTFAResults(model, load(data, coords=coords))


@pytest.mark.parametrize("threshold", ["75%", 0.26, 0.1, 0.0])
@pytest.mark.parametrize("embedded", [False, True])
def test_connectivity_ticks_are_legible_without_changing_scale(
    result, threshold, embedded
):
    figure, ax = plt.subplots(figsize=(5, 2.5)) if embedded else (None, None)
    display = result.plot_connectivity(threshold=threshold, snap=False, ax=ax)
    try:
        cbar = display._cbar
        ticks = cbar.get_ticks()
        np.testing.assert_allclose(ticks[[0, -1]], [-0.27, 0.27], atol=1e-12)
        assert 0 in ticks
        assert np.all(np.diff(ticks) > 0)
        if threshold == 0.1:
            np.testing.assert_allclose(ticks, [-0.27, -0.1, 0, 0.1, 0.27])
        elif threshold in ("75%", 0.26):
            np.testing.assert_allclose(ticks, [-0.27, 0, 0.27])
        canvas = cbar.ax.figure.canvas
        canvas.draw()
        bounds = sorted(
            (
                label.get_window_extent(canvas.get_renderer())
                for label in cbar.ax.get_yticklabels()
                if label.get_visible() and label.get_text()
            ),
            key=lambda box: box.y0,
        )
        assert all(a.y1 < b.y0 for a, b in zip(bounds, bounds[1:], strict=False))
        # Zero threshold renders all three nonzero factor connections.
        if threshold == 0.0:
            assert len(display.axes["z"].ax.lines) == 3
    finally:
        display.close()
        if figure is not None:
            plt.close(figure)


def test_tick_helper_accepts_real_display_without_colorbar():
    from nilearn.plotting import plot_glass_brain

    display = plot_glass_brain(None, colorbar=False)
    try:
        _space_connectome_ticks(display)
        assert not hasattr(display, "_cbar")
    finally:
        display.close()
