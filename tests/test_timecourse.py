"""Tests for htfa.timecourse: summaries, smoothing and the slide-style plot."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from PIL import Image  # noqa: E402

from htfa import plot_timecourse  # noqa: E402
from htfa.timecourse import (  # noqa: E402
    SLIDE_COLORS,
    edge_count,
    global_connectivity,
    moving_average,
)


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def test_global_connectivity_includes_diagonal():
    E = np.array([[[1.0, 0.5, -0.2], [0.5, 1.0, 0.1], [-0.2, 0.1, 1.0]]])
    # upper triangle with diagonal: 1, .5, -.2, 1, .1, 1 -> mean 3.4 / 6
    np.testing.assert_allclose(global_connectivity(E), [3.4 / 6])
    E2 = np.concatenate([E, np.full((1, 3, 3), np.nan)])
    out = global_connectivity(E2)
    assert out[0] == pytest.approx(3.4 / 6) and np.isnan(out[1])
    with pytest.raises(ValueError, match=r"\(T, K, K\)"):
        global_connectivity(E[0])


def test_edge_count_threshold_strict():
    E = np.zeros((3, 4, 4))
    E[0, 0, 1] = E[0, 1, 0] = 0.3
    E[0, 2, 3] = E[0, 3, 2] = -0.5
    E[1, 0, 1] = E[1, 1, 0] = 0.25  # equal to threshold: not counted
    E[2, 0, 2] = np.nan
    np.testing.assert_array_equal(edge_count(E, 0.25)[:2], [2, 0])
    assert np.isnan(edge_count(E, 0.25)[2])
    np.testing.assert_array_equal(edge_count(E, 0.4)[:2], [1, 0])


def test_moving_average_matlab_smooth():
    y = np.array([1.0, 2, 6, 4, 10, 0, 3])
    # MATLAB smooth(y, 5): spans 1, 3, 5, 5, 5, 3, 1
    expect = [1, 3, (1 + 2 + 6 + 4 + 10) / 5, (2 + 6 + 4 + 10 + 0) / 5,
              (6 + 4 + 10 + 0 + 3) / 5, (10 + 0 + 3) / 3, 3]  # fmt: skip
    np.testing.assert_allclose(moving_average(y, 5), expect)
    np.testing.assert_allclose(moving_average(y, 1), y)
    two = moving_average(np.stack([y, 2 * y]), 3)  # along the last axis
    np.testing.assert_allclose(two[1], 2 * two[0])
    for bad in (0, 4, -3):
        with pytest.raises(ValueError, match="positive odd"):
            moving_average(y, bad)


def _png(fig, path):
    fig.savefig(path, dpi=60, facecolor=fig.get_facecolor())
    return np.asarray(Image.open(path).convert("RGB")).astype(int)


def _has(a, color, tol=12):
    target = np.array(to_rgb(color)) * 255
    return int((np.abs(a - target).max(axis=2) <= tol).sum())


def test_plot_timecourse_slide_style(tmp_path):
    rng = np.random.default_rng(0)
    x = np.arange(-10, 21)
    base = np.exp(-(((x - 4) / 3.0) ** 2))
    rem = 0.2 + 0.1 * base + 0.02 * rng.standard_normal((12, x.size))
    forg = 0.2 + 0.2 * base + 0.02 * rng.standard_normal((12, x.size))
    fig = plot_timecourse(
        {"Remember": rem, "Forget": forg}, x, smooth=5, shared_before=0,
        events=[0], periods=[(0, 5)], xlabel="TR relative to cue",
        ylabel="Global connectivity", title="Network after the cue",
    )  # fmt: skip
    ax = fig.axes[0]
    a = _png(fig, tmp_path / "tc.png")
    print({k: _has(a, v) for k, v in SLIDE_COLORS.items()})
    assert tuple(a[2, 2]) == tuple(int(round(c * 255)) for c in to_rgb("#EBEBEB"))
    for key in ("red", "blue", "event", "dark"):
        assert _has(a, SLIDE_COLORS[key]) > 20, key
    # in-plot bold labels in the condition colors, no legend
    labels = {t.get_text(): t for t in ax.texts}
    assert set(labels) == {"Remember", "Forget"} and ax.get_legend() is None
    assert labels["Remember"].get_fontweight() == "bold"
    assert to_rgb(labels["Forget"].get_color()) == to_rgb(SLIDE_COLORS["blue"])
    assert not ax.spines["top"].get_visible() and not ax.spines["right"].get_visible()
    # the plotted condition means are the smoothed means, from x = -1 on
    lines = [ln for ln in ax.get_lines() if ln.get_linewidth() == 2]
    rem_line = next(ln for ln in lines if to_rgb(ln.get_color()) == to_rgb("#E8202A"))
    np.testing.assert_allclose(rem_line.get_xdata(), x[x >= -1])
    np.testing.assert_allclose(
        rem_line.get_ydata(), moving_average(rem, 5).mean(axis=0)[x >= -1]
    )
    dark = [ln for ln in ax.get_lines() if to_rgb(ln.get_color()) == to_rgb("#333333")]
    assert len(dark) == 1 and dark[0].get_xdata().max() == -1
    assert ax.get_xlabel() == "TR relative to cue"


def test_plot_timecourse_single_curves_and_errors(tmp_path):
    fig, ax = plt.subplots()
    out = plot_timecourse({"A": np.arange(5.0), "B": np.ones(5)}, ax=ax,
                          colors={"B": "green"})  # fmt: skip
    assert out is fig and len(ax.collections) == 0  # no ribbons for single curves
    assert to_rgb(ax.get_lines()[1].get_color()) == to_rgb("green")
    with pytest.raises(ValueError, match="different lengths"):
        plot_timecourse({"A": np.ones(5), "B": np.ones(6)})
    with pytest.raises(ValueError, match="x has shape"):
        plot_timecourse({"A": np.ones(5)}, np.arange(4))
    with pytest.raises(ValueError, match="no conditions"):
        plot_timecourse({})
    with pytest.raises(ValueError, match=r"expected \(S, T\)"):
        plot_timecourse({"A": np.ones((2, 2, 2))})
