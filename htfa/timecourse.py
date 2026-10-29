"""Connectivity time courses: per-frame summaries, smoothing and a slide-style plot.

`plot_timecourse` draws conditions as mean +/- SEM ribbons over subjects with
a light-gray background, red / blue conditions, a dark shared pre-event curve,
magenta event lines, tan shaded periods, bold in-plot labels, and large
sentence-case fonts.
"""

# Adapted from ContextLab/directed-forgetting-network-dynamics, commit 31a2b66
# (code/dfnet/compare.py and code/scripts/figure_global_connectivity.py),
# MIT License, Copyright (c) 2019 Contextual Dynamics Laboratory.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ._lazy import lazy_import
from ._validation import real_array

Array = np.ndarray

__all__ = [
    "SLIDE_COLORS",
    "edge_count",
    "global_connectivity",
    "moving_average",
    "plot_timecourse",
]

#: Default colors for condition curves, event markers, and shaded periods.
SLIDE_COLORS = {
    "red": "#E8202A",
    "blue": "#3A52A4",
    "dark": "#333333",
    "event": "#FF00A0",
    "period": "#D2A184",
    "background": "#EBEBEB",
}


def _frames(frames: Any) -> Array:
    f = real_array(frames, "frames")
    if f.ndim != 3 or f.shape[1] != f.shape[2]:
        raise ValueError(f"frames must be (T, K, K), got {f.shape}")
    return f


def global_connectivity(frames: Any) -> Array:
    """Mean edge per frame over the upper triangle including the diagonal.

    Diagonal values contribute to the mean. Frames containing NaN give NaN.

    Parameters
    ----------
    frames : array of shape (T, K, K)

    Returns
    -------
    ndarray of shape (T,)
    """
    f = _frames(frames)
    i, j = np.triu_indices(f.shape[-1], k=0)
    return f[:, i, j].mean(axis=1)


def edge_count(frames: Any, threshold: float = 0.25) -> Array:
    """Number of node pairs (upper triangle, no diagonal) with ``|E| > threshold``.

    Frames containing NaN give NaN.

    Returns
    -------
    ndarray of shape (T,)
    """
    f = _frames(frames)
    i, j = np.triu_indices(f.shape[-1], k=1)
    v = f[:, i, j]
    out = (np.abs(v) > threshold).sum(axis=1).astype(np.float64)
    out[np.isnan(v).any(axis=1)] = np.nan
    return out


def moving_average(y: Any, window: int) -> Array:
    """Centered moving average along the last axis.

    Like MATLAB's ``smooth(y, window)``: an odd ``window``, shrinking
    symmetrically near the ends (the first and last points are unchanged).
    """
    if window < 1 or window % 2 == 0:
        raise ValueError(f"window must be a positive odd integer, got {window}")
    y = real_array(y, "time course")
    T = y.shape[-1]
    out = np.empty_like(y)
    h = window // 2
    for t in range(T):
        k = min(h, t, T - 1 - t)
        out[..., t] = y[..., t - k : t + k + 1].mean(axis=-1)
    return out


def _mean_sem(y: Array) -> tuple[Array, Array | None]:
    if y.ndim == 1:
        return y, None
    n = np.sum(~np.isnan(y), axis=0)
    mean = np.nanmean(y, axis=0)
    if y.shape[0] < 2:
        return mean, None
    sd = np.nanstd(y, axis=0, ddof=1)
    return mean, sd / np.sqrt(n)


def plot_timecourse(
    conditions: Mapping[str, Any],
    x: Any = None,
    *,
    colors: Mapping[str, Any] | None = None,
    smooth: int | None = None,
    shared_before: float | None = None,
    events: Sequence[float] = (),
    periods: Sequence[tuple[float, float]] = (),
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    ax: Any = None,
    fontsize: float = 18,
) -> Any:
    """Plot condition time courses as mean +/- SEM ribbons (slide style).

    Labels and annotations describe the supplied analysis. For example, a
    stimulus-response experiment might plot one curve per condition, mark
    stimulus onset with ``events``, and shade its response interval with
    ``periods``. These annotations are optional and do not require a particular
    experimental task. All annotation positions use the units of ``x``.

    Parameters
    ----------
    conditions : mapping of label -> array of shape (S, T) or (T,)
        One curve per subject (rows), or a single curve (no ribbon).
    x : array of shape (T,), optional
        Time axis; default ``0..T-1``.
    colors : mapping of label -> color, optional
        Default: slide red, slide blue, then matplotlib's cycle.
    smooth : int, optional
        Window of a centered moving average (`moving_average`) applied to
        each subject's curve before averaging.
    shared_before : float, optional
        Draw the mean of the conditions as one dark curve before this x value;
        individual condition curves begin one sample before it. This optional
        display choice can represent a shared baseline interval; it does not
        test whether the conditions are equal there.
    events : sequence of float
        x positions of magenta vertical lines, for example stimulus onsets.
    periods : sequence of (start, end)
        x ranges shaded tan, for example intervals in which a response is
        measured.
    xlabel, ylabel, title : str, optional
    ax : matplotlib Axes, optional
    fontsize : float
        Base font size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = lazy_import("matplotlib.pyplot", "plot", "plotting")
    if not conditions:
        raise ValueError("no conditions to plot")
    curves = {}
    for label, y in conditions.items():
        y = real_array(y, "time course")
        if y.ndim not in (1, 2):
            raise ValueError(f"{label!r}: expected (S, T) or (T,), got {y.shape}")
        curves[label] = moving_average(y, smooth) if smooth else y
    T = {c.shape[-1] for c in curves.values()}
    if len(T) != 1:
        raise ValueError(f"conditions have different lengths: {sorted(T)}")
    n = T.pop()
    xs = np.arange(n, dtype=np.float64) if x is None else real_array(x, "x")
    if xs.shape != (n,):
        raise ValueError(f"x has shape {xs.shape}, expected ({n},)")
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    defaults = [SLIDE_COLORS["red"], SLIDE_COLORS["blue"], *cycle]
    col = {
        label: (colors or {}).get(label, defaults[k % len(defaults)])
        for k, label in enumerate(curves)
    }
    bg = SLIDE_COLORS["background"]
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=bg)
    else:
        fig = ax.figure
    ax.set_facecolor(bg)
    stats = {label: _mean_sem(y) for label, y in curves.items()}
    keep = np.ones(n, dtype=bool)
    if shared_before is not None:
        pre = xs < shared_before
        means = np.array([m for m, _ in stats.values()])
        sems = [e for _, e in stats.values()]
        pre_m = means.mean(axis=0)
        ax.plot(xs[pre], pre_m[pre], color=SLIDE_COLORS["dark"], lw=1.5)
        if all(e is not None for e in sems):
            pre_e = np.sqrt(np.sum(np.square(sems), axis=0)) / len(sems)
            ax.fill_between(
                xs[pre],
                (pre_m - pre_e)[pre],
                (pre_m + pre_e)[pre],
                color=SLIDE_COLORS["dark"],
                lw=0,
                alpha=0.35,
            )
        first = int(np.argmax(~pre)) if (~pre).any() else n
        keep = np.arange(n) >= max(first - 1, 0)
    for label, (m, e) in stats.items():
        if e is not None:
            ax.fill_between(
                xs[keep],
                (m - e)[keep],
                (m + e)[keep],
                color=col[label],
                lw=0,
                alpha=0.4,
            )
        ax.plot(xs[keep], m[keep], color=col[label], lw=2)
    for start, end in periods:
        ax.axvspan(start, end, color=SLIDE_COLORS["period"], alpha=0.8, lw=0, zorder=0)
    for at in events:
        ax.axvline(at, color=SLIDE_COLORS["event"], lw=3, zorder=1)
    for k, label in enumerate(curves):
        ax.text(
            0.97,
            0.95 - 0.08 * k,
            label,
            color=col[label],
            fontweight="bold",
            fontsize=fontsize,
            transform=ax.transAxes,
            ha="right",
            va="top",
        )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=fontsize * 0.85)
    ax.set_xlim(xs[0], xs[-1])
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    if title:
        ax.set_title(title, fontsize=fontsize * 0.85)
    fig.tight_layout()
    return fig
