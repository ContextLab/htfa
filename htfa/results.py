"""Results container: factors, reconstruction, NIfTI export, connectivity, plots."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._kernels import factors as _factors
from ._lazy import lazy_import
from .connectivity import dynamic_connectivity, isfc, static_connectivity
from .htfa import HTFA
from .io import Subject
from .tfa import TFA, standardize_data

Array = np.ndarray

__all__ = ["HTFAResults"]


def _scatter_field(ax: Any, coords: Array, values: Array) -> None:
    """Draw ``values`` at 2D ``coords`` (row, column) as squares."""
    ax.scatter(coords[:, 1], coords[:, 0], c=values, marker="s", s=12, cmap="gray")
    ax.set_aspect("equal")
    ax.invert_yaxis()


class HTFAResults:
    """Fitted model together with the subjects it was fitted on.

    Parameters
    ----------
    model : TFA or HTFA
        A fitted estimator.
    subjects : list of htfa.io.Subject
        The subjects the model was fitted on, in order.

    Attributes
    ----------
    model, subjects
        As given.
    centers, widths, weights : list
        Per-subject centers ``(K, D)``, widths ``(K,)`` and weights ``(K, T)``.
    global_centers, global_widths : ndarray or None
        The HTFA template; ``None`` for a single-subject TFA fit.

    Examples
    --------
    >>> import numpy as np
    >>> from htfa import fit
    >>> res = fit(np.random.default_rng(0).random((6, 6, 6, 8)) + 1, K=2,
    ...           standardize=False)
    >>> res.reconstruct().shape[1]
    8
    """

    def __init__(self, model: TFA | HTFA, subjects: list[Subject]) -> None:
        self.model = model
        self.subjects = subjects
        if isinstance(model, HTFA):
            self.centers = list(model.centers_)
            self.widths = list(model.widths_)
            self.weights = list(model.weights_)
            self.global_centers: Array | None = model.global_centers_
            self.global_widths: Array | None = model.global_widths_
        else:
            self.centers = [model.centers_]
            self.widths = [model.widths_]
            self.weights = [model.weights_]
            self.global_centers = None
            self.global_widths = None
        if len(self.centers) != len(subjects):
            raise ValueError(
                f"model has {len(self.centers)} subjects but {len(subjects)} given"
            )

    def factors(self, subject: int = 0) -> Array:
        """Factor images ``(V, K)`` of ``subject`` at its coordinates."""
        return _factors(
            self.subjects[subject].coords, self.centers[subject], self.widths[subject]
        )

    def reconstruct(self, subject: int = 0) -> Array:
        """Reconstruction ``(V, T)`` in the standardized space of the fit."""
        return self.factors(subject) @ self.weights[subject]

    def _values(self, subject: int, what: str) -> Array:
        if what == "reconstruction":
            return self.reconstruct(subject)
        if what == "factors":
            return self.factors(subject)
        raise ValueError(f"what must be 'reconstruction' or 'factors', got {what!r}")

    def to_nifti(self, subject: int = 0, what: str = "reconstruction") -> Any:
        """Reconstruction or factors as a NIfTI image (NaN outside the mask)."""
        if self.subjects[subject].affine is None:
            raise ValueError("subject has no affine (not NIfTI input); use to_image")
        return self.subjects[subject].to_image(self._values(subject, what))

    def to_image(self, subject: int = 0, what: str = "reconstruction") -> Array:
        """Reconstruction or factors as an array in image space (NaN outside mask)."""
        img = self.subjects[subject].to_image(self._values(subject, what))
        return np.asarray(getattr(img, "dataobj", img), dtype=np.float64)

    def connectivity(self, subject: int = 0) -> Array:
        """Static factor connectivity ``(K, K)`` of ``subject``."""
        return static_connectivity(self.weights[subject])

    def dynamic_connectivity(self, subject: int = 0, window: int = 20) -> Array:
        """Sliding-window factor connectivity ``(T - window + 1, K, K)``."""
        return dynamic_connectivity(self.weights[subject], window)

    def isfc(self, window: int | None = None) -> Array:
        """Inter-subject factor correlation across all subjects."""
        return isfc(self.weights, window)

    def _dim(self) -> int:
        D = int(self.centers[0].shape[1])
        if D not in (2, 3):
            raise ValueError(f"plotting supports 2D and 3D coordinates; got {D}D")
        return D

    def plot_nodes(self, subject: Any = "global", ax: Any = None, **kwargs: Any) -> Any:
        """Plot factor centers (``"global"`` template plus subjects, or one subject)."""
        D = self._dim()
        glob = subject == "global" and self.global_centers is not None
        if subject == "global" and not glob:
            subject = 0
        plt = lazy_import("matplotlib.pyplot", "plot", "plotting")
        K = self.centers[0].shape[0]
        if D == 3:
            plotting = lazy_import("nilearn.plotting", "plot", "3D plotting")
            none = np.zeros((K, K))
            if glob:
                disp = plotting.plot_connectome(
                    none,
                    self.global_centers,
                    node_color="black",
                    node_size=50,
                    axes=ax,
                    colorbar=False,
                    **kwargs,
                )
                colors = plt.get_cmap("Spectral")(np.linspace(0, 1, len(self.centers)))
                for c, col in zip(self.centers, colors, strict=True):
                    disp.add_markers(c, marker_color=[col], marker_size=20)
            else:
                disp = plotting.plot_connectome(
                    none, self.centers[subject], axes=ax, colorbar=False, **kwargs
                )
            return disp
        fig, ax = (ax.figure, ax) if ax is not None else plt.subplots()
        s = self.subjects[0 if glob else subject]
        _scatter_field(ax, s.coords, s.data.mean(axis=1))
        if glob:
            colors = plt.get_cmap("Spectral")(np.linspace(0, 1, len(self.centers)))
            for c, col in zip(self.centers, colors, strict=True):
                ax.scatter(c[:, 1], c[:, 0], color=[col], s=20)
            gc = self.global_centers
            assert gc is not None
            ax.scatter(gc[:, 1], gc[:, 0], c="k", s=50)
        else:
            ax.scatter(
                self.centers[subject][:, 1], self.centers[subject][:, 0], c="r", s=30
            )
        return fig

    def plot_connectivity(
        self, subject: int = 0, threshold: Any = "75%", ax: Any = None
    ) -> Any:
        """Plot factor connectivity as edges between factor centers."""
        D = self._dim()
        conn = self.connectivity(subject)
        np.fill_diagonal(conn, 0)
        centers = self.centers[subject]
        if D == 3:
            plotting = lazy_import("nilearn.plotting", "plot", "3D plotting")
            return plotting.plot_connectome(
                conn, centers, edge_threshold=threshold, axes=ax
            )
        plt = lazy_import("matplotlib.pyplot", "plot", "plotting")
        fig, ax = (ax.figure, ax) if ax is not None else plt.subplots()
        s = self.subjects[subject]
        _scatter_field(ax, s.coords, s.data.mean(axis=1))
        a = np.abs(conn)
        if isinstance(threshold, str) and threshold.endswith("%"):
            cut = np.percentile(a[np.triu_indices_from(a, 1)], float(threshold[:-1]))
        else:
            cut = float(threshold)
        cmap = plt.get_cmap("coolwarm")
        for i, j in zip(*np.triu_indices_from(conn, 1), strict=True):
            if a[i, j] >= cut:
                ax.plot(
                    centers[[i, j], 1],
                    centers[[i, j], 0],
                    color=cmap((conn[i, j] + 1) / 2),
                    lw=1 + 3 * a[i, j],
                )
        ax.scatter(centers[:, 1], centers[:, 0], c="k", s=20)
        return fig

    def plot_reconstruction(self, subject: int = 0, sample: int = 0) -> Any:
        """Original (standardized) data vs reconstruction of one sample."""
        D = self._dim()
        plt = lazy_import("matplotlib.pyplot", "plot", "plotting")
        s = self.subjects[subject]
        orig = standardize_data(s.data, self.model.standardize)[:, sample]
        recon = self.reconstruct(subject)[:, sample]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        titles = ("original", "reconstruction")
        if D == 3:
            plotting = lazy_import("nilearn.plotting", "plot", "3D plotting")
            if s.affine is None:
                raise ValueError("3D plot_reconstruction needs NIfTI input (affine)")
            for ax, v, t in zip(axes, (orig, recon), titles, strict=True):
                img = s.to_image(v)
                img = type(img)(np.nan_to_num(np.asarray(img.dataobj)), img.affine)
                plotting.plot_glass_brain(img, figure=fig, axes=ax, title=t)
            return fig
        for ax, v, t in zip(axes, (orig, recon), titles, strict=True):
            if s.image_shape is not None:
                ax.imshow(np.asarray(s.to_image(v)), cmap="gray")
            else:
                _scatter_field(ax, s.coords, v)
            ax.set_title(t)
        return fig
