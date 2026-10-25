"""Results container: factors, reconstruction, NIfTI export, connectivity, plots."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._kernels import factors as _factors
from ._lazy import lazy_import
from .brainnet import animate_network, plot_network
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
    >>> img = np.random.default_rng(0).random((6, 6, 6, 8)) + 1
    >>> res = fit(img, K=2, standardize=False, init="power", max_iter=5)
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
        """Factor images of one subject at that subject's coordinates.

        Parameters
        ----------
        subject : int
            Index into ``subjects``.

        Returns
        -------
        ndarray, shape (V, K)
            Column ``k`` is the radial basis function of factor ``k``.

        Examples
        --------
        >>> import numpy as np
        >>> import htfa
        >>> img = np.random.default_rng(0).random((6, 6, 6, 8)) + 1
        >>> res = htfa.fit(img, K=2, standardize=False, init="power", max_iter=5,
        ...                random_state=0)
        >>> res.factors().shape
        (216, 2)
        """
        return _factors(
            self.subjects[subject].coords, self.centers[subject], self.widths[subject]
        )

    def reconstruct(self, subject: int = 0) -> Array:
        """Reconstruction ``factors(subject) @ weights[subject]``.

        The result is in the standardized (z-scored) space the model was fitted
        in, not in the units of the raw input.

        Parameters
        ----------
        subject : int
            Index into ``subjects``.

        Returns
        -------
        ndarray, shape (V, T)

        Examples
        --------
        >>> import numpy as np
        >>> import htfa
        >>> img = np.random.default_rng(0).random((6, 6, 6, 8)) + 1
        >>> res = htfa.fit(img, K=2, standardize=False, init="power", max_iter=5,
        ...                random_state=0)
        >>> res.reconstruct().shape
        (216, 8)
        """
        return self.factors(subject) @ self.weights[subject]

    def _values(self, subject: int, what: str) -> Array:
        if what == "reconstruction":
            return self.reconstruct(subject)
        if what == "factors":
            return self.factors(subject)
        raise ValueError(f"what must be 'reconstruction' or 'factors', got {what!r}")

    def to_nifti(self, subject: int = 0, what: str = "reconstruction") -> Any:
        """Reconstruction or factors as a NIfTI image (NaN outside the mask).

        Parameters
        ----------
        subject : int
            Index into ``subjects``.
        what : {'reconstruction', 'factors'}
            The ``(V, T)`` reconstruction, or the ``(V, K)`` factor images.

        Returns
        -------
        nibabel.Nifti1Image
            Image with the subject's affine; the last axis is ``T`` or ``K``.

        Raises
        ------
        ValueError
            If the subject was not read from NIfTI input (no affine), or ``what``
            is not one of the two options.

        Examples
        --------
        >>> import nibabel as nib
        >>> import numpy as np
        >>> import htfa
        >>> data = np.random.default_rng(0).random((6, 6, 6, 8)) + 1
        >>> img = nib.Nifti1Image(data, np.eye(4))
        >>> res = htfa.fit(img, K=2, standardize=False, init="power", max_iter=5,
        ...                random_state=0)
        >>> res.to_nifti(what="factors").shape
        (6, 6, 6, 2)
        """
        if self.subjects[subject].affine is None:
            raise ValueError("subject has no affine (not NIfTI input); use to_image")
        return self.subjects[subject].to_image(self._values(subject, what))

    def to_image(self, subject: int = 0, what: str = "reconstruction") -> Array:
        """Reconstruction or factors as an array in image space.

        Parameters
        ----------
        subject : int
            Index into ``subjects``.
        what : {'reconstruction', 'factors'}
            The ``(V, T)`` reconstruction, or the ``(V, K)`` factor images.

        Returns
        -------
        ndarray
            Shape ``image_shape + (T,)`` or ``image_shape + (K,)``; NaN outside
            the mask.

        Raises
        ------
        ValueError
            If the subject has no image space (data given as ``(V, T)`` plus
            ``coords``), or ``what`` is not one of the two options.

        Examples
        --------
        >>> import numpy as np
        >>> import htfa
        >>> img = np.random.default_rng(0).random((6, 6, 6, 8)) + 1
        >>> res = htfa.fit(img, K=2, standardize=False, init="power", max_iter=5,
        ...                random_state=0)
        >>> res.to_image().shape
        (6, 6, 6, 8)
        """
        img = self.subjects[subject].to_image(self._values(subject, what))
        return np.asarray(getattr(img, "dataobj", img), dtype=np.float64)

    def connectivity(self, subject: int = 0) -> Array:
        """Static factor connectivity of one subject.

        Parameters
        ----------
        subject : int
            Index into ``subjects``.

        Returns
        -------
        ndarray, shape (K, K)
            Pearson correlation between the subject's factor time courses
            (`htfa.static_connectivity` of ``weights[subject]``).

        Examples
        --------
        >>> import numpy as np
        >>> import htfa
        >>> img = np.random.default_rng(0).random((6, 6, 6, 8)) + 1
        >>> res = htfa.fit(img, K=2, standardize=False, init="power", max_iter=5,
        ...                random_state=0)
        >>> res.connectivity().shape
        (2, 2)
        """
        return static_connectivity(self.weights[subject])

    def dynamic_connectivity(self, subject: int = 0, window: int = 20) -> Array:
        """Sliding-window factor connectivity of one subject.

        Parameters
        ----------
        subject : int
            Index into ``subjects``.
        window : int
            Window length in samples, ``2 <= window <= T``.

        Returns
        -------
        ndarray, shape (T - window + 1, K, K)
            See `htfa.dynamic_connectivity`.

        Examples
        --------
        >>> import numpy as np
        >>> import htfa
        >>> img = np.random.default_rng(0).random((6, 6, 6, 8)) + 1
        >>> res = htfa.fit(img, K=2, standardize=False, init="power", max_iter=5,
        ...                random_state=0)
        >>> res.dynamic_connectivity(window=5).shape
        (4, 2, 2)
        """
        return dynamic_connectivity(self.weights[subject], window)

    def isfc(self, window: int | None = None) -> Array:
        """Inter-subject factor correlation across all subjects.

        Parameters
        ----------
        window : int, optional
            If given, sliding-window ISFC.

        Returns
        -------
        ndarray, shape (K, K) or (T - window + 1, K, K)
            See `htfa.isfc`. Needs at least two subjects with equal ``T``.

        Examples
        --------
        >>> import numpy as np
        >>> import htfa
        >>> rng = np.random.default_rng(0)
        >>> imgs = [rng.random((6, 6, 6, 8)) + 1 for _ in range(2)]
        >>> res = htfa.fit(imgs, K=2, standardize=False, init="power",
        ...                max_global_iter=2, max_local_iter=2, random_state=0)
        >>> res.isfc().shape
        (2, 2)
        """
        return isfc(self.weights, window)

    def _dim(self) -> int:
        D = int(self.centers[0].shape[1])
        if D not in (2, 3):
            raise ValueError(f"plotting supports 2D and 3D coordinates; got {D}D")
        return D

    def plot_nodes(self, subject: Any = "global", ax: Any = None, **kwargs: Any) -> Any:
        """Plot factor centers.

        Parameters
        ----------
        subject : {'global'} or int
            ``'global'`` draws the HTFA template in black and each subject's
            centers in a distinct color; for a TFA fit it draws subject 0. An
            integer draws that subject's centers.
        ax : matplotlib Axes, optional
            Axes to draw on (2D coordinates, or the nilearn display's axes in 3D).
        **kwargs
            Passed to ``nilearn.plotting.plot_connectome`` for 3D coordinates.

        Returns
        -------
        matplotlib.figure.Figure or nilearn display
            A figure for 2D coordinates; a nilearn glass-brain display for 3D.

        Raises
        ------
        ValueError
            If the coordinates are not 2D or 3D.

        Notes
        -----
        Needs the ``plot`` extra (matplotlib and, for 3D, nilearn).

        Examples
        --------
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import numpy as np
        >>> import htfa
        >>> X, R, _ = htfa.make_tfa_data((6, 6), 8, K=2, random_state=0)
        >>> res = htfa.fit(X, K=2, coords=R, standardize=False, init="power",
        ...                max_iter=5, random_state=0)
        >>> len(res.plot_nodes().axes)
        1
        """
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
                disp.add_markers(
                    self.global_centers, marker_color="black", marker_size=50
                )
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
                ax.scatter(c[:, 1], c[:, 0], color=[col], s=20, zorder=3)
            gc = self.global_centers
            assert gc is not None
            ax.scatter(gc[:, 1], gc[:, 0], c="k", s=50, zorder=4)
        else:
            ax.scatter(
                self.centers[subject][:, 1], self.centers[subject][:, 0], c="r", s=30
            )
        return fig

    def plot_connectivity(
        self, subject: int = 0, threshold: Any = "75%", ax: Any = None
    ) -> Any:
        """Plot factor connectivity as edges between factor centers.

        Parameters
        ----------
        subject : int
            Index into ``subjects``.
        threshold : str or float
            A percentage string such as ``'75%'`` keeps the edges whose absolute
            correlation is at or above that percentile; a number keeps edges
            with absolute correlation at least that large.
        ax : matplotlib Axes, optional
            Axes to draw on.

        Returns
        -------
        matplotlib.figure.Figure or nilearn display
            A figure for 2D coordinates; a nilearn glass-brain display for 3D.

        Raises
        ------
        ValueError
            If the coordinates are not 2D or 3D.

        Notes
        -----
        Needs the ``plot`` extra.

        Examples
        --------
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import numpy as np
        >>> import htfa
        >>> X, R, _ = htfa.make_tfa_data((6, 6), 8, K=3, random_state=0)
        >>> res = htfa.fit(X, K=3, coords=R, standardize=False, init="power",
        ...                max_iter=5, random_state=0)
        >>> len(res.plot_connectivity(threshold=0.0).axes)
        1
        """
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
        """Original (standardized) data next to the reconstruction of one sample.

        Parameters
        ----------
        subject : int
            Index into ``subjects``.
        sample : int
            Index of the time point or image to show.

        Returns
        -------
        matplotlib.figure.Figure
            Two panels, "original" and "reconstruction". 3D data are drawn as
            glass-brain images and need NIfTI input.

        Raises
        ------
        ValueError
            If the coordinates are not 2D or 3D, or 3D data have no affine.

        Notes
        -----
        Needs the ``plot`` extra.

        Examples
        --------
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import numpy as np
        >>> import htfa
        >>> X, R, _ = htfa.make_tfa_data((6, 6), 8, K=2, random_state=0)
        >>> res = htfa.fit(X, K=2, coords=R, standardize=False, init="power",
        ...                max_iter=5, random_state=0)
        >>> len(res.plot_reconstruction().axes)
        2
        """
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

    def _network_subject(self, subject: Any) -> Any:
        return 0 if subject == "global" and self.global_centers is None else subject

    def plot_network(
        self, subject: Any = "global", connectivity: Any = None, **kwargs: Any
    ) -> Any:
        """BrainNet Viewer-style plot of factor centers and connectivity.

        ``connectivity`` defaults to `isfc` for ``"global"`` (the HTFA
        template; subject 0 for a TFA fit) and to `connectivity` otherwise.
        Node sizes are the factor widths. Other keyword arguments go to
        `htfa.brainnet.plot_network`. Needs the ``brain3d`` extra.
        """
        subject = self._network_subject(subject)
        if connectivity is None:
            connectivity = (
                self.isfc() if subject == "global" else self.connectivity(subject)
            )
        return plot_network(self, connectivity, subject=subject, **kwargs)

    def animate_network(
        self,
        out: Any,
        subject: Any = "global",
        window: int = 20,
        connectivity: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Animate sliding-window connectivity BrainNet Viewer-style (mp4/GIF).

        ``connectivity`` defaults to ``isfc(window)`` for ``"global"`` and to
        ``dynamic_connectivity(subject, window)`` otherwise. Other keyword
        arguments go to `htfa.brainnet.animate_network`. Needs the ``brain3d``
        extra.
        """
        subject = self._network_subject(subject)
        if connectivity is None:
            connectivity = (
                self.isfc(window)
                if subject == "global"
                else self.dynamic_connectivity(subject, window)
            )
        return animate_network(self, connectivity, out, subject=subject, **kwargs)
