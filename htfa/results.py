"""Results container: factors, reconstruction, NIfTI export, connectivity, plots."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._kernels import Kernel
from ._kernels import factors as _factors
from ._lazy import lazy_import
from ._spatial import ComponentNodes, check_level, component_mode
from .brainnet import animate_network, effective_nodes, plot_network
from .connectivity import dynamic_connectivity, isfc, static_connectivity
from .groups import GroupTest, group_permutation_test
from .htfa import HTFA
from .io import Subject
from .tfa import TFA, component_factors, standardize_data

Array = np.ndarray

__all__ = ["HTFAResults"]


def _scatter_field(ax: Any, coords: Array, values: Array, **kwargs: Any) -> Any:
    """Draw ``values`` at 2D ``coords`` (row, column) as squares.

    Keyword arguments (``cmap``, default ``"gray"``, ``vmin``, ``vmax``) go to
    ``ax.scatter``; the scatter collection is returned.
    """
    kwargs.setdefault("cmap", "gray")
    sc = ax.scatter(coords[:, 1], coords[:, 0], c=values, marker="s", s=12, **kwargs)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    return sc


def _space_connectome_ticks(display: Any) -> None:
    """Keep color limits and zero legible when threshold ticks crowd them.

    Nilearn exposes no tick-location argument for ``plot_connectome``. Access
    its Matplotlib colorbar in this one guarded helper, then use the colorbar's
    public tick API. Displays without that optional attribute are unaffected.
    Only labels change: the threshold, color normalization and edges stay fixed.
    """
    cbar = getattr(display, "_cbar", None)
    if cbar is None or not all(
        hasattr(cbar, name) for name in ("get_ticks", "set_ticks", "ax", "norm")
    ):
        return
    lower, upper = cbar.norm.vmin, cbar.norm.vmax
    if lower is None or upper is None or not np.isfinite([lower, upper]).all():
        return
    span = upper - lower
    if span <= 0:
        return
    selected = [float(lower), float(upper)]
    if lower < 0 < upper:
        selected.append(0.0)
    # A fractional floor handles ordinary figures; font size and axes height
    # make the margin more conservative for a compact user-supplied axes.
    labels = cbar.ax.get_yticklabels()
    font_points = max((label.get_fontsize() for label in labels), default=10.0)
    height = cbar.ax.get_window_extent().height
    label_pixels = font_points * cbar.ax.figure.dpi / 72
    fraction = max(0.1, 1.5 * label_pixels / max(height, 1.0))
    for tick in cbar.get_ticks():
        if (
            lower < tick < upper
            and min(abs(tick - x) for x in selected) >= fraction * span
        ):
            selected.append(float(tick))
    cbar.set_ticks(sorted(selected))


def _symmetric_limit(values: list[Array]) -> float:
    """Shared color limit: the 99th percentile of ``|values|`` over all panels.

    The percentile keeps a few extreme (noise) voxels from washing out the
    rest; if it is zero the largest ``|value|`` is used, and 1 for all-zero
    data.
    """
    a = np.abs(np.concatenate([np.ravel(v) for v in values]))
    a = a[np.isfinite(a)]
    if a.size == 0 or a.max() == 0:
        return 1.0
    vmax = float(np.percentile(a, 99))
    return vmax if vmax > 0 else float(a.max())


# Over the MNI152 template, |values| below this fraction of the shared color
# limit are transparent (in every panel) so that the anatomy shows.
_BG_THRESHOLD = 0.1


def _in_mni_space(coords: Array) -> bool:
    """Whether 3D world ``coords`` (mm) span the MNI152 brain.

    True when the bounding box of ``coords`` holds at least half of the voxels
    of nilearn's MNI152 brain mask (2 mm). A whole-brain image in MNI space
    passes; a small or shifted grid (such as a synthetic cube at the origin)
    does not, so drawing it over the MNI152 template would misplace it.
    """
    datasets = lazy_import("nilearn.datasets", "plot", "3D plotting")
    mask = datasets.load_mni152_brain_mask(resolution=2)
    ijk = np.argwhere(np.asarray(mask.dataobj) > 0)
    xyz = ijk @ mask.affine[:3, :3].T + mask.affine[:3, 3]
    inside = np.all((xyz >= coords.min(axis=0)) & (xyz <= coords.max(axis=0)), axis=1)
    return bool(inside.mean() >= 0.5)


@dataclass(frozen=True)
class _Group:
    """A group template as a plotting target (index into ``group_keys_``)."""

    index: int


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
    group_centers, group_widths : dict
        The group templates of an HTFA fit with ``groups`` (empty otherwise),
        ``(K, D)`` and ``(K,)`` per group, in global template order. Keys are
        the group labels for top-level groups and label paths (tuples) for
        deeper ones, as in ``HTFA.group_keys_``; the plotting methods take
        them as ``group=``.

    kernel : htfa._kernels.Kernel
        The fit's ``kernel`` and ``n_components`` (``'rbf'`` and 1 for a model
        without them).
    component_centers, component_widths, mixing_weights : list or None
        Per-subject components (K, M, D), (K, M) and (K, M) of a fit with
        another kernel or several components; ``None`` for single RBFs, whose
        ``centers`` and ``widths`` define the factors.

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
        keys = (
            getattr(model, "group_keys_", [])
            if getattr(model, "group_tree_", None) is not None
            else []
        )
        self.group_centers = {k: model.group_centers_[i] for i, k in enumerate(keys)}
        self.group_widths = {k: model.group_widths_[i] for i, k in enumerate(keys)}

        self.kernel = Kernel(
            getattr(model, "kernel", "rbf"), getattr(model, "n_components", 1)
        )
        self.component_centers: list[Array] | None = None
        self.component_widths: list[Array] | None = None
        self.mixing_weights: list[Array] | None = None
        if not self.kernel.plain_rbf:
            many = isinstance(model, HTFA)
            for name in ("component_centers", "component_widths", "mixing_weights"):
                value = getattr(model, name + "_")
                setattr(self, name, list(value) if many else [value])
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
        coords = self.subjects[subject].coords
        if self.component_centers is None:
            return _factors(coords, self.centers[subject], self.widths[subject])
        assert self.component_widths is not None and self.mixing_weights is not None
        return component_factors(
            coords,
            self.component_centers[subject],
            self.component_widths[subject],
            self.mixing_weights[subject],
            self.kernel,
        )

    def _shape(self, subject: Any) -> tuple[str, Array, Array, Array] | None:
        """Kernel name and components of ``subject`` (``'global'``: the
        template) for the effective nodes; None for single RBFs."""
        if self.component_centers is None:
            return None
        assert self.component_widths is not None and self.mixing_weights is not None
        if isinstance(subject, _Group):
            i = subject.index
            return (
                self.kernel.name,
                self.model.group_component_centers_[i],
                self.model.group_component_widths_[i],
                self.model.group_mixing_weights_[i],
            )
        if subject == "global" and isinstance(self.model, HTFA):
            return (
                self.kernel.name,
                self.model.global_component_centers_,
                self.model.global_component_widths_,
                self.model.global_mixing_weights_,
            )
        s = 0 if subject == "global" else subject
        return (
            self.kernel.name,
            self.component_centers[s],
            self.component_widths[s],
            self.mixing_weights[s],
        )

    def reconstruct(self, subject: int = 0) -> Array:
        """Reconstruction ``factors(subject) @ weights[subject]``.

        The result uses the space fitted by the model: z-scored values when
        standardization is enabled, or the input scale otherwise.

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

        Samples must refer to matching events or times across subjects, such as
        the same moments of a shared film; equal series lengths alone are not
        sufficient. The fitted HTFA template supplies the common factor order.

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
        >>> import warnings
        >>> with warnings.catch_warnings():
        ...     warnings.simplefilter("ignore")  # 2 global iterations of noise
        ...     res = htfa.fit(imgs, K=2, standardize=False, init="power",
        ...                    max_global_iter=2, max_local_iter=2, random_state=0)
        >>> res.isfc().shape
        (2, 2)
        """
        return isfc(self.weights, window)

    def _group_index(self, group: Any) -> int:
        """Index of ``group`` (a key of ``group_centers``) in ``group_keys_``."""
        if isinstance(group, np.generic):
            group = group.item()
        keys = list(self.group_centers)
        if not keys:
            raise ValueError("the model was fitted without groups")
        if group not in self.group_centers:
            raise ValueError(f"unknown group {group!r}; groups: {keys}")
        return keys.index(group)

    def group_subjects(self, group: Any) -> list[int]:
        """Indices of the subjects in ``group`` (and its subgroups).

        Parameters
        ----------
        group : label or tuple of labels
            A key of ``group_centers``.

        Returns
        -------
        list of int

        Examples
        --------
        >>> import warnings
        >>> import htfa
        >>> Xs, Rs, groups, _ = htfa.make_group_data(
        ...     (2, 2), grid_shape=(8, 8), n_samples=10, K=2, random_state=0)
        >>> with warnings.catch_warnings():
        ...     warnings.simplefilter("ignore")  # 2 global iterations only
        ...     res = htfa.fit(Xs, coords=Rs, K=2, groups=["a", "a", "b", "b"],
        ...                    template_update="fixed", max_global_iter=2,
        ...                    max_local_iter=3, init="power", standardize=False,
        ...                    random_state=0)
        >>> res.group_subjects("b")
        [2, 3]
        """
        index = self._group_index(group)
        tree = self.model.group_tree_
        assert tree is not None
        return [int(s) for s in tree.members(index + 1)]

    def group_permutation_test(
        self,
        n_permutations: int = 100,
        level: int = 0,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> GroupTest:
        """`htfa.group_permutation_test` of the model on its subjects' data.

        Parameters
        ----------
        n_permutations, level, random_state, verbose
            As in `htfa.group_permutation_test`.

        Returns
        -------
        htfa.groups.GroupTest
        """
        return group_permutation_test(
            self.model,
            [s.data for s in self.subjects],
            [s.coords for s in self.subjects],
            n_permutations=n_permutations,
            level=level,
            random_state=random_state,
            verbose=verbose,
        )

    def _target(self, subject: Any, group: Any) -> Any:
        """``subject``, or the group template ``group`` when it is given."""
        if group is None:
            return subject
        if not (isinstance(subject, str) and subject == "global"):
            raise ValueError("pass either subject or group, not both")
        return _Group(self._group_index(group))

    def _members(self, target: Any) -> list[int]:
        """Subjects drawn with a template: all, or the group's."""
        if isinstance(target, _Group):
            return self.group_subjects(list(self.group_centers)[target.index])
        return list(range(len(self.centers)))

    def _dim(self) -> int:
        D = int(self.centers[0].shape[1])
        if D not in (2, 3):
            raise ValueError(f"plotting supports 2D and 3D coordinates; got {D}D")
        return D

    def _component_nodes(self, subject: Any, mode: str) -> ComponentNodes | None:
        shape = self._shape(subject)
        if not component_mode(mode, shape is not None):
            return None
        if shape is None:
            c, w = self._fitted_nodes(subject)
            shape = ("rbf", c[:, None], w[:, None], np.ones((len(c), 1)))
        return ComponentNodes.from_shape(shape)

    def _lobe_mask(self, subject: Any) -> Any:
        """Preserve native voxel affines when subjects share an image grid."""
        members = (
            self._members(subject)
            if isinstance(subject, _Group) or subject == "global"
            else [subject]
        )
        subjects = [self.subjects[i] for i in members]
        first = subjects[0]
        if first.affine is not None and all(
            s.affine is not None
            and s.mask is not None
            and s.image_shape == first.image_shape
            and np.array_equal(s.affine, first.affine)
            for s in subjects
        ):
            return (
                np.any(np.asarray([s.mask for s in subjects]), axis=0),
                first.affine,
            )
        return self._display_mask(subject)

    def _plot_components(
        self,
        subject: Any,
        components: ComponentNodes,
        connectivity: Any,
        threshold: Any,
        ax: Any,
        snap: bool,
        mask: Any,
        lobe_level: float,
        kwargs: dict,
    ) -> Any:
        """Shared glass-brain/2D component plots; edges repeat factor values."""
        level = check_level(lobe_level)
        mask = self._lobe_mask(subject) if mask is None else mask
        centers = components.centers
        if snap:
            centers, _, _ = effective_nodes(
                centers, components.widths, mask, components.kernel
            )
        conn = None
        if connectivity is not None:
            original = np.asarray(connectivity)
            cut = threshold
            if isinstance(cut, str) and cut.endswith("%"):
                edges = np.abs(original[np.triu_indices(len(original), 1)])
                cut = np.percentile(edges, float(cut[:-1])) if len(edges) else 0.0
            conn = components.connectivity(original)
        if self._dim() == 3:
            plotting = lazy_import("nilearn.plotting", "plot", "3D component plotting")
            disp = plotting.plot_glass_brain(
                None, axes=ax, plot_abs=False, colorbar=False, **kwargs
            )
            try:
                for i, img in components.images(mask):
                    peak = float(components.mixing[i])
                    if float(np.max(np.abs(np.asarray(img.dataobj)))) >= level * peak:
                        disp.add_overlay(
                            img,
                            threshold=level * peak,
                            cmap="RdBu_r",
                            vmin=-peak,
                            vmax=peak,
                            colorbar=False,
                            **{
                                "transparency"
                                if "transparency"
                                in inspect.signature(disp.add_overlay).parameters
                                else "alpha": float(0.2 + 0.6 * components.mixing[i])
                            },
                        )
                if conn is not None:
                    disp.add_graph(
                        conn,
                        centers,
                        node_size=0,
                        edge_threshold=cut,
                        edge_kwargs={"linewidth": 1.0, "alpha": 0.7},
                    )
            except BaseException:
                disp.close()
                raise
            return disp
        plt = lazy_import("matplotlib.pyplot", "plot", "component plotting")
        fig, ax = (ax.figure, ax) if ax is not None else plt.subplots()
        from .brainnet import _mask_coords

        coords = _mask_coords(mask, 2)[0]
        for i in range(len(centers)):
            v = components.values(coords, i)
            peak = float(components.mixing[i])
            if np.any(np.abs(v) >= level * peak):
                keep = np.abs(v) >= level * peak
                ax.scatter(
                    coords[keep, 1],
                    coords[keep, 0],
                    c=v[keep],
                    marker="s",
                    s=12,
                    cmap="RdBu_r",
                    vmin=-peak,
                    vmax=peak,
                    alpha=0.2 + 0.6 * components.mixing[i],
                )
        if conn is not None:
            cmap = plt.get_cmap("coolwarm")
            for row, col in zip(*np.triu_indices_from(conn, 1), strict=True):
                i, j = int(row), int(col)
                if conn[i, j] != 0 and abs(conn[i, j]) >= float(cut):
                    ax.plot(
                        centers[[i, j], 1],
                        centers[[i, j], 0],
                        color=cmap((conn[i, j] + 1) / 2),
                        lw=1 + 3 * abs(conn[i, j]),
                    )
        ax.set_aspect("equal")
        if not ax.yaxis_inverted():
            ax.invert_yaxis()
        return fig

    def plot_nodes(
        self,
        subject: Any = "global",
        ax: Any = None,
        snap: bool = True,
        mask: Any = None,
        group: Any = None,
        node_mode: str = "auto",
        lobe_level: float = 0.05,
        **kwargs: Any,
    ) -> Any:
        """Plot factor centers or spatial component lobes.

        Parameters
        ----------
        subject : {'global'} or int
            ``'global'`` draws the HTFA template in black and each subject's
            centers in a distinct color; for a TFA fit it draws subject 0. An
            integer draws that subject's centers.
        group : label or tuple, optional
            A key of ``group_centers``: draws that group template in black and
            its subjects' centers in distinct colors (``subject`` must then be
            ``'global'``).
        node_mode : {'auto', 'components', 'summary'}
            ``'auto'`` draws non-RBF kernels and mixtures as signed spatial
            component lobes; single RBFs retain the centroid view.
            ``'components'`` always draws lobes and ``'summary'`` draws one
            center per factor. A template's component view draws that
            template's lobes, without overlaying the subjects' profiles.
            Connectivity edges repeat the factor value between every pair
            of components in different factors, without within-factor edges.
        lobe_level : float
            Hide spatial loadings below this fraction of each component's
            unmasked peak (default 0.05). Blue is negative, red positive;
            opacity is 0.2 + 0.6 times the mixing weight. Glass-brain views
            use signed maximum-absolute projections: overlapping lobes along
            a viewing ray can obscure one another. ``snap`` moves edge
            anchors, never the fitted fields.
        ax : matplotlib Axes, optional
            Axes to draw on (2D coordinates, or the nilearn display's axes in 3D).
        snap : bool
            Draw each node at its in-brain RBF-weighted centroid (see
            `display_nodes`); ``False`` draws the fitted centers. Default
            ``True``.
        mask : optional
            In-brain voxels for ``snap`` (see `display_nodes`). Default: each
            subject's own voxels, and all (the group's) subjects' voxels for the
            template.
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
        subject = self._target(subject, group)
        components = self._component_nodes(subject, node_mode)
        if components is not None:
            return self._plot_components(
                subject, components, None, 0, ax, snap, mask, lobe_level, kwargs
            )
        glob = isinstance(subject, _Group) or (
            subject == "global" and self.global_centers is not None
        )
        if not glob and subject == "global":
            subject = 0
        plt = lazy_import("matplotlib.pyplot", "plot", "plotting")
        K = self.centers[0].shape[0]
        members = self._members(subject)
        if glob:
            gcenters = self._plotted_centers(subject, snap, mask)
            subj_centers = [self._plotted_centers(s, snap, mask) for s in members]
        else:
            centers = self._plotted_centers(subject, snap, mask)
        if D == 3:
            plotting = lazy_import("nilearn.plotting", "plot", "3D plotting")
            none = np.zeros((K, K))
            if glob:
                disp = plotting.plot_connectome(
                    none,
                    gcenters,
                    node_color="black",
                    node_size=50,
                    axes=ax,
                    colorbar=False,
                    **kwargs,
                )
                colors = plt.get_cmap("Spectral")(np.linspace(0, 1, len(members)))
                for c, col in zip(subj_centers, colors, strict=True):
                    disp.add_markers(c, marker_color=[col], marker_size=20)
                disp.add_markers(gcenters, marker_color="black", marker_size=50)
            else:
                disp = plotting.plot_connectome(
                    none, centers, axes=ax, colorbar=False, **kwargs
                )
            return disp
        fig, ax = (ax.figure, ax) if ax is not None else plt.subplots()
        s = self.subjects[members[0] if glob else subject]
        _scatter_field(ax, s.coords, s.data.mean(axis=1))
        if glob:
            colors = plt.get_cmap("Spectral")(np.linspace(0, 1, len(members)))
            for c, col in zip(subj_centers, colors, strict=True):
                ax.scatter(c[:, 1], c[:, 0], color=[col], s=20, zorder=3)
            ax.scatter(gcenters[:, 1], gcenters[:, 0], c="k", s=50, zorder=4)
        else:
            ax.scatter(centers[:, 1], centers[:, 0], c="r", s=30)
        return fig

    def plot_connectivity(
        self,
        subject: int = 0,
        threshold: Any = "75%",
        ax: Any = None,
        snap: bool = True,
        mask: Any = None,
        node_mode: str = "auto",
        lobe_level: float = 0.05,
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
        node_mode : {'auto', 'components', 'summary'}
            ``'auto'`` draws non-RBF kernels and mixtures as signed spatial
            component lobes; single RBFs retain the centroid view.
            ``'components'`` always draws lobes and ``'summary'`` draws one
            center per factor. A template's component view draws that
            template's lobes, without overlaying the subjects' profiles.
            Connectivity edges repeat the factor value between every pair
            of components in different factors, without within-factor edges.
        lobe_level : float
            Hide spatial loadings below this fraction of each component's
            unmasked peak (default 0.05). Blue is negative, red positive;
            opacity is 0.2 + 0.6 times the mixing weight. Glass-brain views
            use signed maximum-absolute projections: overlapping lobes along
            a viewing ray can obscure one another. ``snap`` moves edge
            anchors, never the fitted fields.
        ax : matplotlib Axes, optional
            Axes to draw on.
        snap : bool
            Draw each node at its in-brain RBF-weighted centroid (see
            `display_nodes`); ``False`` draws the fitted centers. Default
            ``True``.
        mask : optional
            In-brain voxels for ``snap``; default the subject's own voxels.

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
        components = self._component_nodes(subject, node_mode)
        if components is not None:
            return self._plot_components(
                subject, components, conn, threshold, ax, snap, mask, lobe_level, {}
            )
        centers = self._plotted_centers(subject, snap, mask)
        if D == 3:
            plotting = lazy_import("nilearn.plotting", "plot", "3D plotting")
            if ax is None:
                disp = plotting.plot_connectome(conn, centers, edge_threshold=threshold)
                _space_connectome_ticks(disp)
                return disp
            # In a given (often small) axes, scale the L/R labels with the
            # axes width (nilearn uses 12 pt) so they clear the outlines.
            disp = plotting.plot_connectome(
                conn, centers, edge_threshold=threshold, axes=ax, annotate=False
            )
            width_pt = ax.get_position().width * ax.figure.get_figwidth() * 72
            disp.annotate(size=float(np.clip(width_pt / 30, 6, 12)))
            _space_connectome_ticks(disp)
            return disp
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

    def plot_reconstruction(
        self,
        subject: int = 0,
        sample: int = 0,
        truth: Any = None,
        view: str = "auto",
        cmap: str = "RdBu_r",
    ) -> Any:
        """Input data in the fitting scale next to one reconstructed sample.

        All panels share one symmetric color range, ``[-vmax, vmax]`` with
        ``vmax`` the 99th percentile of the absolute values over all panels
        (so a few extreme noise voxels do not wash out the rest), a diverging
        colormap with signs kept, and one colorbar: zero is white in every
        panel and equal colors mean equal values. Nothing is thresholded,
        except over the MNI152 template (see ``view``), where values with
        ``|value|`` below 10% of ``vmax`` are left transparent in every panel
        so that the anatomy shows.

        Parameters
        ----------
        subject : int
            Index into ``subjects``.
        sample : int
            Index of the time point or image to show.
        truth : array, optional
            A reference on the subject's voxels, such as the noiseless signal
            of simulated data, shown as a third panel ("truth") on the same
            scale: ``(V,)`` for this sample or ``(V, T)``, from which column
            ``sample`` is taken. It should be on the scale of the data shown
            (standardized if the fit standardized the data).
        view : {'auto', 'glass', 'slices'}
            For 3D data. ``'slices'`` and ``'auto'`` (default, the same)
            draw orthogonal slices (nilearn ``plot_stat_map``), all panels
            cut at the voxel with the largest absolute reconstruction. When
            the image is in MNI space, meaning the bounding box of the
            subject's voxels holds at least half of nilearn's MNI152 brain
            mask, the slices are drawn over the MNI152 template; otherwise
            without a background. ``'glass'`` draws glass-brain projections
            (``plot_glass_brain``); a projection of one noisy sample takes
            the largest value along each ray and so looks saturated, which is
            why it is not the default. Ignored for 2D data.
        cmap : str
            Diverging matplotlib colormap. Default ``"RdBu_r"``.

        Returns
        -------
        matplotlib.figure.Figure
            One panel each for "original", "reconstruction" and (if given)
            "truth" and one shared colorbar (the figure's last axes). 3D data
            need NIfTI input and are drawn one panel per row.

        Raises
        ------
        ValueError
            If the coordinates are not 2D or 3D, 3D data have no affine,
            ``view`` is unknown, or ``truth`` does not have one value per voxel.

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
        >>> fig = res.plot_reconstruction()
        >>> [ax.get_title() for ax in fig.axes[:2]]
        ['original', 'reconstruction']
        """
        if view not in ("auto", "glass", "slices"):
            raise ValueError(f"view must be 'auto', 'glass' or 'slices'; got {view!r}")
        D = self._dim()
        s = self.subjects[subject]
        if D == 3 and s.affine is None:
            raise ValueError("3D plot_reconstruction needs NIfTI input (affine)")
        orig = standardize_data(s.data, self.model.standardize)[:, sample]
        recon = self.reconstruct(subject)[:, sample]
        panels = {"original": orig, "reconstruction": recon}
        if truth is not None:
            truth = np.asarray(truth, dtype=np.float64)
            if truth.shape[0] != len(orig) or truth.ndim not in (1, 2):
                raise ValueError(
                    f"truth must be (V,) or (V, T) with V = {len(orig)}; "
                    f"got {truth.shape}"
                )
            panels["truth"] = truth if truth.ndim == 1 else truth[:, sample]
        vmax = _symmetric_limit(list(panels.values()))
        plt = lazy_import("matplotlib.pyplot", "plot", "plotting")
        n = len(panels)
        if D == 3:
            plotting = lazy_import("nilearn.plotting", "plot", "3D plotting")
            glass = view == "glass"
            brain = not glass and _in_mni_space(s.coords)
            cut = s.coords[np.argmax(np.abs(recon))]
            # one row per panel, each as wide as the figure, and one colorbar
            fig = plt.figure(figsize=(9, 2.9 * n))
            for i, (title, v) in enumerate(panels.items()):
                ax = fig.add_axes((0.0, 1 - (i + 1) / n, 0.9, 0.86 / n))
                img = s.to_image(v)
                img = type(img)(np.nan_to_num(np.asarray(img.dataobj)), img.affine)
                common = dict(
                    figure=fig, axes=ax, colorbar=False, cmap=cmap, vmin=-vmax,
                    vmax=vmax, symmetric_cbar=True, black_bg=False,
                )  # fmt: skip
                if glass:
                    plotting.plot_glass_brain(
                        img, plot_abs=False, threshold=None, **common
                    )
                elif brain:
                    plotting.plot_stat_map(
                        img, cut_coords=cut, draw_cross=False,
                        threshold=_BG_THRESHOLD * vmax, **common,
                    )  # fmt: skip
                else:
                    plotting.plot_stat_map(
                        img, bg_img=None, cut_coords=cut, draw_cross=False,
                        threshold=None, **common,
                    )  # fmt: skip
                ax.set_title(title, fontsize=13, loc="left")
            cax = fig.add_axes((0.92, 0.1, 0.02, 0.8))
            mpl_colors = lazy_import("matplotlib.colors", "plot", "plotting")
            sm = plt.cm.ScalarMappable(
                norm=mpl_colors.Normalize(-vmax, vmax), cmap=cmap
            )
            fig.colorbar(sm, cax=cax)
            return fig
        fig, axes = plt.subplots(1, n, figsize=(4.5 * n + 1, 4))
        for ax, (title, v) in zip(axes, panels.items(), strict=True):
            if s.image_shape is not None:
                im = ax.imshow(
                    np.asarray(s.to_image(v)), cmap=cmap, vmin=-vmax, vmax=vmax
                )
            else:
                im = _scatter_field(ax, s.coords, v, cmap=cmap, vmin=-vmax, vmax=vmax)
            ax.set_title(title)
        fig.colorbar(im, ax=list(axes), shrink=0.8)
        return fig

    def _network_inputs(
        self, subject: Any = "global", connectivity: Any = None, window: Any = None
    ) -> tuple[Array, Array, Array]:
        """Centers, node sizes and connectivity for the network wrappers.

        ``"global"`` means the HTFA template (``global_centers``,
        ``global_widths``) with `isfc`; a TFA fit has no template, so it falls
        back to subject 0. A group template uses `isfc` of the group's
        subjects. A subject uses its own centers and widths with
        `connectivity`. With ``window`` (animations) the defaults are
        ``isfc(window)`` and ``dynamic_connectivity(subject, window)``.
        A given ``connectivity`` is returned unchanged.
        """
        if subject == "global" and self.global_centers is None:
            subject = 0
        centers, sizes = self._fitted_nodes(subject)
        if isinstance(subject, _Group):
            if connectivity is None:
                members = self._members(subject)
                connectivity = isfc([self.weights[s] for s in members], window)
        elif subject == "global":
            if connectivity is None:
                connectivity = self.isfc(window)
        else:
            if connectivity is None:
                connectivity = (
                    self.connectivity(subject)
                    if window is None
                    else self.dynamic_connectivity(subject, window)
                )
        return centers, sizes, connectivity

    def plot_network(
        self,
        subject: Any = "global",
        connectivity: Any = None,
        group: Any = None,
        **kwargs: Any,
    ) -> Any:
        """BrainNet Viewer-style plot of factor centers and connectivity.

        ``connectivity`` defaults to `isfc` for ``"global"`` (the HTFA
        template; subject 0 for a TFA fit) and to `connectivity` otherwise.
        ``group`` (a key of ``group_centers``; ``subject`` must then be
        ``'global'``) draws that group template, with the `isfc` of the
        group's subjects by default and its subjects' voxels as the mask.
        Non-RBF kernels and mixtures default to spatial component lobes,
        repeating each factor edge from every component with the same value.
        ``node_mode='summary'`` restores one sphere per factor; see
        `htfa.brainnet.plot_network` for ``lobe_level`` and sign colors.
        Node sizes in summary mode are the factor widths unless ``sizes`` is given.
        Nodes are
        drawn as their in-brain parts by default (``snap=True``; see
        `display_nodes`, which returns the drawn centers and widths): the
        mask defaults to the subject's voxels, or all subjects' voxels for
        the template. ``snap=False`` draws the fitted centers and widths.
        Other keyword arguments go to `htfa.brainnet.plot_network`. Needs the
        ``brain3d`` extra.
        """
        subject = self._target(subject, group)
        if subject == "global" and self.global_centers is None:
            subject = 0
        centers, widths, conn = self._network_inputs(subject, connectivity)
        self._network_defaults(subject, widths, kwargs)
        return plot_network(self, conn, subject=subject, **kwargs)

    def animate_network(
        self,
        out: Any,
        subject: Any = "global",
        window: int = 20,
        connectivity: Any = None,
        group: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Animate sliding-window connectivity BrainNet Viewer-style (mp4/GIF).

        ``connectivity`` defaults to ``isfc(window)`` for ``"global"`` and to
        ``dynamic_connectivity(subject, window)`` otherwise; ``group`` as in
        `plot_network`. Nodes are
        snapped by default as in `plot_network`. Other keyword arguments go to
        `htfa.brainnet.animate_network`, among them ``fps`` (frame rate),
        ``duration`` (seconds; frames interpolated between windows, see
        ``interpolation``) and ``spin`` (turns about x, y, z). Needs the
        ``brain3d`` extra.

        Examples
        --------
        >>> res.animate_network("isfc.mp4", fps=15, duration=8,
        ...                     spin=(0, 0, 1))  # doctest: +SKIP
        """
        subject = self._target(subject, group)
        if subject == "global" and self.global_centers is None:
            subject = 0
        centers, widths, conn = self._network_inputs(subject, connectivity, window)
        self._network_defaults(subject, widths, kwargs)
        return animate_network(self, conn, out, subject=subject, **kwargs)

    def _network_defaults(
        self, subject: Any, widths: Array, kwargs: dict[str, Any]
    ) -> None:
        """Default widths and, when snapping, mask for the network wrappers."""
        if kwargs.get("snap", True) and kwargs.get("mask") is None:
            kwargs["mask"] = (
                self._lobe_mask(subject)
                if self._shape(subject) is not None
                else self._display_mask(subject)
            )

    def _display_mask(self, subject: Any = "global") -> Array:
        """Voxels nodes snap into: the subject's, or all (the group's) subjects'
        (template)."""
        if isinstance(subject, _Group):
            parts = [self.subjects[s].coords for s in self._members(subject)]
            return np.unique(np.vstack(parts), axis=0)
        if subject == "global" and self.global_centers is None:
            subject = 0
        if subject == "global":
            return np.unique(np.vstack([s.coords for s in self.subjects]), axis=0)
        coords: Array = self.subjects[subject].coords
        return coords

    def _plotted_centers(self, subject: Any, snap: bool, mask: Any) -> Array:
        """Centers a node plot draws: display_nodes, or the fit's with snap off."""
        if snap:
            return self.display_nodes(subject, mask)[0]
        return self._fitted_nodes(subject)[0]

    def _fitted_nodes(self, subject: Any) -> tuple[Array, Array]:
        if isinstance(subject, _Group):
            key = list(self.group_centers)[subject.index]
            return self.group_centers[key], self.group_widths[key]
        if subject == "global" and self.global_centers is not None:
            assert self.global_widths is not None
            return self.global_centers, self.global_widths
        s = 0 if subject == "global" else subject
        return self.centers[s], self.widths[s]

    def display_nodes(
        self, subject: Any = "global", mask: Any = None, group: Any = None
    ) -> tuple[Array, Array, dict[str, Array]]:
        """Node centers and widths as the built-in plots draw them.

        Each node is drawn as the part of its RBF inside the brain: the
        RBF-weighted centroid of the in-brain voxels and the width
        ``lambda_eff = 2 * sigma_eff**2`` from the weighted spread (see
        `htfa.brainnet.effective_nodes`). The fit is not changed. This
        returns the node geometry of a plot with the default ``snap=True``.
        Fits with
        another ``kernel`` or several components use their own profiles
        (see `htfa.brainnet.effective_nodes`).

        Parameters
        ----------
        subject : {'global'} or int
            ``'global'`` is the HTFA template (subject 0 for a TFA fit).
        mask : optional
            In-brain voxels (``(V, D)`` coordinates, NIfTI image or
            ``(volume, affine)``). Default: the subject's voxel coordinates,
            or the union of all subjects' coordinates for the template.
        group : label or tuple, optional
            A key of ``group_centers``: that group template, with the union of
            its subjects' coordinates as the default mask (``subject`` must then
            be ``'global'``).

        Returns
        -------
        centers : ndarray, shape (K, D)
        widths : ndarray, shape (K,)
            ``lambda_eff``.
        info : dict of ndarray, shape (K,)
            ``distance`` moved, ``snapped`` and ``ambiguous`` flags,
            ``mass_fraction``, ``sigma`` (``sigma_eff``) and ``width_ratio``
            (``lambda_eff / lambda``); see `htfa.brainnet.effective_nodes`.

        Examples
        --------
        >>> import numpy as np
        >>> import htfa
        >>> X, R, _ = htfa.make_tfa_data((6, 6), 8, K=2, random_state=0)
        >>> res = htfa.fit(X, K=2, coords=R, standardize=False, init="power",
        ...                max_iter=5, random_state=0)
        >>> centers, widths, info = res.display_nodes()
        >>> centers.shape, sorted(info)[:2]
        ((2, 2), ['ambiguous', 'distance'])
        """
        subject = self._target(subject, group)
        centers, widths = self._fitted_nodes(subject)
        if mask is None:
            mask = self._display_mask(subject)
        shape = self._shape(subject)
        if shape is not None:
            name, comp_c, comp_w, pi = shape
            return effective_nodes(comp_c, comp_w, mask, name, pi)
        return effective_nodes(centers, widths, mask)
