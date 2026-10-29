"""Nested groups of subjects for HTFA: the template tree and a permutation test.

`HTFA.fit` takes ``groups`` (one label per subject, or a list of label sequences
for deeper nesting, top level first). The templates then form a tree: the global
template at the top, one template per group below it (level by level), and the
subjects below their innermost group. Every node's prior is a Gaussian centred on
its parent with the subject-level covariance ``Sigma`` of the HTFA fit (the same
covariance at every level), and the global template has a flat prior. Given the
subjects' matched local estimates ``x_s``, the templates minimizing::

    sum_s (x_s - c_group(s))^T Sigma^-1 (x_s - c_group(s))
    + sum_groups (c_g - c_parent(g))^T Sigma^-1 (c_g - c_parent(g))

solve ``A c = b`` per coordinate, where ``A`` is the tree's graph Laplacian (unit
edge weights) plus the number of subjects on the diagonal of each innermost
group, and ``b`` sums the subjects' estimates at their group (`GroupTree.solve`).
The solution does not depend on ``Sigma``; the posterior covariance of node ``v``
is ``(A^-1)_vv Sigma``. With no groups the tree is the global template alone and
this is the mean of the local estimates, ``template_update='fixed'``'s update.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import clone

from htfa._matching import match_factors
from htfa.connectivity import static_connectivity

if TYPE_CHECKING:
    from htfa.htfa import HTFA

Array = NDArray[np.float64]

__all__ = ["GroupTest", "GroupTree", "group_permutation_test", "group_tree"]

# Fisher z of correlations clipped to +-(1 - FISHER_CLIP) (as in htfa.connectivity).
FISHER_CLIP = 1e-7


def _label(value: Any) -> Hashable:
    """A group label as a Python ``str`` or ``int`` (NumPy scalars converted)."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(
            f"group labels must be strings or integers; got {value!r} "
            f"({type(value).__name__})"
        )
    return value


def _levels(groups: Any, n_subjects: int) -> list[list[Hashable]]:
    """``groups`` as a list of levels, each a list of one label per subject."""
    seq = list(groups.tolist() if isinstance(groups, np.ndarray) else groups)
    if not seq:
        raise ValueError("groups must not be empty")
    nested = all(not isinstance(g, (str, bytes)) and np.ndim(g) == 1 for g in seq)
    levels = [list(g) for g in seq] if nested else [seq]
    for j, level in enumerate(levels):
        if len(level) != n_subjects:
            raise ValueError(
                f"groups level {j} has {len(level)} labels for {n_subjects} subjects"
            )
    return [[_label(v) for v in level] for level in levels]


def group_key(path: tuple[Hashable, ...]) -> Hashable:
    """The key of a group: its label at the top level, else its label path."""
    return path[0] if len(path) == 1 else path


@dataclass(frozen=True)
class GroupTree:
    """The template tree of a grouped HTFA fit (see the module description).

    Attributes
    ----------
    labels : list of list
        One list of labels per level (top level first), one label per subject.
    paths : list of tuple
        Node ``v``'s label path; node 0 is the global template ``()``, then the
        groups level by level in order of first appearance among the subjects.
    parent : ndarray of int, shape (N,)
        Parent node of each node (-1 for the global template).
    subject_node : ndarray of int, shape (n_subjects,)
        Each subject's innermost group.
    precision : ndarray, shape (N, N)
        ``A``: the tree's graph Laplacian plus the subject counts.
    variance : ndarray, shape (N,)
        ``diag(A^-1)``: node ``v``'s posterior covariance is ``variance[v] Sigma``.
    """

    labels: list[list[Hashable]]
    paths: list[tuple[Hashable, ...]]
    parent: NDArray[np.intp]
    subject_node: NDArray[np.intp]
    precision: Array
    variance: Array

    @property
    def n_nodes(self) -> int:
        return len(self.paths)

    def solve(self, values: Array) -> Array:
        """Node values minimizing the tree objective for subject ``values``.

        Parameters
        ----------
        values : ndarray, shape (n_subjects, ...)
            The subjects' matched local estimates (centers or widths).

        Returns
        -------
        ndarray, shape (N, ...)
            Node 0 is the global template.
        """
        b = np.zeros((self.n_nodes,) + values.shape[1:])
        np.add.at(b, self.subject_node, values)
        flat = np.linalg.solve(self.precision, b.reshape(self.n_nodes, -1))
        return flat.reshape(b.shape)

    def within_groups(self, values: Array) -> Array:
        """``values`` (n_subjects, ...) minus the median of the subject's innermost
        group (deviations for ``template_update='empirical'``)."""
        out = np.array(values, dtype=np.float64)
        for v in np.unique(self.subject_node):
            members = self.subject_node == v
            out[members] -= np.median(values[members], axis=0)
        return out

    def members(self, node: int) -> NDArray[np.intp]:
        """Subjects in node ``node``'s subtree (all subjects for node 0)."""
        path = self.paths[node]
        rows = [
            s
            for s in range(len(self.subject_node))
            if tuple(level[s] for level in self.labels[: len(path)]) == path
        ]
        return np.asarray(rows, dtype=np.intp)


def group_tree(
    groups: Any, n_subjects: int, subject_ids: Sequence[Hashable] | None = None
) -> GroupTree:
    """Build the template tree of a grouped HTFA fit.

    Parameters
    ----------
    groups : sequence of labels, or sequence of sequences of labels
        One label (``str`` or ``int``) per subject, or a list of such sequences,
        one per level, top level first (for example ``[site, diagnosis]``).
        Groups must be nested: every label of a level belongs to one label of
        the level above (use distinct labels such as ``'A-patient'`` and
        ``'B-patient'`` for groups under different parents).
    n_subjects : int
        Number of subjects.
    subject_ids : sequence, optional
        One unique identifier per subject (per entry of ``X``). Repeated
        identifiers raise ``ValueError``: repeated runs are not independent
        subjects and must be combined before fitting.

    Returns
    -------
    GroupTree

    Raises
    ------
    ValueError
        On labels that are not strings or integers, a level with the wrong
        number of labels, a label under two parents (crossed levels), or a
        subject identifier in two groups (crossed designs, such as the same
        people before and after a treatment, are not supported).

    Examples
    --------
    >>> tree = group_tree(["child", "adult", "child", "adult"], 4)
    >>> tree.paths
    [(), ('child',), ('adult',)]
    >>> tree.solve(np.array([1.0, 4.0, 3.0, 6.0]))  # global, then group means
    array([3.5, 2.5, 4.5])
    """
    levels = _levels(groups, n_subjects)
    for j in range(1, len(levels)):
        parents: dict[Hashable, set[Hashable]] = {}
        for s in range(n_subjects):
            parents.setdefault(levels[j][s], set()).add(levels[j - 1][s])
        for label, above in parents.items():
            if len(above) > 1:
                raise ValueError(
                    f"groups must be nested: label {label!r} of level {j} appears "
                    f"under {sorted(map(repr, above))} of level {j - 1}; give "
                    "groups under different parents distinct labels (crossed "
                    "designs are not supported)"
                )
    paths: list[tuple[Hashable, ...]] = [()]
    for j in range(len(levels)):
        for s in range(n_subjects):
            path = tuple(level[s] for level in levels[: j + 1])
            if path not in paths:
                paths.append(path)
    index = {p: v for v, p in enumerate(paths)}
    subject_node = np.array(
        [index[tuple(level[s] for level in levels)] for s in range(n_subjects)],
        dtype=np.intp,
    )
    if subject_ids is not None:
        if len(subject_ids) != n_subjects:
            raise ValueError(
                f"subject_ids has {len(subject_ids)} entries for {n_subjects} subjects"
            )
        seen: dict[Hashable, int] = {}
        for s, sid in enumerate(subject_ids):
            v = int(subject_node[s])
            if seen.setdefault(sid, v) != v:
                raise ValueError(
                    f"subject {sid!r} appears in groups "
                    f"{group_key(paths[seen[sid]])!r} and {group_key(paths[v])!r}; "
                    "only nested designs are supported (each subject in one group), "
                    "not crossed ones such as pre/post on the same people"
                )
        if len(seen) != n_subjects:
            raise ValueError(
                "subject IDs are repeated; groups require one independent "
                "entry per subject "
                "(combine repeated runs before fitting)"
            )
    parent = np.array([-1] + [index[p[:-1]] for p in paths[1:]], dtype=np.intp)
    A = np.zeros((len(paths), len(paths)))
    for v in range(1, len(paths)):
        p = parent[v]
        A[v, v] += 1.0
        A[p, p] += 1.0
        A[v, p] -= 1.0
        A[p, v] -= 1.0
    np.add.at(A, (subject_node, subject_node), 1.0)
    return GroupTree(
        levels, paths, parent, subject_node, A, np.diag(np.linalg.inv(A)).copy()
    )


@dataclass
class GroupTest:
    """Result of `group_permutation_test`.

    The statistics are per factor, displayed in the supplied model's global
    template order. Matching across refits uses a fixed, label-independent
    stage-0 reference; ``reference_indices`` identifies each column's reference
    factor. For the groups
    compared (those of ``level``, against their siblings under the same parent),
    each is the subject-weighted root-mean-square distance of the group values
    from their subject-weighted mean, pooled over parents:
    ``sqrt(sum_g n_g |v_g - mean_v|^2 / sum_g n_g)``, with ``v_g`` the group
    template's center (``center``, in coordinate units), its log width
    (``width``), or the row of the group's mean Fisher-z static connectivity
    without the diagonal (``connectivity``; the mean over the group's
    subjects). For two groups of sizes ``n_a`` and ``n_b`` this is
    ``sqrt(n_a n_b) / (n_a + n_b)`` times the distance between them.

    Attributes
    ----------
    level : int
        The level of ``groups`` whose labels were permuted.
    groups : list
        Keys of the groups compared (see `htfa.HTFAResults.group_centers`).
    center, width, connectivity : ndarray, shape (K,)
        Observed statistics (from a refit with the true labels).
    center_null, width_null, connectivity_null : ndarray, shape (P, K)
        The statistics of the P permutations.
    permuted_labels : ndarray, shape (P, n_subjects)
        Each permutation's labels at ``level`` (as strings).
    random_state : int or None
    seconds : float
        Time taken by the refits.
    reference_centers : ndarray, shape (K, D)
        Label-independent factor centers from the first subject's stage-0 fit,
        in its original order. Empty for saved results without this metadata.
    reference_indices : ndarray, shape (K,)
        Reference index for each statistic column. The same column permutation
        is applied to the observed statistics and every null sample. Empty for
        saved results without this metadata.
    """

    level: int
    groups: list[Hashable]
    center: Array
    width: Array
    connectivity: Array
    center_null: Array
    width_null: Array
    connectivity_null: Array
    permuted_labels: NDArray[np.str_]
    random_state: int | None
    seconds: float
    reference_centers: Array = dataclasses.field(
        default_factory=lambda: np.empty((0, 0), dtype=float)
    )
    reference_indices: NDArray[np.intp] = dataclasses.field(
        default_factory=lambda: np.empty(0, dtype=np.intp)
    )

    def pvalues(self, stat: str = "center", fwe: bool = False) -> Array:
        """Per-factor permutation p-values ``(1 + #{null >= observed}) / (1 + P)``.

        Parameters
        ----------
        stat : {'center', 'width', 'connectivity'}
        fwe : bool
            Family-wise over the K factors: compare each observed value with the
            largest statistic over factors of each permutation (max-statistic
            correction). Default ``False`` (per factor, uncorrected).
            Uncorrected values concern prespecified reference factors; selecting
            a factor after inspecting its effect requires multiplicity control.

        Returns
        -------
        ndarray, shape (K,)
            NaN when the observed statistic or a required null statistic is
            nonfinite. With ``fwe=True``, every null factor must be finite,
            because each permutation contributes its maximum over all factors.
        """
        if stat not in ("center", "width", "connectivity"):
            raise ValueError(
                f"stat must be 'center', 'width' or 'connectivity'; got {stat!r}"
            )
        observed = getattr(self, stat)
        null = getattr(self, f"{stat}_null")
        ref = null.max(axis=1, keepdims=True) if fwe else null
        p = (1.0 + (ref >= observed[None, :]).sum(axis=0)) / (1.0 + len(null))
        null_valid = np.isfinite(null).all() if fwe else np.isfinite(null).all(axis=0)
        valid = np.isfinite(observed) & null_valid
        return np.where(valid, p, np.nan)

    def to_arrays(self) -> dict[str, Array]:
        """The fields as NumPy arrays (``np.savez`` without pickling); the
        group keys are stored as strings, nested keys joined with ``'/'``."""
        out = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if f.name == "groups":
                value = [
                    "/".join(map(str, g)) if isinstance(g, tuple) else str(g)
                    for g in value
                ]
            elif f.name == "random_state" and value is None:
                value = -1
            out[f.name] = np.asarray(value)
        return out

    @classmethod
    def from_arrays(cls, arrays: Any) -> GroupTest:
        """A `GroupTest` from `to_arrays` output (or an ``np.load`` of it).

        Group keys come back as strings (nested keys joined with ``'/'``), a
        ``random_state`` of -1 as ``None``; other entries of ``arrays`` are
        ignored.
        """
        kw = {
            f.name: arrays[f.name] for f in dataclasses.fields(cls) if f.name in arrays
        }
        kw["level"] = int(kw["level"])
        kw["groups"] = [str(g) for g in kw["groups"]]
        seed = int(kw["random_state"])
        kw["random_state"] = None if seed == -1 else seed
        kw["seconds"] = float(kw["seconds"])
        return cls(**kw)


def _permuted_labels(
    labels: list[list[Hashable]], level: int, rng: np.random.Generator
) -> list[list[Hashable]]:
    """Labels with those of ``level`` shuffled among the units under each parent.

    The units are the groups of the next level (each moved with its subjects),
    or the subjects when ``level`` is the innermost level; the parent is the
    label path above ``level``. Group sizes in units are kept.
    """
    n = len(labels[0])
    unit_depth = level + 2 if level + 1 < len(labels) else None
    units: dict[tuple[Hashable, ...], dict[tuple[Hashable, ...], Hashable]] = {}
    for s in range(n):
        parent = tuple(lab[s] for lab in labels[:level])
        unit = tuple(lab[s] for lab in labels[:unit_depth]) if unit_depth else (s,)
        units.setdefault(parent, {})[unit] = labels[level][s]
    new_label: dict[tuple[Hashable, ...], Hashable] = {}
    for under in units.values():
        keys = list(under)
        values = [under[u] for u in keys]
        for u, i in zip(keys, rng.permutation(len(keys)), strict=True):
            new_label[u] = values[i]
    out = [list(lab) for lab in labels]
    for s in range(n):
        unit = tuple(lab[s] for lab in labels[:unit_depth]) if unit_depth else (s,)
        out[level][s] = new_label[unit]
    return out


def _between(values: Array, sizes: Array, parents: Sequence[Hashable]) -> Array:
    """Per-factor weighted RMS spread of group ``values`` (G, K, ...) about their
    weighted mean under each parent (see `GroupTest`)."""
    total = np.zeros(values.shape[1])
    for p in dict.fromkeys(parents):
        idx = np.array([q == p for q in parents])
        n = sizes[idx].reshape((-1,) + (1,) * (values.ndim - 1))
        v = values[idx]
        mean = (n * v).sum(axis=0) / n.sum()
        sq = (v - mean) ** 2
        total += (n * sq).sum(axis=tuple(i for i in range(values.ndim) if i != 1))
    return np.sqrt(total / sizes.sum())


def _statistics(
    model: HTFA, level: int, perm: NDArray[np.intp]
) -> tuple[list[Hashable], Array, Array, Array]:
    """Group keys and the center, width and connectivity statistics of a grouped
    fit, its factors reordered by ``perm``."""
    tree = model.group_tree_
    assert tree is not None
    nodes = [v for v in range(1, tree.n_nodes) if len(tree.paths[v]) == level + 1]
    sizes = np.array([len(tree.members(v)) for v in nodes], dtype=np.float64)
    parents = [tree.paths[v][:-1] for v in nodes]
    centers = model.group_centers_[np.array(nodes) - 1][:, perm]
    widths = model.group_widths_[np.array(nodes) - 1][:, perm]
    K = len(perm)
    off = ~np.eye(K, dtype=bool)
    z = []
    for v in nodes:
        rows = []
        for s in tree.members(v):
            c = static_connectivity(model.weights_[s][perm])
            rows.append(np.arctanh(np.clip(c, -1 + FISHER_CLIP, 1 - FISHER_CLIP)))
        z.append(np.mean(rows, axis=0)[off].reshape(K, K - 1))
    return (
        [group_key(tree.paths[v]) for v in nodes],
        _between(centers, sizes, parents),
        _between(np.log(widths), sizes, parents),
        _between(np.array(z), sizes, parents),
    )


def _permutation_reference(
    free_fit: Sequence[Array], global_centers: Array
) -> tuple[Array, NDArray[np.intp]]:
    """Label-independent stage-0 anchor and a common display permutation."""
    centers = np.asarray(free_fit[0])
    reference = (
        centers.copy()
        if centers.ndim == 2
        else np.einsum("km,kmd->kd", free_fit[2], centers)
    )
    return reference, match_factors(global_centers, reference)


def group_permutation_test(
    model: HTFA,
    X: Sequence[Any],
    coords: Sequence[Any],
    n_permutations: int = 100,
    level: int = 0,
    random_state: int | None = None,
    verbose: bool = False,
) -> GroupTest:
    """Permutation test of differences between groups, per factor.

    The labels of ``level`` are shuffled among the units under each parent
    (the groups of the next level, moved with their subjects, or the subjects
    at the innermost level), and the model is refitted with the shuffled
    labels: the same estimator parameters, ``random_state`` included, and the
    model's unpenalized stage-0 fits (``free_fits_``) as ``starts``, since those
    do not depend on the labels. The observed statistics come from the same
    refit with the true labels (which, for an integer ``random_state`` and a
    fit made without ``starts`` or ``init_template``, reproduces ``model``).
    Every refit's global template is matched to the first subject's fixed
    stage-0 centers (mixture-weighted centers for mixture factors). This
    reference does not depend on the group labels. A single common permutation
    displays all statistics in the supplied model's global template order;
    ``reference_indices`` records the underlying reference identity of each
    column. See `GroupTest` for the statistics.

    Any caller-supplied ``starts`` must be independent of the tested labels;
    label-informed initialization invalidates this conditional permutation
    procedure. Group labels must be exchangeable within the specified units
    under the null. Per-factor hypotheses should be specified before examining
    their effects; use max-statistic correction for selection across factors.

    Parameters
    ----------
    model : HTFA
        A fitted HTFA with ``groups``.
    X, coords : list of array_like
        The data and coordinates ``model`` was fitted on.
    n_permutations : int
        Number of permutations P (default 100). The smallest attainable p-value
        is ``1 / (P + 1)``.
    level : int
        Which level of ``groups`` to test (0, the top level, by default).
    random_state : int, optional
        Seed of the permutations (the refits use ``model.random_state``).
    verbose : bool
        Print the progress of the refits.

    Returns
    -------
    GroupTest

    Raises
    ------
    ValueError
        If ``model`` was fitted without groups, ``level`` is out of range,
        no parent at ``level`` has two groups, or ``n_permutations < 1``.

    Examples
    --------
    >>> import warnings
    >>> from htfa import HTFA
    >>> from htfa.synthetic import make_htfa_data
    >>> Xs, Rs, _ = make_htfa_data(4, (8, 8), 15, K=2, random_state=0)
    >>> model = HTFA(K=2, template_update="fixed", max_global_iter=2,
    ...              max_local_iter=3, init="power", standardize=False,
    ...              random_state=0)
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")  # 2 global iterations only
    ...     model = model.fit(Xs, Rs, groups=["a", "a", "b", "b"])
    ...     test = group_permutation_test(model, Xs, Rs, n_permutations=3,
    ...                                   random_state=0)
    >>> test.groups, test.center_null.shape, test.pvalues("center").shape
    (['a', 'b'], (3, 2), (2,))
    """
    tree = getattr(model, "group_tree_", None)
    if tree is None:
        raise ValueError("model was fitted without groups; pass groups to HTFA.fit")
    if (
        isinstance(level, (bool, np.bool_))
        or not isinstance(level, (int, np.integer))
        or not 0 <= level < len(tree.labels)
    ):
        raise ValueError(
            f"level must be in [0, {len(tree.labels) - 1}] for groups with "
            f"{len(tree.labels)} level(s); got {level}"
        )
    if (
        isinstance(n_permutations, (bool, np.bool_))
        or not isinstance(n_permutations, (int, np.integer))
        or n_permutations < 1
    ):
        raise ValueError(f"n_permutations must be >= 1, got {n_permutations}")
    children: dict[tuple[Hashable, ...], int] = {}
    for p in tree.paths[1:]:
        if len(p) == level + 1:
            children[p[:-1]] = children.get(p[:-1], 0) + 1
    if max(children.values()) < 2:
        raise ValueError(
            f"level {level} has no two groups under one parent; nothing to compare"
        )
    rng = np.random.default_rng(random_state)
    reference, display_order = _permutation_reference(
        model.free_fits_[0], model.global_centers_
    )
    t0 = time.perf_counter()

    def refit(
        labels: list[list[Hashable]],
    ) -> tuple[list[Hashable], Array, Array, Array]:
        fitted = clone(model).fit(X, coords, starts=model.free_fits_, groups=labels)
        perm = match_factors(reference, fitted.global_centers_)[display_order]
        return _statistics(fitted, level, perm)

    keys, center, width, conn = refit(tree.labels)
    nulls = []
    shuffled = []
    for i in range(n_permutations):
        labels = _permuted_labels(tree.labels, level, rng)
        shuffled.append([str(v) for v in labels[level]])
        nulls.append(refit(labels)[1:])
        if verbose:
            print(
                f"permutation {i + 1}/{n_permutations}: "
                f"{time.perf_counter() - t0:.0f} s",
                flush=True,
            )
    return GroupTest(
        level=level,
        groups=keys,
        center=center,
        width=width,
        connectivity=conn,
        center_null=np.array([n[0] for n in nulls]),
        width_null=np.array([n[1] for n in nulls]),
        connectivity_null=np.array([n[2] for n in nulls]),
        permuted_labels=np.array(shuffled),
        random_state=random_state,
        seconds=time.perf_counter() - t0,
        reference_centers=reference,
        reference_indices=display_order,
    )
