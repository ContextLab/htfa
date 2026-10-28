"""Tests for nested group templates (HTFA.fit(groups=...)) and the permutation test.

Real fits of synthetic data with known group differences; no mocks. The
bitwise equivalence of fits without groups to the code before groups existed
was checked with ``scripts/validation_groups.py nogroups`` / ``compare`` (see
docs/validation.rst); here a fit with ``groups=None`` is compared with one
that does not pass the argument, and the structure of the grouped update is
checked exactly.
"""

import time
import warnings

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402

import htfa  # noqa: E402
from htfa import HTFA, group_permutation_test, make_group_data  # noqa: E402
from htfa.groups import (  # noqa: E402
    GroupTest,
    _permuted_labels,
    _statistics,
    group_tree,
)
from htfa.synthetic import make_htfa_data, match_factors  # noqa: E402

K = 5
SIZES = (4, 5)
SHIFT = 3.0  # grid units, factor 0, group 1, first axis
SCALE = 2.0  # factor 1's width in group 1
ALPHA = 0.05
N_PERM = 39  # smallest p-value 1 / 40 = 0.025
# Group templates are pulled towards their parent by one pseudo-subject: for two
# groups of sizes n_a and n_b the estimated difference is the subjects' mean
# difference times `_shrink` (solve the tree for two leaves; 0.816 for 4 and 5).
# Measured (seed 0, fast and exact alike): every estimated group difference
# within 0.041 of the shrunk difference of the true subject means; factor 0's
# -3.05 against 0.816 * -3.0 = -2.45 for the group templates (the subjects'
# means differ by 3.77: jitter 0.5 per subject, sd 0.34 for the difference of
# means); width ratio 1.79, the other factors' 0.985 to 1.013.
DIFF_TOL = 0.15
SHIFT_TOL = 1.0  # about 3 sd of the subject-mean jitter
RATIO_RANGE = (1.5, 2.2)


def _shrink(n_a: int, n_b: int) -> float:
    tree = group_tree([0] * n_a + [1] * n_b, n_a + n_b)
    out = tree.solve(np.r_[np.zeros(n_a), np.ones(n_b)])
    return float(out[2] - out[1])


def _data(seed: int = 0, shift: float = SHIFT, scale: float = SCALE, sizes=SIZES):
    center_shift = np.zeros((len(sizes), K, 2))
    width_scale = np.ones((len(sizes), K))
    # The global centers do not depend on the shifts: move factor 0 towards the
    # middle of the grid (first axis), so that the shifted center stays inside.
    centers = make_group_data(sizes, (16, 16), 2, K=K, random_state=seed)[3]
    toward = -np.sign(centers["global_centers"][0, 0] - 7.5)
    center_shift[1, 0, 0] = toward * shift
    width_scale[1, 1] = scale
    return make_group_data(
        sizes, grid_shape=(16, 16), n_samples=40, K=K, center_shift=center_shift,
        width_scale=width_scale, center_jitter=0.5, random_state=seed,
    )  # fmt: skip


def _fit(Xs, Rs, groups, mode="fast", n_jobs=None, seed=0, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return HTFA(
            K=K, template_update=kw.pop("update", "fixed"), mode=mode,
            standardize=False, n_jobs=n_jobs, random_state=seed, **kw,
        ).fit(Xs, Rs, groups=groups)  # fmt: skip


def _test(model, Xs, Rs, n_permutations=N_PERM, seed=0, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return group_permutation_test(
            model, Xs, Rs, n_permutations=n_permutations, random_state=seed, **kw
        )


# ---------------------------------------------------------------- the tree


def test_group_tree_two_groups_closed_form():
    tree = group_tree(["c", "a", "c", "a", "a"], 5)
    print("paths", tree.paths, "parent", tree.parent, "\nA\n", tree.precision)
    assert tree.paths == [(), ("c",), ("a",)]
    np.testing.assert_array_equal(tree.parent, [-1, 0, 0])
    np.testing.assert_array_equal(tree.subject_node, [1, 2, 1, 2, 2])
    expected_A = np.array([[2.0, -1, -1], [-1, 3, 0], [-1, 0, 4]])
    np.testing.assert_array_equal(tree.precision, expected_A)
    np.testing.assert_allclose(tree.variance, np.diag(np.linalg.inv(expected_A)))
    x = np.array([1.0, 10.0, 3.0, 20.0, 30.0])
    out = tree.solve(x)
    # Stationarity of the objective: c0 = mean(c_c, c_a); (n_g + 1) c_g = sum + c0
    c0, cc, ca = out
    print("solution", out)
    assert np.isclose(c0, (cc + ca) / 2)
    assert np.isclose(3 * cc, 1 + 3 + c0)
    assert np.isclose(4 * ca, 10 + 20 + 30 + c0)
    # vector-valued (K, D) estimates are solved per coordinate
    rng = np.random.default_rng(0)
    xs = rng.normal(size=(5, 3, 2))
    out = tree.solve(xs)
    for k in range(3):
        for d in range(2):
            np.testing.assert_allclose(out[:, k, d], tree.solve(xs[:, k, d]))


def test_group_tree_single_group_is_the_mean():
    x = np.random.default_rng(1).normal(size=(6, 4, 3))
    tree = group_tree(["all"] * 6, 6)
    out = tree.solve(x)
    np.testing.assert_allclose(out[0], x.mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(out[1], x.mean(axis=0), atol=1e-12)
    # posterior variance factors: global 1/6 + 1, group 1/6
    np.testing.assert_allclose(tree.variance, [1 / 6 + 1, 1 / 6])


def test_group_tree_nested_matches_brute_force_minimizer():
    site = ["A", "A", "A", "B", "B", "B", "B"]
    dx = ["A-ctl", "A-pat", "A-pat", "B-ctl", "B-ctl", "B-pat", "B-pat"]
    tree = group_tree([site, dx], 7)
    print(tree.paths)
    assert tree.paths == [
        (), ("A",), ("B",), ("A", "A-ctl"), ("A", "A-pat"), ("B", "B-ctl"),
        ("B", "B-pat"),
    ]  # fmt: skip
    np.testing.assert_array_equal(tree.parent, [-1, 0, 0, 1, 1, 2, 2])
    np.testing.assert_array_equal(tree.members(1), [0, 1, 2])
    np.testing.assert_array_equal(tree.members(6), [5, 6])
    np.testing.assert_array_equal(tree.members(0), np.arange(7))
    x = np.random.default_rng(2).normal(size=7)
    # minimize sum_s (x_s - c_leaf)^2 + sum_edges (c_v - c_parent)^2 by least squares
    rows, rhs = [], []
    for s, v in enumerate(tree.subject_node):
        r = np.zeros(7)
        r[v] = 1.0
        rows.append(r)
        rhs.append(x[s])
    for v in range(1, 7):
        r = np.zeros(7)
        r[v], r[tree.parent[v]] = 1.0, -1.0
        rows.append(r)
        rhs.append(0.0)
    brute = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
    print("tree", tree.solve(x), "\nbrute", brute)
    np.testing.assert_allclose(tree.solve(x), brute, atol=1e-12)


@pytest.mark.parametrize(
    "groups, ids, match",
    [
        (["a", "b"], None, "has 2 labels for 3 subjects"),
        ([0.5, 1.5, 0.5], None, "strings or integers"),
        ([True, False, True], None, "strings or integers"),
        ([], None, "must not be empty"),
        ([["s1", "s1", "s2"], ["pre", "post", "pre"]], None, "must be nested"),
        ([["x", "x", "y"], ["a", "b"]], None, "level 1 has 2 labels"),
        (["a", "b", "a"], ["p1", "p1", "p2"], "subject 'p1' appears in groups"),
        (["a", "b", "a"], ["p1", "p2"], "subject_ids has 2 entries"),
    ],
)
def test_group_tree_errors(groups, ids, match):
    with pytest.raises(ValueError, match=match) as err:
        group_tree(groups, 3, ids)
    print(err.value)


def test_group_tree_accepts_numpy_and_integer_labels():
    tree = group_tree(np.array([2, 1, 2]), 3)
    assert tree.paths == [(), (2,), (1,)]
    assert all(type(p[0]) is int for p in tree.paths[1:])
    tree = group_tree(np.array([["a", "a", "b"], ["x", "y", "z"]]), 3)
    assert tree.paths[-1] == ("b", "z")
    # repeated runs of one subject in one group are allowed
    group_tree(["a", "a", "b"], 3, subject_ids=["p1", "p1", "p2"])


# ---------------------------------------------------------- the estimator


def test_htfa_group_errors():
    Xs, Rs, groups, _ = _data()
    with pytest.raises(ValueError, match="template_update='fixed' or 'empirical'"):
        HTFA(K=K).fit(Xs, Rs, groups=groups)
    with pytest.raises(ValueError, match="subject_ids is only used with groups"):
        HTFA(K=K, template_update="fixed").fit(Xs, Rs, subject_ids=list(range(9)))
    with pytest.raises(ValueError, match="not crossed ones"):
        HTFA(K=K, template_update="fixed").fit(
            Xs, Rs, groups=groups, subject_ids=[0] * 9
        )
    with pytest.raises(ValueError, match="groups and subject_ids need several"):
        htfa.fit(Xs[0], coords=Rs[0], K=K, groups=["a"])
    plain = _fit(Xs, Rs, None, max_global_iter=1)
    with pytest.raises(ValueError, match="fitted without groups"):
        group_permutation_test(plain, Xs, Rs)
    model = _fit(Xs, Rs, groups, max_global_iter=2)
    with pytest.raises(ValueError, match="level must be in"):
        group_permutation_test(model, Xs, Rs, level=1)
    with pytest.raises(ValueError, match="n_permutations must be >= 1"):
        group_permutation_test(model, Xs, Rs, n_permutations=0)
    nested = _fit(Xs, Rs, [["s1"] * 4 + ["s2"] * 5, groups], max_global_iter=1)
    with pytest.raises(ValueError, match="no two groups under one parent"):
        group_permutation_test(nested, Xs, Rs, level=1)


def test_no_groups_fit_is_unchanged():
    """``groups=None`` runs the code path of fits without groups."""
    Xs, Rs, _ = make_htfa_data(3, (12, 12), 30, K=4, random_state=0)
    for update in ("brainiak", "fixed", "empirical"):
        kw = dict(K=4, template_update=update, max_global_iter=3, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            a = HTFA(**kw).fit(Xs, Rs)
            b = HTFA(**kw).fit(Xs, Rs, groups=None)
        assert a.group_tree_ is None and not hasattr(a, "group_centers_")
        np.testing.assert_array_equal(a.global_centers_, b.global_centers_)
        np.testing.assert_array_equal(a.global_centers_cov_, b.global_centers_cov_)
        for s in range(3):
            np.testing.assert_array_equal(a.weights_[s], b.weights_[s])


@pytest.mark.parametrize("mode", ["fast", "exact"])
def test_grouped_fit_recovers_known_shifts_and_test_detects_them(mode):
    Xs, Rs, groups, truth = _data()
    t0 = time.perf_counter()
    model = _fit(Xs, Rs, groups, mode=mode)
    print(f"{mode}: fit {time.perf_counter() - t0:.1f} s, {model.n_iter_} iterations")
    assert model.group_keys_ == [0, 1]
    assert model.group_centers_.shape == (2, K, 2)
    assert model.group_centers_cov_.shape == (2, K, 2, 2)
    tree = model.group_tree_
    # The templates are the tree solution for the final local estimates.
    nodes = tree.solve(np.stack(model.centers_))
    np.testing.assert_allclose(model.global_centers_, nodes[0], atol=1e-12)
    np.testing.assert_allclose(model.group_centers_, nodes[1:], atol=1e-12)
    widths = tree.solve(np.stack(model.widths_))
    np.testing.assert_allclose(model.group_widths_, widths[1:], atol=1e-12)
    # Uncertainty: (A^-1)_vv Sigma, Sigma the subject-level prior covariance.
    np.testing.assert_allclose(
        model.group_centers_cov_[1, 0], tree.variance[2] * model.prior_centers_cov_
    )
    np.testing.assert_allclose(
        model.global_centers_cov_[0], tree.variance[0] * model.prior_centers_cov_
    )
    # With two groups the global template is a mean of two group templates, each
    # Sigma (between groups) + Sigma / n_g (subjects) from it: its variance is
    # 1 / sum_g (1 + 1 / n_g)^-1 times Sigma, more than a group template's.
    v0 = 1.0 / sum(1.0 / (1.0 + 1.0 / n) for n in SIZES)
    assert tree.variance[0] == pytest.approx(v0, rel=1e-12)
    assert np.all(tree.variance[1:] < v0)
    # Recovery, in truth order.
    perm = match_factors(truth["global_centers"], model.global_centers_)
    gc = model.group_centers_[:, perm]
    diff = gc[1] - gc[0]
    expected = _shrink(*SIZES) * (
        np.mean(truth["centers"][4:], axis=0) - np.mean(truth["centers"][:4], axis=0)
    )
    print("estimated group differences\n", np.round(diff, 3))
    print("expected (shrunk subject-mean differences)\n", np.round(expected, 3))
    true_shift = truth["group_centers"][1, 0, 0] - truth["group_centers"][0, 0, 0]
    assert abs(true_shift) == SHIFT
    print("max |diff - expected|", np.abs(diff - expected).max())
    np.testing.assert_allclose(diff, expected, atol=DIFF_TOL)
    assert abs(diff[0, 0] - _shrink(*SIZES) * true_shift) < SHIFT_TOL

    ratio = model.group_widths_[1, perm] / model.group_widths_[0, perm]
    print("width ratios", np.round(ratio, 3))
    assert RATIO_RANGE[0] < ratio[1] < RATIO_RANGE[1]
    assert np.all(np.abs(np.log(ratio[[0, 2, 3, 4]])) < np.log(1.25))
    # The permutation test finds the moved factor's center and the scaled width.
    test = _test(model, Xs, Rs)
    p_center = test.pvalues("center")[perm]
    p_width = test.pvalues("width")[perm]
    print(f"{test.seconds:.1f} s; center p {p_center}; width p {p_width}")
    print("center p (FWE)", test.pvalues("center", fwe=True)[perm])
    assert p_center[0] <= ALPHA and p_width[1] <= ALPHA
    assert test.pvalues("center", fwe=True)[perm][0] <= ALPHA
    assert test.pvalues("width", fwe=True)[perm][1] <= ALPHA
    assert np.argmax(test.center[perm]) == 0 and np.argmax(test.width[perm]) == 1


@pytest.mark.slow
def test_no_difference_false_positive_rate_near_nominal():
    """With no true difference, per-factor p-values below ALPHA occur at about
    ALPHA. Measured over 20 seeds (5 factors each): see the printout; the
    larger study (200 seeds) is in docs/validation.rst."""
    t0 = time.perf_counter()
    hits = {"center": 0, "width": 0, "connectivity": 0}
    fwe = {"center": 0, "width": 0, "connectivity": 0}
    n_seeds = 20
    for seed in range(n_seeds):
        Xs, Rs, groups, _ = _data(seed=seed, shift=0.0, scale=1.0)
        model = _fit(Xs, Rs, groups, seed=seed)
        test = _test(model, Xs, Rs, n_permutations=19, seed=seed)
        for stat in hits:
            hits[stat] += int((test.pvalues(stat) <= ALPHA).sum())
            fwe[stat] += int((test.pvalues(stat, fwe=True) <= ALPHA).any())
    n = n_seeds * K
    print(f"{time.perf_counter() - t0:.0f} s; per-factor rejections of {n}: {hits}; "
          f"family-wise of {n_seeds}: {fwe}")  # fmt: skip
    # P(X >= 13 | n=100, p=0.05) = 0.0010; P(X >= 6 | n=20, p=0.05) = 0.0003
    for stat in hits:
        assert hits[stat] <= 12, stat
        assert fwe[stat] <= 5, stat
    assert sum(hits.values()) >= 4  # not degenerate (all p-values 1)


@pytest.mark.parametrize("mode", ["fast", "exact"])
def test_grouped_fit_and_test_independent_of_n_jobs(mode):
    Xs, Rs, groups, _ = _data(seed=1)
    a = _fit(Xs, Rs, groups, mode=mode, n_jobs=1, max_global_iter=3)
    b = _fit(Xs, Rs, groups, mode=mode, n_jobs=3, max_global_iter=3)
    np.testing.assert_array_equal(a.group_centers_, b.group_centers_)
    np.testing.assert_array_equal(a.group_widths_, b.group_widths_)
    np.testing.assert_array_equal(a.global_centers_, b.global_centers_)
    for s in range(len(Xs)):
        np.testing.assert_array_equal(a.centers_[s], b.centers_[s])
        np.testing.assert_array_equal(a.weights_[s], b.weights_[s])
    ta = _test(a, Xs, Rs, n_permutations=3)
    tb = _test(b, Xs, Rs, n_permutations=3)
    for name in ("center", "width", "connectivity", "center_null", "width_null"):
        np.testing.assert_array_equal(getattr(ta, name), getattr(tb, name))
    np.testing.assert_array_equal(ta.permuted_labels, tb.permuted_labels)


def test_observed_statistics_reproduce_the_model():
    """The test's refit with the true labels reproduces the model (integer
    random_state, no starts): its statistics equal those of the model."""
    Xs, Rs, groups, _ = _data(seed=2)
    model = _fit(Xs, Rs, groups, max_global_iter=3)
    test = _test(model, Xs, Rs, n_permutations=2)
    keys, center, width, conn = _statistics(model, 0, np.arange(K))
    assert keys == test.groups == [0, 1]
    np.testing.assert_array_equal(test.center, center)
    np.testing.assert_array_equal(test.width, width)
    np.testing.assert_array_equal(test.connectivity, conn)


def test_empirical_update_with_groups():
    Xs, Rs, groups, truth = _data(seed=3)
    model = _fit(Xs, Rs, groups, update="empirical")
    print("empirical prior sd", np.sqrt(np.diag(model.prior_centers_cov_)))
    perm = match_factors(truth["global_centers"], model.global_centers_)
    diff = model.group_centers_[1, perm] - model.group_centers_[0, perm]
    print("group differences", np.round(diff, 3))
    assert abs(abs(diff[0, 0]) - _shrink(*SIZES) * SHIFT) < SHIFT_TOL
    # the dispersion is estimated from deviations within the innermost groups
    tree = model.group_tree_
    x = np.array([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    np.testing.assert_array_equal(
        tree.within_groups(x), [-1.5, -0.5, 0.5, 1.5, -20, -10, 0, 10, 20]
    )


def test_deeper_nesting_fit_and_restricted_permutations():
    sizes = (3, 3, 3, 3)
    center_shift = np.zeros((4, K, 2))
    center_shift[[1, 3], 0, 0] = SHIFT  # the patients of both sites
    Xs, Rs, _, truth = make_group_data(
        sizes, grid_shape=(16, 16), n_samples=40, K=K, center_shift=center_shift,
        center_jitter=0.5, random_state=4,
    )  # fmt: skip
    site = ["A"] * 6 + ["B"] * 6
    dx = ["A-ctl"] * 3 + ["A-pat"] * 3 + ["B-ctl"] * 3 + ["B-pat"] * 3
    model = _fit(Xs, Rs, [site, dx])
    assert model.group_keys_ == [
        "A", "B", ("A", "A-ctl"), ("A", "A-pat"), ("B", "B-ctl"), ("B", "B-pat")
    ]  # fmt: skip
    tree = model.group_tree_
    nodes = tree.solve(np.stack(model.centers_))
    np.testing.assert_allclose(model.group_centers_, nodes[1:], atol=1e-12)
    perm = match_factors(truth["global_centers"], model.global_centers_)
    gc = model.group_centers_[:, perm]
    print("factor 0 x by group:", np.round(gc[:, 0, 0], 2))
    for site_node, ctl, pat in ((0, 2, 3), (1, 4, 5)):
        assert gc[pat, 0, 0] - gc[ctl, 0, 0] > 1.5
        # a site template lies between its two subgroups
        assert gc[ctl, 0, 0] < gc[site_node, 0, 0] < gc[pat, 0, 0]
    # restricted permutations: level 1 shuffles subjects within each site,
    # level 0 moves whole subgroups between sites
    rng = np.random.default_rng(0)
    for _ in range(20):
        lab = _permuted_labels(tree.labels, 1, rng)
        assert lab[0] == site
        for s in range(12):
            assert lab[1][s].startswith(site[s])
        assert sorted(lab[1]) == sorted(dx)
        lab = _permuted_labels(tree.labels, 0, rng)
        assert lab[1] == dx
        for block in range(4):
            assert len(set(lab[0][3 * block : 3 * block + 3])) == 1
        assert sorted(lab[0]) == sorted(site)
        group_tree(lab, 12)  # still nested
    test = _test(model, Xs, Rs, n_permutations=9, level=1)
    print("level 1 center statistic", np.round(test.center[perm], 3))
    assert test.groups == [
        ("A", "A-ctl"),
        ("A", "A-pat"),
        ("B", "B-ctl"),
        ("B", "B-pat"),
    ]
    assert np.argmax(test.center[perm]) == 0
    assert test.pvalues("center")[perm][0] == pytest.approx(0.1)  # 1 / (9 + 1)
    level0 = _test(model, Xs, Rs, n_permutations=3, level=0)
    assert level0.groups == ["A", "B"] and level0.center_null.shape == (3, K)


def test_group_test_pvalues_and_arrays():
    null = np.array([[1.0, 5.0], [2.0, 0.5], [3.0, 0.1]])
    t = GroupTest(
        level=0, groups=["a", ("b", "c")], center=np.array([2.5, 1.0]),
        width=np.zeros(2), connectivity=np.zeros(2), center_null=null,
        width_null=np.zeros((3, 2)), connectivity_null=np.zeros((3, 2)),
        permuted_labels=np.array([["a"], ["b"], ["a"]]), random_state=None,
        seconds=1.0,
    )  # fmt: skip
    np.testing.assert_allclose(t.pvalues("center"), [(1 + 1) / 4, (1 + 1) / 4])
    # family-wise: each permutation's maximum is 5, 2, 3
    np.testing.assert_allclose(t.pvalues("center", fwe=True), [(1 + 2) / 4, 1.0])
    np.testing.assert_allclose(t.pvalues("width"), [1.0, 1.0])  # ties count
    with pytest.raises(ValueError, match="stat must be"):
        t.pvalues("size")
    arrays = t.to_arrays()
    assert list(arrays["groups"]) == ["a", "b/c"]
    assert int(arrays["random_state"]) == -1


def test_group_test_saves_without_pickle(tmp_path):
    Xs, Rs, groups, _ = _data(seed=5)
    model = _fit(Xs, Rs, ["x" if g else "y" for g in groups], max_global_iter=2)
    test = _test(model, Xs, Rs, n_permutations=2)
    np.savez(tmp_path / "t.npz", **test.to_arrays())
    with np.load(tmp_path / "t.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(z["center_null"], test.center_null)
        assert list(z["groups"]) == ["y", "x"]
        assert z["permuted_labels"].shape == (2, 9)


# --------------------------------------------------------------- results


def test_results_group_templates_and_plots():
    Xs, Rs, groups, _ = _data(seed=6)
    labels = ["ctl" if g == 0 else "pat" for g in groups]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        res = htfa.fit(
            Xs, coords=Rs, K=K, groups=labels, template_update="fixed",
            standardize=False, random_state=0,
        )  # fmt: skip
    model = _fit(Xs, Rs, labels)
    np.testing.assert_array_equal(res.model.group_centers_, model.group_centers_)
    assert list(res.group_centers) == ["ctl", "pat"]
    np.testing.assert_array_equal(res.group_widths["pat"], model.group_widths_[1])
    assert res.group_subjects("pat") == [4, 5, 6, 7, 8]
    centers, widths, info = res.display_nodes(group="pat")
    assert centers.shape == (K, 2)
    centers_raw = res._fitted_nodes(res._target("global", "pat"))[0]
    np.testing.assert_array_equal(centers_raw, model.group_centers_[1])
    # plot_nodes(group=...) draws the group template last, after its subjects
    fig = res.plot_nodes(group="pat", snap=False)
    offsets = [c.get_offsets()[:, ::-1] for c in fig.axes[0].collections]
    assert len(offsets) == 1 + 1 + 5  # field, 5 subjects, template
    np.testing.assert_allclose(offsets[-1], model.group_centers_[1])
    np.testing.assert_allclose(offsets[1], model.centers_[4])
    with pytest.raises(ValueError, match="either subject or group"):
        res.display_nodes(0, group="pat")
    with pytest.raises(ValueError, match="unknown group 'x'"):
        res.plot_nodes(group="x")
    test = res.group_permutation_test(n_permutations=2, random_state=0)
    assert test.groups == ["ctl", "pat"]
    plain = htfa.HTFAResults(_fit(Xs, Rs, None, max_global_iter=1), res.subjects)
    assert plain.group_centers == {}
    with pytest.raises(ValueError, match="fitted without groups"):
        plain.display_nodes(group="ctl")


def test_results_group_network_uses_group_isfc():
    pytest.importorskip("pyvista")
    from htfa import brainnet

    Xs, Rs, groups, _ = make_group_data(
        (2, 3), grid_shape=(8, 8, 8), n_samples=30, K=3, random_state=7
    )
    Rs = [R * 4.0 - 14.0 for R in Rs]  # a small box in mm
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        res = htfa.fit(
            Xs, coords=Rs, K=3, groups=groups, template_update="fixed",
            max_global_iter=2, standardize=False, random_state=0,
        )  # fmt: skip
    members = res.group_subjects(1)
    conn = htfa.isfc([res.weights[s] for s in members])
    a = res.plot_network(group=1, mesh=None, snap=False, threshold=0.0)
    b = brainnet.plot_network(
        res.group_centers[1], conn, widths=res.group_widths[1], mesh=None,
        snap=False, threshold=0.0,
    )  # fmt: skip
    print(a.size, b.size)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


# ------------------------------------------------------------- generator


def test_make_group_data_truth_and_errors():
    Xs, Rs, groups, truth = _data()
    assert groups == [0] * 4 + [1] * 5 and len(Xs) == 9
    np.testing.assert_allclose(
        np.abs(truth["group_centers"][1, 0] - truth["global_centers"][0]), [SHIFT, 0]
    )
    np.testing.assert_allclose(
        truth["group_widths"][1, 1] / truth["global_widths"][1], SCALE
    )
    # subjects scatter around their group template (jitter 0.5)
    sub = np.stack(truth["centers"][4:])[:, 0]
    print("group 1 factor 0 subject centers", np.round(sub, 2))
    assert np.abs(sub.mean(axis=0) - truth["group_centers"][1, 0]).max() < 1.0
    again = _data()
    np.testing.assert_array_equal(Xs[3], again[0][3])
    with pytest.raises(ValueError, match="group_sizes must be positive"):
        make_group_data((3, 0), grid_shape=(8, 8), K=2)
    with pytest.raises(ValueError, match="center_shift must be"):
        make_group_data((2, 2), grid_shape=(8, 8), K=2, center_shift=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="width_scale must be positive"):
        make_group_data((2, 2), grid_shape=(8, 8), K=2, width_scale=np.zeros((2, 2)))
