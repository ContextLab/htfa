"""Folder-derived trees use actual paths and preserve independent subjects."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from htfa import HTFA, HTFAResults, fit, load
from htfa.groups import GroupTree, group_permutation_test, group_tree

FAST = dict(
    K=2,
    max_global_iter=1,
    max_local_iter=2,
    random_state=4,
    standardize=False,
    init="power",
)
pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")


def save(root, relative, seed=0):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.random.default_rng(seed).uniform(1, 2, (3, 3, 3, 6))
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    return path


def test_recursive_natural_order_paths_and_hidden_entries(tmp_path):
    for name in ["group10/sub2.nii.gz", "group2/sub10.nii", "group2/sub2.NII.GZ"]:
        save(tmp_path, name)
    save(tmp_path, ".hidden/a.nii")
    save(tmp_path, "group2/.hidden.nii")
    (tmp_path / "empty").mkdir()
    (tmp_path / "group2" / "notes.txt").write_text("not imaging data")
    subjects = load(tmp_path)
    assert [Path(s.source).relative_to(tmp_path).as_posix() for s in subjects] == [
        "group2/sub2.NII.GZ",
        "group2/sub10.nii",
        "group10/sub2.nii.gz",
    ]
    assert [s.group_path for s in subjects] == [("group2",), ("group2",), ("group10",)]
    assert all(s.data.shape == (27, 6) for s in subjects)


def test_folder_symlink_cycle_is_skipped(tmp_path):
    save(tmp_path, "group/a.nii")
    try:
        (tmp_path / "group" / "cycle").symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")
    assert len(load(tmp_path)) == 1


def test_folder_lists_get_root_names_but_file_lists_do_not(tmp_path):
    a = save(tmp_path, "condition-a/a.nii")
    b = save(tmp_path, "condition-b/b.nii.gz")
    assert [s.group_path for s in load([a.parent, b.parent])] == [
        ("condition-a",),
        ("condition-b",),
    ]
    assert [s.group_path for s in load([a, b])] == [(), ()]
    assert load(a.parent)[0].group_path == ()
    (stacked,) = load(tmp_path, stack=True)
    assert stacked.group_path == () and stacked.data.shape == (27, 12)


def test_ragged_tree_and_repeated_leaf_names_are_actual_nodes(tmp_path):
    for i, name in enumerate(
        ["root.nii", "a/direct.nii", "a/control/s1.nii", "b/control/s2.nii.gz"]
    ):
        save(tmp_path, name, i)
    res = fit(tmp_path, **FAST)
    tree = res.model.group_tree_
    assert res.model.template_update == "fixed"
    assert tree.paths == [(), ("a",), ("b",), ("a", "control"), ("b", "control")]
    # Natural depth-first ordering is a/control, a/direct, b/control, root.
    assert [tree.paths[v] for v in tree.subject_node] == [
        ("a", "control"),
        ("a",),
        ("b", "control"),
        (),
    ]
    np.testing.assert_array_equal(tree.members(1), [0, 1])
    np.testing.assert_array_equal(tree.members(0), np.arange(4))
    assert np.isfinite(res.model.group_centers_).all()
    assert set(res.group_centers) == {"a", "b", ("a", "control"), ("b", "control")}


def test_explicit_groups_and_opt_out_override_folder_metadata(tmp_path):
    save(tmp_path, "a/s1.nii")
    save(tmp_path, "b/s2.nii")
    flat = fit(tmp_path, groups=False, **FAST)
    assert flat.model.group_tree_ is None
    assert flat.model.template_update == "brainiak"
    explicit = fit(tmp_path, groups=["x", "x"], template_update="fixed", **FAST)
    assert explicit.model.group_tree_.paths == [(), ("x",)]
    empirical = fit(tmp_path, template_update="empirical", **FAST)
    assert empirical.model.template_update == "empirical"
    with pytest.raises(ValueError, match="groups need template_update"):
        fit(tmp_path, template_update="brainiak", **FAST)


def test_nested_single_subject_still_fits_tfa(tmp_path):
    save(tmp_path, "a/b/one.nii")
    result = fit(tmp_path, K=2, init="power", max_iter=2, random_state=0)
    assert result.subjects[0].group_path == ("a", "b")
    assert len(result.centers) == 1


def test_duplicate_root_names_need_explicit_groups(tmp_path):
    a = save(tmp_path, "one/control/a.nii")
    b = save(tmp_path, "two/control/b.nii")
    with pytest.raises(ValueError, match="different directories"):
        fit([a.parent, b.parent], **FAST)
    res = fit(
        [a.parent, b.parent], groups=["one", "two"], template_update="fixed", **FAST
    )
    assert res.model.group_tree_.paths == [(), ("one",), ("two",)]
    with pytest.raises(ValueError, match="duplicate input files"):
        fit([a.parent, a.parent], **FAST)


def test_path_tree_rebuild_validation_and_exact_ragged_solution():
    tree = GroupTree.from_paths([(), ("a",), ("a", "b")])
    assert tree.paths == [(), ("a",), ("a", "b")]
    # Three observations with unit precision, coupled by two unit-precision edges.
    np.testing.assert_allclose(tree.solve(np.array([0.0, 3.0, 6.0])), [1.5, 3.0, 4.5])
    copied = group_tree(tree, 3, ["s1", "s2", "s3"])
    np.testing.assert_array_equal(copied.precision, tree.precision)
    with pytest.raises(ValueError, match="one path per subject"):
        group_tree(tree, 2)
    with pytest.raises(ValueError, match="repeated"):
        group_tree(GroupTree.from_paths([("a",), ("a",)]), 2, ["x", "x"])
    with pytest.raises(ValueError, match="sequence of labels"):
        GroupTree.from_paths(["a", "b"])
    with pytest.raises(ValueError, match="at least one"):
        GroupTree.from_paths([])


@pytest.mark.parametrize(
    "paths,message",
    [
        (["a/direct.nii", "b/child/sub.nii"], "equally deep"),
        (["a/control/sub.nii", "b/control/sub.nii"], "distinct labels"),
    ],
)
def test_permutation_rejects_unsupported_exchangeability(tmp_path, paths, message):
    for i, name in enumerate(paths):
        save(tmp_path, name, i)
    result = fit(tmp_path, **FAST)
    with pytest.raises(ValueError, match=message):
        group_permutation_test(
            result.model,
            [s.data for s in result.subjects],
            [s.coords for s in result.subjects],
            n_permutations=1,
        )


@pytest.mark.parametrize("stack", [False, True])
def test_empty_inputs_raise_actionable_error(stack):
    with pytest.raises(ValueError, match="at least one subject"):
        load([], stack=stack)


def test_ambiguous_root_names_with_distinct_descendants_are_not_merged(tmp_path):
    a = save(tmp_path, "one/control/subgroup-a/a.nii")
    b = save(tmp_path, "two/control/subgroup-b/b.nii")
    with pytest.raises(ValueError, match="different directories"):
        fit([a.parent.parent, b.parent.parent], **FAST)


def test_explicit_files_share_global_parent_without_directory_collision(tmp_path):
    a = save(tmp_path, "one/a.nii")
    b = save(tmp_path, "two/b.nii")
    c = save(tmp_path, "three/condition/c.nii")
    result = fit([a, b, c.parent], **FAST)
    assert [s.group_path for s in result.subjects] == [(), (), ("condition",)]
    assert result.model.group_tree_.paths == [(), ("condition",)]
    np.testing.assert_array_equal(result.model.group_tree_.subject_node, [0, 0, 1])


@pytest.mark.parametrize("components", [1, 2])
def test_global_only_path_tree_matches_fixed_ungrouped_fit(components):
    rng = np.random.default_rng(0)
    data = [rng.normal(size=(16, 8)) for _ in range(2)]
    coords = np.indices((4, 4)).reshape(2, -1).T
    options = dict(FAST, template_update="fixed", n_components=components)
    tree = GroupTree.from_paths([(), ()])
    grouped = HTFA(**options).fit(data, [coords] * 2, groups=tree)
    plain = HTFA(**options).fit(data, [coords] * 2)
    np.testing.assert_array_equal(grouped.global_centers_, plain.global_centers_)
    np.testing.assert_array_equal(grouped.weights_, plain.weights_)
    assert grouped.group_mixing_weights_.shape == (0, 2, components)
    results = HTFAResults(grouped, load(data, coords=[coords] * 2))
    assert results.group_centers == {}
    with pytest.raises(ValueError, match="0 level"):
        group_permutation_test(grouped, data, [coords] * 2, n_permutations=1)
