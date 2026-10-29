"""Public inference boundaries exercised independently of model fitting."""

import nibabel as nib
import numpy as np
import pytest

from htfa.groups import GroupTest
from htfa.io import load


def _test_result(observed, null):
    observed, null = np.asarray(observed), np.asarray(null)
    return GroupTest(
        level=0,
        groups=["a", "b"],
        center=observed,
        width=observed,
        connectivity=observed,
        center_null=null,
        width_null=null,
        connectivity_null=null,
        permuted_labels=np.empty((len(null), 0), dtype=str),
        random_state=0,
        seconds=0.0,
    )


@pytest.mark.parametrize("stat", ["center", "width", "connectivity"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_undefined_observed_statistic_does_not_become_significant(stat, bad):
    result = _test_result([bad, 2.0], np.zeros((100, 2)))
    for fwe in (False, True):
        p = result.pvalues(stat, fwe=fwe)
        assert np.isnan(p[0])
        assert p[1] == pytest.approx(1 / 101)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_undefined_null_statistic_is_not_dropped_from_denominator(bad):
    null = np.zeros((100, 2))
    null[0, 0] = bad
    result = _test_result([2.0, 2.0], null)
    p = result.pvalues()
    assert np.isnan(p[0])
    assert p[1] == pytest.approx(1 / 101)
    # A maximum over an undefined factor is itself undefined; correction cannot
    # silently discard that factor from the family.
    assert np.isnan(result.pvalues(fwe=True)).all()


def test_permutation_pvalues_count_ties_and_include_observation():
    result = _test_result([2.0, 4.0], [[2.0, 1.0], [0.0, 3.0], [3.0, 4.0]])
    np.testing.assert_allclose(result.pvalues(), [3 / 4, 2 / 4])
    np.testing.assert_allclose(result.pvalues(fwe=True), [1, 2 / 4])


def test_stacking_equal_shapes_does_not_ignore_spatial_registration():
    a = nib.Nifti1Image(np.ones((3, 4, 5)), np.eye(4))
    affine = np.eye(4)
    affine[0, 3] = 20
    b = nib.Nifti1Image(np.ones((3, 4, 5)) * 2, affine)
    with pytest.raises(ValueError, match="different affines"):
        load([a, b], stack=True)
    with pytest.raises(ValueError, match="with and without"):
        load([a, np.ones((3, 4, 5))], stack=True)
    # Same-grid images retain the intended time ordering and coordinates.
    c = nib.Nifti1Image(np.ones((3, 4, 5)) * 3, np.eye(4))
    (subject,) = load([a, c], stack=True)
    np.testing.assert_array_equal(subject.data, np.tile([1, 3], (60, 1)))


def test_stacking_explicit_coordinates_requires_same_order():
    data = np.ones((4, 2))
    coords = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    with pytest.raises(ValueError, match="different coordinates"):
        load([data, data], coords=[coords, coords[::-1]], stack=True)


@pytest.mark.parametrize("kind", ["image", "path", "masker"])
def test_mask_same_shape_different_affine_is_rejected(tmp_path, kind):
    img = nib.Nifti1Image(np.ones((3, 4, 5)), np.eye(4))
    affine = np.eye(4)
    affine[1, 3] = 20
    mask = nib.Nifti1Image(np.ones((3, 4, 5)), affine)
    if kind == "path":
        path = tmp_path / "mask.nii.gz"
        nib.save(mask, path)
        mask = path
    elif kind == "masker":
        from nilearn.maskers import NiftiMasker

        mask = NiftiMasker(mask_img=mask).fit()
    with pytest.raises(ValueError, match="mask affine"):
        load(img, mask=mask)


def test_mask_affine_roundoff_is_allowed():
    img = nib.Nifti1Image(np.ones((3, 4, 5)), np.eye(4))
    affine = np.eye(4)
    affine[1, 3] = 1e-7
    mask = nib.Nifti1Image(np.ones((3, 4, 5)), affine)
    (subject,) = load(img, mask=mask)
    assert subject.data.shape == (60, 1)


@pytest.mark.parametrize(
    "api,which",
    [
        (api, which)
        for api in ["TFA", "HTFA", "fit", "load", "standardize"]
        for which in ["data", "coords"]
        if (api, which) != ("standardize", "coords")
    ],
)
def test_real_data_apis_reject_complex_values(api, which):
    import htfa
    from htfa.tfa import standardize_data

    x = np.ones((4, 3))
    coords = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    if which == "data":
        x = x + 1j
    else:
        coords = coords + 1j
    with pytest.raises(ValueError, match="real-valued"):
        if api == "TFA":
            htfa.TFA(K=1).fit(x, coords)
        elif api == "HTFA":
            htfa.HTFA(K=1).fit([x, x], [coords, coords])
        elif api == "fit":
            htfa.fit(x, K=1, coords=coords)
        elif api == "load":
            htfa.load(x, coords=coords)
        else:
            standardize_data(x, False)


def test_complex_nifti_is_rejected_before_get_fdata_discards_it():
    img = nib.Nifti1Image(np.ones((3, 4, 5), dtype=complex) * (1 + 2j), np.eye(4))
    with pytest.raises(ValueError, match="real-valued"):
        load(img)


@pytest.mark.parametrize("api", ["static", "dynamic", "isfc", "global", "smooth"])
def test_real_connectivity_apis_reject_complex_values(api):
    import htfa
    from htfa.timecourse import global_connectivity, moving_average

    weights = np.ones((2, 5), dtype=complex) * (1 + 2j)
    with pytest.raises(ValueError, match="real-valued"):
        if api == "static":
            htfa.static_connectivity(weights)
        elif api == "dynamic":
            htfa.dynamic_connectivity(weights, 3)
        elif api == "isfc":
            htfa.isfc([weights, weights])
        elif api == "global":
            global_connectivity(np.ones((3, 2, 2), dtype=complex) * (1 + 2j))
        else:
            moving_average(weights, 3)


def test_observed_reference_matching_is_not_transitive():
    """A moving reference changes permutation ranks on a three-state orbit."""
    from htfa._matching import match_factors
    from htfa.groups import _permutation_reference

    angles = np.deg2rad([0, 120, 240])
    centers = [
        np.stack([[np.cos(a), np.sin(a)], [-np.cos(a), -np.sin(a)]]) for a in angles
    ]
    statistics = np.tile([1.0, 0.0], (3, 1))
    # All pairwise matches swap factors: assignments cannot be composed into
    # one consistent identity for each factor across the three label states.
    np.testing.assert_array_equal(match_factors(centers[0], centers[1]), [1, 0])
    np.testing.assert_array_equal(match_factors(centers[1], centers[2]), [1, 0])
    np.testing.assert_array_equal(match_factors(centers[0], centers[2]), [1, 0])
    canonical = None
    for observed in range(3):
        moving = np.array(
            [
                statistics[i, match_factors(centers[observed], c)]
                for i, c in enumerate(centers)
            ]
        )
        # The observed-defined first factor gets p=1/3 in every state, despite
        # the unaligned first statistic being constant over the orbit.
        assert (moving[:, 0] >= 1).mean() == pytest.approx(1 / 3)
        reference, display = _permutation_reference(
            (centers[0], np.ones(2)), centers[observed]
        )
        fixed = np.array(
            [
                statistics[i, match_factors(reference, c)[display]]
                for i, c in enumerate(centers)
            ]
        )
        restored = fixed[:, np.argsort(display)]
        if canonical is None:
            canonical = restored
        else:
            np.testing.assert_array_equal(restored, canonical)
    # For a fixed reference factor, the complete-orbit ranks are superuniform.
    exact_p = (canonical[:, None, :] >= canonical[None, :, :]).mean(axis=0)
    for alpha in (1 / 3, 2 / 3):
        assert np.all((exact_p <= alpha).mean(axis=0) <= alpha)


def test_permutation_reference_uses_mixture_weighted_stage_zero_centers():
    from htfa.groups import _permutation_reference

    components = np.array([[[0.0, 0.0], [4.0, 0.0]], [[0.0, 4.0], [4.0, 4.0]]])
    mixing = np.array([[0.75, 0.25], [0.25, 0.75]])
    expected = np.array([[1.0, 0.0], [3.0, 4.0]])
    reference, order = _permutation_reference(
        (components, np.ones((2, 2)), mixing), expected[::-1]
    )
    np.testing.assert_allclose(reference, expected, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(order, [1, 0])


@pytest.mark.parametrize("scale", [0.1, 10.0, 1e-100, 1e100])
def test_permutation_reference_normalizes_custom_start_weights(scale):
    from htfa.groups import _permutation_reference

    centers = np.array([[[0.0, 0.0], [4.0, 0.0]], [[0.0, 4.0], [4.0, 4.0]]])
    mixing = np.array([[0.75, 0.25], [0.25, 0.75]])
    global_centers = np.array([[3.0, 4.0], [1.0, 0.0]])
    reference, order = _permutation_reference(
        (centers, np.ones((2, 2)), mixing), global_centers
    )
    scaled_reference, scaled_order = _permutation_reference(
        (centers, np.ones((2, 2)), mixing * [[scale], [1 / scale]]), global_centers
    )
    # Custom starts use relative positive weights. Normalization must match the
    # estimator's logit/softmax path, including row-specific scaling.
    np.testing.assert_allclose(scaled_reference, reference, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(scaled_order, order)


def test_group_refit_statistics_share_reference_across_observed_labelings(tmp_path):
    import warnings

    from sklearn.base import clone

    import htfa

    xs, rs, _ = htfa.make_htfa_data(4, (6, 6), 12, K=2, random_state=12)
    model = htfa.HTFA(
        K=2,
        template_update="fixed",
        max_global_iter=1,
        max_local_iter=2,
        template_max_iter=2,
        init="power",
        standardize=False,
        random_state=0,
        n_threads=1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(xs, rs, groups=["a", "a", "b", "b"])
        first = htfa.group_permutation_test(
            model, xs, rs, n_permutations=2, random_state=3
        )
        relabeled = clone(model).fit(
            xs, rs, starts=model.free_fits_, groups=list(first.permuted_labels[0])
        )
        second = htfa.group_permutation_test(
            relabeled, xs, rs, n_permutations=1, random_state=4
        )
    np.testing.assert_array_equal(first.reference_centers, second.reference_centers)
    first_order, second_order = (
        np.argsort(first.reference_indices),
        np.argsort(second.reference_indices),
    )
    for stat in ("center", "width", "connectivity"):
        np.testing.assert_allclose(
            getattr(first, stat + "_null")[0, first_order],
            getattr(second, stat)[second_order],
            atol=1e-12,
        )
    path = tmp_path / "reference.npz"
    np.savez(path, **first.to_arrays())
    with np.load(path, allow_pickle=False) as arrays:
        restored = GroupTest.from_arrays(arrays)
    np.testing.assert_array_equal(restored.reference_centers, first.reference_centers)
    np.testing.assert_array_equal(restored.reference_indices, first.reference_indices)
    legacy = {
        k: v for k, v in first.to_arrays().items() if not k.startswith("reference_")
    }
    assert GroupTest.from_arrays(legacy).reference_centers.size == 0


@pytest.mark.parametrize("api", ["TFA", "HTFA-starts", "HTFA-template"])
@pytest.mark.parametrize(
    "components,which",
    [(1, 0), (1, 1), (2, 0), (2, 1), (2, 2)],
)
def test_initialization_rejects_complex_arrays_before_cast(api, components, which):
    import htfa

    data = np.arange(12, dtype=float).reshape(4, 3)
    coords = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    if components == 1:
        start = [np.array([[0.5, 0.5]]), np.ones(1)]
    else:
        start = [np.full((1, 2, 2), 0.5), np.ones((1, 2)), np.full((1, 2), 0.5)]
    start[which] = start[which] + 1j
    with pytest.raises(ValueError, match="real-valued"):
        if api == "TFA":
            htfa.TFA(K=1, n_components=components, backend="numpy").fit(
                data, coords, start=tuple(start)
            )
        else:
            options = (
                {"starts": [tuple(start), tuple(start)]}
                if api == "HTFA-starts"
                else {"init_template": tuple(start)}
            )
            htfa.HTFA(K=1, n_components=components, backend="numpy").fit(
                [data, data], [coords, coords], **options
            )
