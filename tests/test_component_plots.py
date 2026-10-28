"""Component geometry and repeated edges, including real renderers and group fits."""

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pytest

from htfa import HTFA, TFA, HTFAResults
from htfa._spatial import ComponentNodes
from htfa.brainnet import _component_surfaces, _network_geometry
from htfa.io import Subject
from htfa.synthetic import make_htfa_data


def _components(kernel="rbf"):
    return ComponentNodes.from_shape(
        (
            kernel,
            np.array(
                [
                    [[-12.0, -8.0, 0.0], [12.0, -8.0, 0.0]],
                    [[-12.0, 8.0, 0.0], [12.0, 8.0, 0.0]],
                ]
            ),
            np.full((2, 2), 16.0),
            np.array([[0.25, 0.75], [0.6, 0.4]]),
        )
    )


def _mask():
    return np.stack(
        np.meshgrid(
            np.arange(-24.0, 25.0, 2),
            np.arange(-20.0, 21.0, 2),
            np.arange(-12.0, 13.0, 2),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)


@pytest.mark.parametrize("kernel", ["rbf", "laplace", "mexican_hat"])
def test_component_images_are_signed_weighted_profiles(kernel):
    components = _components(kernel)
    coords = _mask()
    for i, img in components.images(coords):
        ijk = (
            nib.affines.apply_affine(np.linalg.inv(img.affine), coords)
            .round()
            .astype(int)
        )
        z = ((coords - components.centers[i]) ** 2).sum(axis=1) / components.widths[i]
        expected = {
            "rbf": lambda z=z: np.exp(-z),
            "laplace": lambda z=z: np.exp(0.1 - np.sqrt(z + 0.01)),
            "mexican_hat": lambda z=z: (1 - 2 * z / 3) * np.exp(-z),
        }[kernel]()
        np.testing.assert_allclose(
            np.asarray(img.dataobj)[tuple(ijk.T)],
            expected * components.mixing[i],
            atol=1e-15,
        )
        if kernel == "mexican_hat":
            assert np.min(img.dataobj) < 0 < np.max(img.dataobj)


def test_component_edges_repeat_values_without_self_or_within_factor_edges():
    comp = _components()
    E = np.array([[1.0, -0.65], [-0.65, 1.0]])
    expanded = comp.connectivity(E)
    np.testing.assert_array_equal(expanded[:2, 2:], np.full((2, 2), -0.65))
    np.testing.assert_array_equal(expanded[:2, :2], 0)
    np.testing.assert_array_equal(expanded[2:, 2:], 0)
    movie = comp.connectivity(np.stack([E, E * 0.5]))
    np.testing.assert_array_equal(movie[0], expanded)
    np.testing.assert_array_equal(movie[1], expanded * 0.5)


def test_component_images_preserve_rotated_affine_and_mask():
    affine = np.array(
        [
            [0.0, -2.0, 0.0, 12.0],
            [3.0, 0.0, 0.0, -12.0],
            [0.0, 0.0, 4.0, -8.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    mask = np.ones((9, 11, 5), dtype=np.uint8)
    mask[3:5, 4:6, :] = 0
    comp = _components("mexican_hat")
    for i, img in comp.images(nib.Nifti1Image(mask, affine)):
        np.testing.assert_array_equal(img.affine, affine)
        np.testing.assert_array_equal(np.asarray(img.dataobj)[mask == 0], 0)
        ijk = np.argwhere(mask)
        expected = comp.values(nib.affines.apply_affine(affine, ijk), i)
        np.testing.assert_allclose(np.asarray(img.dataobj)[mask != 0], expected)


def test_component_surfaces_keep_separate_lobes_and_negative_surrounds():
    pytest.importorskip("pyvista")
    comp = _components("mexican_hat")
    surfaces = _component_surfaces(comp, _mask(), 0.05)
    assert {(i, sign) for i, sign, _ in surfaces} == {
        (i, sign) for i in range(4) for sign in (-1, 1)
    }
    for i, _sign, surface in surfaces:
        assert surface.n_points > 20
        # No surface silently moves to a factor's midpoint.
        assert np.linalg.norm(np.asarray(surface.center) - comp.centers[i]) < 1e-5


@pytest.fixture(scope="module")
def result():
    comp = _components("mexican_hat")
    coords = _mask()
    F = (
        np.column_stack([comp.values(coords, i) for i in range(4)])
        .reshape(len(coords), 2, 2)
        .sum(axis=2)
    )
    W = np.random.default_rng(3).normal(size=(2, 20))
    data = F @ W
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = TFA(
            K=2,
            kernel="mexican_hat",
            n_components=2,
            standardize=False,
            max_iter=3,
            random_state=0,
        ).fit(
            data,
            coords,
            start=(
                comp.centers.reshape(2, 2, 3),
                comp.widths.reshape(2, 2),
                comp.mixing.reshape(2, 2),
            ),
        )
    return HTFAResults(model, [Subject(data, coords)])


def test_glass_brain_component_nodes_and_repeated_edges(result):
    disp = result.plot_connectivity(threshold=0.0, snap=False)
    # Two factors x two components: four cross-factor edges in each projection.
    assert len(disp.axes["z"].ax.lines) == 4
    assert len(disp.axes["z"].ax.images) == 4
    assert any(np.ma.min(im.get_array()) < 0 for im in disp.axes["z"].ax.images)
    disp.close()
    summary = result.plot_connectivity(threshold=0.0, snap=False, node_mode="summary")
    assert len(summary.axes["z"].ax.lines) == 1
    summary.close()


def test_brainnet_component_geometry_and_real_static_animation(result, tmp_path):
    pytest.importorskip("pyvista")
    from htfa import brainnet

    args = (result, None, None, 0, False, _mask(), False, False)
    centers, _, _, components, surfaces = _network_geometry(*args, "auto", 0.05)
    assert centers.shape == (4, 3) and len(surfaces) == 8
    np.testing.assert_array_equal(components.parents, [0, 0, 1, 1])
    E = np.array([[0.0, 0.6], [0.6, 0.0]])
    image = result.plot_network(
        subject=0, connectivity=E, mesh=None, snap=False, size=(240, 180)
    )
    direct = brainnet.plot_network(
        result.model, E, mask=_mask(), mesh=None, snap=False, size=(240, 180)
    )
    np.testing.assert_array_equal(np.asarray(image), np.asarray(direct))
    movie = result.animate_network(
        tmp_path / "components.gif",
        subject=0,
        connectivity=E,
        n_frames=2,
        mesh=None,
        snap=False,
        size=(240, 180),
    )
    from PIL import Image

    with Image.open(movie) as im:
        assert im.n_frames >= 1
    image.save(tmp_path / "components.png")


@pytest.mark.parametrize("kernel", ["rbf", "laplace", "mexican_hat"])
@pytest.mark.parametrize("mode", ["fast", "exact"])
def test_group_mixtures_propagate_components_and_plot_group(kernel, mode):
    Xs, Rs, _ = make_htfa_data(
        4, (8, 8), 12, K=2, kernel=kernel, n_components=2, random_state=4
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = HTFA(
            K=2,
            kernel=kernel,
            n_components=2,
            mode=mode,
            template_update="fixed",
            max_global_iter=2,
            max_local_iter=3,
            template_max_iter=3,
            standardize=False,
            random_state=0,
        ).fit(Xs, Rs, groups=[0, 0, 1, 1])
    assert model.group_component_centers_.shape == (2, 2, 2, 2)
    np.testing.assert_allclose(model.group_mixing_weights_.sum(axis=2), 1)
    np.testing.assert_allclose(
        model.group_centers_,
        (model.group_mixing_weights_[..., None] * model.group_component_centers_).sum(
            axis=2
        ),
    )
    res = HTFAResults(model, [Subject(x, r) for x, r in zip(Xs, Rs, strict=True)])
    comp = res._component_nodes(res._target("global", 1), "auto")
    np.testing.assert_array_equal(
        comp.centers, model.group_component_centers_[1].reshape(4, 2)
    )
    plt.close(res.plot_nodes(group=1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(Xs, Rs)
    assert model.group_tree_ is None and not hasattr(model, "group_component_centers_")
