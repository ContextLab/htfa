"""Tests for HTFAResults: factors, reconstruction, export, connectivity, plots."""

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

import warnings  # noqa: E402

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402

from htfa import fit  # noqa: E402
from htfa.synthetic import make_htfa_data, make_tfa_data  # noqa: E402
from htfa.tfa import standardize_data  # noqa: E402

PNG_MIN_BYTES = 5 * 1024
AFFINE = np.array([[2.0, 0, 0, -10], [0, 2.0, 0, -12], [0, 0, 2.0, -8], [0, 0, 0, 1]])
OPTS = dict(max_global_iter=3, max_local_iter=5, standardize=False, random_state=0, K=3)


def _fit_quiet(*args, **kwargs):
    """Fit with few iterations on purpose (speed); silence the expected warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return fit(*args, **kwargs)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def res3d(tmp_path_factory):
    d = tmp_path_factory.mktemp("nii")
    grid = (12, 12, 12)
    Xs, _, _ = make_htfa_data(2, grid, n_samples=40, K=3, noise=0.05, random_state=1)
    for i, X in enumerate(Xs):
        nib.save(nib.Nifti1Image(X.reshape(*grid, -1), AFFINE), d / f"s{i}.nii.gz")
    return _fit_quiet(d, **OPTS)


@pytest.fixture(scope="module")
def res2d():
    """Two 2D subjects given as (V, T) + coords (no image space)."""
    Xs, Rs, _ = make_htfa_data(2, (14, 14), n_samples=30, K=3, random_state=3)
    return _fit_quiet(Xs, coords=Rs, **OPTS)


@pytest.fixture(scope="module")
def res2d_img(tmp_path_factory):
    """One 2D subject from a PNG stack (has image space)."""
    d = tmp_path_factory.mktemp("png")
    X, R, _ = make_tfa_data((14, 14), n_samples=10, K=3, random_state=4)
    X = (X - X.min()) / (X.max() - X.min())
    for t in range(X.shape[1]):
        arr = (X[:, t].reshape(14, 14) * 255).astype(np.uint8)
        Image.fromarray(arr).save(d / f"f{t:02d}.png")
    return fit(d, K=3, stack=True, standardize=True, random_state=0)


def test_factors_and_reconstruct_shapes(res3d):
    V, T = res3d.subjects[0].data.shape
    assert res3d.factors(0).shape == (V, 3)
    recon = res3d.reconstruct(1)
    assert recon.shape == res3d.subjects[1].data.shape
    np.testing.assert_allclose(recon, res3d.factors(1) @ res3d.weights[1])
    r = np.corrcoef(recon.ravel(), res3d.subjects[1].data.ravel())[0, 1]
    assert r > 0.8, r  # measured: 0.892 (3 global iters, seed 0)


def test_to_nifti_shapes_and_affine(res3d):
    img = res3d.to_nifti(0)
    assert isinstance(img, nib.Nifti1Image)
    assert img.shape == (12, 12, 12, 40)
    np.testing.assert_allclose(img.affine, AFFINE)
    fac = res3d.to_nifti(0, "factors")
    assert fac.shape == (12, 12, 12, 3)
    arr = res3d.to_image(0)
    assert arr.shape == (12, 12, 12, 40)


def test_to_nifti_2d_raises(res2d_img):
    with pytest.raises(ValueError, match="affine"):
        res2d_img.to_nifti(0)
    assert res2d_img.to_image(0).shape == (14, 14, 10)


def test_bad_what_raises(res3d):
    with pytest.raises(ValueError, match="what"):
        res3d.to_image(0, "bogus")


def test_connectivity_methods(res3d):
    assert res3d.connectivity(0).shape == (3, 3)
    assert res3d.dynamic_connectivity(0, window=10).shape == (31, 3, 3)
    assert res3d.isfc().shape == (3, 3)


@pytest.mark.parametrize("kind", ["nodes", "connectivity", "reconstruction"])
@pytest.mark.parametrize("dim", ["2D", "2Dimage", "3D"])
def test_plots_render_png(kind, dim, res2d, res2d_img, res3d, tmp_path):
    res = {"2D": res2d, "2Dimage": res2d_img, "3D": res3d}[dim]
    out = {
        "nodes": lambda: res.plot_nodes("global"),
        "connectivity": lambda: res.plot_connectivity(0),
        "reconstruction": lambda: res.plot_reconstruction(0, sample=0),
    }[kind]()
    path = tmp_path / f"{kind}_{dim}.png"
    out.savefig(path)
    assert path.exists() and path.stat().st_size > PNG_MIN_BYTES, path.stat().st_size


def test_plot_reconstruction_3d_without_affine_leaves_no_figure():
    """The no-affine error is raised before a figure is created."""
    Xs, Rs, _ = make_htfa_data(2, (6, 6, 6), n_samples=10, K=2, random_state=0)
    res = _fit_quiet(Xs, coords=Rs, **{**OPTS, "K": 2})
    before = plt.get_fignums()
    with pytest.raises(ValueError, match="affine"):
        res.plot_reconstruction(0)
    print("open figures before/after:", before, plt.get_fignums())
    assert plt.get_fignums() == before


@pytest.fixture(scope="module")
def res_mni(tmp_path_factory):
    """One subject on a 10 mm grid that spans the MNI152 brain."""
    d = tmp_path_factory.mktemp("mni")
    grid = (20, 24, 19)  # x -98..92, y -134..96, z -72..108 mm
    aff = np.array(
        [[10.0, 0, 0, -98], [0, 10.0, 0, -134], [0, 0, 10.0, -72], [0, 0, 0, 1]]
    )
    X, _, _ = make_tfa_data(grid, n_samples=8, K=3, random_state=2)
    nib.save(nib.Nifti1Image(X.reshape(*grid, -1), aff), d / "mni.nii.gz")
    return _fit_quiet(
        d / "mni.nii.gz", K=3, standardize=False, max_iter=5, random_state=0
    )


def _panel_norms(fig):
    """(vmin, vmax) of every image and colorbar mesh in the figure."""
    norms = []
    for ax in fig.axes:
        for art in list(ax.images) + [
            c for c in ax.collections if type(c).__name__ == "QuadMesh"
        ]:
            norms.append((art.norm.vmin, art.norm.vmax))
    return norms


def _is_glass(fig):
    """Glass-brain panels draw the brain outline as PathPatches."""
    return any(type(a).__name__ == "PathPatch" for a in fig.findobj())


def _expected_vmax(res, sample, extra=()):
    s = res.subjects[0]
    orig = standardize_data(s.data, res.model.standardize)[:, sample]
    vals = [orig, res.reconstruct(0)[:, sample], *extra]
    return float(np.percentile(np.abs(np.concatenate(vals)), 99))


@pytest.mark.parametrize("view", ["glass", "slices"])
def test_plot_reconstruction_3d_shared_symmetric_limits(res3d, view):
    """Every image and colorbar of both panels uses [-vmax, vmax], vmax = p99."""
    fig = res3d.plot_reconstruction(0, sample=3, view=view)
    norms = _panel_norms(fig)
    vmax = _expected_vmax(res3d, 3)
    print(view, "norms:", norms, "expected vmax:", vmax)
    # 2 panels x 3 orthogonal views + 1 shared colorbar
    assert len(norms) == 7
    np.testing.assert_allclose(norms, [(-vmax, vmax)] * 7, rtol=1e-6)


def test_plot_reconstruction_3d_keeps_signs(res3d):
    """Slices show negative and positive values of the reconstruction."""
    recon = res3d.reconstruct(0)[:, 3]
    print("reconstruction range:", recon.min(), recon.max())
    assert recon.min() < 0 < recon.max()
    fig = res3d.plot_reconstruction(0, sample=3, view="slices")
    lows = [np.nanmin(im.get_array()) for ax in fig.axes for im in ax.images]
    print("per-slice minima:", lows)
    assert min(lows) < 0


def _has_background(fig):
    """Slices over a template draw two images (template, data) per slice axes."""
    return any(len(ax.images) >= 2 for ax in fig.axes)


def test_plot_reconstruction_view_auto(res3d, res_mni):
    """'auto' draws slices; MNI data get the template background, others none."""
    small = res3d.plot_reconstruction(0)
    mni = res_mni.plot_reconstruction(0)
    print("small grid: glass", _is_glass(small), "background", _has_background(small))
    print("MNI grid: glass", _is_glass(mni), "background", _has_background(mni))
    assert not _is_glass(small) and not _has_background(small)
    assert not _is_glass(mni) and _has_background(mni)
    glass = res_mni.plot_reconstruction(0, view="glass")
    assert _is_glass(glass) and not _has_background(glass)
    assert not _is_glass(res3d.plot_reconstruction(0, view="slices"))


def test_plot_reconstruction_one_colorbar_and_titles(res3d):
    """3D: one row per panel, legible titles and a single colorbar."""
    fig = res3d.plot_reconstruction(0, truth=res3d.reconstruct(0))
    titles = [ax.get_title(loc="left") for ax in fig.axes if ax.get_title(loc="left")]
    bars = [ax for ax in fig.axes if ax.collections and not ax.images]
    print("titles:", titles, "| colorbars:", len(bars))
    assert titles == ["original", "reconstruction", "truth"]
    assert len(bars) == 1


def test_plot_reconstruction_bad_view_raises(res3d):
    with pytest.raises(ValueError, match="view"):
        res3d.plot_reconstruction(0, view="surface")


def test_plot_reconstruction_truth_panel(res3d):
    """A (V, T) truth adds a third panel on the same scale; bad shapes raise."""
    truth = res3d.reconstruct(0) * 2
    fig = res3d.plot_reconstruction(0, sample=3, truth=truth, view="slices")
    vmax = _expected_vmax(res3d, 3, [truth[:, 3]])
    norms = _panel_norms(fig)
    print("norms:", norms, "expected vmax:", vmax)
    assert len(norms) == 10
    np.testing.assert_allclose(norms, [(-vmax, vmax)] * 10, rtol=1e-6)
    with pytest.raises(ValueError, match="truth"):
        res3d.plot_reconstruction(0, truth=truth[:-1])


@pytest.mark.parametrize("dim", ["2D", "2Dimage"])
def test_plot_reconstruction_2d_shared_limits_and_signs(dim, res2d, res2d_img):
    res = {"2D": res2d, "2Dimage": res2d_img}[dim]
    s = res.subjects[0]
    fig = res.plot_reconstruction(0, sample=2)
    orig = standardize_data(s.data, res.model.standardize)[:, 2]
    vmax = _expected_vmax(res, 2)
    panels = [a for a in fig.axes if a.get_title()]
    print(dim, "titles:", [a.get_title() for a in panels], "vmax:", vmax)
    assert [a.get_title() for a in panels] == ["original", "reconstruction"]
    assert len(fig.axes) == 3  # two panels and one shared colorbar
    mappables = [a.images[0] if a.images else a.collections[0] for a in panels]
    for m in mappables:
        print("norm:", m.norm.vmin, m.norm.vmax)
        assert (m.norm.vmin, m.norm.vmax) == pytest.approx((-vmax, vmax))
    shown = np.ravel(mappables[1].get_array())
    recon = res.reconstruct(0)[:, 2]
    # the values drawn are the signed reconstruction, not its absolute value
    np.testing.assert_allclose(np.sort(shown), np.sort(recon))
    np.testing.assert_allclose(
        np.sort(np.ravel(mappables[0].get_array())), np.sort(orig)
    )
    assert fig.axes[2].get_ylim() == pytest.approx((-vmax, vmax))


def _lr_fontsizes(fig):
    return [
        t.get_fontsize()
        for a in fig.axes
        for t in a.texts
        if t.get_text() in ("L", "R")
    ]


def test_3d_connectivity_in_given_axes_scales_labels(res3d, tmp_path):
    """In a small axes the L/R labels shrink with it so they clear the outlines."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    res3d.plot_connectivity(0, ax=axes[2])
    fig.savefig(tmp_path / "row.png", dpi=100)
    small = _lr_fontsizes(fig)
    fig, ax = plt.subplots(figsize=(6, 3))
    res3d.plot_connectivity(0, ax=ax)
    single = _lr_fontsizes(fig)
    own = _lr_fontsizes(res3d.plot_connectivity(0).frame_axes.figure)
    print("L/R font sizes: 1x3 row", small, "6x3 figure", single, "own figure", own)
    assert small and all(s <= 9 for s in small)
    assert all(s_ > s for s_, s in zip(single, small, strict=True))
    assert own and all(s == 12 for s in own)  # nilearn's default, unchanged


def _offsets(fig):
    """(row, column) of each scatter collection on the figure's first axes."""
    return [c.get_offsets()[:, ::-1] for c in fig.axes[0].collections]


def test_2d_node_plots_draw_display_nodes(res2d):
    """plot_nodes / plot_connectivity draw display_nodes; snap=False the fit."""
    r = res2d
    g, _, info = r.display_nodes()
    print("template moved (coordinate units):", info["distance"])
    off = _offsets(r.plot_nodes("global"))
    np.testing.assert_allclose(off[-1], g)  # template, drawn last
    for s in range(2):
        np.testing.assert_allclose(off[1 + s], r.display_nodes(s)[0])
    off = _offsets(r.plot_nodes("global", snap=False))
    np.testing.assert_allclose(off[-1], r.global_centers)
    np.testing.assert_allclose(off[1], r.centers[0])
    off = _offsets(r.plot_connectivity(1, threshold=0.0))
    np.testing.assert_allclose(off[-1], r.display_nodes(1)[0])
    off = _offsets(r.plot_connectivity(1, threshold=0.0, snap=False))
    np.testing.assert_allclose(off[-1], r.centers[1])
    # a given mask replaces the default
    mask = r.subjects[0].coords[r.subjects[0].coords[:, 0] < 7]
    off = _offsets(r.plot_connectivity(0, threshold=0.0, mask=mask))
    np.testing.assert_allclose(off[-1], r.display_nodes(0, mask=mask)[0])
    assert (off[-1][:, 0] < 7.5).all()


def test_3d_glass_brain_draws_display_nodes(res3d, tmp_path):
    """The 3D glass-brain plot equals plot_connectome at display_nodes."""
    from nilearn import plotting

    r = res3d
    conn = r.connectivity(0)
    np.fill_diagonal(conn, 0)
    c_eff, _, info = r.display_nodes(0)
    print("subject 0 moved (mm):", info["distance"], "ratio", info["width_ratio"])
    paths = {}
    for name, make in {
        "snap": lambda: r.plot_connectivity(0),
        "raw": lambda: r.plot_connectivity(0, snap=False),
        "ref_snap": lambda: plotting.plot_connectome(conn, c_eff, edge_threshold="75%"),
        "ref_raw": lambda: plotting.plot_connectome(
            conn, r.centers[0], edge_threshold="75%"
        ),
    }.items():
        paths[name] = tmp_path / f"{name}.png"
        make().savefig(paths[name])
        plt.close("all")
    arr = {k: np.asarray(Image.open(p).convert("RGB")) for k, p in paths.items()}
    np.testing.assert_array_equal(arr["snap"], arr["ref_snap"])
    np.testing.assert_array_equal(arr["raw"], arr["ref_raw"])
    if info["distance"].max() > 1.0:
        assert not np.array_equal(arr["snap"], arr["raw"])
