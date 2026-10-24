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
