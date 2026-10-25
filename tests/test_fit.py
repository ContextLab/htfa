"""Tests for htfa.fit: numpy paths, errors and end-to-end NIfTI / PNG fits."""

import nibabel as nib
import numpy as np
import pytest
from PIL import Image

from htfa import HTFA, TFA, fit
from htfa._matching import match_factors
from htfa.synthetic import make_htfa_data, make_tfa_data

TFA_FAST = dict(standardize=False, random_state=0)
FAST = dict(max_global_iter=6, max_local_iter=10, **TFA_FAST)
NIFTI_GLOBAL_CENTER_TOL_MM = 0.6  # measured: 0.212 (max over 3 factors, seed 1)
PNG_CENTER_TOL_PX = 1.5  # measured: 0.593 (max over 3 factors, seed 2)


def test_fit_numpy_single_subject_returns_tfa():
    X, R, _ = make_tfa_data((8, 8, 8), n_samples=20, K=2, random_state=0)
    res = fit(X, K=2, coords=R, **TFA_FAST)
    assert isinstance(res.model, TFA)
    assert res.global_centers is None and len(res.centers) == 1
    assert res.reconstruct().shape == X.shape


# HTFA is deliberately stopped after a few iterations to keep the test fast
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_fit_numpy_multi_subject_returns_htfa():
    Xs, Rs, _ = make_htfa_data(2, (8, 8, 8), n_samples=20, K=2, random_state=0)
    res = fit(Xs, K=2, coords=Rs, **FAST)
    assert isinstance(res.model, HTFA)
    assert len(res.centers) == 2 and res.global_centers.shape == (2, 3)
    assert res.isfc().shape == (2, 2)


def test_fit_unknown_kwarg_raises():
    X, R, _ = make_tfa_data((6, 6, 6), n_samples=10, K=2, random_state=0)
    with pytest.raises(TypeError, match="bogus_kw"):
        fit(X, K=2, coords=R, bogus_kw=1)


def _write_nifti_folder(tmp_path, grid, affine, Xs):
    for i, X in enumerate(Xs):
        img = nib.Nifti1Image(X.reshape(*grid, -1).astype(np.float32), affine)
        nib.save(img, tmp_path / f"sub-{i:02d}.nii.gz")


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_fit_nifti_folder_end_to_end(tmp_path):
    grid = (12, 12, 12)
    Xs, Rs, truth = make_htfa_data(
        2, grid, n_samples=40, K=3, noise=0.05, random_state=1
    )
    affine = np.array(
        [[2.0, 0, 0, -10], [0, 2.0, 0, -12], [0, 0, 2.0, -8], [0, 0, 0, 1]]
    )
    _write_nifti_folder(tmp_path, grid, affine, Xs)
    res = fit(tmp_path, K=3, **FAST)
    # global centers are compared with the mean of the true subject centers
    mean_true = np.mean(truth["centers"], axis=0)
    true_mm = mean_true @ affine[:3, :3].T + affine[:3, 3]
    est = res.global_centers
    err = np.linalg.norm(est[match_factors(true_mm, est)] - true_mm, axis=1)
    print("global center errors (mm):", err)
    assert err.max() < NIFTI_GLOBAL_CENTER_TOL_MM
    img = res.to_nifti(0)
    assert img.shape == (*grid, 40)
    np.testing.assert_allclose(img.affine, affine)


def test_fit_png_stack_end_to_end(tmp_path):
    grid = (16, 16)
    X, R, truth = make_tfa_data(grid, n_samples=12, K=3, noise=0.05, random_state=2)
    X = (X - X.min()) / (X.max() - X.min())
    for t in range(X.shape[1]):
        arr = (X[:, t].reshape(grid) * 255).astype(np.uint8)
        Image.fromarray(arr).save(tmp_path / f"frame-{t:03d}.png")
    res = fit(tmp_path, K=3, stack=True, standardize=True, random_state=0)
    est = res.centers[0]
    true = truth["centers"]
    err = np.linalg.norm(est[match_factors(true, est)] - true, axis=1)
    print("center errors (px):", err)
    assert err.max() < PNG_CENTER_TOL_PX
    assert res.to_image(0, "factors").shape == (*grid, 3)


def test_fit_single_subject_rejects_htfa_kwarg():
    X, R, _ = make_tfa_data((6, 6, 6), n_samples=10, K=2, random_state=0)
    with pytest.raises(TypeError, match="max_global_iter"):
        fit(X, K=2, coords=R, max_global_iter=3)


def test_public_api():
    import htfa

    names = [
        "HTFA", "HTFAResults", "Subject", "TFA", "animate_network",
        "dynamic_connectivity", "fit", "isfc", "load", "make_htfa_data",
        "make_tfa_data", "plot_network", "plot_timecourse", "set_autoinstall",
        "static_connectivity",
    ]  # fmt: skip
    for name in names:
        assert hasattr(htfa, name), name
        assert name in htfa.__all__, name
