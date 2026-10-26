"""Tests for htfa.datasets (nilearn's development_fmri, preprocessed for HTFA)."""

import time

import numpy as np
import pytest

import htfa
from htfa import datasets


def test_regress_confounds_removes_trend_and_confounds():
    rng = np.random.default_rng(0)
    T, V = 120, 50
    confounds = rng.standard_normal((T, 3))
    signal = rng.standard_normal((V, T))
    trend = np.linspace(-1, 1, T)
    X = (
        signal
        + 5.0
        + rng.standard_normal((V, 1)) * trend
        + rng.standard_normal((V, 3)) @ confounds.T * 2.0
    ).astype(np.float32)
    out = datasets.regress_confounds(X, confounds)
    assert out.shape == X.shape
    Z = np.column_stack([np.ones(T), trend, confounds])
    # residuals are orthogonal to the intercept, trend and confounds
    resid_proj = np.abs(out @ Z).max()
    print("max |residual . regressor|:", resid_proj)
    assert resid_proj < 1e-6 * np.abs(X).sum()
    # and what is left is the part of the signal outside their span
    beta, *_ = np.linalg.lstsq(Z, signal.T, rcond=None)
    np.testing.assert_allclose(out, signal - (Z @ beta).T, atol=1e-4)


def test_regress_confounds_handles_constant_confound():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((10, 40))
    conf = np.column_stack([np.zeros(40), rng.standard_normal(40)])
    out = datasets.regress_confounds(X, conf)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out.mean(axis=1), 0, atol=1e-10)


def test_regress_confounds_shape_errors():
    with pytest.raises(ValueError, match="time points"):
        datasets.regress_confounds(np.zeros((5, 10)), np.zeros((9, 2)))


def test_datasets_is_reachable_from_package():
    assert htfa.datasets is datasets
    assert callable(htfa.datasets.fetch_development_fmri)


@pytest.mark.network
def test_fetch_development_fmri_adults(tmp_path):
    t0 = time.perf_counter()
    ds = datasets.fetch_development_fmri(n_subjects=2, cache_dir=tmp_path)
    first = time.perf_counter() - t0
    print(ds.subject_ids, [x.shape for x in ds.data], f"{first:.1f} s")
    assert len(ds.subject_ids) == len(ds.data) == len(ds.coords) == 2
    assert len(ds.confounds) == 2 and ds.tr == 2.0
    assert list(ds.phenotypic["Child_Adult"]) == ["adult", "adult"]
    assert list(ds.phenotypic["participant_id"]) == ds.subject_ids
    for X, R, C in zip(ds.data, ds.coords, ds.confounds, strict=True):
        V, T = X.shape
        assert T == 168 and X.dtype == np.float32 and V > 10000
        assert R.shape == (V, 3)
        assert np.abs(R).max() < 120  # MNI mm
        assert C.shape[0] == T and np.isfinite(C).all()
        # cleaned: every voxel's mean is regressed out, no constant voxels
        assert np.abs(X.mean(axis=1)).max() < 1e-2 * X.std()
        assert (X.std(axis=1) > 0).all()
    # cached per subject, and a second call reads the cache
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        f"{s}.npz" for s in ds.subject_ids
    )
    t0 = time.perf_counter()
    raw = datasets.fetch_development_fmri(n_subjects=2, cache_dir=tmp_path, clean=False)
    print(f"cached, raw: {time.perf_counter() - t0:.1f} s")
    assert raw.subject_ids == ds.subject_ids
    np.testing.assert_array_equal(raw.coords[0], ds.coords[0])
    # raw intensities are positive BOLD values; cleaning them gives the clean data
    assert raw.data[0].mean() > 100
    np.testing.assert_allclose(
        datasets.regress_confounds(raw.data[0], raw.confounds[0]),
        ds.data[0],
        atol=1e-3,
    )


@pytest.mark.network
def test_fetch_development_fmri_files_and_children(tmp_path):
    files = datasets.development_fmri_files(n_subjects=4, adults_only=False)
    print(files)
    assert len(files) == 4
    for sid, bold, conf in files:
        assert bold.name.startswith(sid) and bold.exists()
        assert conf.name.startswith(sid) and conf.exists()
    ds = datasets.fetch_development_fmri(
        n_subjects=4, adults_only=False, cache_dir=tmp_path
    )
    groups = set(ds.phenotypic["Child_Adult"])
    print(ds.subject_ids, groups)
    assert ds.subject_ids == [sid for sid, _, _ in files]
    assert groups == {"adult", "child"}
