"""Tests for htfa.datasets (nilearn's development_fmri, preprocessed for HTFA)."""

import re
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
    # cached per subject (<id>-<12 hex key>.npz, no temporary files left),
    # and a second call reads the cache
    names = sorted(p.name for p in tmp_path.iterdir())
    print(names)
    assert len(names) == 2
    for name, sid in zip(names, sorted(ds.subject_ids), strict=True):
        assert re.fullmatch(rf"{sid}-[0-9a-f]{{12}}\.npz", name), name
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


@pytest.mark.network
def test_cache_key_and_corrupt_cache(tmp_path, monkeypatch):
    (sid, bold, conf), *_ = datasets.development_fmri_files(n_subjects=1)
    X, R, C = datasets.load_development_fmri_subject(bold, conf, tmp_path)
    (cached,) = tmp_path.iterdir()
    # a truncated cache file (as from a killed write) is recomputed and replaced
    cached.write_bytes(cached.read_bytes()[:1000])
    X2, R2, C2 = datasets.load_development_fmri_subject(bold, conf, tmp_path)
    np.testing.assert_array_equal(X2, X)
    np.testing.assert_array_equal(R2, R)
    np.testing.assert_array_equal(C2, C)
    with np.load(cached) as z:
        np.testing.assert_array_equal(z["X"], X)
    # other preprocessing settings get another cache file
    monkeypatch.setitem(datasets._PREPROCESSING, "min_mean_fraction", 0.2)
    X3, _, _ = datasets.load_development_fmri_subject(bold, conf, tmp_path)
    names = sorted(p.name for p in tmp_path.iterdir())
    print(names, X.shape, X3.shape)
    assert len(names) == 2 and all(n.startswith(sid) for n in names)
    assert X3.shape[0] < X.shape[0]  # a stricter mean threshold drops voxels


def _reference_preprocessing(bold, confounds):
    """The masking and cleaning of scripts/validation_data.py at commit 65badf5,
    copied verbatim (load_subject and clean), as the reference."""
    import nibabel as nib
    import pandas as pd
    from nilearn.datasets import load_mni152_brain_mask
    from nilearn.image import resample_to_img

    img = nib.load(bold)
    mask_img = resample_to_img(
        load_mni152_brain_mask(resolution=2),
        img.slicer[..., 0],
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    data = np.asarray(img.dataobj, dtype=np.float32)
    mask = (np.asarray(mask_img.dataobj) > 0) & (data.std(axis=-1) > 0)
    mean = data.mean(axis=-1)
    mask &= mean > 0.1 * np.median(mean[mask])
    q = data.shape[-1] // 4
    for seg in (data[..., i * q : (i + 1) * q] for i in range(4)):
        sd = seg.std(axis=-1)
        mask &= sd > 0.01 * np.median(sd[mask])
    ijk = np.argwhere(mask)
    coords = nib.affines.apply_affine(img.affine, ijk)
    X = data[mask]
    conf = pd.read_csv(confounds, sep="\t")
    conf = conf.fillna(0.0).to_numpy(dtype=np.float64)

    T = X.shape[1]
    t = np.linspace(-1.0, 1.0, T)
    Z = np.column_stack([np.ones(T), t, conf])
    Z = (Z - Z.mean(0)) / np.where(Z.std(0) > 0, Z.std(0), 1.0)
    Z[:, 0] = 1.0
    beta, *_ = np.linalg.lstsq(Z, X.T.astype(np.float64), rcond=None)
    return X, coords, conf, X - (Z @ beta).T


@pytest.mark.network
def test_matches_the_original_validation_preprocessing(tmp_path):
    (sid, bold, conf), *_ = datasets.development_fmri_files(n_subjects=1)
    X, R, C, cleaned = _reference_preprocessing(bold, conf)
    raw = datasets.fetch_development_fmri(n_subjects=1, clean=False, cache_dir=tmp_path)
    print(sid, X.shape, raw.data[0].shape)
    assert raw.subject_ids == [sid]
    new_arrays = (raw.data[0], raw.coords[0], raw.confounds[0])
    for new, ref in zip(new_arrays, (X, R, C), strict=True):
        assert new.dtype == ref.dtype and np.array_equal(new, ref)
    np.testing.assert_array_equal(datasets.regress_confounds(raw.data[0], C), cleaned)
    ds = datasets.fetch_development_fmri(n_subjects=1, cache_dir=tmp_path)
    np.testing.assert_array_equal(ds.data[0], cleaned.astype(np.float32))


def test_fetch_tutorial_fit_unknown_name(tmp_path):
    with pytest.raises(ValueError, match="unknown tutorial fit 'nope'"):
        datasets.fetch_tutorial_fit("nope", cache_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_fetch_tutorial_fit_rejects_a_cached_file_with_the_wrong_checksum(tmp_path):
    name = "development_fmri_6adults_K30"
    filename, expected = datasets.TUTORIAL_FITS[name]
    bad = tmp_path / filename
    np.savez(bad, weights=np.zeros((6, 30, 168)))  # a real .npz, but not the asset
    print("local file:", bad, "size:", bad.stat().st_size)
    with pytest.raises(ValueError, match="REFIT = True") as err:
        datasets.fetch_tutorial_fit(name, cache_dir=tmp_path)
    print("error:", err.value)
    assert expected in str(err.value)
    assert not bad.exists()  # deleted, so the next call downloads it again


@pytest.mark.network
def test_fetch_tutorial_fit_downloads_verifies_and_caches(tmp_path, monkeypatch):
    from htfa._download import DATA_DIR_ENV, sha256

    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    name = "development_fmri_6adults_K30"  # the smallest asset (0.2 MB)
    filename, expected = datasets.TUTORIAL_FITS[name]
    t0 = time.perf_counter()
    fit = datasets.fetch_tutorial_fit(name)
    print(f"download and load: {time.perf_counter() - t0:.1f} s")
    path = tmp_path / "tutorial_fits" / filename
    assert sha256(path) == expected
    assert [p.name for p in path.parent.iterdir()] == [filename]  # no .part left
    print({k: v.shape for k, v in fit.items()})
    assert fit["weights"].shape == (6, 30, 168)
    assert fit["centers"].shape == (6, 30, 3)
    assert fit["global_centers"].shape == (30, 3)
    assert int(fit["n_iter"]) == 6 and not bool(fit["converged"])
    assert list(fit["subject_ids"]) == [f"sub-pixar{i}" for i in range(123, 129)]
    assert len(str(fit["htfa_commit"])) == 40
    mtime = path.stat().st_mtime_ns
    again = datasets.fetch_tutorial_fit(name)  # from the cache
    assert path.stat().st_mtime_ns == mtime
    np.testing.assert_array_equal(again["weights"], fit["weights"])
