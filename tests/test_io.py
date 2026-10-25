"""Tests for htfa.io: real files in tmp_path, no mocks."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from nibabel.affines import apply_affine
from PIL import Image

from htfa.io import Subject, load

AFFINE = np.array(
    [[0.0, -2.0, 0.0, 10.0], [2.5, 0.0, 0.0, -5.0], [0.0, 0.0, 3.0, 7.0], [0, 0, 0, 1]]
)


def _vol(shape, seed=0):
    return np.random.default_rng(seed).uniform(1, 2, shape)


def _save(path, arr, affine=AFFINE):
    nib.save(nib.Nifti1Image(np.asarray(arr, dtype=np.float32), affine), str(path))
    return path


def test_load_4d_nifti_coords_from_affine(tmp_path):
    arr = _vol((4, 3, 2, 5))
    p = _save(tmp_path / "a.nii", arr)
    (s,) = load(p)
    assert isinstance(s, Subject)
    assert s.data.dtype == np.float64 and s.coords.dtype == np.float64
    assert s.data.shape == (24, 5) and s.coords.shape == (24, 3)
    ijk = np.argwhere(s.mask)
    np.testing.assert_allclose(s.coords, apply_affine(AFFINE, ijk))
    np.testing.assert_allclose(s.affine, AFFINE)
    assert s.image_shape == (4, 3, 2)
    assert s.source == str(p)
    np.testing.assert_allclose(s.data, arr.astype(np.float32)[s.mask], rtol=1e-6)


def test_load_3d_nifti_is_one_sample(tmp_path):
    (s,) = load(_save(tmp_path / "a.nii", _vol((3, 3, 3))))
    assert s.data.shape == (27, 1)


def test_load_nifti_gz_and_folder_of_subjects(tmp_path):
    for i in (1, 2, 10):
        _save(tmp_path / f"s{i}.nii.gz", _vol((2, 2, 2), i))
    subs = load(tmp_path)
    assert len(subs) == 3
    assert [Path(s.source).name for s in subs] == [
        "s1.nii.gz",
        "s2.nii.gz",
        "s10.nii.gz",
    ]


def test_load_folder_stack_true(tmp_path):
    for i in range(10):
        _save(tmp_path / f"v{i:02d}.nii", _vol((2, 3, 2), i))
    (s,) = load(tmp_path, stack=True)
    assert s.data.shape == (12, 10)


def test_load_png_jpg_tiff(tmp_path):
    rgb = np.random.default_rng(1).integers(0, 256, (5, 6, 3), dtype=np.uint8)
    Image.fromarray(rgb).save(tmp_path / "c.png")
    Image.fromarray(rgb[..., 0]).save(tmp_path / "g.png")
    Image.fromarray(rgb).save(tmp_path / "c.jpg", quality=100)
    import tifffile

    t16 = np.random.default_rng(2).integers(0, 65535, (5, 6), dtype=np.uint16)
    tifffile.imwrite(tmp_path / "d.tif", t16)

    (g,) = load(tmp_path / "g.png")
    assert g.data.shape == (30, 1) and g.coords.shape == (30, 2)
    np.testing.assert_allclose(g.data[:, 0], rgb[..., 0].ravel().astype(float))

    (c,) = load(tmp_path / "c.png")
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    np.testing.assert_allclose(c.data[:, 0], luma.ravel())
    assert c.affine is None and c.image_shape == (5, 6)

    (sep,) = load(tmp_path / "c.png", channels="separate")
    assert sep.data.shape == (30, 3)
    np.testing.assert_allclose(sep.data, rgb.reshape(30, 3).astype(float))

    (j,) = load(tmp_path / "c.jpg")
    assert j.data.shape == (30, 1)

    (t,) = load(tmp_path / "d.tif")
    np.testing.assert_allclose(t.data[:, 0], t16.ravel().astype(float))
    assert t.data.max() > 255


def test_load_uint8_image_is_float64(tmp_path):
    Image.fromarray(np.full((3, 3), 200, dtype=np.uint8)).save(tmp_path / "g.png")
    (s,) = load(tmp_path / "g.png")
    assert s.data.dtype == np.float64 and s.coords.dtype == np.float64


def test_load_image_spacing(tmp_path):
    Image.fromarray(np.full((2, 3), 9, dtype=np.uint8)).save(tmp_path / "g.png")
    (s,) = load(tmp_path / "g.png", spacing=(2.0, 0.5))
    np.testing.assert_allclose(s.coords[-1], [2.0, 1.0])
    (s,) = load(tmp_path / "g.png", spacing=3)
    np.testing.assert_allclose(s.coords[-1], [3.0, 6.0])


def test_load_folder_of_images_stack(tmp_path):
    for i in range(4):
        Image.fromarray(np.full((3, 3), 10 * i + 1, dtype=np.uint8)).save(
            tmp_path / f"f{i}.png"
        )
    assert len(load(tmp_path)) == 4
    (s,) = load(tmp_path, stack=True)
    assert s.data.shape == (9, 4)


def test_load_nibabel_object_and_mixed_list(tmp_path):
    p = _save(tmp_path / "a.nii", _vol((3, 3, 3), 1))
    img = nib.Nifti1Image(_vol((3, 3, 3, 4), 2), AFFINE)
    arr = _vol((7, 6), 3)
    coords = _vol((7, 3), 4)
    subs = load([p, img, arr], coords=coords)
    assert len(subs) == 3
    assert [s.data.shape for s in subs] == [(27, 1), (27, 4), (7, 6)]
    np.testing.assert_allclose(subs[2].coords, coords)
    assert subs[2].image_shape is None


def test_load_arrays_without_coords():
    (v,) = load(_vol((3, 4, 5)))
    assert v.data.shape == (60, 1) and v.coords.shape == (60, 3)
    (im,) = load(_vol((3, 4)))
    assert im.data.shape == (12, 1) and im.coords.shape == (12, 2)
    (t,) = load(_vol((3, 4, 5, 6)))
    assert t.data.shape == (60, 6)
    (st,) = load([_vol((3, 4), i) for i in range(5)], stack=True)
    assert st.data.shape == (12, 5)


def test_load_niftimasker_mask(tmp_path):
    from nilearn.maskers import NiftiMasker

    arr = _vol((4, 4, 4, 3))
    p = _save(tmp_path / "a.nii", arr)
    m = np.zeros((4, 4, 4))
    m[1:3, 1:3, 1:3] = 1
    mask_img = nib.Nifti1Image(m, AFFINE)
    masker = NiftiMasker(mask_img=mask_img).fit()
    (s,) = load(p, mask=masker)
    assert s.data.shape == (8, 3)
    (s2,) = load(p, mask=mask_img)
    np.testing.assert_array_equal(s.mask, s2.mask)
    _save(tmp_path / "m.nii", m)
    (s3,) = load(p, mask=tmp_path / "m.nii")
    np.testing.assert_array_equal(s.mask, s3.mask)
    with pytest.raises(ValueError, match="fitted"):
        load(p, mask=NiftiMasker())


def test_load_nilearn_epi_and_background_masks(tmp_path):
    arr = np.zeros((10, 10, 10, 3))
    arr[2:8, 2:8, 2:8] = _vol((6, 6, 6, 3)) * 100
    p = _save(tmp_path / "a.nii", arr, np.eye(4))
    for kind in ("epi", "background"):
        (s,) = load(p, mask=kind)
        assert 0 < s.data.shape[0] <= 1000
        assert s.mask.shape == (10, 10, 10)


def test_load_nifti_masks_nan_and_zero_voxels(tmp_path):
    arr = _vol((3, 3, 3, 4))
    arr[0, 0, 0] = 0
    arr[1, 1, 1, 2] = np.nan
    (s,) = load(_save(tmp_path / "a.nii", arr))
    assert not s.mask[0, 0, 0] and not s.mask[1, 1, 1]
    assert s.data.shape == (25, 4) and np.isfinite(s.data).all()


def test_load_nan_inside_explicit_mask_raises(tmp_path):
    arr = _vol((3, 3, 3, 2))
    arr[1, 1, 1, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        load(_save(tmp_path / "a.nii", arr), mask=np.ones((3, 3, 3), bool))


def test_load_folder_ignores_sidecars_and_case(tmp_path):
    _save(tmp_path / "a.NII.GZ", _vol((2, 2, 2)))
    _save(tmp_path / "b.nii", _vol((2, 2, 2)))
    (tmp_path / "a.json").write_text("{}")
    (tmp_path / ".DS_Store").write_bytes(b"x")
    assert len(load(tmp_path)) == 2
    import tifffile

    d = tmp_path / "t"
    d.mkdir()
    tifffile.imwrite(d / "x.TIF", np.ones((3, 3), dtype=np.uint16))
    (d / "x.json").write_text("{}")
    assert len(load(d)) == 1


def test_to_image_roundtrip(tmp_path):
    arr = _vol((3, 3, 3, 4))
    arr[0, 0, 0] = 0
    (s,) = load(_save(tmp_path / "a.nii", arr))
    out = s.to_image(s.data)
    assert isinstance(out, nib.Nifti1Image)
    got = out.get_fdata()
    np.testing.assert_allclose(got[s.mask], arr.astype(np.float32)[s.mask], rtol=1e-6)
    assert np.isnan(got[0, 0, 0]).all()
    np.testing.assert_allclose(out.affine, AFFINE)
    one = s.to_image(s.data[:, 0])
    assert one.shape == (3, 3, 3)


def test_to_image_array_without_affine():
    img = _vol((4, 5))
    img[0, 0] = np.nan  # outside the default mask (zeros are kept for 2D images)
    (s,) = load(img)
    out = s.to_image(s.data[:, 0])
    assert isinstance(out, np.ndarray) and out.shape == (4, 5)
    assert np.isnan(out[0, 0])
    np.testing.assert_allclose(out[s.mask.reshape(4, 5)], img[np.isfinite(img)])


def test_to_image_requires_image_space():
    (s,) = load(_vol((5, 2)), coords=_vol((5, 3)))
    with pytest.raises(ValueError, match="image"):
        s.to_image(s.data)


def test_load_rejects_mixed_dimensionality(tmp_path):
    with pytest.raises(ValueError, match="dimension"):
        load([_vol((3, 3, 3)), _vol((3, 3))])


def test_load_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "missing.nii")
    with pytest.raises(ValueError, match="no NIfTI or image files"):
        load(tmp_path)
    (tmp_path / "x.txt").write_text("hi")
    with pytest.raises(ValueError, match="unrecognized"):
        load(tmp_path / "x.txt")
    with pytest.raises(ValueError, match="channels"):
        load(_vol((3, 3)), channels="rgb")
    with pytest.raises(ValueError, match="spacing"):
        load(_vol((3, 3)), spacing=(1, 2, 3))


def test_load_tiff_pages_and_channels(tmp_path):
    import tifffile

    rng = np.random.default_rng(5)
    pages = rng.integers(0, 60000, (3, 4, 3), dtype=np.uint16)  # 3 pages, width 3
    tifffile.imwrite(tmp_path / "pages.tif", pages, photometric="minisblack")
    (s,) = load(tmp_path / "pages.tif")
    assert s.data.shape == (12, 3) and s.image_shape == (4, 3)
    np.testing.assert_allclose(s.data, np.moveaxis(pages, 0, -1).reshape(12, 3))

    rgb = rng.integers(0, 256, (4, 5, 3), dtype=np.uint8)
    luma = (rgb * np.array([0.299, 0.587, 0.114])).sum(-1).ravel()
    tifffile.imwrite(tmp_path / "rgb.tif", rgb, photometric="rgb")
    tifffile.imwrite(
        tmp_path / "planar.tif",
        np.moveaxis(rgb, -1, 0),
        photometric="rgb",
        planarconfig="separate",
    )
    for name in ("rgb.tif", "planar.tif"):
        (s,) = load(tmp_path / name)
        np.testing.assert_allclose(s.data[:, 0], luma)
        (s,) = load(tmp_path / name, channels="separate")
        np.testing.assert_allclose(s.data, rgb.reshape(20, 3))

    multi = rng.integers(0, 256, (2, 4, 5, 3), dtype=np.uint8)
    tifffile.imwrite(tmp_path / "multi.tif", multi, photometric="rgb")
    (s,) = load(tmp_path / "multi.tif")
    assert s.data.shape == (20, 2)
    (s,) = load(tmp_path / "multi.tif", channels="separate")
    assert s.data.shape == (20, 6)

    ga = rng.integers(0, 256, (4, 5, 2), dtype=np.uint8)
    tifffile.imwrite(
        tmp_path / "ga.tif", ga, extrasamples=["unassalpha"], photometric="minisblack"
    )
    (s,) = load(tmp_path / "ga.tif")
    np.testing.assert_allclose(s.data[:, 0], ga[..., 0].ravel())


def test_load_coords_default_screening():
    data = _vol((6, 3))
    data[1] = 0
    data[4, 2] = np.nan
    coords = _vol((6, 3), 9)
    (s,) = load(data, coords=coords)
    keep = [0, 2, 3, 5]
    np.testing.assert_allclose(s.data, data[keep])
    np.testing.assert_allclose(s.coords, coords[keep])


def test_load_folder_skips_dot_files(tmp_path):
    _save(tmp_path / "a.nii", _vol((2, 2, 2)))
    (tmp_path / "._a.nii").write_bytes(b"junk")
    assert len(load(tmp_path)) == 1


def test_default_mask_keeps_black_pixels_of_2d_images(tmp_path):
    """mask=None keeps every finite pixel of an image (black pixels are data)."""
    img = np.random.default_rng(0).integers(1, 256, (30, 30)).astype(np.uint8)
    img[:3, :3] = 0  # 9 black pixels
    rgb = np.repeat(img[..., None], 3, axis=-1)
    tif = np.stack([img, img])
    Image.fromarray(img).save(tmp_path / "a.png")
    Image.fromarray(rgb).save(tmp_path / "b.png")
    import tifffile

    tifffile.imwrite(tmp_path / "c.tif", tif)
    for subject in [
        *load([tmp_path / "a.png", tmp_path / "b.png", tmp_path / "c.tif"]),
        *load(img),  # a bare 2D array without coords is an image too
        *load(tmp_path, stack=True),
    ]:
        print(subject.source, subject.data.shape)
        assert subject.data.shape[0] == 900
        assert subject.mask.all()
    bad = img.astype(np.float64)
    bad[5, 5] = np.nan
    (subject,) = load(bad)
    assert subject.data.shape == (899, 1) and not subject.mask[5, 5]


def test_default_mask_still_drops_zero_voxels_of_volumes(tmp_path):
    vol = _vol((4, 4, 4, 3))
    vol[0, 0, 0] = 0.0
    nib.save(nib.Nifti1Image(vol, AFFINE), tmp_path / "v.nii.gz")
    for subject in (load(tmp_path / "v.nii.gz")[0], load(vol)[0]):
        assert subject.data.shape == (63, 3) and not subject.mask[0, 0, 0]
