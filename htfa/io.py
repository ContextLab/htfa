"""Load NIfTI files, images, nibabel objects and arrays as :class:`Subject` objects."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

import nibabel as nib
import numpy as np

from htfa._lazy import lazy_import

NIFTI_EXTS = (".nii.gz", ".nii")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif")
TIFF_EXTS = (".tif", ".tiff")
LUMA = np.array([0.299, 0.587, 0.114])  # ITU-R 601


@dataclass
class Subject:
    """One subject's data in voxel/pixel-by-sample form.

    Parameters
    ----------
    data : numpy.ndarray
        Values, shape ``(V, T)``, float64.
    coords : numpy.ndarray
        Coordinates of the ``V`` locations, shape ``(V, D)``, float64.
    affine : numpy.ndarray or None
        ``(4, 4)`` voxel-to-world affine, for NIfTI-derived subjects.
    mask : numpy.ndarray or None
        Boolean mask in image space (``image_shape``) selecting the ``V`` locations.
    image_shape : tuple or None
        Shape of the image grid the data came from.
    source : str or None
        Path the data were read from, if any.

    Examples
    --------
    >>> import numpy as np
    >>> from htfa.io import load
    >>> (subject,) = load(np.random.rand(4, 5) + 1)
    >>> subject.data.shape, subject.coords.shape
    ((20, 1), (20, 2))
    """

    data: np.ndarray
    coords: np.ndarray
    affine: np.ndarray | None = None
    mask: np.ndarray | None = None
    image_shape: tuple[int, ...] | None = None
    source: str | None = None

    def to_image(self, values: np.ndarray) -> Any:
        """Place ``values`` back into image space, with NaN outside the mask.

        Parameters
        ----------
        values : numpy.ndarray
            Shape ``(V,)`` or ``(V, T)``.

        Returns
        -------
        nibabel.Nifti1Image or numpy.ndarray
            A NIfTI image when the subject has an affine, else an array of
            ``image_shape`` (with a trailing ``T`` axis for 2D ``values``).

        Examples
        --------
        >>> import numpy as np
        >>> from htfa.io import load
        >>> (subject,) = load(np.random.rand(4, 5) + 1)
        >>> subject.to_image(subject.data[:, 0]).shape
        (4, 5)
        """
        if self.image_shape is None or self.mask is None:
            raise ValueError("subject has no image space (loaded from (V, T) + coords)")
        values = np.asarray(values, dtype=np.float64)
        if values.ndim not in (1, 2) or values.shape[0] != self.data.shape[0]:
            raise ValueError(
                f"values must have shape ({self.data.shape[0]},) or "
                f"({self.data.shape[0]}, T); got {values.shape}"
            )
        out = np.full(self.image_shape + values.shape[1:], np.nan)
        out[self.mask] = values
        if self.affine is not None:
            return nib.Nifti1Image(out, self.affine)
        return out


@dataclass
class _Grid:
    values: np.ndarray  # spatial dims + (T,)
    affine: np.ndarray | None
    source: str | None
    coords: np.ndarray | None = None  # given coords: spatial dims == (V,)
    # mask=None drops all-zero locations (out-of-brain screening for volumes);
    # False for 2D images, where black pixels are data.
    screen_zeros: bool = True


def _natural(name: str) -> list[Any]:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name.lower())]


def _ext(name: str, exts: tuple[str, ...]) -> bool:
    return name.lower().endswith(exts)


def _from_nifti(img: Any, source: str | None) -> _Grid:
    values = np.asarray(img.get_fdata(dtype=np.float64))
    if values.ndim == 3:
        values = values[..., None]
    if values.ndim != 4:
        raise ValueError(f"NIfTI data must be 3D or 4D; got {values.ndim}D")
    return _Grid(values, np.asarray(img.affine, dtype=np.float64), source)


def _to_samples(arr: np.ndarray, channels: str) -> np.ndarray:
    """Image array ``(H, W[, C])`` to float64 ``(H, W, T)``."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 2:
        return arr[..., None]
    if channels == "separate":
        return arr
    if arr.shape[-1] >= 3:
        return (arr[..., :3] @ LUMA)[..., None]
    return arr[..., :1]


def _read_tiff(path: str, channels: str) -> np.ndarray:
    """TIFF to ``(H, W, T)``; page axes (not Y, X or S) are samples."""
    with lazy_import("tifffile", "images", "TIFF images").TiffFile(path) as tif:
        series = tif.series[0]
        axes, arr = series.axes, np.asarray(series.asarray())
    if "S" not in axes:
        arr, axes = arr[..., None], axes + "S"
    pages = [a for a in axes if a not in "YXS"]
    arr = np.transpose(arr, [axes.index(a) for a in (*pages, "Y", "X", "S")])
    arr = arr.reshape((-1, *arr.shape[-3:]))
    if arr.shape[-1] in (2, 4):  # alpha is dropped, as for PIL images
        arr = arr[..., : arr.shape[-1] - 1]
    return np.concatenate([_to_samples(page, channels) for page in arr], axis=-1)


def _read_image(path: str, channels: str) -> np.ndarray:
    if _ext(path, TIFF_EXTS):
        return _read_tiff(path, channels)
    pil = lazy_import("PIL.Image", "images", "reading images")
    with pil.open(path) as im:
        im.seek(0)
        if im.mode in ("P", "PA", "CMYK", "RGBA", "RGBX", "YCbCr"):
            im = im.convert("RGB")
        elif im.mode in ("1", "LA"):
            im = im.convert("L")
        arr = np.asarray(im)
    return _to_samples(arr, channels)


def _read_file(path: str, channels: str) -> _Grid:
    if _ext(path, NIFTI_EXTS):
        return _from_nifti(nib.load(path), path)
    if _ext(path, IMAGE_EXTS):
        return _Grid(_read_image(path, channels), None, path, screen_zeros=False)
    raise ValueError(f"unrecognized file type: {path}")


def _from_array(arr: Any, coords: Any) -> _Grid:
    arr = np.asarray(arr, dtype=np.float64)
    if coords is not None:
        coords = np.asarray(coords, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2 or coords.ndim != 2 or coords.shape[0] != arr.shape[0]:
            raise ValueError(
                f"with coords, data must be (V, T) matching coords (V, D); got "
                f"data {arr.shape}, coords {coords.shape}"
            )
        return _Grid(arr, None, None, coords)
    if arr.ndim not in (2, 3, 4):
        raise ValueError(f"cannot interpret a {arr.ndim}D array without coords")
    if arr.ndim == 2:
        return _Grid(arr[..., None], None, None, screen_zeros=False)
    return _Grid(arr[..., None] if arr.ndim < 4 else arr, None, None)


def _grids(item: Any, coords: Any, channels: str) -> list[_Grid]:
    if isinstance(item, (str, os.PathLike)):
        path = os.fspath(item)
        if os.path.isdir(path):
            names = sorted(
                (
                    n
                    for n in os.listdir(path)
                    if not n.startswith(".") and _ext(n, NIFTI_EXTS + IMAGE_EXTS)
                ),
                key=_natural,
            )
            if not names:
                raise ValueError(f"no NIfTI or image files in folder {path}")
            return [_read_file(os.path.join(path, n), channels) for n in names]
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return [_read_file(path, channels)]
    if isinstance(item, nib.spatialimages.SpatialImage):
        return [_from_nifti(item, None)]
    return [_from_array(item, coords)]


def _ijk(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(mask).astype(np.float64)


def _spacing(spacing: Any, ndim: int) -> np.ndarray:
    if spacing is None:
        return np.ones(ndim)
    sp = np.asarray(spacing, dtype=np.float64)
    if sp.ndim == 0:
        return np.full(ndim, float(sp))
    if sp.shape != (ndim,):
        raise ValueError(
            f"spacing must be a scalar or have {ndim} entries; got {sp.shape}"
        )
    return sp


def _mask_from_image(img: Any) -> np.ndarray:
    data = np.asarray(img.get_fdata())
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    return data != 0


def _resolve_mask(mask: Any, grid: _Grid) -> np.ndarray | None:
    """Spatial boolean mask for ``grid``."""
    spatial = grid.values.shape[:-1]
    if mask is None:
        v = grid.values
        finite = np.isfinite(v).all(-1)
        return finite & (v != 0).any(-1) if grid.screen_zeros else finite
    if isinstance(mask, str) and mask in ("epi", "background"):
        if grid.affine is None:
            raise ValueError(f"mask={mask!r} needs NIfTI input")
        masking = lazy_import("nilearn.masking", "nilearn", "automatic masks")
        func = (
            masking.compute_epi_mask
            if mask == "epi"
            else masking.compute_background_mask
        )
        out = _mask_from_image(func(nib.Nifti1Image(grid.values, grid.affine)))
    elif isinstance(mask, (str, os.PathLike)):
        out = _mask_from_image(nib.load(os.fspath(mask)))
    elif hasattr(mask, "fit") and hasattr(mask, "transform"):
        if getattr(mask, "mask_img_", None) is None:
            raise ValueError("the NiftiMasker must be fitted (mask_img_ is missing)")
        out = _mask_from_image(mask.mask_img_)
    elif isinstance(mask, nib.spatialimages.SpatialImage):
        out = _mask_from_image(mask)
    else:
        out = np.asarray(mask).astype(bool)
    if out.shape != spatial:
        raise ValueError(f"mask shape {out.shape} does not match data shape {spatial}")
    if not np.isfinite(grid.values[out]).all():
        raise ValueError("non-finite values inside the mask; adjust the mask or data")
    return out


def _subject(grid: _Grid, mask: Any, spacing: Any) -> Subject:
    keep = _resolve_mask(mask, grid)
    if grid.coords is not None:
        return Subject(
            grid.values[keep], grid.coords[keep], mask=keep, source=grid.source
        )
    assert keep is not None
    spatial = grid.values.shape[:-1]
    ijk = _ijk(keep)
    if grid.affine is not None:
        coords = ijk @ grid.affine[:3, :3].T + grid.affine[:3, 3]
    else:
        coords = ijk * _spacing(spacing, len(spatial))
    return Subject(
        grid.values[keep], coords, grid.affine, keep, tuple(spatial), grid.source
    )


def _stack(grids: list[_Grid]) -> _Grid:
    first = grids[0]
    for g in grids[1:]:
        if g.values.shape[:-1] != first.values.shape[:-1]:
            raise ValueError(
                f"cannot stack: shapes {first.values.shape[:-1]} and "
                f"{g.values.shape[:-1]} differ"
            )
        if (g.coords is None) != (first.coords is None):
            raise ValueError("cannot stack image data with (V, T) + coords data")
    return _Grid(
        np.concatenate([g.values for g in grids], axis=-1),
        first.affine,
        first.source,
        first.coords,
        first.screen_zeros,
    )


def load(
    data: Any,
    coords: Any = None,
    mask: Any = None,
    stack: bool = False,
    channels: str = "gray",
    spacing: Any = None,
) -> list[Subject]:
    """Load inputs into a list of :class:`Subject` objects.

    Parameters
    ----------
    data : path, folder, nibabel image, numpy.ndarray, or list of these
        Each file or list element is one subject. A 4D NIfTI is one subject's time
        series; a 3D NIfTI, 3D array or 2D image is one subject with one sample. A
        folder expands to its NIfTI/image files (natural filename order; other files
        are ignored). Without ``coords``, arrays are image grids: ``(H, W)``,
        ``(X, Y, Z)`` or ``(X, Y, Z, T)``. With ``coords`` a 2D array is ``(V, T)``.
    coords : array_like or list of array_like, optional
        ``(V, D)`` coordinates for ``(V, T)`` arrays; a list gives one per element.
    mask : None, path, nibabel image, fitted NiftiMasker, ndarray, 'epi' or 'background'
        ``None`` keeps voxels that are finite and not all zero across samples
        (NIfTI, 3D/4D arrays and ``(V, T)`` arrays); for 2D images (PNG, JPEG,
        TIFF, ... files and bare 2D arrays without ``coords``) it keeps every
        finite pixel, since black pixels are data there.
        Otherwise nonzero voxels of the mask are kept; ``'epi'`` and
        ``'background'`` use nilearn's automatic masks (NIfTI input only).
    stack : bool
        Treat every 3D volume / 2D image (all files, folder contents and list
        elements) as samples of ONE subject's time series.
    channels : {'gray', 'separate'}
        Color images become ITU-R 601 grayscale, or each channel becomes a sample.
    spacing : float or sequence of float, optional
        Scales pixel coordinates for images and bare arrays (not NIfTI).

    Returns
    -------
    list of Subject

    Raises
    ------
    ValueError
        On unrecognized inputs, shape mismatches, or subjects with different
        coordinate dimensionality.

    Examples
    --------
    >>> import numpy as np
    >>> from htfa.io import load
    >>> subjects = load([np.random.rand(4, 5) + 1, np.random.rand(4, 5) + 1])
    >>> len(subjects), subjects[0].data.shape
    (2, (20, 1))
    """
    if channels not in ("gray", "separate"):
        raise ValueError(f"channels must be 'gray' or 'separate'; got {channels!r}")
    items = list(data) if isinstance(data, (list, tuple)) else [data]
    per_item = (
        isinstance(coords, (list, tuple))
        and len(coords) == len(items)
        and all(np.ndim(c) == 2 for c in coords)
    )
    grids: list[_Grid] = []
    for i, item in enumerate(items):
        grids += _grids(item, coords[i] if per_item else coords, channels)
    if stack:
        grids = [_stack(grids)]
    subjects = [_subject(g, mask, spacing) for g in grids]
    dims = {s.coords.shape[1] for s in subjects}
    if len(dims) > 1:
        raise ValueError(
            f"subjects have different coordinate dimensions: {sorted(dims)}"
        )
    return subjects
