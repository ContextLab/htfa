"""Shared component geometry for network plots (one edge anchor per component)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import nibabel as nib
import numpy as np

from ._kernels import Kernel


@dataclass(frozen=True)
class ComponentNodes:
    """Flattened components in factor-major order, retaining their factor IDs."""

    kernel: str
    centers: np.ndarray
    widths: np.ndarray
    mixing: np.ndarray
    parents: np.ndarray

    @classmethod
    def from_shape(cls, shape: tuple) -> ComponentNodes:
        name, centers, widths, mixing = shape
        centers, widths, mixing = (
            np.asarray(v, dtype=float) for v in (centers, widths, mixing)
        )
        if (
            centers.ndim != 3
            or widths.shape != centers.shape[:2]
            or mixing.shape != widths.shape
        ):
            raise ValueError(
                "component centers, widths and mixing weights have incompatible shapes"
            )
        if (
            not all(np.all(np.isfinite(v)) for v in (centers, widths, mixing))
            or np.any(widths <= 0)
            or np.any(mixing <= 0)
        ):
            raise ValueError(
                "components must be finite with positive widths and mixing weights"
            )
        Kernel(name, centers.shape[1])
        return cls(
            name,
            centers.reshape(-1, centers.shape[-1]),
            widths.ravel(),
            mixing.ravel(),
            np.repeat(np.arange(len(centers)), centers.shape[1]),
        )

    def connectivity(self, values: Any) -> np.ndarray:
        """Repeat each factor edge unchanged; omit edges within the same factor."""
        a = np.asarray(values, dtype=float)
        K = int(self.parents.max()) + 1
        if a.ndim not in (2, 3) or a.shape[-2:] != (K, K):
            raise ValueError(f"connectivity must end in ({K}, {K}), got {a.shape}")
        out = a[..., self.parents[:, None], self.parents[None, :]].copy()
        out[..., self.parents[:, None] == self.parents[None, :]] = 0
        return out

    def values(self, coords: np.ndarray, index: int) -> np.ndarray:
        """Signed, mixing-weighted component profile at world coordinates."""
        z = np.sum((coords - self.centers[index]) ** 2, axis=1) / self.widths[index]
        return self.mixing[index] * Kernel(self.kernel).profile(z, coords.shape[1])

    def images(self, mask: Any) -> Iterator[tuple[int, Any]]:
        """Signed component images on the mask's native grid (zero outside).

        Coordinate masks must be an axis-aligned regular lattice. NIfTI masks
        retain their affine, including rotations and anisotropic voxel sizes.
        Each image is made separately, avoiding a V-by-K-by-M allocation.
        """
        coords, indices, shape, affine = _grid(mask)
        for i in range(len(self.centers)):
            data = np.zeros(shape)
            data[tuple(indices.T)] = self.values(coords, i)
            yield i, nib.Nifti1Image(data, affine)


def component_mode(mode: str, has_shape: bool) -> bool:
    if mode not in ("auto", "components", "summary"):
        raise ValueError("node_mode must be 'auto', 'components' or 'summary'")
    return mode == "components" or (mode == "auto" and has_shape)


def check_level(level: float) -> float:
    if not np.isfinite(level) or not 0 < level < 1:
        raise ValueError(
            "lobe_level must be a finite fraction strictly between 0 and 1"
        )
    return float(level)


def _grid(mask: Any) -> tuple[np.ndarray, np.ndarray, tuple, np.ndarray]:
    if isinstance(mask, tuple) or (
        hasattr(mask, "affine") and hasattr(mask, "dataobj")
    ):
        volume, affine = (
            mask if isinstance(mask, tuple) else (mask.dataobj, mask.affine)
        )
        volume, affine = np.asarray(volume), np.asarray(affine, dtype=float)
        if (
            volume.ndim != 3
            or affine.shape != (4, 4)
            or not np.all(np.isfinite(affine))
        ):
            raise ValueError("component lobes need a 3D mask and finite 4x4 affine")
        indices = np.argwhere(np.isfinite(volume) & (volume != 0))
        shape = volume.shape
        coords = nib.affines.apply_affine(affine, indices)
    else:
        from .brainnet import _grid_spacing

        coords = np.asarray(mask, dtype=float)
        if (
            coords.ndim != 2
            or coords.shape[1] != 3
            or len(coords) < 2
            or not np.all(np.isfinite(coords))
        ):
            raise ValueError("component lobes need finite (V, 3) mask coordinates")
        spacing = _grid_spacing(coords)
        origin = coords.min(axis=0)
        index = (coords - origin) / spacing
        if not np.allclose(index, np.round(index), atol=1e-4, rtol=0):
            raise ValueError(
                "component mask coordinates must form a regular grid; "
                "pass a NIfTI mask to preserve a rotated affine"
            )
        indices = np.round(index).astype(int)
        shape = tuple(indices.max(axis=0) + 1)
        affine = np.diag([*spacing, 1.0])
        affine[:3, 3] = origin
    if not len(indices):
        raise ValueError("mask has no voxels")
    if np.prod(shape) > 16_000_000:
        raise ValueError(
            "component mask bounding grid exceeds 16 million voxels; use a coarser mask"
        )
    return coords, indices, shape, affine
