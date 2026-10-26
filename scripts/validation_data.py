"""Shared data loading for the real-fMRI validation scripts.

The data are the first twelve adult participants of nilearn's
``development_fmri`` dataset (Pixar "Partly Cloudy" viewing, 168 volumes,
TR = 2 s, 4 mm MNI space), downloaded if missing and masked by `htfa.datasets`
(see `htfa.datasets.fetch_development_fmri` for the mask), and cached as ``.npz``
in the output directory given by ``HTFA_VALIDATION_OUT`` (default:
``./validation_out``).

Also provides the preprocessing shared by the scripts: per-segment confound
regression (`clean`, which is `htfa.datasets.regress_confounds`) and z-scoring
with parameters estimated on training samples (`zscore_with`).
"""

import os
from functools import cache
from pathlib import Path

import numpy as np

from htfa.datasets import development_fmri_files, load_development_fmri_subject
from htfa.datasets import regress_confounds as clean

OUT = Path(os.environ.get("HTFA_VALIDATION_OUT", "validation_out"))
STATIC = Path(__file__).resolve().parents[1] / "docs" / "_static" / "validation"
N_ADULTS = 12  # sub-pixar123 to sub-pixar134 (docs/validation.rst)

__all__ = [
    "N_ADULTS",
    "OUT",
    "STATIC",
    "adult_ids",
    "clean",
    "load_all",
    "load_subject",
    "zscore_with",
]


@cache
def _files() -> dict[str, tuple[Path, Path]]:
    return {s: (bold, conf) for s, bold, conf in development_fmri_files(N_ADULTS)}


def adult_ids() -> list[str]:
    """IDs of the ``N_ADULTS`` adults (their runs are downloaded if missing)."""
    return list(_files())


def load_subject(sid: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``X`` (V, T) float32, ``coords`` (V, 3) in mm and ``confounds`` (T, C)."""
    return load_development_fmri_subject(*_files()[sid], cache_dir=OUT)


def zscore_with(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """``(X - mean) / std`` per voxel with given parameters (std 0 -> 1)."""
    return (X - mean[:, None]) / np.where(std > 0, std, 1.0)[:, None]


def load_all(
    n: int | None = None,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """``(id, X, coords, confounds)`` for the first ``n`` adults."""
    return [(s, *load_subject(s)) for s in adult_ids()[:n]]


if __name__ == "__main__":
    for sid, X, R, c in load_all():
        print(sid, X.shape, R.shape, c.shape)
