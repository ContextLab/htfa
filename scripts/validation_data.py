"""Shared data loading for the real-fMRI validation scripts.

Loads the adult participants of nilearn's ``development_fmri`` dataset (Pixar
"Partly Cloudy" viewing, 168 volumes, TR = 2 s, 4 mm MNI space) that are present
in nilearn's data directory, masks them with the MNI152 brain mask resampled to
the data grid (minus voxels whose mean intensity is below 10% of the median, which
lie outside the subject's brain, and voxels nearly constant over any
quarter of the run), and caches ``(X, coords, confounds)`` per subject as ``.npz`` in
the output directory given by ``HTFA_VALIDATION_OUT`` (default: ``./validation_out``).

Also provides the preprocessing shared by the scripts: per-segment confound
regression (`clean`) and z-scoring with parameters estimated on training samples
(`zscore_with`).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(os.environ.get("HTFA_VALIDATION_OUT", "validation_out"))
STATIC = Path(__file__).resolve().parents[1] / "docs" / "_static" / "validation"


def _data_dir() -> Path:
    from nilearn.datasets import get_data_dirs

    return Path(get_data_dirs()[0]) / "development_fmri"


def adult_ids() -> list[str]:
    """IDs of the adult participants whose preprocessed run is downloaded."""
    d = _data_dir()
    pheno = pd.read_csv(d / "participants.tsv", sep="\t")
    adults = pheno.loc[pheno["Child_Adult"] == "adult", "participant_id"]
    return [
        s
        for s in adults
        if (
            d / f"{s}_task-pixar_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        ).exists()
    ]


def load_subject(sid: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``X`` (V, T) float32, ``coords`` (V, 3) in mm and ``confounds`` (T, C)."""
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / f"{sid}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["X"], z["coords"], z["confounds"]
    import nibabel as nib
    from nilearn.datasets import load_mni152_brain_mask
    from nilearn.image import resample_to_img

    d = _data_dir()
    img = nib.load(
        d / f"{sid}_task-pixar_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    )
    mask_img = resample_to_img(
        load_mni152_brain_mask(resolution=2),
        img.slicer[..., 0],
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    data = np.asarray(img.dataobj, dtype=np.float32)
    mask = (np.asarray(mask_img.dataobj) > 0) & (data.std(axis=-1) > 0)
    # Voxels of the MNI mask outside the subject's own brain hold near-zero values
    # with near-zero variance; z-scoring would blow them up to unit-variance noise.
    mean = data.mean(axis=-1)
    mask &= mean > 0.1 * np.median(mean[mask])
    # Some edge voxels are constant over part of the run (any quarter).
    q = data.shape[-1] // 4
    for seg in (data[..., i * q : (i + 1) * q] for i in range(4)):
        sd = seg.std(axis=-1)
        mask &= sd > 0.01 * np.median(sd[mask])
    ijk = np.argwhere(mask)
    coords = nib.affines.apply_affine(img.affine, ijk)
    X = data[mask]
    conf = pd.read_csv(
        d / f"{sid}_task-pixar_desc-reducedConfounds_regressors.tsv", sep="\t"
    )
    confounds = conf.fillna(0.0).to_numpy(dtype=np.float64)
    np.savez(cache, X=X, coords=coords, confounds=confounds)
    return X, coords, confounds


def clean(X: np.ndarray, confounds: np.ndarray) -> np.ndarray:
    """Linear detrend plus confound regression of ``X`` (V, T) on its own samples."""
    T = X.shape[1]
    t = np.linspace(-1.0, 1.0, T)
    Z = np.column_stack([np.ones(T), t, confounds])
    Z = (Z - Z.mean(0)) / np.where(Z.std(0) > 0, Z.std(0), 1.0)
    Z[:, 0] = 1.0
    beta, *_ = np.linalg.lstsq(Z, X.T.astype(np.float64), rcond=None)
    return X - (Z @ beta).T


def zscore_with(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """``(X - mean) / std`` per voxel with given parameters (std 0 -> 1)."""
    return (X - mean[:, None]) / np.where(std > 0, std, 1.0)[:, None]


def load_all(
    n: int | None = None,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """``(id, X, coords, confounds)`` for the first ``n`` downloaded adults."""
    ids = adult_ids()[:n]
    return [(s, *load_subject(s)) for s in ids]


if __name__ == "__main__":
    for sid, X, R, c in load_all():
        print(sid, X.shape, R.shape, c.shape)
