"""Example fMRI data, preprocessed for HTFA.

`fetch_development_fmri` downloads (on first use) nilearn's ``development_fmri``
dataset (Richardson et al., 2018, Nature Communications 9: 1027): 155 children and
adults watching the Pixar short "Partly Cloudy", with 168 brain volumes sampled
every two seconds (the repetition time, TR) in 4 mm MNI anatomical reference
space, with fMRIPrep confounds. This is an illustrative movie-viewing dataset,
not an input requirement of HTFA; other studies can use `htfa.load` or arrays.
Each run is masked with the MNI152 brain mask
resampled to the data grid, minus voxels outside the subject's own brain (mean
intensity below 10% of the median) and voxels nearly constant over any quarter of
the run, and cached as ``.npz``. `regress_confounds` then removes a linear trend
and the confounds.

nilearn (the ``nilearn`` extra) is installed on first use.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._download import data_dir, sha256
from ._lazy import lazy_import

Array = np.ndarray

__all__ = [
    "DevelopmentFMRI",
    "TUTORIAL_FITS",
    "development_fmri_files",
    "fetch_development_fmri",
    "fetch_tutorial_fit",
    "load_development_fmri_subject",
    "regress_confounds",
]

TUTORIAL_DATA_URL = (
    "https://github.com/ContextLab/htfa/releases/download/tutorial-data-v1/"
)
# Saved results of the tutorials (HTFA fits; tutorial 09's embeddings and
# permutation nulls): name -> (file in the tutorial-data-v1 pre-release of the
# htfa repository, SHA-256).
TUTORIAL_FITS = {
    "development_fmri_10children_10adults_K30": (
        "htfa_development_fmri_10children_10adults_K30-01a094f61846.npz",
        "01a094f618466ae4400c4d3583342f64496e84a4733028a95399131b1bd9b468",
    ),
    "development_fmri_children_adults_null_100": (
        "htfa_development_fmri_children_adults_null_100-5bf2ae0b61f7.npz",
        "5bf2ae0b61f729f10be2820ba84cdd5f00448f5b51910024b66d7cca821b30eb",
    ),
    "mitchell2008_K100": (
        "htfa_mitchell2008_K100-de0427fbc847.npz",
        "de0427fbc8476fb5b0233b17eae0bf7f116d874287667b0cb7a285bd2d01b92c",
    ),
    "development_fmri_12adults_K50": (
        "htfa_development_fmri_12adults_K50-939fb70901ca.npz",
        "939fb70901cacb0d800133014d7366e51e28a8b609ca6e5cb1b2f0e991175e61",
    ),
    "development_fmri_6adults_K30": (
        "htfa_development_fmri_6adults_K30-b1b11b99f829.npz",
        "b1b11b99f8296348546b1ad570ee9f89290df0806e2b89844d4d2c346c5de488",
    ),
    "mitchell2008_embeddings": (
        "htfa_mitchell2008_embeddings_Qwen3-Embedding-0.6B.npz",
        "0f61b70541d0e5c3cb428fb9b81f1cb98df2a835bb76cbd7c76d9f13ce6b2e96",
    ),
    "mitchell2008_null_20": (
        "htfa_mitchell2008_null_20-47da3a734a49.npz",
        "47da3a734a49253a82cdceebc94879989543202fbe31a2aa0ba82677fe8f75b0",
    ),
    "mitchell2008_null_100": (
        "htfa_mitchell2008_null_100-3708d4eac165.npz",
        "3708d4eac165fb5aaa5510cd6f8b1f9189d61842e5e1ecb3dfbf1bfff91e1e7d",
    ),
}

TR = 2.0
# Preprocessing settings; they are part of the cache key, so changing any of them
# (or bumping the version after a code change) recomputes the cached runs.
_PREPROCESSING = {
    "version": 1,
    "mni_mask_resolution": 2,
    "min_mean_fraction": 0.1,  # of the median voxel mean
    "min_quarter_sd_fraction": 0.01,  # of the median voxel SD, per quarter
}


@dataclass
class DevelopmentFMRI:
    """Preprocessed ``development_fmri`` subjects (see `fetch_development_fmri`).

    Attributes
    ----------
    subject_ids : list of str
        Participant IDs, for example ``"sub-pixar123"``.
    data : list of arrays of shape (V_s, 168), float32
        In-brain voxel time series per subject.
    coords : list of arrays of shape (V_s, 3)
        Voxel coordinates in MNI mm.
    confounds : list of arrays of shape (168, C)
        fMRIPrep's reduced confound regressors (NaN set to 0).
    phenotypic : pandas.DataFrame
        One row per subject (``participant_id``, ``Age``, ``Child_Adult``, ...).
    tr : float
        Repetition time in seconds (2.0).
    """

    subject_ids: list[str]
    data: list[Array]
    coords: list[Array]
    confounds: list[Array]
    phenotypic: Any
    tr: float = TR


def _datasets() -> Any:
    return lazy_import("nilearn.datasets", "nilearn", "the development_fmri dataset")


def development_fmri_files(
    n_subjects: int | None = None,
    adults_only: bool = True,
    data_dir: str | os.PathLike | None = None,
) -> list[tuple[str, Path, Path]]:
    """Download (if needed) development_fmri runs and return their files.

    Parameters
    ----------
    n_subjects : int, optional
        Number of subjects; all of the age group by default (33 adults, or 155
        children and adults).
    adults_only : bool
        Only the adults (default). ``False`` takes children and adults in
        nilearn's proportions.
    data_dir : path, optional
        nilearn's data directory (default: nilearn's, usually ``~/nilearn_data``).

    Returns
    -------
    list of (str, Path, Path)
        ``(subject_id, bold_file, confounds_file)``, adults first.
    """
    data = _datasets().fetch_development_fmri(
        n_subjects=n_subjects,
        age_group="adult" if adults_only else "both",
        data_dir=None if data_dir is None else str(data_dir),
        verbose=0,
    )
    files = []
    for bold, conf in zip(data.func, data.confounds, strict=True):
        bold = Path(bold)
        files.append((bold.name.split("_")[0], bold, Path(conf)))
    return files


def load_development_fmri_subject(
    bold: str | os.PathLike,
    confounds: str | os.PathLike,
    cache_dir: str | os.PathLike | None = None,
) -> tuple[Array, Array, Array]:
    """Mask one development_fmri run (cached), without cleaning it.

    Parameters
    ----------
    bold : path
        The run's ``*_desc-preproc_bold.nii.gz`` file.
    confounds : path
        Its ``*_desc-reducedConfounds_regressors.tsv`` file.
    cache_dir : path, optional
        Where ``<subject_id>-<key>.npz`` is cached (default: ``htfa`` next to
        ``bold``). The key changes with the preprocessing settings and the input
        files; an unreadable cache file is recomputed.

    Returns
    -------
    X : array of shape (V, T), float32
        In-brain voxel time series (see the module description for the mask).
    coords : array of shape (V, 3)
        Voxel coordinates in MNI mm.
    confounds : array of shape (T, C)
        Confound regressors, NaN set to 0.
    """
    bold, confounds = Path(bold), Path(confounds)
    cache = Path(cache_dir) if cache_dir is not None else bold.parent / "htfa"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / _cache_name(bold, confounds)
    if path.exists():
        try:
            with np.load(path) as z:
                return z["X"], z["coords"], z["confounds"]
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            pass  # unreadable (for example a truncated write): recompute
    import nibabel as nib

    datasets = _datasets()
    image = lazy_import("nilearn.image", "nilearn", "the development_fmri dataset")
    pd = lazy_import("pandas", "nilearn", "the development_fmri dataset")
    img: Any = nib.load(bold)
    mask_img = image.resample_to_img(
        datasets.load_mni152_brain_mask(
            resolution=_PREPROCESSING["mni_mask_resolution"]
        ),
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
    mask &= mean > _PREPROCESSING["min_mean_fraction"] * np.median(mean[mask])
    # Some edge voxels are constant over part of the run (any quarter).
    q = data.shape[-1] // 4
    for seg in (data[..., i * q : (i + 1) * q] for i in range(4)):
        sd = seg.std(axis=-1)
        mask &= sd > _PREPROCESSING["min_quarter_sd_fraction"] * np.median(sd[mask])
    coords = nib.affines.apply_affine(img.affine, np.argwhere(mask))
    X = data[mask]
    conf = pd.read_csv(confounds, sep="\t").fillna(0.0).to_numpy(dtype=np.float64)
    _save_atomic(path, X=X, coords=coords, confounds=conf)
    return X, coords, conf


def _cache_name(bold: Path, confounds: Path) -> str:
    """``<subject>-<key>.npz``; the key covers the settings and the input files."""
    key = json.dumps(
        [_PREPROCESSING, [(f.name, f.stat().st_size) for f in (bold, confounds)]],
        sort_keys=True,
    )
    return (
        f"{bold.name.split('_')[0]}-{hashlib.sha256(key.encode()).hexdigest()[:12]}.npz"
    )


def _save_atomic(path: Path, X: Array, coords: Array, confounds: Array) -> None:
    """``np.savez`` to a temporary file renamed onto ``path`` (no partial files)."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.stem, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez(f, X=X, coords=coords, confounds=confounds)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def regress_confounds(X: Array, confounds: Array) -> Array:
    """Remove an intercept, a linear trend and ``confounds`` from ``X``.

    Ordinary least squares of each voxel on ``[1, t, confounds]`` (the
    regressors standardized), fitted on the samples of ``X`` itself.

    Parameters
    ----------
    X : array of shape (V, T)
        Voxel time series.
    confounds : array of shape (T, C)
        Confound regressors.

    Returns
    -------
    array of shape (V, T), float64
        The residuals.

    Examples
    --------
    >>> import numpy as np
    >>> from htfa.datasets import regress_confounds
    >>> X = np.arange(10.0)[None, :] + 3.0
    >>> np.allclose(regress_confounds(X, np.zeros((10, 1))), 0)
    True
    """
    T = X.shape[1]
    if confounds.shape[0] != T:
        raise ValueError(
            f"X has {T} time points but confounds has {confounds.shape[0]}"
        )
    t = np.linspace(-1.0, 1.0, T)
    Z = np.column_stack([np.ones(T), t, confounds])
    sd = Z.std(0)
    Z = (Z - Z.mean(0)) / np.where(sd > 0, sd, 1.0)
    Z[:, 0] = 1.0
    beta, *_ = np.linalg.lstsq(Z, X.T.astype(np.float64), rcond=None)
    return X - (Z @ beta).T


def fetch_development_fmri(
    n_subjects: int | None = None,
    adults_only: bool = True,
    clean: bool = True,
    data_dir: str | os.PathLike | None = None,
    cache_dir: str | os.PathLike | None = None,
) -> DevelopmentFMRI:
    """nilearn's development_fmri runs, masked and cleaned for HTFA.

    In this example dataset, children and adults watch Pixar's "Partly Cloudy"
    during fMRI acquisition. Each run contains 168 brain volumes sampled two
    seconds apart. This convenience loader is specific to that study; use
    `htfa.load` or arrays for other experimental datasets.

    Downloads the runs that are not yet in nilearn's data directory (about
    25 MB each), masks each one (`load_development_fmri_subject`, cached) and,
    with ``clean``, removes a linear trend and the confounds
    (`regress_confounds`).

    Parameters
    ----------
    n_subjects : int, optional
        Number of subjects; all of the age group by default.
    adults_only : bool
        Only the 33 adults (default); ``False`` adds children.
    clean : bool
        Regress out a linear trend and the confounds (default). ``False``
        returns the masked BOLD intensities.
    data_dir : path, optional
        nilearn's data directory.
    cache_dir : path, optional
        Cache of the masked runs (default: ``development_fmri/htfa`` in the
        data directory).

    Returns
    -------
    DevelopmentFMRI
        ``subject_ids``, ``data``, ``coords``, ``confounds``, ``phenotypic`` and
        ``tr``; ``data`` and ``coords`` go straight to :func:`htfa.fit`.

    Examples
    --------
    >>> import htfa
    >>> ds = htfa.datasets.fetch_development_fmri(n_subjects=3)  # doctest: +SKIP
    >>> res = htfa.fit(ds.data, coords=ds.coords, K=20)  # doctest: +SKIP
    """
    files = development_fmri_files(n_subjects, adults_only, data_dir)
    pd = lazy_import("pandas", "nilearn", "the development_fmri dataset")
    if not files:
        raise RuntimeError("nilearn's fetch_development_fmri returned no runs")
    pheno = pd.read_csv(files[0][1].parent / "participants.tsv", sep="\t").set_index(
        "participant_id", drop=False
    )
    ids, data, coords, confs = [], [], [], []
    for sid, bold, conf in files:
        X, R, C = load_development_fmri_subject(bold, conf, cache_dir)
        if clean:
            X = regress_confounds(X, C).astype(np.float32)
        ids.append(sid)
        data.append(X)
        coords.append(R)
        confs.append(C)
    return DevelopmentFMRI(
        ids, data, coords, confs, pheno.loc[ids].reset_index(drop=True)
    )


def fetch_tutorial_fit(
    name: str, cache_dir: str | os.PathLike | None = None
) -> dict[str, Array]:
    """A saved result of one of the tutorials, downloaded on first use.

    The results (HTFA fits, and tutorials 09/13's permutation nulls and tutorial 09's
    word embeddings) are files of the ``tutorial-data-v1`` pre-release of the htfa
    repository (see ``htfa.datasets.TUTORIAL_FITS``). The file is downloaded
    once, its SHA-256 is checked before it is loaded, and it is loaded with
    ``allow_pickle=False``. Each file records how it was made (for a fit:
    ``settings``, ``htfa_commit``, ``htfa_version``, ``fit_seconds`` and the
    fitted data). A fit may be regenerated when htfa's fitting code changes.

    Parameters
    ----------
    name : str
        A key of ``TUTORIAL_FITS``: ``"mitchell2008_K100"``,
        ``"mitchell2008_embeddings"``, ``"mitchell2008_null_20"`` and
        ``"mitchell2008_null_100"`` (tutorial 09),
        ``"development_fmri_12adults_K50"`` (tutorial 10) or
        ``"development_fmri_6adults_K30"`` (tutorial 11),
        ``"development_fmri_10children_10adults_K30"`` and
        ``"development_fmri_children_adults_null_100"`` (tutorial 13).
    cache_dir : path, optional
        Where the file is kept (default: ``tutorial_fits`` in htfa's data
        folder, ``$HTFA_DATA_DIR`` or ``~/.cache/htfa``).

    Returns
    -------
    dict of str to ndarray
        Every array in the file, for example ``weights`` (S, K, T), ``centers``
        (S, K, D), ``widths`` (S, K), ``global_centers``, ``global_widths``,
        ``n_iter``, ``converged`` and the provenance strings (0-d arrays).

    Raises
    ------
    ValueError
        If ``name`` is unknown, or the file does not match its SHA-256 (it is
        deleted; a later call downloads it again).
    ConnectionError
        If the file is not cached and cannot be downloaded.

    Examples
    --------
    >>> from htfa.datasets import fetch_tutorial_fit
    >>> fit = fetch_tutorial_fit("development_fmri_6adults_K30")  # doctest: +SKIP
    >>> fit["weights"].shape  # doctest: +SKIP
    (6, 30, 168)
    """
    if name not in TUTORIAL_FITS:
        raise ValueError(
            f"unknown tutorial fit {name!r}; choose from {sorted(TUTORIAL_FITS)}"
        )
    filename, expected = TUTORIAL_FITS[name]
    folder = Path(cache_dir) if cache_dir is not None else data_dir() / "tutorial_fits"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    url = TUTORIAL_DATA_URL + filename
    redo = "set REFIT = True in the tutorial to fit HTFA from scratch instead"
    if not path.exists():
        fd, tmp_name = tempfile.mkstemp(prefix=filename, suffix=".part", dir=folder)
        tmp = Path(tmp_name)
        try:
            try:
                with (
                    os.fdopen(fd, "wb") as out,
                    urllib.request.urlopen(url, timeout=60) as response,
                ):
                    shutil.copyfileobj(response, out)
            except OSError as err:
                raise ConnectionError(
                    f"could not download {url} ({err}); {redo}"
                ) from err
            got = sha256(tmp)
            if got != expected:
                raise ValueError(
                    f"{url} has SHA-256 {got}, expected {expected}; {redo}"
                )
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
    got = sha256(path)
    if got != expected:
        path.unlink()
        raise ValueError(
            f"{path} has SHA-256 {got}, expected {expected}; it was deleted, so the "
            f"next call downloads it again. Or {redo}"
        )
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}
