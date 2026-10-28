"""BrainNet Viewer-style network plots and animations.

Nodes are drawn as gray spheres inside a translucent brain surface, and edges
whose absolute weight exceeds a threshold as tubes (red for positive weights,
blue for negative ones). The drawing rules follow BrainNet Viewer 1.43
(Xia, Wang & He, 2013, PLoS ONE 8: e68910; https://www.nitrc.org/projects/bnv/)
with the settings of the 2014 directed-forgetting network movies: white
background, mesh color 0.95 gray at opacity 0.2, node radii scaled linearly
onto 2-7 mm, edge diameter ``|E|`` mm at opacity 0.7, and an orthographic left
lateral view.

Rendering uses pyvista off screen; the ``brain3d`` extra is installed on first
use. On Linux without a display (Colab, containers), VTK's EGL offscreen
window (or else OSMesa) is selected automatically. The brain surfaces are
BrainNet Viewer's own ``.nv`` meshes. BrainNet Viewer is distributed under the
GNU GPL, so htfa does not ship them: they are
downloaded from NITRC on first use, checked against SHA-256 checksums, and
cached (see `mesh_cache_dir`).
"""

# Adapted from ContextLab/directed-forgetting-network-dynamics, commit 31a2b66
# (code/dfnet/render.py and code/dfnet/brainnet_io.py), MIT License,
# Copyright (c) 2019 Contextual Dynamics Laboratory.

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from ._download import DATA_DIR_ENV as DATA_DIR_ENV
from ._download import data_dir
from ._download import sha256 as _sha256
from ._kernels import LAPLACE_EPS, Kernel
from ._lazy import lazy_import

Array = np.ndarray

__all__ = [
    "MESHES",
    "animate_network",
    "annotate_pair",
    "config_style",
    "effective_nodes",
    "fetch_mesh",
    "image_diff",
    "load_mesh",
    "mesh_cache_dir",
    "mni_mask",
    "montage",
    "node_radius",
    "plot_network",
    "read_brainnet_config",
    "read_edge",
    "read_node",
    "read_nv",
    "write_animation",
    "write_edge",
    "write_node",
    "write_nv",
]

# BrainNet Viewer 1.43 (2013-12-27), the release the 2014 movies were made with.
MESH_ARCHIVE_URL = (
    "https://www.nitrc.org/frs/download.php/6140/BrainNetViewer_20131227.zip"
)
MESH_ARCHIVE_SHA256 = "246b10c872966734e2c00b45d6ceee21bf37a02798ee1ba804e6329b0074dd5a"
_MESH_DIR_IN_ARCHIVE = "BrainNetViewer/Data/SurfTemplate/"
#: Mesh name -> (file in BrainNet Viewer 1.43's Data/SurfTemplate, SHA-256).
MESHES = {
    "ch2": (
        "BrainMesh_Ch2withCerebellum.nv",
        "a6bdca96dcb1195829567c1b29d250cae50146c3cd2e7f24b384d3dfdd7d122b",
    ),
    "icbm152": (
        "BrainMesh_ICBM152.nv",
        "d68a41d966a36de91b7ae766d2b7aa5c71120de6c11fa68ea04f8a9c0406bea7",
    ),
    "icbm152_smoothed": (
        "BrainMesh_ICBM152_smoothed.nv",
        "e2ee8c7e91f42b16e19b84228e8bb43c70389911c567fc4d5a9da03288b614ae",
    ),
}

SIZE = (1600, 1000)
# Left lateral framing fitted to the 2014 frames (1600 x 1000): image column
# = 740.27 - 5.5288 * y and row = 513.15 - 5.5288 * z, in MNI mm.
_PX_PER_MM = 5.5288
_FOCAL = np.array(
    [0.0, -(SIZE[0] / 2 - 740.27) / _PX_PER_MM, (513.15 - SIZE[1] / 2) / _PX_PER_MM]
)
_PARALLEL_SCALE = SIZE[1] / 2 / _PX_PER_MM
# Dorsal/ventral views: centered on the Ch2 mesh's y range (-104.4 to 72.7 mm),
# with 5% more room so the 177 mm long brain fits the image height.
_FOCAL_AXIAL = np.array([0.0, -15.85, _FOCAL[2]])
# view -> (direction from the focal point to the camera, camera up vector)
VIEWS = {
    "left": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "lateral": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "right": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "dorsal": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    "ventral": ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    "anterior": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "posterior": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
}
EDGE_OPACITY = 0.7
_DULL = {"ambient": 0.3, "diffuse": 0.8, "specular": 0.0}  # MATLAB `material dull`
_SUPERSAMPLE = 2


def _pil() -> Any:
    return lazy_import("PIL.Image", "brain3d", "network plots")


# --------------------------------------------------------------------- file io


def read_node(path: str | os.PathLike) -> tuple[Array, Array, Array, list[str]]:
    """Read a BrainNet ``.node`` file.

    Returns
    -------
    centers : ndarray, shape (K, 3)
    colors, sizes : ndarray, shape (K,)
    labels : list of str
        ``"-"`` where the file has no label.
    """
    rows = [
        line.split()
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "%"))
    ]
    if not rows or any(len(r) < 5 for r in rows):
        raise ValueError(f"{path}: a .node row needs x y z color size [label]")
    num = np.array([r[:5] for r in rows], dtype=np.float64)
    labels = [" ".join(r[5:]) if len(r) > 5 else "-" for r in rows]
    return num[:, :3], num[:, 3], num[:, 4], labels


def write_node(
    path: str | os.PathLike,
    centers: Array,
    sizes: Array | None = None,
    colors: Array | None = None,
    labels: Sequence[str] | None = None,
) -> Path:
    """Write a BrainNet ``.node`` file (tab separated, two decimals)."""
    centers = _centers(centers)
    K = len(centers)
    sizes = np.ones(K) if sizes is None else _per_node(sizes, K, "sizes")
    colors = np.ones(K) if colors is None else _per_node(colors, K, "colors")
    labels = ["-"] * K if labels is None else [str(s) for s in labels]
    if len(labels) != K:
        raise ValueError(f"labels has {len(labels)} entries for {K} nodes")
    lines = [
        f"{x:.2f}\t{y:.2f}\t{z:.2f}\t{c:.2f}\t{s:.2f}\t{lab}\n"
        for (x, y, z), c, s, lab in zip(centers, colors, sizes, labels, strict=True)
    ]
    Path(path).write_text("".join(lines))
    return Path(path)


def read_edge(path: str | os.PathLike) -> Array:
    """Read a BrainNet ``.edge`` file (whitespace-separated square matrix)."""
    E = np.loadtxt(path, ndmin=2)
    if E.shape[0] != E.shape[1]:
        raise ValueError(f"{path}: edge matrix is {E.shape}, not square")
    return E


def write_edge(path: str | os.PathLike, connectivity: Array) -> Path:
    """Write a BrainNet ``.edge`` file (tab separated, two decimals)."""
    E = np.asarray(connectivity, dtype=np.float64)
    if E.ndim != 2 or E.shape[0] != E.shape[1]:
        raise ValueError(f"connectivity must be (K, K), got {E.shape}")
    np.savetxt(path, E, fmt="%0.2f", delimiter="\t")
    return Path(path)


def read_nv(path: str | os.PathLike) -> tuple[Array, Array]:
    """Read a BrainNet ``.nv`` mesh.

    The format is the vertex count, one ``x y z`` row per vertex, the face
    count and one row of three 1-based vertex indices per face.

    Returns
    -------
    vertices : ndarray, shape (V, 3)
    faces : ndarray of int, shape (F, 3), 0-based
    """
    tokens = [
        line.split()
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "%"))
    ]
    nv = int(tokens[0][0])
    vertices = np.array(tokens[1 : 1 + nv], dtype=np.float64)
    nf = int(tokens[1 + nv][0])
    faces = np.array(tokens[2 + nv : 2 + nv + nf], dtype=np.int64) - 1
    if vertices.shape != (nv, 3) or faces.shape != (nf, 3):
        raise ValueError(f"{path}: malformed .nv file")
    if np.any(faces < 0) or np.any(faces >= nv):
        raise ValueError(f"{path}: face indices out of range")
    return vertices, faces


def write_nv(path: str | os.PathLike, vertices: Array, faces: Array) -> Path:
    """Write a BrainNet ``.nv`` mesh (``faces`` 0-based, written 1-based)."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must be (V, 3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must be (F, 3), got {faces.shape}")
    with open(path, "w") as f:
        f.write(f"{len(vertices)}\n")
        np.savetxt(f, vertices, fmt="%.6f")
        f.write(f"{len(faces)}\n")
        np.savetxt(f, faces + 1, fmt="%d")
    return Path(path)


def read_brainnet_config(path: str | os.PathLike) -> dict[str, Any]:
    """Read a BrainNet Viewer option file (``.mat`` holding struct ``EC``).

    Returns the struct as nested dicts (``cfg["edg"]["draw_threshold"]``, ...).
    """
    import scipy.io

    m = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    if "EC" not in m:
        raise ValueError(f"{path} has no BrainNet option struct 'EC'")

    def todict(x: Any) -> Any:
        if isinstance(x, scipy.io.matlab.mat_struct):
            return {k: todict(getattr(x, k)) for k in x._fieldnames}
        return x

    return dict(todict(m["EC"]))


def config_style(cfg: dict[str, Any]) -> dict[str, Any]:
    """`plot_network` style keywords from a BrainNet option struct.

    Uses the background color, mesh color and opacity, the first node color,
    the edge draw threshold, and the first / last edge colormap rows as the
    positive / negative edge colors (BrainNet's two-color edge mode). The view
    is always left lateral: BrainNet 1.43 uses ``view(-90, 0)`` for
    ``view_direction`` 1, whatever ``view_az`` holds.
    """
    edge_cm = np.atleast_2d(np.asarray(cfg["edg"]["CM"], dtype=np.float64))
    node_cm = np.atleast_2d(np.asarray(cfg["nod"]["CM"], dtype=np.float64))

    def rgb(x: Any) -> tuple[float, ...]:
        return tuple(float(v) for v in np.ravel(x))

    return {
        "background": rgb(cfg["bak"]["color"]),
        "mesh_color": rgb(cfg["msh"]["color"]),
        "mesh_opacity": float(cfg["msh"]["alpha"]),
        "node_color": rgb(node_cm[0]),
        "threshold": float(cfg["edg"]["draw_threshold"]),
        "positive_color": rgb(edge_cm[0]),
        "negative_color": rgb(edge_cm[-1]),
    }


# ---------------------------------------------------------------------- meshes


def mesh_cache_dir() -> Path:
    """Folder holding downloaded meshes.

    ``$HTFA_DATA_DIR/brainnet`` if that variable is set, otherwise
    ``%LOCALAPPDATA%\\htfa\\brainnet`` on Windows and
    ``${XDG_CACHE_HOME:-~/.cache}/htfa/brainnet`` elsewhere.
    """
    return data_dir() / "brainnet"


_ZIP_MAGIC = b"PK\x03\x04"


def _fetch_archive(
    dest: Path, folder: Path, attempts: int = 3, url: str = MESH_ARCHIVE_URL
) -> None:
    """Download the zip archive at ``url`` to ``dest``.

    NITRC sometimes answers with an HTML error page (status 200) instead of
    the archive; a response that does not start with the zip signature is
    retried. The SHA-256 check in `_download_meshes` guards the content.
    """
    from . import __version__

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": f"htfa/{__version__} (+https://github.com/ContextLab/htfa)"
        },
    )
    err: Exception | None = None
    for attempt in range(attempts):
        if attempt:
            time.sleep(2.0 * attempt)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                head = response.read(len(_ZIP_MAGIC))
                if head != _ZIP_MAGIC:
                    kind = response.headers.get("Content-Type", "")
                    raise urllib.error.URLError(
                        f"{response.url} returned {kind!r} content that is not a "
                        "zip archive"
                    )
                with open(dest, "wb") as out:
                    out.write(head)
                    shutil.copyfileobj(response, out)
            return
        except (urllib.error.URLError, OSError) as e:
            err = e
    raise ConnectionError(
        f"could not download the BrainNet Viewer meshes from {url} "
        f"({err}). Download that file by hand and put the .nv files from "
        f"{_MESH_DIR_IN_ARCHIVE} in {folder}, or pass mesh= a path to a .nv file."
    ) from err


def _download_meshes(folder: Path) -> None:
    """Download BrainNet Viewer 1.43 and extract the meshes in `MESHES`."""
    folder.mkdir(parents=True, exist_ok=True)
    print(
        f"htfa: downloading BrainNet Viewer 1.43 (21 MB) from {MESH_ARCHIVE_URL} "
        "for its brain surface meshes (BrainNet Viewer is GPL-licensed; "
        "Xia et al., 2013, PLoS ONE 8: e68910)...",
        file=sys.stderr,
    )
    fd, tmp_name = tempfile.mkstemp(suffix=".zip", dir=folder)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        _fetch_archive(tmp, folder)
        got = _sha256(tmp)
        if got != MESH_ARCHIVE_SHA256:
            raise ValueError(
                f"{MESH_ARCHIVE_URL} has SHA-256 {got}, expected "
                f"{MESH_ARCHIVE_SHA256}; the download is corrupt or the file changed"
            )
        _extract_meshes(tmp, folder)
    finally:
        tmp.unlink(missing_ok=True)


def _extract_meshes(archive_path: Path, folder: Path) -> None:
    """Extract the meshes in `MESHES` from the BrainNet Viewer zip into ``folder``.

    Each file is written to ``<name>.part`` and renamed when complete; a
    ``.part`` left by a failure is removed.
    """
    with zipfile.ZipFile(archive_path) as archive:
        for filename, _ in MESHES.values():
            part = folder / (filename + ".part")
            try:
                with (
                    archive.open(_MESH_DIR_IN_ARCHIVE + filename) as src,
                    open(part, "wb") as dst,
                ):
                    shutil.copyfileobj(src, dst)
                os.replace(part, folder / filename)
            finally:
                part.unlink(missing_ok=True)


def fetch_mesh(name: str = "ch2") -> Path:
    """Path to a BrainNet Viewer mesh, downloading it on first use.

    Parameters
    ----------
    name : {"ch2", "icbm152", "icbm152_smoothed"}
        ``"ch2"`` is ``BrainMesh_Ch2withCerebellum.nv`` (used for the 2014
        movies); the ICBM152 meshes have no cerebellum.

    Raises
    ------
    ConnectionError
        If the mesh is not cached and cannot be downloaded.
    ValueError
        If a downloaded file does not match its SHA-256 checksum.
    """
    if name not in MESHES:
        raise ValueError(f"unknown mesh {name!r}; choose from {sorted(MESHES)}")
    filename, sha = MESHES[name]
    path = mesh_cache_dir() / filename
    if path.exists() and _sha256(path) == sha:
        return path
    _download_meshes(path.parent)
    got = _sha256(path)
    if got != sha:
        raise ValueError(f"{path} has SHA-256 {got}, expected {sha}")
    return path


def load_mesh(mesh: Any) -> tuple[Array, Array] | None:
    """Resolve ``mesh`` to ``(vertices, faces)``.

    ``mesh`` is a name in `MESHES`, a path to a ``.nv`` file, a
    ``(vertices, faces)`` pair (faces 0-based), or ``None`` (no surface).
    """
    if mesh is None:
        return None
    if isinstance(mesh, str) and mesh in MESHES:
        return read_nv(fetch_mesh(mesh))
    if isinstance(mesh, (str, os.PathLike)):
        if not Path(mesh).is_file():
            raise ValueError(
                f"mesh {mesh!r} is neither a file nor one of {sorted(MESHES)}"
            )
        return read_nv(mesh)
    vertices, faces = mesh
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


# ------------------------------------------------------------------- rendering


def node_radius(
    sizes: Any,
    lo: float = 2.0,
    hi: float = 7.0,
    *,
    k: float | None = None,
    b: float | None = None,
    clip: tuple[float, float] | None = None,
) -> Array:
    """Sphere radii in mm from node sizes, BrainNet Viewer's ``k * size + b``.

    By default ``k`` and ``b`` map the smallest size to ``lo`` and the
    largest to ``hi`` (BrainNet "size by value": 2-7 mm). All-equal sizes
    give 4 mm, as in BrainNet Viewer.

    Parameters
    ----------
    sizes : array of shape (K,)
    lo, hi : float
        Radii of the smallest and largest size, ``lo < hi``.
    k, b : float, optional
        A fixed mapping ``k * size + b`` instead (both needed), e.g. the
        ``nod.k`` / ``nod.b`` stored in a BrainNet option file, so that radii
        are comparable across plots.
    clip : (float, float), optional
        Clip the radii to this range afterwards.

    Returns
    -------
    ndarray, shape (K,)

    Examples
    --------
    >>> from htfa.brainnet import node_radius
    >>> node_radius([1.0, 2.0, 3.0]).tolist()
    [2.0, 4.5, 7.0]
    >>> node_radius([1.0, 2.0, 3.0], k=3.0, b=0.0, clip=(0.0, 7.0)).tolist()
    [3.0, 6.0, 7.0]
    """
    s = np.asarray(sizes, dtype=np.float64)
    if (k is None) != (b is None):
        raise ValueError("give both k and b, or neither")
    if not lo < hi:
        raise ValueError(f"need lo < hi, got lo={lo}, hi={hi}")
    if clip is not None and not clip[0] <= clip[1]:
        raise ValueError(f"clip must be (min, max) with min <= max, got {clip}")
    if k is not None and b is not None:
        r = k * s + b
    elif s.max() == s.min():
        r = np.full_like(s, 4.0)
    else:
        slope = (hi - lo) / (s.max() - s.min())
        r = slope * s + (hi - slope * s.max())
    return r if clip is None else np.clip(r, clip[0], clip[1])


def _centers(centers: Any) -> Array:
    c = np.asarray(centers, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 3:
        raise ValueError(f"node centers must be (K, 3) MNI coordinates, got {c.shape}")
    return c


def _per_node(values: Any, K: int, what: str) -> Array:
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.shape != (K,):
        raise ValueError(f"{what} has {v.size} entries for {K} nodes")
    return v


def _nodes(nodes: Any, sizes: Any, subject: Any) -> tuple[Array, Array]:
    """Centers and sizes from an HTFAResults, a fitted TFA/HTFA, or an array."""
    centers, widths = _fit_nodes(nodes, subject)
    K = len(centers)
    if sizes is not None:
        return centers, _per_node(sizes, K, "sizes")
    return centers, np.ones(K) if widths is None else widths


def _fit_nodes(nodes: Any, subject: Any) -> tuple[Array, Array | None]:
    """Centers and widths (``None`` for an array) of ``nodes``."""
    from .htfa import HTFA
    from .results import HTFAResults
    from .tfa import TFA

    if isinstance(nodes, HTFAResults):
        if subject == "global":
            if nodes.global_centers is None:
                raise ValueError("a TFA fit has no global template; pass subject=0")
            centers, widths = nodes.global_centers, nodes.global_widths
        else:
            centers, widths = nodes.centers[subject], nodes.widths[subject]
    elif isinstance(nodes, HTFA):
        if subject == "global":
            centers, widths = nodes.global_centers_, nodes.global_widths_
        else:
            centers, widths = nodes.centers_[subject], nodes.widths_[subject]
    elif isinstance(nodes, TFA):
        centers, widths = nodes.centers_, nodes.widths_
    else:
        centers, widths = nodes, None
    centers = _centers(centers)
    return centers, None if widths is None else _per_node(
        widths, len(centers), "widths"
    )


def _fit_shape(nodes: Any, subject: Any) -> tuple[str, Array, Array, Array] | None:
    """Kernel name and component centers, widths and mixing weights of a fit
    whose factors are not single RBFs (None for those and for arrays)."""
    from .htfa import HTFA
    from .results import HTFAResults
    from .tfa import TFA

    if isinstance(nodes, HTFAResults):
        return nodes._shape(subject)
    if not isinstance(nodes, (HTFA, TFA)):
        return None
    kernel = Kernel(getattr(nodes, "kernel", "rbf"), getattr(nodes, "n_components", 1))
    if kernel.plain_rbf:
        return None
    if isinstance(nodes, TFA):
        parts = (nodes.component_centers_, nodes.component_widths_)
        return (kernel.name, *parts, nodes.mixing_weights_)
    if subject == "global":
        return (
            kernel.name,
            nodes.global_component_centers_,
            nodes.global_component_widths_,
            nodes.global_mixing_weights_,
        )
    return (
        kernel.name,
        nodes.component_centers_[subject],
        nodes.component_widths_[subject],
        nodes.mixing_weights_[subject],
    )


# ------------------------------------------------------------- effective nodes

_TAIL_SIGMAS = 5.0  # RBF weights beyond 5 sigma (exp(-12.5)) are left out
_TAIL_LOG = _TAIL_SIGMAS**2 / 2  # the other kernels: profile values below exp(-12.5)


def _reach(kernel: str, w: float, D: int) -> float:
    """Radius beyond which a component of width ``w`` is left out: its profile
    is below ``exp(-12.5)`` there (``'mexican_hat'``: the positive lobe,
    ``r < sqrt(D w / 2)``)."""
    if kernel == "rbf":
        return math.sqrt(_TAIL_LOG * w)
    if kernel == "laplace":
        return math.sqrt(w * ((_TAIL_LOG + LAPLACE_EPS) ** 2 - LAPLACE_EPS**2))
    return math.sqrt(D * w / 2)


_UNIT_MASS: dict[tuple[str, int], float] = {}


def _unit_mass(kernel: str, D: int) -> float:
    """Integral over R^D of a width-1 component's profile (``'mexican_hat'``:
    of its positive lobe); ``pi**(D / 2)`` for the RBF."""
    if (kernel, D) not in _UNIT_MASS:
        from scipy.integrate import quad

        profile = Kernel(kernel).profile
        upper = math.sqrt(D / 2) if kernel == "mexican_hat" else np.inf
        radial = quad(
            lambda t: float(profile(np.array(t * t), D)) * t ** (D - 1), 0, upper
        )
        sphere = 2 * math.pi ** (D / 2) / math.gamma(D / 2)
        _UNIT_MASS[kernel, D] = sphere * radial[0]
    return _UNIT_MASS[kernel, D]


_TIE_TOL = 1e-3  # distances within 1e-3 voxel count as tied


def mni_mask() -> Array:
    """Voxel centers (V, 3) of nilearn's 2 mm MNI152 brain mask, in MNI mm.

    The default mask for snapping bare node centers (`effective_nodes`,
    `plot_network`): the BrainNet meshes are in MNI space. Needs nilearn (the
    ``nilearn`` extra); the mask ships with nilearn, so nothing is downloaded.

    Raises
    ------
    ImportError
        If nilearn is missing and cannot be installed; pass ``mask=`` or
        ``snap=False`` instead.

    Examples
    --------
    >>> from htfa.brainnet import effective_nodes, mni_mask
    >>> mask = mni_mask()
    >>> mask.shape[1]
    3
    >>> c, _, info = effective_nodes([[0.0, 95.0, 0.0]], None, mask)  # frontal pole
    >>> bool(info["snapped"][0]), bool(c[0, 1] < 75)
    (True, True)
    """
    return _mni_mask().copy()


_MNI_CACHE: list[Array] = []


def _mni_mask() -> Array:
    if not _MNI_CACHE:
        try:
            datasets = lazy_import("nilearn.datasets", "nilearn", "the MNI brain mask")
        except ImportError as err:
            raise ImportError(
                f"{err}\nSnapping nodes into the brain needs a mask: the default "
                "for bare MNI centers is nilearn's MNI152 brain mask. Pass mask= "
                "(voxel coordinates or a NIfTI mask) or snap=False."
            ) from err
        img = datasets.load_mni152_brain_mask(resolution=2)
        _MNI_CACHE.append(_mask_coords(img)[0])
    return _MNI_CACHE[0]


def _mask_coords(mask: Any, dim: int | None = None) -> tuple[Array, Array, float]:
    """Voxel centers, per-axis voxel spacing and voxel volume of a mask.

    ``mask`` is a (V, D) coordinate array, a NIfTI-like image (nonzero
    voxels), or a ``(volume, affine)`` pair. For an image the spacing along
    each world axis is the extent of one voxel along it and the volume is
    ``|det(affine[:3, :3])|``. For coordinates the spacing along an axis is
    the median gap between neighboring voxels on the same grid line, and the
    volume is the product of the spacings; voxels off the grid of the first
    voxel (e.g. a union of subjects' differently aligned grids) are moved to
    the nearest point of that grid, so that every region is counted once.
    ``dim``, if given, is the number of dimensions the mask must have.
    """
    if isinstance(mask, tuple) or (
        hasattr(mask, "affine") and hasattr(mask, "dataobj")
    ):
        volume, affine = (
            mask if isinstance(mask, tuple) else (mask.dataobj, mask.affine)
        )
        out = _volume_coords(np.asarray(volume), np.asarray(affine, dtype=float))
        _check_dim(out[0], dim)
        return out
    coords = np.asarray(mask, dtype=np.float64)
    if coords.ndim != 2:
        raise ValueError(f"mask coordinates must be (V, D), got {coords.shape}")
    _check_dim(coords, dim)
    if len(coords) == 0:
        raise ValueError("mask has no voxels")
    if len(coords) == 1:
        raise ValueError("mask coordinates need at least 2 voxels to set the spacing")
    spacing = _grid_spacing(coords)
    if not np.all(spacing > 0):
        raise ValueError("mask voxels coincide: cannot tell the voxel spacing")
    idx = (coords - coords[0]) / spacing
    if np.abs(idx - np.round(idx)).max() > 1e-3:
        grid = np.unique(np.floor(idx + 0.5), axis=0)
        coords = coords[0] + grid * spacing
    return coords, spacing, float(np.prod(spacing))


def _check_dim(coords: Array, dim: int | None) -> None:
    if dim is not None and coords.shape[1] != dim:
        raise ValueError(
            f"mask has {coords.shape[1]} dimensions but centers have {dim}"
        )


def _grid_spacing(coords: Array) -> Array:
    """Per-axis median gap between neighboring voxels on one grid line."""
    q = np.round(coords, 3)
    spacing = np.empty(q.shape[1])
    for a in range(q.shape[1]):
        other = np.delete(q, a, axis=1)
        order = np.lexsort((q[:, a], *other.T[::-1]))
        same = np.all(other[order][1:] == other[order][:-1], axis=1)
        gaps = np.diff(q[order, a])[same]
        gaps = gaps[gaps > 1e-6]
        if len(gaps):
            spacing[a] = np.median(gaps)
        else:  # no two voxels share a line along this axis
            spacing[a] = np.median(cKDTree(coords).query(coords, k=2)[0][:, 1])
    return spacing


def _volume_coords(volume: Array, affine: Array) -> tuple[Array, Array, float]:
    ijk = np.argwhere(volume.astype(bool))
    if not len(ijk):
        raise ValueError("mask has no voxels")
    A = affine[:3, :3]
    coords = ijk @ A.T + affine[:3, 3]
    return coords, np.abs(A).sum(axis=1), float(abs(np.linalg.det(A)))


def _lowest_tied(tree: Any, points: Array, tol: float, p: float) -> tuple[Array, Array]:
    """Nearest voxel of each point, the lowest voxel index among near-ties.

    Distances within ``tol`` of the nearest count as tied. Returns the voxel
    indices and whether another voxel was tied.
    """
    k = min(8, tree.n)
    d, i = tree.query(points, k=k, p=p)
    d, i = d.reshape(len(points), k), i.reshape(len(points), k)
    tied = d <= d[:, :1] + tol
    best = np.where(tied, i, tree.n).min(axis=1)
    return best, tied.sum(axis=1) > 1


def effective_nodes(
    centers: Any,
    widths: Any,
    mask: Any,
    kernel: str = "rbf",
    mixing_weights: Any = None,
) -> tuple[Array, Array, dict[str, Array]]:
    """In-mask ("effective") centers and widths of RBF nodes, for display.

    A fitted center may lie outside the brain: an RBF there still reaches into
    it, and is how a fit explains a blurry cortical activation. A plot should
    not draw a node where no brain is, so the built-in plots show each node as
    the part of its RBF that lies inside the mask, ``F(v) = exp(-||r_v - c||^2
    / lambda)`` over the mask voxels ``r_v``:

    - center: the F-weighted centroid of the mask voxels. Interior nodes keep
      their center; nodes near the edge move inward. If the centroid falls
      outside the mask (a concave mask: not within half a voxel of a mask
      voxel), it is snapped to the first mask voxel on the segment from the
      centroid to the original center, or, if that segment never enters the
      mask, to the mask voxel nearest the centroid.
    - width: ``lambda_eff = 2 * sigma_eff**2``, where ``sigma_eff**2`` is the
      F-weighted per-axis variance about the centroid, averaged over axes.
      An RBF has per-axis variance ``lambda / 2``, so interior nodes keep
      ``lambda`` (to within 0.2% on 2 mm and 4 mm grids for
      ``20 <= lambda <= 400`` mm^2).

    The fit is not changed. Ties between equally near voxels (e.g. a node on
    the midline between the hemispheres) go to the lowest voxel index and
    are flagged ``ambiguous``.

    Other kernels and mixtures (``kernel``, component ``centers`` (K, M, D))
    use their own profile ``F`` in place of the RBF's, summed over the
    components with the mixing weights: for ``'mexican_hat'`` its positive
    part ``max(F, 0)`` (the centre, not the negative surround, is the node).
    ``lambda_eff = 2 sigma_eff**2`` is then the width of the RBF with the
    same spread: an interior node has ``lambda_eff = 6.02 w`` (D = 2) or
    ``8.01 w`` (D = 3) for ``'laplace'`` (``2 (D + 1) w`` without the
    rounding at its center), and ``0.282 w`` or ``0.351 w`` for
    ``'mexican_hat'`` (by numerical integration). The mass fraction divides by
    the summed masses of the components (positive lobes), each fully inside;
    voxels where every component is below ``exp(-12.5)`` are left out, and a
    node with no voxel left (far outside the mask) is treated as a point.

    Parameters
    ----------
    centers : array of shape (K, D) or (K, M, D)
        Node centers, or component centers of nodes with M components.
    widths : array of shape (K,) or (K, M), or None
        RBF widths ``lambda`` (the BrainIAK convention ``2 * sigma**2``), or
        the components' widths. ``None`` treats the nodes as points: a
        center within half a voxel of the mask stays, any other goes to the
        nearest mask voxel, and the widths and masses are NaN.
    mask : array of shape (V, D), NIfTI image or (volume, affine)
        The in-brain voxels: voxel centers on a regular grid (its spacing
        may differ per axis; a union of differently aligned grids is moved
        onto the grid of the first voxel), or a mask image whose nonzero
        voxels count. "Within half a voxel" is per axis. `mni_mask` gives
        the MNI152 brain.

    Returns
    -------
    centers_eff : ndarray, shape (K, D)
    widths_eff : ndarray, shape (K,)
        ``lambda_eff``.
    info : dict of ndarray, shape (K,)
        ``distance`` (from the original center, in coordinate units),
        ``snapped`` (the centroid fell outside the mask), ``ambiguous``
        (tied nearest voxels), ``mass_fraction`` (in-mask RBF mass over the
        mass of an RBF fully inside, ``(pi * lambda)**(D / 2)`` divided by
        the voxel volume), ``sigma`` (``sigma_eff``) and ``width_ratio``
        (``lambda_eff / lambda``; with components, ``lambda`` is their
        mixing-weighted mean width).
    kernel : {'rbf', 'laplace', 'mexican_hat'}, default='rbf'
        Shape of the nodes (see `htfa.TFA`).
    mixing_weights : array of shape (K, M), optional
        Mixing weights of component centers (default: equal).

    Examples
    --------
    >>> import numpy as np
    >>> from htfa.brainnet import effective_nodes
    >>> ax = np.arange(-20.0, 21.0, 2.0)
    >>> grid = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
    >>> ball = grid[np.linalg.norm(grid, axis=1) <= 20]
    >>> c, w, info = effective_nodes([[0.0, 0.0, 30.0]], [50.0], ball)
    >>> bool(np.linalg.norm(c[0]) < 20), bool(info["snapped"][0])
    (True, False)
    """
    c = np.asarray(centers, dtype=np.float64)
    if c.ndim not in (2, 3):
        raise ValueError(f"centers must be (K, D) or (K, M, D), got {c.shape}")
    shape = Kernel(kernel, 1 if c.ndim == 2 else c.shape[1])
    comps = None
    if c.ndim == 3:
        if widths is None:
            raise ValueError("component centers (K, M, D) need their widths")
        Kc, M, _ = c.shape
        wc = np.asarray(widths, dtype=np.float64).reshape(Kc, M)
        if mixing_weights is None:
            pi = np.full((Kc, M), 1.0 / M)
        else:
            pi = np.asarray(mixing_weights, dtype=np.float64).reshape(Kc, M)
        comps = (c, wc, pi)
        c = np.einsum("km,kmd->kd", pi, c)
        widths = np.sum(pi * wc, axis=1)
        if not np.all(wc > 0):
            raise ValueError("widths must be positive")
    K, D = c.shape
    coords, spacing, voxel_volume = _mask_coords(mask, D)
    w = None if widths is None else _per_node(widths, K, "widths")
    if w is not None and not np.all(w > 0):
        raise ValueError("widths must be positive")
    if comps is None and w is not None:
        comps = (c[:, None], w[:, None], np.ones((K, 1)))
    tree = cKDTree(coords)
    grid = cKDTree(coords / spacing)  # voxel units: "inside" is <= 0.5 per axis
    half = 0.5 * (1 + 1e-9)
    out = c.copy()
    sigma = np.full(K, np.nan)
    mass = np.full(K, np.nan)
    snapped = np.zeros(K, dtype=bool)
    ambiguous = np.zeros(K, dtype=bool)
    if comps is not None and not shape.plain_rbf:
        for k in range(K):
            ck, wk, pk = comps[0][k], comps[1][k], comps[2][k]
            dm = tree.query(ck)[0]
            idx = sorted(
                set().union(
                    *(
                        tree.query_ball_point(ck[m], dm[m] + _reach(kernel, wk[m], D))
                        for m in range(len(wk))
                    )
                )
            )
            r = coords[idx]
            f = np.zeros(len(r))
            for m in range(len(wk)):
                z = np.sum((r - ck[m]) ** 2, axis=1) / wk[m]
                f += pk[m] * shape.profile(z, D)
            if kernel == "mexican_hat":
                f = np.maximum(f, 0.0)
            total = f.sum()
            if not total > 0:
                continue  # nothing of the node inside: a point
            m_k = f @ r / total
            sigma[k] = np.sqrt(f @ np.sum((r - m_k) ** 2, axis=1) / total / D)
            full = _unit_mass(kernel, D) * np.sum(pk * wk ** (D / 2))
            mass[k] = total * voxel_volume / full
            out[k] = m_k
    elif w is not None:
        dmin = tree.query(c)[0]
        for k in range(K):
            s = np.sqrt(w[k] / 2)
            idx = tree.query_ball_point(c[k], dmin[k] + _TAIL_SIGMAS * s)
            r = coords[idx]
            d2 = np.sum((r - c[k]) ** 2, axis=1)
            f = np.exp(-(d2 - dmin[k] ** 2) / w[k])  # scaled by exp(dmin^2 / w)
            total = f.sum()
            m = f @ r / total
            sigma[k] = np.sqrt(f @ np.sum((r - m) ** 2, axis=1) / total / D)
            mass[k] = (
                total
                * np.exp(-(dmin[k] ** 2) / w[k])
                * voxel_volume
                / (np.pi * w[k]) ** (D / 2)
            )
            out[k] = m
    inside = grid.query(out / spacing, p=np.inf)[0] <= half
    for k in np.flatnonzero(~inside).tolist():
        snapped[k] = True
        hit = _first_hit(grid, out[k] / spacing, c[k] / spacing)
        if hit is not None:
            i, tie = _lowest_tied(grid, hit[None], _TIE_TOL, np.inf)
        else:
            i, tie = _lowest_tied(tree, out[k : k + 1], _TIE_TOL * spacing.min(), 2.0)
        out[k], ambiguous[k] = coords[i[0]], tie[0]
    widths_eff = 2 * sigma**2
    info = {
        "distance": np.linalg.norm(out - c, axis=1),
        "snapped": snapped,
        "ambiguous": ambiguous,
        "mass_fraction": mass,
        "sigma": sigma,
        "width_ratio": widths_eff / w if w is not None else np.full(K, np.nan),
    }
    return out, widths_eff, info


def _first_hit(grid: Any, start: Array, stop: Array) -> Array | None:
    """First point within half a voxel of the mask on the segment start-stop.

    Both ends are in voxel units (coordinates / spacing). The walk takes
    quarter-voxel steps, in chunks so that it stops at the first hit, and
    ends where the segment leaves the mask's bounding box.
    """
    pts = grid.data
    lo, hi = pts.min(axis=0) - 0.5, pts.max(axis=0) + 0.5
    d = stop - start
    t_end = 1.0
    for a in np.flatnonzero(d):
        t_a = ((hi[a] if d[a] > 0 else lo[a]) - start[a]) / d[a]
        t_end = min(t_end, max(t_a, 0.0))
    n = max(1, int(np.ceil(t_end * np.linalg.norm(d) / 0.25)))
    t = np.linspace(0.0, t_end, n + 1)
    for chunk in np.array_split(t, max(1, len(t) // 256)):
        seg = start + chunk[:, None] * d
        hit = np.flatnonzero(grid.query(seg, p=np.inf)[0] <= 0.5 * (1 + 1e-9))
        if len(hit):
            return np.asarray(seg[hit[0]])
    return None


def _default_mask(nodes: Any, subject: Any) -> Any:
    """The fit's own voxels for an HTFAResults, else the MNI152 brain."""
    from .results import HTFAResults

    if isinstance(nodes, HTFAResults):
        return nodes._display_mask(subject)
    return _mni_mask()


def _display_nodes(
    nodes: Any,
    sizes: Any,
    widths: Any,
    subject: Any,
    snap: bool,
    mask: Any,
    scale_by_mass: bool = False,
    fade_by_mass: bool | float = False,
) -> tuple[Array, Array, Array | None]:
    """Centers, sizes and per-node opacities (or None) the plots draw."""
    centers, fit_widths = _fit_nodes(nodes, subject)
    K = len(centers)
    w = fit_widths if widths is None else _per_node(widths, K, "widths")
    floor = None
    if fade_by_mass is not False:
        floor = 0.15 if fade_by_mass is True else float(fade_by_mass)
        if not 0 <= floor < 1:
            raise ValueError(
                f"fade_by_mass must be True or an opacity floor in [0, 1), "
                f"got {fade_by_mass!r}"
            )
    if not snap:
        if scale_by_mass or floor is not None:
            raise ValueError("scale_by_mass and fade_by_mass need snap=True")
        return (*_nodes(nodes, w if sizes is None else sizes, subject), None)
    if w is None and (scale_by_mass or floor is not None):
        raise ValueError("scale_by_mass and fade_by_mass need node widths (widths=)")
    if mask is None:
        mask = _default_mask(nodes, subject)
    shape = _fit_shape(nodes, subject) if widths is None else None
    if shape is None:
        centers, w_eff, info = effective_nodes(centers, w, mask)
    else:
        name, comp_c, comp_w, pi = shape
        centers, w_eff, info = effective_nodes(comp_c, comp_w, mask, name, pi)
    if sizes is not None:
        node_sizes = _per_node(sizes, K, "sizes")
    else:
        node_sizes = np.ones(K) if w is None else w_eff
    if scale_by_mass:
        node_sizes = node_sizes * info["mass_fraction"]
    opacity = None
    if floor is not None:
        opacity = floor + (1 - floor) * np.clip(info["mass_fraction"], 0, 1)
    return centers, node_sizes, opacity


def _edge_list(E: Array, threshold: float) -> tuple[Array, Array, Array]:
    i, j = np.triu_indices(len(E), 1)
    w = E[i, j]
    keep = np.abs(w) > threshold
    return i[keep], j[keep], w[keep]


def _tubes(
    centers: Array, radii: Array, i: Array, j: Array, widths: Array, sides: int = 20
) -> Any:
    """Open cylinders from the surface of sphere i to that of sphere j."""
    pv = lazy_import("pyvista", "brain3d", "network plots")
    p, q = centers[i], centers[j]
    u = (q - p) / np.linalg.norm(q - p, axis=1, keepdims=True)
    a = p + u * radii[i, None]
    b = q - u * radii[j, None]
    ref = np.where(np.abs(u[:, [2]]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
    e1 = np.cross(u, ref)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(u, e1)
    th = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ring = (
        np.cos(th)[None, :, None] * e1[:, None]
        + np.sin(th)[None, :, None] * e2[:, None]
    )
    ring = ring * widths[:, None, None]
    pts = np.concatenate([a[:, None] + ring, b[:, None] + ring], axis=1)
    n = len(i)
    base = (np.arange(n) * 2 * sides)[:, None]
    k = np.arange(sides)[None]
    k1 = (k + 1) % sides
    quads = np.stack(
        [
            np.full((n, sides), 4),
            base + k,
            base + k1,
            base + sides + k1,
            base + sides + k,
        ],
        axis=-1,
    )
    return pv.PolyData(pts.reshape(-1, 3), quads.reshape(-1))


# VTK's offscreen OpenGL windows for Linux without a display, in order of
# preference: EGL (libEGL with Mesa's llvmpipe or a GPU driver; preinstalled on
# Colab), then OSMesa.
HEADLESS_WINDOWS = ("vtkEGLRenderWindow", "vtkOSOpenGLRenderWindow")
_GL_PROBE = """
import sys
sys.path[:] = sys.argv[1:]
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkRenderingCore import vtkRenderer, vtkRenderWindow
w = vtkRenderWindow()
w.SetOffScreenRendering(1)
w.SetSize(8, 8)
w.AddRenderer(vtkRenderer())
w.Render()
print(w.GetClassName())
"""
_auto_window: str | None = None  # the window _headless_window selected
_no_window = False  # no offscreen window worked (cleared when DISPLAY appears)


def _gl_window_works(window: str) -> bool:
    """Whether VTK renders with the OpenGL window class ``window``.

    Probed in a subprocess, because VTK crashes the process when the library
    behind a window class is missing. The subprocess gets this process's
    ``sys.path`` (so it finds the same VTK) and runs in a fresh private
    directory.
    """
    env = dict(os.environ, VTK_DEFAULT_OPENGL_WINDOW=window)
    path = [os.path.abspath(p) if p else os.getcwd() for p in sys.path]
    work = tempfile.mkdtemp(prefix="htfa-glprobe-")
    try:
        run = subprocess.run(
            [sys.executable, "-c", _GL_PROBE, *path],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
            cwd=work,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return run.returncode == 0 and run.stdout.split()[-1:] == [window]


def _headless_window() -> str | None:
    """Select VTK's offscreen window on Linux without a display.

    Returns the window class in use (setting ``VTK_DEFAULT_OPENGL_WINDOW`` if it
    was unset), or ``None`` when a display (or another OS) needs no change.
    The choice is re-checked on every call: once ``DISPLAY`` is set (for
    example by starting Xvfb), a window selected here is dropped again.
    """
    global _auto_window, _no_window
    if not sys.platform.startswith("linux"):
        return None
    if os.environ.get("DISPLAY"):
        if _auto_window and os.environ.get("VTK_DEFAULT_OPENGL_WINDOW") == _auto_window:
            del os.environ["VTK_DEFAULT_OPENGL_WINDOW"]
        _auto_window, _no_window = None, False
        return None
    chosen = os.environ.get("VTK_DEFAULT_OPENGL_WINDOW")
    if chosen:
        return chosen
    if not _no_window:
        for window in HEADLESS_WINDOWS:
            if _gl_window_works(window):
                os.environ["VTK_DEFAULT_OPENGL_WINDOW"] = window
                _auto_window = window
                return window
        _no_window = True
    raise RuntimeError(
        "No display (DISPLAY is unset) and no offscreen OpenGL library: VTK needs "
        "Mesa's EGL or OSMesa to render without one. Install EGL with "
        "`apt-get install libegl1 libegl-mesa0` (Debian/Ubuntu; Colab has it) and "
        "restart Python, or set DISPLAY to a running X server such as Xvfb."
    )


class _Scene:
    """Off-screen plotter holding mesh and nodes; edges change per frame."""

    def __init__(
        self,
        centers: Array,
        sizes: Array,
        mesh: tuple[Array, Array] | None,
        view: str,
        size: tuple[int, int],
        node_color: Any,
        mesh_color: Any,
        mesh_opacity: float,
        background: Any,
        node_opacity: Array | None = None,
        radius_range: tuple[float, float] = (2.0, 7.0),
        radius_kb: tuple[float, float] | None = None,
        radius_clip: tuple[float, float] | None = None,
    ) -> None:
        pv = lazy_import("pyvista", "brain3d", "network plots")
        if view not in VIEWS:
            raise ValueError(f"unknown view {view!r}; choose from {sorted(VIEWS)}")
        self.centers = centers
        k, b = (None, None) if radius_kb is None else radius_kb
        self.radii = node_radius(
            sizes, radius_range[0], radius_range[1], k=k, b=b, clip=radius_clip
        )
        if not np.all(self.radii > 0):
            raise ValueError(
                f"node radii must be positive, got min {self.radii.min():.3g} mm; "
                "set radius_clip"
            )
        self.node_opacity = node_opacity
        self.size = (int(size[0]), int(size[1]))
        self.view_dir, self.view_up = (np.asarray(v, float) for v in VIEWS[view])
        self.verts = None if mesh is None else np.asarray(mesh[0], float)
        s = _SUPERSAMPLE
        _headless_window()
        pl = pv.Plotter(
            off_screen=True,
            window_size=[self.size[0] * s, self.size[1] * s],
            lighting="none",
        )
        # Depth peeling (needed for the translucent mesh and edges) draws no
        # translucent geometry into a multisampled framebuffer on Mesa
        # (llvmpipe, Linux CI). The 2x supersampling above anti-aliases.
        pl.render_window.SetMultiSamples(0)
        try:
            self._build(
                pv, pl, mesh, view, node_color, mesh_color, mesh_opacity, background
            )
        except BaseException:
            pl.close()
            raise
        self.pl = pl
        self._edges: Any = None

    def _build(
        self,
        pv: Any,
        pl: Any,
        mesh: tuple[Array, Array] | None,
        view: str,
        node_color: Any,
        mesh_color: Any,
        mesh_opacity: float,
        background: Any,
    ) -> None:
        """Add the background, mesh, nodes, camera and light to ``pl``."""
        centers = self.centers
        pl.set_background(background)
        if mesh is not None:
            verts, faces = mesh
            cells = np.c_[np.full(len(faces), 3), faces].ravel()
            brain = pv.PolyData(verts, cells).compute_normals()
            pl.add_mesh(
                brain,
                color=mesh_color,
                opacity=float(mesh_opacity),
                smooth_shading=True,
                **_DULL,
            )
        spheres = [
            pv.Sphere(radius=r, center=c, theta_resolution=40, phi_resolution=40)
            for c, r in zip(centers, self.radii, strict=True)
        ]
        if self.node_opacity is None:
            pl.add_mesh(
                pv.merge(spheres), color=node_color, smooth_shading=True, **_DULL
            )
        else:
            for sphere, alpha in zip(spheres, self.node_opacity, strict=True):
                pl.add_mesh(
                    sphere,
                    color=node_color,
                    opacity=float(alpha),
                    smooth_shading=True,
                    **_DULL,
                )
        pl.enable_depth_peeling(number_of_peels=8)
        pl.enable_parallel_projection()
        direction, up = VIEWS[view]
        axial = view in ("dorsal", "ventral")
        focal = _FOCAL_AXIAL if axial else _FOCAL
        pl.camera.focal_point = focal
        pl.camera.position = focal + 500.0 * np.asarray(direction)
        pl.camera.up = up
        pl.camera.parallel_scale = _PARALLEL_SCALE * (1.05 if axial else 1.0)
        pl.camera.clipping_range = (1, 1000)
        # MATLAB camlight('right'): 30 degrees right of and above the camera.
        el = az = np.deg2rad(30)
        pl.add_light(
            pv.Light(
                light_type="camera light",
                position=(np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)),
                focal_point=(0, 0, 0),
                intensity=1.0,
                positional=False,
            )
        )

    def set_edges(
        self,
        E: Array,
        threshold: float,
        edge_style: str,
        positive_color: Any,
        negative_color: Any,
    ) -> None:
        if self._edges is not None:
            self.pl.remove_actor(self._edges)
            self._edges = None
        i, j, w = _edge_list(E, threshold)
        if not len(i):
            return
        pos = np.asarray(_rgb(positive_color))
        neg = np.asarray(_rgb(negative_color))
        rgb = np.where((w > 0)[:, None], pos, neg)
        if edge_style == "intensity":
            radius, opacity = 0.5 * np.abs(w), EDGE_OPACITY
        else:
            radius, opacity = np.full(len(w), 0.25), 1.0
        tubes = _tubes(self.centers, self.radii, i, j, radius)
        per_edge = tubes.n_cells // len(i)
        tubes.cell_data["rgb"] = np.round(
            np.repeat(rgb, per_edge, axis=0) * 255
        ).astype(np.uint8)
        self._edges = self.pl.add_mesh(
            tubes,
            scalars="rgb",
            rgb=True,
            opacity=opacity,
            smooth_shading=False,
            show_scalar_bar=False,
            **_DULL,
        )

    def fit_spin(self, rotations: Array) -> tuple[Array, float]:
        """Rotation center and parallel scale for a spinning camera.

        The center is the middle of the mesh's bounding box (of the node
        spheres' without a mesh). The scale is 5% more than the smallest one
        that keeps the mesh and the nodes inside the image for every brain
        rotation in ``rotations`` (shape (n, 3, 3)).
        """
        if self.verts is None:
            lo = (self.centers - self.radii[:, None]).min(axis=0)
            hi = (self.centers + self.radii[:, None]).max(axis=0)
        else:
            lo, hi = self.verts.min(axis=0), self.verts.max(axis=0)
        center = (lo + hi) / 2
        w, h = self.size
        right = np.cross(self.view_dir, self.view_up)
        nodes = self.centers - center
        scale = 0.0
        for start in range(0, len(rotations), 16):
            R = rotations[start : start + 16]
            # image axes in brain coordinates: R^T up and R^T right
            axes = np.einsum("nji,aj->ain", R, np.stack([self.view_up, right]))
            ext = (np.abs(nodes @ axes) + self.radii[:, None]).max(axis=1)
            if self.verts is not None:
                ext = np.maximum(ext, np.abs((self.verts - center) @ axes).max(axis=1))
            scale = max(scale, float(np.max(np.maximum(ext[0], ext[1] * h / w))))
        return center, 1.05 * scale

    def orbit(self, R: Array, center: Array, scale: float) -> None:
        """Place the camera so the brain looks rotated by ``R`` about ``center``."""
        # Camera outside the sphere (radius r about center) that holds the
        # mesh and the nodes; the clipping range spans that sphere.
        r = 1.05 * self._reach(center) + 1.0
        cam = self.pl.camera
        cam.focal_point = center
        cam.position = center + 2.0 * r * (R.T @ self.view_dir)
        cam.up = R.T @ self.view_up
        cam.parallel_scale = scale
        cam.clipping_range = (r, 3.0 * r)

    def _reach(self, center: Array) -> float:
        """Largest distance (mm) from ``center`` to the mesh or a node surface."""
        d = np.linalg.norm(self.centers - center, axis=1) + self.radii
        reach = float(d.max()) if d.size else 0.0
        if self.verts is not None:
            reach = max(reach, float(np.linalg.norm(self.verts - center, axis=1).max()))
        return reach

    def image(self) -> Any:
        Image = _pil()
        self.pl.render()
        shot = self.pl.screenshot(None, return_img=True)[..., :3]
        return Image.fromarray(np.ascontiguousarray(shot)).resize(self.size, Image.BOX)

    def close(self) -> None:
        self.pl.close()


def _rgb(color: Any) -> tuple[float, float, float]:
    pv = lazy_import("pyvista", "brain3d", "network plots")
    r, g, b = pv.Color(color).float_rgb
    return float(r), float(g), float(b)


def _connectivity(connectivity: Any, K: int, ndim: int) -> Array:
    E = np.asarray(connectivity, dtype=np.float64)
    if E.ndim != ndim or E.shape[-2:] != (K, K):
        want = "(K, K)" if ndim == 2 else "(T, K, K) (or (K, K) with duration)"
        raise ValueError(f"connectivity must be {want} with K={K}, got {E.shape}")
    if not np.all(np.isfinite(E)):
        raise ValueError("connectivity contains NaN or inf")
    return E


MATLAB_JPEG_QUALITY = 75  # MATLAB's default JPEG export quality (2014 frames)


def _is_jpeg(path: Path) -> bool:
    return path.suffix.lower() in (".jpg", ".jpeg")


def _jpeg(img: Any, quality: int) -> Any:
    """``img`` after a JPEG round trip with 4:2:0 chroma, as MATLAB writes."""
    import io

    Image = _pil()
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality), subsampling=2)
    buf.seek(0)
    with Image.open(buf) as im:
        return im.convert("RGB")


def _save(img: Any, path: Path, jpeg_quality: int | None) -> None:
    if _is_jpeg(path):
        quality = MATLAB_JPEG_QUALITY if jpeg_quality is None else jpeg_quality
        img.save(path, quality=int(quality), subsampling=2)
    else:
        img.save(path)


def _style(kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split keyword arguments into scene and edge options, with defaults."""
    scene = {
        "view": kwargs.pop("view", "lateral"),
        "size": kwargs.pop("size", SIZE),
        "node_color": kwargs.pop("node_color", (0.502, 0.502, 0.502)),
        "mesh_color": kwargs.pop("mesh_color", (0.95, 0.95, 0.95)),
        "mesh_opacity": kwargs.pop("mesh_opacity", 0.2),
        "background": kwargs.pop("background", "white"),
        "radius_range": tuple(kwargs.pop("radius_range", (2.0, 7.0))),
        "radius_kb": kwargs.pop("radius_kb", None),
        "radius_clip": kwargs.pop("radius_clip", None),
    }
    edges = {
        "threshold": float(kwargs.pop("threshold", 0.25)),
        "edge_style": kwargs.pop("edge_style", "intensity"),
        "positive_color": kwargs.pop("positive_color", "red"),
        "negative_color": kwargs.pop("negative_color", "blue"),
    }
    if kwargs:
        raise TypeError(f"unexpected keyword arguments: {sorted(kwargs)}")
    if edges["edge_style"] not in ("intensity", "flat"):
        raise ValueError(
            f"edge_style must be 'intensity' or 'flat', got {edges['edge_style']!r}"
        )
    return scene, edges


def plot_network(
    nodes: Any,
    connectivity: Any = None,
    *,
    sizes: Any = None,
    widths: Any = None,
    subject: Any = "global",
    mesh: Any = "ch2",
    snap: bool = True,
    mask: Any = None,
    scale_by_mass: bool = False,
    fade_by_mass: bool | float = False,
    out: str | os.PathLike | None = None,
    jpeg_quality: int | None = None,
    **style: Any,
) -> Any:
    """Render nodes and thresholded edges BrainNet Viewer-style.

    Parameters
    ----------
    nodes : HTFAResults, TFA, HTFA or array of shape (K, 3)
        Node centers in MNI mm. For a fit, the centers of ``subject`` are
        used and its widths set the node sizes.
    connectivity : array of shape (K, K), optional
        Edge weights (upper triangle used). ``None`` draws nodes only.
    sizes : array of shape (K,), optional
        Node sizes; mapped linearly onto 2-7 mm radii (`node_radius`; see the
        ``radius_*`` style options).
        Defaults to the (effective, with ``snap``) widths, or equal sizes
        (4 mm) when no widths are known.
    widths : array of shape (K,), optional
        RBF widths of array ``nodes`` (a fit's own widths are used
        otherwise).
    subject : "global" or int
        Which centers to use when ``nodes`` is a fit.
    mesh : str, path, (vertices, faces) or None
        Brain surface; see `load_mesh`. Default ``"ch2"``.
    snap : bool
        Draw each node as its in-brain part (`effective_nodes`): the
        RBF-weighted centroid of the mask voxels, with width
        ``lambda_eff``. Default ``True``; ``False`` draws the fitted centers
        and widths as they are. The fit itself is never changed, and
        `effective_nodes` (or `htfa.HTFAResults.display_nodes`) returns the
        drawn values. Nodes without widths (a bare array without ``widths``)
        are points: they stay if inside the mask and otherwise move to the
        nearest mask voxel.
    mask : array of shape (V, 3), NIfTI image or (volume, affine), optional
        The in-brain voxels for ``snap``. Default: for an HTFAResults, the
        subject's voxel coordinates (all subjects' for ``"global"``);
        otherwise nilearn's 2 mm MNI152 brain mask (`mni_mask`), because the
        BrainNet meshes are in MNI space. Raises ``ImportError`` if that needs
        nilearn and nilearn cannot be installed.
    scale_by_mass : bool
        Multiply the node sizes by the in-mask RBF mass fraction before they
        are mapped to radii (needs widths and ``snap``).
    fade_by_mass : bool or float
        Node opacity ``floor + (1 - floor) * mass_fraction`` with floor 0.15
        (``True``) or the given floor, so nodes mostly outside the brain stay
        faintly visible (needs widths and ``snap``).
    out : path, optional
        Save the image there: PNG, or JPEG for a ``.jpg``/``.jpeg`` suffix
        (quality ``jpeg_quality``, default 75, 4:2:0 chroma, like MATLAB's
        export of the 2014 frames).
    jpeg_quality : int, optional
        If given, the image is JPEG-compressed at this quality before it is
        returned or saved. JPEG at 75 makes thin edges paler, as in the 2014
        BrainNet frames.
    **style
        ``threshold`` (0.25; edges with ``|E| > threshold`` are drawn),
        ``view`` (``"lateral"`` = ``"left"``, ``"right"``, ``"dorsal"``,
        ``"ventral"``, ``"anterior"``, ``"posterior"``), ``size``
        ((1600, 1000) pixels), ``edge_style`` (``"intensity"``: diameter
        ``|E|`` mm at opacity 0.7; ``"flat"``: 0.5 mm, opaque),
        ``positive_color`` ("red"), ``negative_color`` ("blue"),
        ``node_color`` (0.502 gray), ``mesh_color`` (0.95 gray),
        ``mesh_opacity`` (0.2), ``background`` ("white"),
        ``radius_range`` ((2, 7) mm: radii of the smallest and largest
        size), ``radius_kb`` (a fixed ``(k, b)``: radius ``k * size + b``
        mm, comparable across plots), ``radius_clip`` (``(min, max)`` mm
        applied to the radii); see `node_radius`. See also
        `config_style`.

    Returns
    -------
    PIL.Image.Image
        The RGB image.
    """
    scene_opts, edge_opts = _style(dict(style))
    centers, node_sizes, opacity = _display_nodes(
        nodes, sizes, widths, subject, snap, mask, scale_by_mass, fade_by_mass
    )
    E = None
    if connectivity is not None:
        E = _connectivity(connectivity, len(centers), 2)
    scene = _Scene(
        centers, node_sizes, load_mesh(mesh), **scene_opts, node_opacity=opacity
    )
    try:
        if E is not None:
            scene.set_edges(E, **edge_opts)
        img = scene.image()
    finally:
        scene.close()
    if jpeg_quality is not None:
        img = _jpeg(img, jpeg_quality)
    if out is not None:
        _save(img, Path(out), jpeg_quality)
    return img


def _check_fps(fps: float) -> None:
    if not (math.isfinite(float(fps)) and float(fps) > 0):
        raise ValueError(f"fps must be a positive, finite number, got {fps!r}")


def _movie_length(
    duration: float | None, n_frames: int | None, fps: float
) -> int | None:
    """Number of movie frames set by ``duration`` or ``n_frames`` (else None)."""
    _check_fps(fps)
    if duration is not None and n_frames is not None:
        raise ValueError("give duration or n_frames, not both")
    if duration is not None:
        if not (math.isfinite(float(duration)) and float(duration) > 0):
            raise ValueError(
                f"duration must be a positive, finite number of seconds, "
                f"got {duration!r}"
            )
        n = round(float(duration) * fps)
        if n < 1:
            raise ValueError(f"duration {duration} s at {fps} fps gives no frames")
        return n
    if n_frames is not None and (int(n_frames) != n_frames or n_frames < 1):
        raise ValueError(f"n_frames must be a positive integer, got {n_frames!r}")
    return None if n_frames is None else int(n_frames)


def _spin_rotations(spin: Any, n: int) -> Array:
    """Brain rotation of each of ``n`` frames, shape (n, 3, 3).

    Frame k turns the brain by ``spin * k / n`` full turns about the x, then
    the y, then the z axis (right-hand rule).
    """
    turns = np.asarray(spin, dtype=np.float64)
    if turns.shape != (3,) or not np.all(np.isfinite(turns)):
        raise ValueError(
            f"spin must be three finite numbers of turns (x, y, z), got {spin!r}"
        )
    c, s = (f(2 * np.pi * np.outer(np.arange(n) / n, turns)) for f in (np.cos, np.sin))
    one = np.ones(n)

    def rot(a: int, b: int, axis: int) -> Array:
        R = np.zeros((n, 3, 3))
        R[:, axis, axis] = one
        R[:, a, a] = R[:, b, b] = c[:, axis]
        R[:, a, b], R[:, b, a] = -s[:, axis], s[:, axis]
        return R

    return rot(0, 1, 2) @ rot(2, 0, 1) @ rot(1, 2, 0)


def _nearest(t: float, T: int) -> int:
    """Index of the window nearest to time ``t`` (halves round up)."""
    return min(int(np.floor(t + 0.5)), T - 1)


def _at_time(E: Array, t: float, interpolation: str) -> Array:
    """Connectivity at fractional window index ``t``."""
    if interpolation == "nearest":
        return E[_nearest(t, len(E))]
    lo = min(int(np.floor(t)), len(E) - 1)
    frac = t - lo
    if frac == 0:
        return E[lo]
    return (1 - frac) * E[lo] + frac * E[lo + 1]


def animate_network(
    nodes: Any,
    connectivity: Any,
    out: str | os.PathLike,
    *,
    fps: float = 10,
    duration: float | None = None,
    n_frames: int | None = None,
    interpolation: str = "linear",
    spin: Sequence[float] | None = None,
    sizes: Any = None,
    widths: Any = None,
    subject: Any = "global",
    mesh: Any = "ch2",
    snap: bool = True,
    mask: Any = None,
    scale_by_mass: bool = False,
    fade_by_mass: bool | float = False,
    captions: Sequence[str] | Callable[[float], str] | None = None,
    frames_dir: str | os.PathLike | None = None,
    jpeg_quality: int | None = None,
    **style: Any,
) -> Path:
    """Render a ``(T, K, K)`` connectivity series as an mp4 or GIF.

    By default the movie has one frame per connectivity window, and the
    camera, mesh, nodes and threshold stay fixed. ``duration`` (or
    ``n_frames``) sets the number of frames instead, interpolating between
    windows, and ``spin`` turns the brain.

    Parameters
    ----------
    nodes, sizes, widths, subject, mesh, snap, mask, scale_by_mass, fade_by_mass
        As in `plot_network` (``snap`` is on by default).
    **style
        As in `plot_network`.
    connectivity : array of shape (T, K, K), or (K, K) with duration
        One edge matrix per window. A single ``(K, K)`` matrix needs
        ``duration`` or ``n_frames`` and gives fixed edges (for a spin).
    out : path
        ``.mp4`` (H.264 via imageio-ffmpeg) or ``.gif``. An mp4 is encoded
        while frames render, holding one frame in memory at a time. A GIF
        needs all frames for its shared palette, about 4.8 MB per
        1600 x 1000 frame (see `write_animation`).
    fps : float
        Frame rate of the movie, in frames per second (default 10).
    duration : float, optional
        Length of the movie in seconds: it gets ``round(duration * fps)``
        frames whatever T is, so the windows play in slow motion or fast
        forward. Movie frame ``i`` of ``n`` shows window time
        ``t = i * (T - 1) / (n - 1)`` (all of the windows, first to last).
    n_frames : int, optional
        The number of movie frames, instead of ``duration``.
    interpolation : {"linear", "nearest"}
        How a fractional ``t`` is drawn: ``"linear"`` (default) mixes the
        two neighbouring windows, ``(1 - f) * E[floor(t)] + f * E[floor(t)
        + 1]`` with ``f = t - floor(t)``, so edges fade in and out
        smoothly as their weights cross ``threshold``; ``"nearest"`` shows
        the nearest window (halves round up).
    spin : (x, y, z) turns, optional
        Number of full turns of the brain about the MNI x, y and z axes over
        the whole movie, through the center of the mesh's bounding box (of
        the nodes without a mesh). Fractions are allowed. Movie frame ``k``
        of ``n`` turns the brain by ``spin * k / n`` turns, about x first,
        then y, then z (right-hand rule: ``(0, 0, 0.25)`` brings the front
        of the brain towards the camera in the left lateral view). The turn
        is made by orbiting the camera (with its light) around the brain,
        with the parallel projection kept and the zoom fixed at the largest
        one that keeps the brain and nodes inside every frame. With a whole
        number of turns the movie loops seamlessly: frame ``n`` would be
        frame 0 again. ``None`` (default) keeps the fixed camera of
        ``view``; any spin, even ``(0, 0, 0)``, uses the centered, fitted
        zoom.
    captions : sequence of T str, or callable, optional
        Text shown in a white band above each frame (not over the brain).
        A sequence gives the caption of each window (a movie frame takes
        that of the nearest window); a callable is called with each frame's
        window time ``t`` (a float) and returns its caption.
    frames_dir : path, optional
        Also save each frame there as ``1.png``, ``2.png``, ... (``.jpg``
        with ``jpeg_quality``).
    jpeg_quality : int, optional
        JPEG-compress each frame at this quality before it goes into the
        movie (75 reproduces the look of the 2014 frames).

    Returns
    -------
    pathlib.Path
        ``out``.

    Examples
    --------
    A 6 s movie of 30 windows at 15 frames per second (90 frames, three per
    window, interpolated), turning once about the vertical axis:

    >>> animate_network(centers, E, "net.mp4", fps=15, duration=6,
    ...                 spin=(0, 0, 1))  # doctest: +SKIP

    A 4 s spin of one static network:

    >>> animate_network(centers, C, "spin.gif", fps=12, duration=4,
    ...                 spin=(0, 0, 1), size=(480, 300))  # doctest: +SKIP
    """
    scene_opts, edge_opts = _style(dict(style))
    if interpolation not in ("linear", "nearest"):
        raise ValueError(
            f"interpolation must be 'linear' or 'nearest', got {interpolation!r}"
        )
    n = _movie_length(duration, n_frames, fps)
    out = Path(out)
    if out.suffix.lower() not in (".mp4", ".gif"):
        raise ValueError(f"out must end in .mp4 or .gif, got {out.name!r}")
    centers, node_sizes, opacity = _display_nodes(
        nodes, sizes, widths, subject, snap, mask, scale_by_mass, fade_by_mass
    )
    K = len(centers)
    if n is not None and np.ndim(connectivity) == 2:
        E = _connectivity(connectivity, K, 2)[None]
    else:
        E = _connectivity(connectivity, K, 3)
    T = len(E)
    if callable(captions):
        caption: Callable[[float], str] | None = captions
    elif captions is not None:
        if len(captions) != T:
            raise ValueError(f"{len(captions)} captions for {T} frames")
        seq = list(captions)

        def caption(t: float) -> str:
            return seq[_nearest(t, T)]

    else:
        caption = None
    times = np.arange(T, dtype=np.float64) if n is None else np.linspace(0, T - 1, n)
    rotations = None if spin is None else _spin_rotations(spin, len(times))
    folder = None if frames_dir is None else Path(frames_dir)
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
    scene = _Scene(
        centers, node_sizes, load_mesh(mesh), **scene_opts, node_opacity=opacity
    )

    def frames() -> Iterator[Any]:
        if rotations is not None:
            center, scale = scene.fit_spin(rotations)
        for k, t in enumerate(times):
            if rotations is not None:
                scene.orbit(rotations[k], center, scale)
            scene.set_edges(_at_time(E, float(t), interpolation), **edge_opts)
            img = scene.image()
            if caption is not None:
                img = _with_header(img, [caption(float(t))])
            if jpeg_quality is not None:
                img = _jpeg(img, jpeg_quality)
            if folder is not None:
                ext = ".png" if jpeg_quality is None else ".jpg"
                _save(img, folder / f"{k + 1}{ext}", jpeg_quality)
            yield img

    try:
        return write_animation(frames(), out, fps=fps)
    finally:
        scene.close()


# ------------------------------------------------------------------ animations


def write_animation(
    images: Iterable[Any], out: str | os.PathLike, fps: float = 10
) -> Path:
    """Write PIL images as an mp4 (H.264, yuv420p) or looping GIF, by suffix.

    ``images`` may be any iterable, such as a generator. An mp4 is written
    frame by frame as the iterable yields, so only one frame is held in
    memory at a time.

    GIFs use one palette for all frames (white, white-to-red and white-to-blue
    ramps, plus median-cut colors of all frames), without dithering, so colors
    do not flicker between frames. Choosing that palette needs every frame,
    so a GIF holds all frames in memory: about 4.8 MB per 1600 x 1000 frame
    (300 frames: about 1.4 GB). Use mp4 for long movies. Pillow stores
    identical consecutive GIF frames once, with their durations added, so
    such a GIF has fewer frames but the same running time.
    """
    _check_fps(fps)
    out = Path(out)
    suffix = out.suffix.lower()
    if suffix not in (".mp4", ".gif"):
        raise ValueError(f"out must end in .mp4 or .gif, got {out.name!r}")
    if suffix == ".gif":
        frames = [im.convert("RGB") for im in images]
        if not frames:
            raise ValueError("no frames to write")
        if len({f.size for f in frames}) != 1:
            raise ValueError("all frames must have the same size")
        _write_gif(frames, out, fps)
        return out
    iio = lazy_import("imageio.v2", "brain3d", "mp4 export")
    lazy_import("imageio_ffmpeg", "brain3d", "mp4 export")
    size = None
    with iio.get_writer(
        out,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=2,
        pixelformat="yuv420p",
    ) as writer:
        for im in images:
            frame = im.convert("RGB")
            if size is None:
                size = frame.size
            elif frame.size != size:
                raise ValueError("all frames must have the same size")
            w, h = size
            writer.append_data(np.asarray(frame.crop((0, 0, w - w % 2, h - h % 2))))
    if size is None:
        out.unlink(missing_ok=True)
        raise ValueError("no frames to write")
    return out


def _write_gif(frames: list[Any], out: Path, fps: float) -> None:
    Image = _pil()
    w, h = frames[0].size
    tw, th = max(1, w // 2), max(1, h // 2)
    mosaic = Image.new("RGB", (tw, th * len(frames)))
    for k, f in enumerate(frames):
        mosaic.paste(f.resize((tw, th), Image.BOX), (0, k * th))
    t = np.linspace(0, 1, 16)[:, None]
    ramps = np.vstack(
        [[255.0, 255.0, 255.0]]
        + [255 * (1 - t) + np.array(c) * t for c in ([255, 0, 0], [0, 0, 255])]
    )
    n_free = 255 - len(ramps)  # index 255 is reserved for transparency
    rest = np.asarray(
        mosaic.quantize(colors=n_free, method=Image.Quantize.MEDIANCUT).getpalette()[
            : 3 * n_free
        ]
    ).reshape(-1, 3)
    palette = np.vstack([np.round(ramps), rest]).astype(np.int64)
    transparent = len(palette)
    flat_palette = np.vstack([palette, [[255, 0, 255]]]).ravel().tolist()
    tree = cKDTree(palette)
    indexed = []
    for f in frames:
        a = np.asarray(f).reshape(-1, 3).astype(np.int64)
        packed = (a[:, 0] << 16) | (a[:, 1] << 8) | a[:, 2]
        uniq, inv = np.unique(packed, return_inverse=True)
        cols = np.c_[uniq >> 16, (uniq >> 8) & 255, uniq & 255]
        idx = tree.query(cols)[1].astype(np.uint8)
        indexed.append(idx[inv.ravel()].reshape(h, w))
    pages = []
    for k, a in enumerate(indexed):
        b = a.copy()
        if k:
            b[a == indexed[k - 1]] = transparent
        page = Image.fromarray(b, "P")
        page.putpalette(flat_palette)
        pages.append(page)
    pages[0].save(
        out,
        save_all=True,
        append_images=pages[1:],
        duration=int(round(1000 / fps)),
        loop=0,
        transparency=transparent,
        disposal=1,
        optimize=False,
    )


# --------------------------------------------------------------------- panels


def image_diff(a: Any, b: Any, size: tuple[int, int] = (400, 250)) -> float:
    """Mean absolute RGB difference (0-255) of two images resized to ``size``.

    ``a`` and ``b`` are paths or PIL images.
    """
    Image = _pil()

    def arr(x: Any) -> Array:
        im = Image.open(x) if isinstance(x, (str, os.PathLike)) else x
        return np.asarray(
            im.convert("RGB").resize(size, Image.BILINEAR), dtype=np.float64
        )

    return float(np.abs(arr(a) - arr(b)).mean())


def _open(x: Any) -> Any:
    Image = _pil()
    if isinstance(x, (str, os.PathLike)):
        with Image.open(x) as im:
            return im.convert("RGB")
    return x.convert("RGB")


def _content_box(images: Sequence[Any], margin: int = 10) -> tuple[int, int, int, int]:
    """Smallest box holding every non-white pixel of all images, plus a margin."""
    w, h = images[0].size
    mask = np.zeros((h, w), dtype=bool)
    for im in images:
        if im.size != (w, h):
            raise ValueError("all images must have the same size")
        mask |= np.asarray(im).min(axis=2) < 250
    if not mask.any():
        return 0, 0, w, h
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    return (
        max(0, int(cols[0]) - margin),
        max(0, int(rows[0]) - margin),
        min(w, int(cols[-1]) + 1 + margin),
        min(h, int(rows[-1]) + 1 + margin),
    )


def _font(size: int) -> Any:
    font = lazy_import("PIL.ImageFont", "brain3d", "annotations")
    return font.load_default(size=size)


def _center_text(
    draw: Any, xc: float, y: float, text: str, font: Any, fill: Any
) -> None:
    draw.text(
        (xc - draw.textlength(text, font=font) / 2, y), text, font=font, fill=fill
    )


def _with_header(img: Any, lines: Sequence[str], fontsize: int | None = None) -> Any:
    """``img`` under a white band holding centered text lines."""
    Image = _pil()
    draw_mod = lazy_import("PIL.ImageDraw", "brain3d", "annotations")
    w, h = img.size
    size = fontsize or max(16, w // 32)
    font = _font(size)
    head = int(size * (0.5 + 1.3 * len(lines)))
    canvas = Image.new("RGB", (w, head + h), "white")
    canvas.paste(img, (0, head))
    d = draw_mod.Draw(canvas)
    for k, line in enumerate(lines):
        _center_text(d, w / 2, size * (0.25 + 1.3 * k), line, font, (0, 0, 0))
    return canvas


def annotate_pair(
    left: Any,
    right: Any,
    titles: tuple[str, str],
    caption: str | None = None,
    width: int = 1200,
    crop: tuple[int, int, int, int] | None = None,
) -> Any:
    """Two frames side by side under a header band with titles and a caption.

    Parameters
    ----------
    left, right : path or PIL image
        Frames of equal size.
    titles : (str, str)
        Panel titles; a ``"\\n"`` starts a smaller subtitle line.
    caption : str, optional
        A line centered under the titles (e.g. a time counter).
    width : int
        Output width in pixels.
    crop : (left, top, right, bottom), optional
        Region of each frame to show. Default: the box around all non-white
        pixels of both frames.

    Returns
    -------
    PIL.Image.Image
        Text sits only in the header band, never over the brain.
    """
    Image = _pil()
    draw_mod = lazy_import("PIL.ImageDraw", "brain3d", "annotations")
    ims = [_open(left), _open(right)]
    box = crop or _content_box(ims)
    pane_w = width // 2
    pane_h = round(pane_w * (box[3] - box[1]) / (box[2] - box[0]))
    tsize = max(16, width // 32)
    ssize = int(tsize * 0.8)
    n_sub = max(len(t.split("\n")) for t in titles) - 1
    head = int(
        tsize * 1.5
        + n_sub * ssize * 1.3
        + (ssize * 1.6 if caption else 0)
        + tsize * 0.4
    )
    ft, fs = _font(tsize), _font(ssize)
    canvas = Image.new("RGB", (width, head + pane_h), "white")
    d = draw_mod.Draw(canvas)
    for k, (im, title) in enumerate(zip(ims, titles, strict=True)):
        canvas.paste(
            im.crop(box).resize((pane_w, pane_h), Image.LANCZOS), (k * pane_w, head)
        )
        lines = title.split("\n")
        xc = k * pane_w + pane_w / 2
        _center_text(d, xc, int(tsize * 0.3), lines[0], ft, (0, 0, 0))
        for m, line in enumerate(lines[1:]):
            _center_text(
                d, xc, int(tsize * 1.5 + m * ssize * 1.3), line, fs, (60, 60, 60)
            )
    if caption:
        _center_text(
            d, width / 2, int(tsize * 1.6 + n_sub * ssize * 1.3), caption, fs, (0, 0, 0)
        )
    d.line([(pane_w, head), (pane_w, head + pane_h)], fill=(200, 200, 200), width=2)
    return canvas


def montage(
    images: Sequence[Sequence[Any]],
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str | None = None,
    cell_width: int = 330,
    crop: tuple[int, int, int, int] | None = None,
    out: str | os.PathLike | None = None,
) -> Any:
    """Grid of frames with row labels on the left and column labels on top.

    Parameters
    ----------
    images : rows of paths or PIL images, all the same size
    row_labels, col_labels : sequences of str
        One per row / column; ``"\\n"`` splits a row label into lines.
    title : str, optional
        Line above the column labels.
    cell_width : int
        Width of each cell in pixels.
    crop : (left, top, right, bottom), optional
        Region of each frame to show; default the box around all non-white
        pixels of all frames.
    out : path, optional
        Save the montage there.

    Returns
    -------
    PIL.Image.Image
    """
    Image = _pil()
    draw_mod = lazy_import("PIL.ImageDraw", "brain3d", "annotations")
    grid = [[_open(x) for x in row] for row in images]
    if len(grid) != len(row_labels) or any(len(r) != len(col_labels) for r in grid):
        raise ValueError("images must be len(row_labels) rows of len(col_labels)")
    box = crop or _content_box([im for row in grid for im in row])
    cell_h = round(cell_width * (box[3] - box[1]) / (box[2] - box[0]))
    tsize, lsize = 40, 34
    ft, fl = _font(tsize), _font(lsize)
    label_w = (
        14
        + int(max(fl.getlength(line) for lab in row_labels for line in lab.split("\n")))
        + 20
    )
    head = int(tsize * 1.8) + (int(tsize * 1.6) if title else 0)
    W, H = label_w + cell_width * len(col_labels), head + cell_h * len(grid)
    canvas = Image.new("RGB", (W, H), "white")
    d = draw_mod.Draw(canvas)
    if title:
        _center_text(d, W / 2, int(tsize * 0.3), title, ft, (0, 0, 0))
    for c, lab in enumerate(col_labels):
        xc = label_w + c * cell_width + cell_width / 2
        _center_text(d, xc, head - int(tsize * 1.5), lab, ft, (0, 0, 0))
    for r, (row, lab) in enumerate(zip(grid, row_labels, strict=True)):
        for c, im in enumerate(row):
            cell = im.crop(box).resize((cell_width, cell_h), Image.LANCZOS)
            canvas.paste(cell, (label_w + c * cell_width, head + r * cell_h))
        lines = lab.split("\n")
        yc = head + r * cell_h + cell_h / 2 - len(lines) * lsize * 0.6
        for k, line in enumerate(lines):
            d.text((14, yc + k * lsize * 1.2), line, font=fl, fill=(0, 0, 0))
    if out is not None:
        canvas.save(out)
    return canvas
