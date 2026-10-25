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
use. The brain surfaces are BrainNet Viewer's own ``.nv`` meshes. BrainNet
Viewer is distributed under the GNU GPL, so htfa does not ship them: they are
downloaded from NITRC on first use, checked against SHA-256 checksums, and
cached (see `mesh_cache_dir`).
"""

# Adapted from ContextLab/directed-forgetting-network-dynamics, commit 31a2b66
# (code/dfnet/render.py and code/dfnet/brainnet_io.py), MIT License,
# Copyright (c) 2019 Contextual Dynamics Laboratory.

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from ._lazy import lazy_import

Array = np.ndarray

__all__ = [
    "MESHES",
    "animate_network",
    "annotate_pair",
    "config_style",
    "fetch_mesh",
    "image_diff",
    "load_mesh",
    "mesh_cache_dir",
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
DATA_DIR_ENV = "HTFA_DATA_DIR"

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
    root = os.environ.get(DATA_DIR_ENV)
    if root:
        return Path(root).expanduser() / "brainnet"
    if sys.platform == "win32" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "htfa" / "brainnet"
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base).expanduser() / "htfa" / "brainnet"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


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


def node_radius(sizes: Array, lo: float = 2.0, hi: float = 7.0) -> Array:
    """Map node sizes linearly onto ``[lo, hi]`` mm (BrainNet: size by value).

    All-equal sizes give 4 mm, as in BrainNet Viewer.
    """
    s = np.asarray(sizes, dtype=np.float64)
    if s.max() == s.min():
        return np.full_like(s, 4.0)
    k = (hi - lo) / (s.max() - s.min())
    return k * s + (hi - k * s.max())


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
    K = len(centers)
    if sizes is not None:
        return centers, _per_node(sizes, K, "sizes")
    return centers, np.ones(K) if widths is None else _per_node(widths, K, "widths")


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
    ) -> None:
        pv = lazy_import("pyvista", "brain3d", "network plots")
        if view not in VIEWS:
            raise ValueError(f"unknown view {view!r}; choose from {sorted(VIEWS)}")
        self.centers = centers
        self.radii = node_radius(sizes)
        self.size = (int(size[0]), int(size[1]))
        s = _SUPERSAMPLE
        pl = pv.Plotter(
            off_screen=True,
            window_size=[self.size[0] * s, self.size[1] * s],
            lighting="none",
        )
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
        spheres = pv.merge(
            [
                pv.Sphere(radius=r, center=c, theta_resolution=40, phi_resolution=40)
                for c, r in zip(centers, self.radii, strict=True)
            ]
        )
        pl.add_mesh(spheres, color=node_color, smooth_shading=True, **_DULL)
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
        want = "(K, K)" if ndim == 2 else "(T, K, K)"
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
    subject: Any = "global",
    mesh: Any = "ch2",
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
        Node sizes; mapped linearly onto 2-7 mm radii (`node_radius`).
        Defaults to the fit's widths, or equal sizes (4 mm) for arrays.
    subject : "global" or int
        Which centers to use when ``nodes`` is a fit.
    mesh : str, path, (vertices, faces) or None
        Brain surface; see `load_mesh`. Default ``"ch2"``.
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
        ``mesh_opacity`` (0.2), ``background`` ("white"). See also
        `config_style`.

    Returns
    -------
    PIL.Image.Image
        The RGB image.
    """
    scene_opts, edge_opts = _style(dict(style))
    centers, node_sizes = _nodes(nodes, sizes, subject)
    E = None
    if connectivity is not None:
        E = _connectivity(connectivity, len(centers), 2)
    scene = _Scene(centers, node_sizes, load_mesh(mesh), **scene_opts)
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


def animate_network(
    nodes: Any,
    connectivity: Any,
    out: str | os.PathLike,
    *,
    fps: float = 10,
    sizes: Any = None,
    subject: Any = "global",
    mesh: Any = "ch2",
    captions: Sequence[str] | None = None,
    frames_dir: str | os.PathLike | None = None,
    jpeg_quality: int | None = None,
    **style: Any,
) -> Path:
    """Render a ``(T, K, K)`` connectivity series as an mp4 or GIF.

    The camera, mesh, nodes and threshold stay fixed across frames.

    Parameters
    ----------
    nodes, sizes, subject, mesh, **style
        As in `plot_network`.
    connectivity : array of shape (T, K, K)
        One edge matrix per frame.
    out : path
        ``.mp4`` (H.264 via imageio-ffmpeg) or ``.gif``. An mp4 is encoded
        while frames render, holding one frame in memory at a time. A GIF
        needs all frames for its shared palette, about 4.8 MB per
        1600 x 1000 frame (see `write_animation`).
    fps : float
        Frames per second.
    captions : sequence of T str, optional
        Text shown in a white band above each frame (not over the brain).
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
    """
    scene_opts, edge_opts = _style(dict(style))
    centers, node_sizes = _nodes(nodes, sizes, subject)
    E = _connectivity(connectivity, len(centers), 3)
    if captions is not None and len(captions) != len(E):
        raise ValueError(f"{len(captions)} captions for {len(E)} frames")
    out = Path(out)
    if out.suffix.lower() not in (".mp4", ".gif"):
        raise ValueError(f"out must end in .mp4 or .gif, got {out.name!r}")
    folder = None if frames_dir is None else Path(frames_dir)
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
    scene = _Scene(centers, node_sizes, load_mesh(mesh), **scene_opts)

    def frames() -> Iterator[Any]:
        for t, frame in enumerate(E):
            scene.set_edges(frame, **edge_opts)
            img = scene.image()
            if captions is not None:
                img = _with_header(img, [captions[t]])
            if jpeg_quality is not None:
                img = _jpeg(img, jpeg_quality)
            if folder is not None:
                ext = ".png" if jpeg_quality is None else ".jpg"
                _save(img, folder / f"{t + 1}{ext}", jpeg_quality)
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
    (300 frames: about 1.4 GB). Use mp4 for long movies.
    """
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
