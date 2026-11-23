"""Tests for htfa.brainnet: BrainNet Viewer files, meshes, rendering, animation.

Everything renders for real (pyvista off screen) and writes real files. The
mesh tests download BrainNet Viewer 1.43 from NITRC on first use (cached in
``htfa.brainnet.mesh_cache_dir()``) and skip only when that download fails.
The 2014 comparison reads the original BrainNet frames from a local copy of
the directed-forgetting-network-dynamics repository and skips without it.
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from sklearn.exceptions import ConvergenceWarning

import htfa
from htfa import brainnet as bn
from htfa.synthetic import make_htfa_data, make_tfa_data

GT_2014 = Path(
    os.environ.get(
        "HTFA_BRAINNET_2014",
        Path.home() / "directed-forgetting-network-dynamics/data/ground_truth_2014",
    )
)
# Pixel position of MNI (y, z) in the 1600 x 1000 left lateral view.
U0, V0, SCALE = 740.27, 513.15, 5.5288


def _px(y, z):
    return int(round(U0 - SCALE * y)), int(round(V0 - SCALE * z))


def _rgb(img):
    return np.asarray(img.convert("RGB")).astype(int)


def _red(a):
    return (a[..., 0] - a[..., 2]) > 80


def _blue(a):
    return (a[..., 2] - a[..., 0]) > 80


@pytest.fixture(scope="module")
def ch2():
    try:
        return bn.load_mesh("ch2")
    except ConnectionError as err:
        pytest.skip(f"BrainNet Viewer mesh download unavailable (offline?): {err}")


# Three horizontal edges in the x = 0 plane: positive at z = 50, negative at
# z = 0, and one at z = -40 whose weight is varied per test.
CENTERS = np.array(
    [[0, -60, 50], [0, 40, 50], [0, -60, 0], [0, 40, 0], [0, -60, -40], [0, 40, -40]],
    dtype=float,
)


def _conn(third=0.0):
    E = np.zeros((6, 6))
    E[0, 1] = E[1, 0] = 0.8
    E[2, 3] = E[3, 2] = -0.8
    E[4, 5] = E[5, 4] = third
    return E


def _window(a, y, z, half=6):
    c, r = _px(y, z)
    return a[r - half : r + half + 1, c - half : c + half + 1]


# ----------------------------------------------------------------- file io


def test_node_edge_roundtrip(tmp_path):
    c = np.array([[-53.75, -47.07, -16.59], [10.0, 20.5, 30.25]])
    p = bn.write_node(
        tmp_path / "a.node", c, sizes=[183.55, 7], labels=["Temporal_Inf_L", "x"]
    )
    centers, colors, sizes, labels = bn.read_node(p)
    print(p.read_text())
    np.testing.assert_allclose(centers, c)
    np.testing.assert_allclose(colors, [1, 1])
    np.testing.assert_allclose(sizes, [183.55, 7])
    assert labels == ["Temporal_Inf_L", "x"]
    E = np.array([[1.0, -0.334], [-0.334, 1.0]])
    q = bn.write_edge(tmp_path / "a.edge", E)
    np.testing.assert_allclose(bn.read_edge(q), np.round(E, 2))


def test_node_defaults_and_errors(tmp_path):
    p = bn.write_node(tmp_path / "b.node", np.zeros((3, 3)))
    _, colors, sizes, labels = bn.read_node(p)
    assert labels == ["-"] * 3 and np.all(sizes == 1) and np.all(colors == 1)
    with pytest.raises(ValueError, match="sizes has 2 entries for 3 nodes"):
        bn.write_node(tmp_path / "c.node", np.zeros((3, 3)), sizes=[1, 2])
    with pytest.raises(ValueError, match="labels has 1 entries"):
        bn.write_node(tmp_path / "c.node", np.zeros((3, 3)), labels=["a"])
    with pytest.raises(ValueError, match=r"\(K, 3\)"):
        bn.write_node(tmp_path / "c.node", np.zeros((3, 2)))
    bad = tmp_path / "bad.node"
    bad.write_text("1 2 3\n")
    with pytest.raises(ValueError, match="x y z color size"):
        bn.read_node(bad)
    rect = tmp_path / "rect.edge"
    rect.write_text("1 2 3\n4 5 6\n")
    with pytest.raises(ValueError, match="not square"):
        bn.read_edge(rect)
    with pytest.raises(ValueError, match=r"\(K, K\)"):
        bn.write_edge(tmp_path / "x.edge", np.zeros((2, 3)))


def test_nv_roundtrip_and_validation(tmp_path):
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.5]])
    f = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    p = bn.write_nv(tmp_path / "t.nv", v, f)
    text = p.read_text().split("\n")
    assert text[0] == "4" and text[5] == "4" and text[6] == "1 2 3"  # 1-based
    v2, f2 = bn.read_nv(p)
    np.testing.assert_allclose(v2, v)
    np.testing.assert_array_equal(f2, f)
    bad = tmp_path / "bad.nv"
    bad.write_text("3\n0 0 0\n1 0 0\n0 1 0\n1\n1 2 4\n")
    with pytest.raises(ValueError, match="out of range"):
        bn.read_nv(bad)
    with pytest.raises(ValueError, match=r"\(F, 3\)"):
        bn.write_nv(tmp_path / "x.nv", v, f[:, :2])


@pytest.mark.skipif(not GT_2014.is_dir(), reason=f"2014 frames not found at {GT_2014}")
def test_read_2014_node_file():
    centers, colors, sizes, labels = bn.read_node(GT_2014 / "frames/cue_r/1.node")
    assert centers.shape == (100, 3) and np.all(colors == 1)
    assert sizes.min() == pytest.approx(181.04) and sizes.max() == pytest.approx(209.61)
    assert labels[0] == "Temporal_Inf_L"
    assert bn.read_edge(GT_2014 / "frames/cue_r/28.edge").shape == (100, 100)


# ------------------------------------------------------------------ meshes


def test_node_radius_reproduces_2014_k_b():
    sizes = np.array([181.04, 195.0, 209.61])
    r = bn.node_radius(sizes)
    # 2014 DF_config.mat: nod.k = 0.17500875, nod.b = -29.683584
    np.testing.assert_allclose(r, 0.17500875043752173 * sizes - 29.683584179208935)
    assert r.min() == pytest.approx(2.0) and r.max() == pytest.approx(7.0)
    np.testing.assert_array_equal(bn.node_radius([3.0, 3.0]), [4.0, 4.0])


def test_node_radius_range_fixed_k_b_and_clip():
    """The min-max range, a fixed BrainNet k * size + b, and clipping."""
    s = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(bn.node_radius(s, lo=1.0, hi=5.0), [1.0, 3.0, 5.0])
    k, b = 0.17500875043752173, -29.683584179208935
    sizes = np.array([150.0, 181.04, 209.61, 250.0])
    np.testing.assert_allclose(bn.node_radius(sizes, k=k, b=b), k * sizes + b)
    clipped = bn.node_radius(sizes, k=k, b=b, clip=(2.0, 7.0))
    print("fixed k, b radii", k * sizes + b, "clipped", clipped)
    np.testing.assert_allclose(clipped, np.clip(k * sizes + b, 2.0, 7.0))
    np.testing.assert_allclose(bn.node_radius(s, clip=(3.0, 6.0)), [3.0, 4.5, 6.0])
    # equal sizes: BrainNet's 4 mm, clipped like any radius
    np.testing.assert_array_equal(bn.node_radius([3.0, 3.0], clip=(5, 9)), [5.0, 5.0])
    with pytest.raises(ValueError, match="both k and b"):
        bn.node_radius(s, k=1.0)
    with pytest.raises(ValueError, match="lo < hi"):
        bn.node_radius(s, lo=5.0, hi=5.0)
    with pytest.raises(ValueError, match="clip"):
        bn.node_radius(s, clip=(4.0, 1.0))


def _gray_run(a, y, z):
    """Width in pixels of the gray run through the image point of (y, z)."""
    c, r = _px(y, z)
    row = a[r]
    gray = (row.max(axis=1) < 235) & (np.abs(row[:, 0] - row[:, 2]) < 10)
    lo = hi = c
    while gray[lo - 1]:
        lo -= 1
    while gray[hi + 1]:
        hi += 1
    return hi - lo + 1


def test_plot_network_radius_options():
    """radius_range, radius_kb and radius_clip set the drawn sphere radii."""
    centers = np.array([[0.0, -40.0, 0.0], [0.0, 40.0, 0.0]])
    sizes = np.array([1.0, 2.0])

    def radii_px(**style):
        a = _rgb(bn.plot_network(centers, sizes=sizes, mesh=None, snap=False, **style))
        return [_gray_run(a, y, 0) for y in (-40, 40)]

    for style, radii in [
        ({}, (2.0, 7.0)),
        ({"radius_range": (5.0, 10.0)}, (5.0, 10.0)),
        ({"radius_kb": (6.0, 0.0)}, (6.0, 12.0)),
        ({"radius_kb": (6.0, 0.0), "radius_clip": (3.0, 9.0)}, (6.0, 9.0)),
    ]:
        got = radii_px(**style)
        expect = [2 * r * SCALE for r in radii]
        print(style, "diameter px", got, "expected", np.round(expect, 1))
        np.testing.assert_allclose(got, expect, atol=3)
    with pytest.raises(ValueError, match="radii must be positive"):
        bn.plot_network(centers, sizes=sizes, mesh=None, radius_kb=(1.0, -1.5))


def test_fetch_mesh_checksums(ch2):
    verts, faces = ch2
    print("ch2 mesh:", verts.shape, faces.shape, verts.min(0), verts.max(0))
    assert verts.shape == (77124, 3) and faces.shape == (154244, 3)
    for name, (filename, sha) in bn.MESHES.items():
        path = bn.fetch_mesh(name)
        assert path.name == filename and path.parent == bn.mesh_cache_dir()
        assert bn._sha256(path) == sha


def test_mesh_cache_dir_env(monkeypatch, tmp_path):
    monkeypatch.setenv(bn.DATA_DIR_ENV, str(tmp_path))
    assert bn.mesh_cache_dir() == tmp_path / "brainnet"


def test_load_mesh_inputs(tmp_path):
    assert bn.load_mesh(None) is None
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.0]])
    f = np.array([[0, 1, 2]])
    p = bn.write_nv(tmp_path / "tri.nv", v, f)
    for mesh in (p, str(p), (v, f)):
        v2, f2 = bn.load_mesh(mesh)
        np.testing.assert_allclose(v2, v)
        np.testing.assert_array_equal(f2, f)
    with pytest.raises(ValueError, match="neither a file nor one of"):
        bn.load_mesh("brain")
    with pytest.raises(ValueError, match="unknown mesh"):
        bn.fetch_mesh("fsaverage")


@pytest.mark.network
def test_fetch_mesh_repairs_corrupt_cache(monkeypatch, tmp_path):
    """A cached file with the wrong checksum is downloaded again."""
    monkeypatch.setenv(bn.DATA_DIR_ENV, str(tmp_path))
    folder = bn.mesh_cache_dir()
    folder.mkdir(parents=True)
    filename, sha = bn.MESHES["ch2"]
    (folder / filename).write_text("not a mesh\n")
    path = bn.fetch_mesh("ch2")
    assert bn._sha256(path) == sha
    assert sorted(p.name for p in folder.iterdir()) == sorted(
        f for f, _ in bn.MESHES.values()
    )


@pytest.fixture
def http_folder(tmp_path):
    """A real local HTTP server serving files from a folder.

    Responses carry the Content-Type named by a ``<file>.type`` sidecar.
    """
    import http.server
    import threading

    root = tmp_path / "www"
    root.mkdir()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            path = root / self.path.lstrip("/")
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", (root / (path.name + ".type")).read_text())
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield root, f"http://127.0.0.1:{server.server_address[1]}/"
    finally:
        server.shutdown()
        server.server_close()


def test_fetch_archive_checks_zip_signature(http_folder, tmp_path):
    import zipfile

    root, base = http_folder
    with zipfile.ZipFile(root / "a.zip", "w") as z:
        z.writestr("x.txt", "hello")
    # a real zip passes whatever its Content-Type variant
    for kind in ("application/zip; charset=binary", "application/x-zip-compressed"):
        (root / "a.zip.type").write_text(kind)
        dest = tmp_path / "got.zip"
        bn._fetch_archive(dest, tmp_path, attempts=1, url=base + "a.zip")
        assert dest.read_bytes() == (root / "a.zip").read_bytes()
    # an HTML error page labelled as a zip is rejected
    (root / "err.html").write_text("<html>NITRC Under Maintenance</html>")
    (root / "err.html.type").write_text("application/zip")
    with pytest.raises(ConnectionError, match="not a zip archive"):
        bn._fetch_archive(
            tmp_path / "bad.zip", tmp_path, attempts=2, url=base + "err.html"
        )


def test_extract_meshes_cleans_up_on_failure(tmp_path):
    """A corrupt second mesh fails mid-copy and leaves no .part file behind."""
    import zipfile

    first, second = [f for f, _ in bn.MESHES.values()][:2]
    archive = tmp_path / "a.zip"
    tri = "3\n0 0 0\n1 0 0\n0 1 0\n1\n1 2 3\n"
    marker = "CORRUPTME" * 50
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as z:
        z.writestr(bn._MESH_DIR_IN_ARCHIVE + first, tri)
        z.writestr(bn._MESH_DIR_IN_ARCHIVE + second, tri + marker)
    raw = archive.read_bytes()
    archive.write_bytes(raw.replace(marker.encode(), marker.lower().encode()))
    out = tmp_path / "cache"
    out.mkdir()
    with pytest.raises(zipfile.BadZipFile, match="CRC"):
        bn._extract_meshes(archive, out)
    print(sorted(p.name for p in out.iterdir()))
    assert sorted(p.name for p in out.iterdir()) == [first]  # complete file kept
    assert bn.read_nv(out / first)[1].shape == (1, 3)


# --------------------------------------------------------------- rendering


def test_plot_network_edges_colors_threshold(tmp_path):
    out = tmp_path / "net.png"
    # raw centers: (0, 40, -40) lies outside the MNI brain
    img = bn.plot_network(
        CENTERS, _conn(0.3), mesh=None, out=out, threshold=0.35, snap=False
    )
    assert out.stat().st_size > 5000
    with Image.open(out) as saved:
        assert saved.size == (1600, 1000) and saved.mode == "RGB"
    a = _rgb(img)
    # white background (no dark mode): every corner pixel is pure white
    for r, c in [(0, 0), (0, -1), (-1, 0), (-1, -1)]:
        assert tuple(a[r, c]) == (255, 255, 255)
    pos, neg, sub = (_window(a, -10, z) for z in (50, 0, -40))
    print("red/blue px at +edge", _red(pos).sum(), _blue(pos).sum())
    print("red/blue px at -edge", _red(neg).sum(), _blue(neg).sum())
    print("min value at sub-threshold edge", sub.min())
    assert _red(pos).sum() >= 10 and _blue(pos).sum() == 0
    assert _blue(neg).sum() >= 10 and _red(neg).sum() == 0
    assert sub.min() == 255  # |0.3| <= 0.35: not drawn
    # gray node sphere (radius 4 mm) centered where the camera maps (y, z)
    c, r = _px(-60, 0)
    node = a[r - 5 : r + 6, c - 5 : c + 6]
    assert np.all(np.abs(node[..., 0] - node[..., 2]) < 10) and node.max() < 200
    # lower threshold: the 0.3 edge appears (red)
    a2 = _rgb(
        bn.plot_network(CENTERS, _conn(0.3), mesh=None, threshold=0.25, snap=False)
    )
    assert _red(_window(a2, -10, -40)).sum() >= 5
    # strict inequality: |E| == threshold is not drawn
    a3 = _rgb(
        bn.plot_network(CENTERS, _conn(0.3), mesh=None, threshold=0.3, snap=False)
    )
    assert _window(a3, -10, -40).min() == 255


def test_plot_network_with_mesh_and_styles(ch2, tmp_path):
    img = bn.plot_network(CENTERS, _conn(), mesh=ch2, out=tmp_path / "brain.png")
    a = _rgb(img)
    assert tuple(a[0, 0]) == (255, 255, 255) and tuple(a[-1, -1]) == (255, 255, 255)
    # translucent 0.95-gray surface: light gray inside the silhouette, away
    # from nodes and edges (MNI y = -80, z = 25: occipital cortex)
    c, r = _px(-80, 25)
    patch = a[r - 3 : r + 4, c - 3 : c + 4]
    print("mesh patch", patch.mean(axis=(0, 1)))
    assert 170 < patch.mean() < 250
    assert np.abs(patch[..., 0] - patch[..., 2]).max() < 6
    silhouette = (a.min(axis=2) < 250).mean()
    print("silhouette fraction", silhouette)
    assert 0.25 < silhouette < 0.6
    assert _red(_window(a, -10, 50)).sum() >= 10
    flat = _rgb(bn.plot_network(CENTERS, _conn(), mesh=None, edge_style="flat"))
    plain = _rgb(bn.plot_network(CENTERS, _conn(), mesh=None))

    # "intensity": an open tube at opacity 0.7 shows two walls over white,
    # so green stays near 0.09 * 255 = 23; "flat" is opaque (green 0)
    g_flat = _window(flat, -10, 50)[..., 1].min()
    g_plain = _window(plain, -10, 50)[..., 1].min()
    print("min green flat/intensity", g_flat, g_plain)
    assert g_flat <= 5 and 15 <= g_plain <= 40
    custom = _rgb(
        bn.plot_network(
            CENTERS, _conn(), mesh=None, positive_color="green", negative_color="orange"
        )
    )
    g = _window(custom, -10, 50)
    assert ((g[..., 1] - g[..., 0]) > 60).sum() >= 10 and _red(g).sum() == 0


def test_views_frame_the_brain(ch2):
    """Each view keeps the whole Ch2 surface inside the image."""
    for view in ("left", "right", "dorsal", "ventral", "anterior", "posterior"):
        a = _rgb(bn.plot_network(np.zeros((1, 3)), mesh=ch2, view=view))
        nz = a.min(axis=2) < 250
        rows, cols = np.flatnonzero(nz.any(1)), np.flatnonzero(nz.any(0))
        print(view, rows[0], rows[-1], cols[0], cols[-1])
        assert rows[0] > 5 and rows[-1] < 995 and cols[0] > 5 and cols[-1] < 1595
    left = _rgb(bn.plot_network(np.zeros((1, 3)), mesh=ch2, view="left"))
    right = _rgb(bn.plot_network(np.zeros((1, 3)), mesh=ch2, view="right"))
    assert bn.image_diff(Image.fromarray(left[:, ::-1].astype(np.uint8)),
                         Image.fromarray(right.astype(np.uint8))) < 5  # fmt: skip


def test_failed_scene_setup_closes_plotter():
    """An error after the Plotter exists (bad color) still closes it."""
    from pyvista.plotting import _ALL_PLOTTERS

    before = set(_ALL_PLOTTERS)
    for bad in ({"background": "notacolor"}, {"node_color": "notacolor"}):
        with pytest.raises(ValueError):
            bn.plot_network(CENTERS, mesh=None, **bad)
    print("open plotters before/after:", len(before), len(_ALL_PLOTTERS))
    assert set(_ALL_PLOTTERS) == before


def test_plot_network_input_errors():
    with pytest.raises(ValueError, match=r"\(K, 3\) MNI"):
        bn.plot_network(np.zeros((4, 2)), mesh=None)
    with pytest.raises(ValueError, match=r"\(K, K\) with K=6"):
        bn.plot_network(CENTERS, np.zeros((5, 5)), mesh=None)
    with pytest.raises(ValueError, match="NaN"):
        bn.plot_network(CENTERS, np.full((6, 6), np.nan), mesh=None)
    with pytest.raises(ValueError, match="unknown view"):
        bn.plot_network(CENTERS, mesh=None, view="top")
    with pytest.raises(ValueError, match="edge_style"):
        bn.plot_network(CENTERS, mesh=None, edge_style="bold")
    with pytest.raises(TypeError, match="unexpected keyword arguments"):
        bn.plot_network(CENTERS, mesh=None, colour="red")
    with pytest.raises(ValueError, match="sizes has 2 entries"):
        bn.plot_network(CENTERS, mesh=None, sizes=[1, 2])


@pytest.mark.skipif(not GT_2014.is_dir(), reason=f"2014 frames not found at {GT_2014}")
def test_reproduces_2014_frames(ch2, tmp_path):
    """Render the saved 2014 .node/.edge files and compare with the 2014 JPEGs.

    The reference implementation measured mean pixel differences of 2.0-2.7
    (0-255 scale) on these frames.
    """
    centers, _, sizes, _ = bn.read_node(GT_2014 / "frames/cue_r/1.node")
    diffs = {}
    for frame in (1, 25, 28, 35):
        E = bn.read_edge(GT_2014 / f"frames/cue_r/{frame}.edge")
        out = tmp_path / f"{frame}.png"
        bn.plot_network(centers, E, sizes=sizes, mesh=ch2, out=out)
        diffs[frame] = bn.image_diff(out, GT_2014 / f"frames/cue_r/{frame}.jpg")
    other = bn.image_diff(
        GT_2014 / "frames/cue_f/28.jpg", GT_2014 / "frames/cue_r/28.jpg"
    )
    print("image diff vs 2014:", diffs, "cue_f vs cue_r frame 28:", other)
    assert all(d < 2.75 for d in diffs.values()), diffs
    assert diffs[28] < other / 2


# --------------------------------------------------------------- animation


def _frames():
    return np.stack([_conn(), -_conn(), _conn(0.6)])


def test_animate_network_gif(tmp_path):
    out = bn.animate_network(
        CENTERS, _frames(), tmp_path / "a.gif", mesh=None, fps=5,
        frames_dir=tmp_path / "frames", snap=False,
    )  # fmt: skip
    assert sorted(p.name for p in (tmp_path / "frames").iterdir()) == [
        "1.png",
        "2.png",
        "3.png",
    ]
    with Image.open(out) as g:
        assert g.n_frames == 3 and g.size == (1600, 1000)
        assert g.info["duration"] == 200
        decoded = []
        for k in range(3):
            g.seek(k)
            decoded.append(_rgb(g))
    for k, a in enumerate(decoded):
        png = _rgb(Image.open(tmp_path / "frames" / f"{k + 1}.png"))
        print(
            k, "max |gif - png|", np.abs(a - png).max(), "mean", np.abs(a - png).mean()
        )
        assert np.abs(a - png).mean() < 1.0
    # colors follow the sign of each frame
    assert _red(_window(decoded[0], -10, 50)).sum() >= 10
    assert _blue(_window(decoded[1], -10, 50)).sum() >= 10
    assert _red(_window(decoded[2], -10, -40)).sum() >= 10


def test_animate_network_mp4_and_captions(tmp_path):
    import imageio.v2 as iio

    out = bn.animate_network(
        CENTERS, _frames(), tmp_path / "a.mp4", mesh=None, fps=10,
        captions=["TR 0", "TR 1", "TR 2"], frames_dir=tmp_path / "f",
    )  # fmt: skip
    assert out.stat().st_size > 1000
    reader = iio.get_reader(out, "ffmpeg")
    try:
        assert reader.count_frames() == 3
        assert reader.get_meta_data()["fps"] == 10
        first = reader.get_data(0)
    finally:
        reader.close()
    with Image.open(tmp_path / "f" / "1.png") as png:
        w, h = png.size
        assert w == 1600 and h > 1000  # header band above the frame
        band = _rgb(png)[: h - 1000]
    assert (band.min(axis=2) < 128).any()  # caption text is in the band
    assert first.shape[:2] == (h, w)
    # the decoded video shows the positive edge in red below the header
    assert _red(_window(first[h - 1000 :].astype(int), -10, 50)).sum() >= 10


def test_animate_network_errors(tmp_path):
    with pytest.raises(ValueError, match=r"\(T, K, K\)"):
        bn.animate_network(CENTERS, _conn(), tmp_path / "a.gif", mesh=None)
    with pytest.raises(ValueError, match="2 captions for 3 frames"):
        bn.animate_network(
            CENTERS, _frames(), tmp_path / "a.gif", mesh=None, captions=["a", "b"]
        )
    with pytest.raises(ValueError, match=r"\.mp4 or \.gif"):
        bn.animate_network(CENTERS, _frames(), tmp_path / "a.avi", mesh=None)
    with pytest.raises(ValueError, match="no frames"):
        bn.write_animation([], tmp_path / "x.gif")
    with pytest.raises(ValueError, match="no frames"):
        bn.write_animation(iter([]), tmp_path / "x.mp4")
    assert not (tmp_path / "x.mp4").exists()
    a, b = Image.new("RGB", (40, 30)), Image.new("RGB", (42, 30))
    for suffix in (".gif", ".mp4"):
        with pytest.raises(ValueError, match="same size"):
            bn.write_animation([a, b], tmp_path / f"y{suffix}")


def test_write_animation_mp4_streams_frames(tmp_path):
    """Each mp4 frame is released once encoded: no frame list is kept."""
    import gc
    import weakref

    import imageio.v2 as iio

    refs = []
    alive_when_yielding = []

    def frames(n=40):
        for k in range(n):
            gc.collect()
            alive_when_yielding.append(sum(r() is not None for r in refs))
            a = np.full((120, 160, 3), 255, np.uint8)
            a[:, : 4 * (k + 1)] = (200, 0, 0)  # a growing red bar
            img = Image.fromarray(a)
            refs.append(weakref.ref(img))
            yield img
            del img

    out = bn.write_animation(frames(), tmp_path / "s.mp4", fps=20)
    print("frames alive when the next one was requested:", alive_when_yielding)
    # when frame k is requested, earlier frames have been freed
    assert max(alive_when_yielding) <= 1
    reader = iio.get_reader(out, "ffmpeg")
    try:
        assert reader.count_frames() == 40
        assert reader.get_meta_data()["fps"] == 20
        last = reader.get_data(39)
    finally:
        reader.close()
    assert (last[60, :155, 0] > 150).all() and (last[60, :155, 2] < 80).all()


def _square_frames(n=12):
    """Real 320 x 200 frames: a red square moving over a white background."""
    out = []
    for k in range(n):
        a = np.full((200, 320, 3), 255, np.uint8)
        a[60:100, 10 + 20 * k : 50 + 20 * k] = (200, 30, 30)
        out.append(Image.fromarray(a))
    return out


def test_write_animation_crf_size(tmp_path):
    """A higher CRF gives a smaller file; both decode to 12 frames of 320 x 200."""
    import imageio.v2 as iio

    sizes = {}
    for crf in (10, 40):
        out = bn.write_animation(_square_frames(), tmp_path / f"c{crf}.mp4", crf=crf)
        sizes[crf] = out.stat().st_size
        decoded = iio.mimread(out)
        assert len(decoded) == 12 and decoded[0].shape[:2] == (200, 320)
    print("file sizes by crf:", sizes)
    assert sizes[40] < sizes[10]


def test_write_animation_default_unchanged(tmp_path):
    """The default call equals a direct imageio encode with the 0.1.0 settings."""
    import imageio.v2 as iio

    legacy = tmp_path / "legacy.mp4"
    with iio.get_writer(
        legacy, fps=10, codec="libx264", quality=8, macro_block_size=2,
        pixelformat="yuv420p",
    ) as writer:  # fmt: skip
        for im in _square_frames():
            w, h = im.size
            writer.append_data(np.asarray(im.crop((0, 0, w - w % 2, h - h % 2))))
    a = bn.write_animation(_square_frames(), tmp_path / "a.mp4").read_bytes()
    c = bn.write_animation(_square_frames(), tmp_path / "c.mp4", crf=23).read_bytes()
    print(
        "sizes (0.1.0 writer, default, crf=23):",
        len(legacy.read_bytes()),
        len(a),
        len(c),
    )
    assert a == legacy.read_bytes()
    assert a != c


def test_write_animation_crf_and_faststart(tmp_path):
    """crf and faststart together: the crf takes effect and moov precedes mdat."""
    sizes = {}
    for crf in (10, 40):
        out = bn.write_animation(
            _square_frames(), tmp_path / f"b{crf}.mp4", crf=crf, faststart=True
        )
        data = out.read_bytes()
        sizes[crf] = len(data)
        moov, mdat = data.find(b"moov"), data.find(b"mdat")
        print(f"crf={crf}: size {len(data)}, moov {moov}, mdat {mdat}")
        assert 0 < moov < mdat
    assert sizes[40] < sizes[10]


def test_write_animation_faststart(tmp_path):
    """faststart puts the moov atom before mdat; the default puts it after."""
    for faststart in (True, False):
        data = bn.write_animation(
            _square_frames(), tmp_path / f"f{faststart}.mp4", faststart=faststart
        ).read_bytes()
        moov, mdat = data.find(b"moov"), data.find(b"mdat")
        print(f"faststart={faststart}: moov at {moov}, mdat at {mdat}")
        assert -1 not in (moov, mdat)
        assert (moov < mdat) is faststart


def test_write_animation_options_validated(tmp_path):
    frames = _square_frames(2)
    for crf in (52, -1, 2.5):
        with pytest.raises(ValueError, match=str(crf)):
            bn.write_animation(frames, tmp_path / "x.mp4", crf=crf)
    with pytest.raises(ValueError, match="crf"):
        bn.write_animation(frames, tmp_path / "x.gif", crf=23)
    with pytest.raises(ValueError, match="faststart"):
        bn.write_animation(frames, tmp_path / "x.gif", faststart=True)


def test_animate_network_long_mp4(tmp_path):
    """A 40-frame animate_network mp4 decodes to 40 frames with changing edges."""
    import imageio.v2 as iio

    T = 40
    E = np.stack([_conn() * np.sign(np.sin(t / 3.0) + 1e-9) for t in range(T)])
    out = bn.animate_network(CENTERS, E, tmp_path / "long.mp4", mesh=None,
                             size=(400, 250), fps=10)  # fmt: skip
    reader = iio.get_reader(out, "ffmpeg")
    try:
        assert reader.count_frames() == T
        colors = []
        for t in (0, 10):  # sin > 0 at t = 0 (red top edge), < 0 at t = 10 (blue)
            f = reader.get_data(t).astype(int)
            top = f[int(V0 / 4 - SCALE / 4 * 50) - 3 : int(V0 / 4 - SCALE / 4 * 50) + 4]
            colors.append((_red(top).sum(), _blue(top).sum()))
    finally:
        reader.close()
    print("red/blue px on the top edge at t=0 and t=10:", colors)
    assert colors[0][0] > 20 and colors[0][1] == 0
    assert colors[1][1] > 20 and colors[1][0] == 0


# ------------------------------------------- duration, interpolation, spin


def _pngs(folder, n):
    return [_rgb(Image.open(folder / f"{k + 1}.png")) for k in range(n)]


def test_animate_network_duration_frame_count(tmp_path):
    """duration sets round(duration * fps) frames for GIF and mp4 alike."""
    import imageio.v2 as iio

    for duration, fps, want in [(1.2, 5, 6), (0.9, 10, 9)]:
        gif = bn.animate_network(
            CENTERS,
            _frames(),
            tmp_path / "d.gif",
            mesh=None,
            size=(200, 125),
            fps=fps,
            duration=duration,
        )
        with Image.open(gif) as g:
            print("gif", duration, fps, "->", g.n_frames, g.info["duration"], "ms")
            assert g.n_frames == want and g.info["duration"] == 1000 / fps
        mp4 = bn.animate_network(
            CENTERS,
            _frames(),
            tmp_path / "d.mp4",
            mesh=None,
            size=(200, 125),
            fps=fps,
            duration=duration,
        )
        reader = iio.get_reader(mp4, "ffmpeg")
        try:
            print("mp4", duration, fps, "->", reader.count_frames())
            assert reader.count_frames() == want
            assert reader.get_meta_data()["fps"] == fps
        finally:
            reader.close()
    # n_frames instead of duration: 7 frames out of 3 connectivity frames
    bn.animate_network(
        CENTERS,
        _frames(),
        tmp_path / "n.gif",
        mesh=None,
        size=(200, 125),
        n_frames=7,
        frames_dir=tmp_path / "n",
    )
    assert len(list((tmp_path / "n").iterdir())) == 7


def test_animate_network_linear_interpolation(tmp_path):
    """Between two windows, frames show the linearly interpolated matrix.

    Edge 4-5 goes from 0 to 1 (threshold 0.45), edges 0-1 and 2-3 change
    sign. On 5 output frames at t = 0, 0.25, 0.5, 0.75, 1: edge 4-5 is 0,
    0.25, 0.5, 0.75, 1, so it appears from the middle frame on, and the top
    edge is 0.8, 0.4, 0, -0.4, -0.8: drawn red, hidden, hidden, hidden, blue.
    """
    E0, E1 = _conn(0.0), -_conn()
    E1[4, 5] = E1[5, 4] = 1.0
    series = np.stack([E0, E1])
    kw = dict(mesh=None, snap=False, threshold=0.45, size=(400, 250))
    bn.animate_network(
        CENTERS,
        series,
        tmp_path / "lin.gif",
        n_frames=5,
        frames_dir=tmp_path / "lin",
        **kw,
    )
    got = _pngs(tmp_path / "lin", 5)
    for k, t in enumerate(np.linspace(0, 1, 5)):
        want = _rgb(bn.plot_network(CENTERS, (1 - t) * E0 + t * E1, **kw))
        diff = np.abs(got[k] - want)
        print(
            f"frame {k} (t={t}): max |frame - plot_network(interpolated)|", diff.max()
        )
        assert diff.max() == 0
    # visibility follows the interpolated weight; rows of z = 50, 0, -40 in
    # the 400 x 250 image, between the nodes (y = -60 and 40)
    bottom = [_red(a[180:189, 150:250]).sum() for a in got]
    print("red px of edge 4-5 per frame:", bottom)
    assert bottom[0] == bottom[1] == 0 and min(bottom[2:]) > 5
    upper = [a[55:133, 150:250].min() for a in got]
    print("darkest value at z = 50..0 per frame:", upper)
    assert upper[1] == upper[2] == upper[3] == 255  # |w| <= 0.4: hidden
    assert _red(got[0][55:64, 150:250]).sum() > 5
    assert _blue(got[4][55:64, 150:250]).sum() > 5
    # "nearest": frames 0-1 show window 0 and frames 2-4 window 1 (t >= 0.5)
    bn.animate_network(
        CENTERS,
        series,
        tmp_path / "near.gif",
        n_frames=5,
        interpolation="nearest",
        frames_dir=tmp_path / "near",
        **kw,
    )
    near = _pngs(tmp_path / "near", 5)
    a0, a1 = (
        _rgb(bn.plot_network(CENTERS, E0, **kw)),
        _rgb(bn.plot_network(CENTERS, E1, **kw)),
    )
    assert [np.abs(f - a0).max() for f in near[:2]] == [0, 0]
    assert [np.abs(f - a1).max() for f in near[2:]] == [0, 0, 0]


def test_animate_network_captions_follow_time(tmp_path):
    """Sequence captions use the nearest window; a callable gets the time."""
    seen = []

    def caption(t):
        seen.append(t)
        return f"t = {t:.2f}"

    bn.animate_network(
        CENTERS,
        _frames(),
        tmp_path / "c.gif",
        mesh=None,
        size=(200, 125),
        n_frames=5,
        captions=caption,
    )
    print("caption times:", seen)
    assert np.allclose(seen, [0, 0.5, 1, 1.5, 2])
    bn.animate_network(
        CENTERS,
        _frames(),
        tmp_path / "s.gif",
        mesh=None,
        size=(200, 125),
        n_frames=5,
        captions=["a", "b", "c"],
        frames_dir=tmp_path / "s",
    )
    # frames 1 and 2 (t = 0.5, 1) both take caption "b" (nearest window: 1)
    bands = [a[:-125] for a in _pngs(tmp_path / "s", 5)]
    assert np.array_equal(bands[1], bands[2])
    assert not np.array_equal(bands[0], bands[1])
    assert not np.array_equal(bands[2], bands[3])


def test_animate_network_static_with_duration(tmp_path):
    """A (K, K) matrix with duration gives a movie of fixed edges."""
    out = bn.animate_network(
        CENTERS,
        _conn(),
        tmp_path / "s.gif",
        mesh=None,
        size=(200, 125),
        fps=4,
        duration=1.0,
        frames_dir=tmp_path / "f",
    )
    frames = _pngs(tmp_path / "f", 4)
    assert len(list((tmp_path / "f").iterdir())) == 4
    # Pillow stores identical consecutive GIF frames once, with their
    # durations added, so the GIF still plays for 1 s
    with Image.open(out) as g:
        total = 0
        for k in range(g.n_frames):
            g.seek(k)
            total += g.info["duration"]
    print("GIF frames stored:", g.n_frames, "total duration (ms):", total)
    assert total == 1000
    assert all(np.array_equal(frames[0], f) for f in frames[1:])
    assert _red(frames[0]).sum() > 20 and _blue(frames[0]).sum() > 20


def test_animate_network_option_errors(tmp_path):
    out = tmp_path / "e.gif"
    with pytest.raises(ValueError, match=r"\(T, K, K\)"):
        bn.animate_network(CENTERS, _conn(), out, mesh=None, spin=(0, 0, 1))
    with pytest.raises(ValueError, match="not both"):
        bn.animate_network(CENTERS, _frames(), out, mesh=None, duration=1, n_frames=3)
    with pytest.raises(ValueError, match="duration"):
        bn.animate_network(CENTERS, _frames(), out, mesh=None, duration=0.01)
    with pytest.raises(ValueError, match="n_frames"):
        bn.animate_network(CENTERS, _frames(), out, mesh=None, n_frames=0)
    with pytest.raises(ValueError, match="interpolation"):
        bn.animate_network(CENTERS, _frames(), out, mesh=None, interpolation="cubic")
    with pytest.raises(ValueError, match="spin"):
        bn.animate_network(CENTERS, _frames(), out, mesh=None, spin=(0, 1))
    with pytest.raises(ValueError, match="spin"):
        bn.animate_network(CENTERS, _frames(), out, mesh=None, spin=(0, np.nan, 1))
    assert not out.exists()


def test_animate_network_spin_returns_to_start(tmp_path):
    """Frame k shows k / n_frames of the turns: a whole turn is the start view.

    With spin=(0, 0, 2) over 8 frames, frame 4 is one full turn about z and
    frame 2 half a turn; spin=(0, 0, 1) over 4 frames visits the same
    orientations (so the same zoom) and reaches half a turn at frame 2, the
    same camera. Around x and y as well.
    """
    kw = dict(mesh=None, snap=False, size=(400, 250))
    for axis in range(3):
        two, one = np.zeros(3), np.zeros(3)
        two[axis], one[axis] = 2, 1
        bn.animate_network(
            CENTERS,
            _conn(),
            tmp_path / "a.gif",
            spin=tuple(two),
            n_frames=8,
            frames_dir=tmp_path / f"two{axis}",
            **kw,
        )
        bn.animate_network(
            CENTERS,
            _conn(),
            tmp_path / "b.gif",
            spin=tuple(one),
            n_frames=4,
            frames_dir=tmp_path / f"one{axis}",
            **kw,
        )
        f2, f1 = _pngs(tmp_path / f"two{axis}", 8), _pngs(tmp_path / f"one{axis}", 4)
        d_turn = np.abs(f2[4] - f2[0]).mean()
        d_half = np.abs(f2[2] - f1[2]).mean()
        d_moved = np.abs(f2[1] - f2[0]).mean()
        print(
            f"axis {axis}: |turn - start| {d_turn:.4f}, "
            f"|half (2 turns) - half (1 turn)| {d_half:.4f}, "
            f"|quarter - start| {d_moved:.3f}"
        )
        assert d_turn < 0.05 and d_half < 0.05
        assert d_moved > 0.5


def test_animate_network_spin_half_turn_views(tmp_path):
    """Half a turn about z shows the other side: the left view mirrored.

    A quarter turn looks along the y axis, onto the x = 0 plane that holds
    all nodes edge-on, so the drawing collapses to a narrow vertical strip.
    """
    bn.animate_network(
        CENTERS,
        _conn(),
        tmp_path / "h.gif",
        mesh=None,
        snap=False,
        size=(400, 250),
        n_frames=4,
        spin=(0, 0, 1),
        frames_dir=tmp_path / "h",
    )
    f = _pngs(tmp_path / "h", 4)

    def cols(a):
        c = np.flatnonzero((a.min(axis=2) < 250).any(0))
        return c[0], c[-1]

    widths = [cols(a)[1] - cols(a)[0] for a in f]
    print("drawn column span per frame (0, 90, 180, 270 deg):", widths)
    assert widths[1] < widths[0] / 4 and widths[3] < widths[0] / 4
    mirrored = bn.image_diff(
        Image.fromarray(f[2][:, ::-1].astype(np.uint8)),
        Image.fromarray(f[0].astype(np.uint8)),
    )
    print("image_diff(180 deg mirrored, 0 deg):", mirrored)
    assert mirrored < 2
    # red top edge stays on top: a turn about z does not flip up and down
    assert _red(f[2][:125]).sum() > 20 and _blue(f[2][:125]).sum() == 0


@pytest.mark.parametrize("spin", [(1, 0, 0), (0, 1, 0), (0, 0, 1)])
def test_animate_network_spin_keeps_brain_in_frame(ch2, tmp_path, spin):
    """No frame of a spin cuts the mesh, and the fit is tight in some frame."""
    w, h = 320, 200
    bn.animate_network(
        CENTERS,
        _conn(),
        tmp_path / "s.gif",
        mesh=ch2,
        size=(w, h),
        n_frames=12,
        spin=spin,
        frames_dir=tmp_path / "f",
    )
    fill = []
    for k, a in enumerate(_pngs(tmp_path / "f", 12)):
        nz = a.min(axis=2) < 250
        rows, cols = np.flatnonzero(nz.any(1)), np.flatnonzero(nz.any(0))
        assert rows[0] > 1 and rows[-1] < h - 2, (k, rows[0], rows[-1])
        assert cols[0] > 1 and cols[-1] < w - 2, (k, cols[0], cols[-1])
        fill.append(max((rows[-1] - rows[0]) / h, (cols[-1] - cols[0]) / w))
    print(spin, "largest extent / image size per frame:", np.round(fill, 3))
    assert max(fill) > 0.85


# ------------------------------------------------------------------ panels


def test_image_diff_known_values(tmp_path):
    a = Image.fromarray(np.zeros((10, 16, 3), np.uint8))
    b = Image.fromarray(np.full((10, 16, 3), 30, np.uint8))
    a.save(tmp_path / "a.png")
    assert bn.image_diff(tmp_path / "a.png", a) == 0.0
    assert bn.image_diff(a, b) == pytest.approx(30.0)


def _boxed(bg, fg, size=(100, 60), rect=(20, 30, 40, 45)):
    im = Image.new("RGB", size, bg)
    im.paste(Image.new("RGB", (rect[2] - rect[0], rect[3] - rect[1]), fg), rect[:2])
    return im


def test_content_box_white():
    im = _boxed("white", (10, 10, 10))
    box = bn.content_box([im])
    print("margin 10:", box)
    assert box == (10, 20, 50, 55)
    assert bn.content_box([im], margin=0) == (20, 30, 40, 45)


def test_content_box_single_image_path_and_union(tmp_path):
    a = _boxed("white", (10, 10, 10))
    b = _boxed("white", (10, 10, 10), rect=(60, 5, 70, 15))
    assert bn.content_box(a, margin=0) == (20, 30, 40, 45)
    a.save(tmp_path / "a.png")
    assert bn.content_box(tmp_path / "a.png", margin=0) == (20, 30, 40, 45)
    assert bn.content_box(str(tmp_path / "a.png"), margin=0) == (20, 30, 40, 45)
    box = bn.content_box([a, b], margin=0)
    print("union:", box)
    assert box == (20, 5, 70, 45)


def test_content_box_background():
    im = _boxed("black", "white")
    assert bn.content_box(im, margin=0, background="black") == (20, 30, 40, 45)
    assert bn.content_box(im, margin=0) == (0, 0, 100, 60)


def test_content_box_all_background_and_sizes():
    assert bn.content_box(Image.new("RGB", (100, 60), "white")) == (0, 0, 100, 60)
    with pytest.raises(ValueError, match="same size"):
        bn.content_box([Image.new("RGB", (10, 10)), Image.new("RGB", (10, 12))])


def test_content_box_sequences_empty_and_margin():
    from collections import UserList

    a = _boxed("white", (10, 10, 10))
    b = _boxed("white", (10, 10, 10), rect=(60, 5, 70, 15))
    assert bn.content_box(UserList([a, b]), margin=0) == (20, 5, 70, 45)
    assert bn.content_box((a,), margin=0) == (20, 30, 40, 45)
    for bad in ([], ()):
        with pytest.raises(ValueError, match="empty"):
            bn.content_box(bad)
    for bad in (2.5, "3", -1, None, True):
        with pytest.raises(ValueError, match="margin") as err:
            bn.content_box(a, margin=bad)
        print("margin", repr(bad), "->", err.value)


def test_content_box_threshold_is_exactly_five():
    for channel in range(3):
        for delta, content in ((5, False), (6, True)):
            color = [255, 255, 255]
            color[channel] -= delta
            box = bn.content_box(_boxed("white", tuple(color)), margin=0)
            print("channel", channel, "delta", delta, "->", box)
            assert box == ((20, 30, 40, 45) if content else (0, 0, 100, 60))
    # same rule against a non-white background, in the other direction
    box = bn.content_box(_boxed((100, 100, 100), (106, 100, 100)), 0, (100, 100, 100))
    assert box == (20, 30, 40, 45)
    box = bn.content_box(_boxed((100, 100, 100), (105, 100, 100)), 0, (100, 100, 100))
    assert box == (0, 0, 100, 60)


def test_content_box_exported():
    assert htfa.content_box is htfa.brainnet.content_box
    assert "content_box" in htfa.__all__ and "content_box" in bn.__all__
    assert not hasattr(bn, "_content_box")


_IMAGE_HELPERS = """
import json, sys
from PIL import Image, ImageDraw
import htfa.brainnet as bn

a = Image.new("RGB", (400, 250), "white")
ImageDraw.Draw(a).ellipse((80, 60, 300, 200), fill=(200, 30, 30))
b = Image.new("RGB", (400, 250), "white")
ImageDraw.Draw(b).rectangle((120, 40, 340, 220), fill=(30, 30, 200))
pair = bn.annotate_pair(a, b, ("Left\\nsub", "Right"), "TR 3", width=800)
grid = bn.montage([[a, b], [b, a]], ["r1", "r2"], ["c1", "c2"], title="T")
print(json.dumps({
    "pair": pair.size, "montage": grid.size,
    "box": bn.content_box([a, b]), "box_white": bn.content_box(a, background="white"),
    "loaded": [m for m in ("pyvista", "vtkmodules", "matplotlib") if m in sys.modules],
}))
"""


def test_image_helpers_need_no_3d_extras():
    """annotate_pair, montage and content_box on white PIL images import neither
    pyvista nor matplotlib (both are 3D-extra dependencies), as in 0.1.0.

    Run in a fresh process, so that nothing imported by other tests counts.
    """
    import json
    import subprocess

    run = subprocess.run(
        [sys.executable, "-c", _IMAGE_HELPERS], capture_output=True, text=True,
        timeout=120,
    )  # fmt: skip
    print("exit code", run.returncode, "stdout:", run.stdout, "stderr:", run.stderr,
          sep="\n")  # fmt: skip
    assert run.returncode == 0
    got = json.loads(run.stdout.splitlines()[-1])
    assert got["loaded"] == []
    assert got["box"] == [70, 30, 351, 231] and got["box_white"] == [70, 50, 311, 211]
    assert got["pair"][0] == 800 and got["montage"][1] > 0


def test_annotate_pair_and_montage(tmp_path):
    left = bn.plot_network(CENTERS, _conn(), mesh=None)
    right = bn.plot_network(CENTERS, -_conn(), mesh=None)
    pair = bn.annotate_pair(
        left, right, ("Original\nsaved files", "Reconstruction"), "TR 3", width=1200
    )
    pair.save(tmp_path / "pair.png")
    box = bn.content_box([left.convert("RGB"), right.convert("RGB")])
    pane_h = round(600 * (box[3] - box[1]) / (box[2] - box[0]))
    head = pair.size[1] - pane_h
    assert pair.size[0] == 1200 and head > 60
    # the brain panes are the frames untouched: text is only in the header
    expect = left.crop(box).resize((600, pane_h), Image.LANCZOS)
    np.testing.assert_array_equal(_rgb(pair)[head:, :600], _rgb(expect))
    assert (_rgb(pair)[:head].min(axis=2) < 128).sum() > 200
    grid = bn.montage(
        [[left, right], [right, left]], ["Remember cue", "Forget\ncue"],
        ["TR 0", "TR 1"], title="Network after the cue", cell_width=200,
        out=tmp_path / "m.png",
    )  # fmt: skip
    cell_h = round(200 * (box[3] - box[1]) / (box[2] - box[0]))
    assert grid.size[1] == int(40 * 1.8) + int(40 * 1.6) + 2 * cell_h
    assert (tmp_path / "m.png").stat().st_size > 5000
    with pytest.raises(ValueError, match="rows of"):
        bn.montage([[left]], ["a", "b"], ["c"])


# ------------------------------------------------------- fits and wrappers


def _fit(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return htfa.fit(*args, **kwargs)


@pytest.fixture(scope="module")
def res_htfa():
    Xs, Rs, _ = make_htfa_data(2, (8, 8, 8), n_samples=30, K=4, random_state=0)
    Rs = [R * 12.0 - np.array([42.0, 60.0, 30.0]) for R in Rs]  # spread over MNI mm
    return _fit(Xs, coords=Rs, K=4, max_global_iter=3, random_state=0)


# The wrapper tests below check the arguments the wrappers resolve, exactly,
# and render once through each wrapper. They do not compare two renders
# pixel for pixel: Apple's software OpenGL renderer on the macOS 3.10/3.11 CI
# runners does not render translucent edges deterministically (measured:
# 25/25/25/5/5 differing pixels across 6 renders of the same scene).


@pytest.fixture(scope="module")
def res_tfa():
    X, R, _ = make_tfa_data((8, 8, 8), n_samples=20, K=3, random_state=2)
    return _fit(X, coords=R * 10.0 - 35.0, K=3, max_iter=5, random_state=0)


def _same(got, expect):
    assert len(got) == len(expect) == 3
    for g, e in zip(got, expect, strict=True):
        assert np.array_equal(g, e), (g, e)


def test_network_inputs_global_and_subject(res_htfa):
    r = res_htfa
    _same(r._network_inputs(), (r.global_centers, r.global_widths, r.isfc()))
    _same(r._network_inputs(subject=1), (r.centers[1], r.widths[1], r.connectivity(1)))
    _same(
        r._network_inputs(window=25),
        (r.global_centers, r.global_widths, r.isfc(25)),
    )
    _same(
        r._network_inputs(subject=0, window=20),
        (r.centers[0], r.widths[0], r.dynamic_connectivity(0, 20)),
    )
    given = np.eye(4)
    assert r._network_inputs(subject=1, connectivity=given)[2] is given
    assert r._network_inputs(window=20)[2].shape == (30 - 20 + 1, 4, 4)
    # estimators and results resolve to the same nodes in htfa.brainnet
    for nodes in (r, r.model):
        centers, sizes = bn._nodes(nodes, None, "global")
        assert np.array_equal(centers, r.global_centers)
        assert np.array_equal(sizes, r.global_widths)
    centers, sizes = bn._nodes(r, None, 1)
    assert np.array_equal(centers, r.centers[1]) and np.array_equal(sizes, r.widths[1])


def test_network_inputs_tfa_falls_back_to_subject_0(res_tfa):
    r = res_tfa
    expect = (r.centers[0], r.widths[0], r.connectivity(0))
    _same(r._network_inputs(), expect)
    _same(r._network_inputs(subject=0), expect)
    _same(
        r._network_inputs(window=10),
        (r.centers[0], r.widths[0], r.dynamic_connectivity(0, 10)),
    )
    centers, sizes = bn._nodes(r.model, None, "global")
    assert np.array_equal(centers, r.centers[0]) and np.array_equal(sizes, r.widths[0])
    with pytest.raises(ValueError, match="no global template"):
        bn.plot_network(r, mesh=None, subject="global")


def _signed_edges(K):
    """Edge 0-1 positive, edge 2-3 negative, nothing else."""
    E = np.zeros((K, K))
    E[0, 1] = E[1, 0] = 0.8
    E[2, 3] = E[3, 2] = -0.8
    return E


def _midpoint_window(a, centers, i, j):
    y, z = (centers[i, 1:] + centers[j, 1:]) / 2
    return _window(a, y, z, half=3)


def test_results_plot_network_renders(res_htfa, tmp_path):
    r = res_htfa
    out = tmp_path / "g.png"
    img = r.plot_network(connectivity=_signed_edges(4), mesh=None, out=out)
    with Image.open(out) as saved:
        assert saved.size == (1600, 1000) and saved.mode == "RGB"
    a = _rgb(img)
    drawn = r.display_nodes()[0]  # nodes are snapped by default
    pos = _midpoint_window(a, drawn, 0, 1)
    neg = _midpoint_window(a, drawn, 2, 3)
    print("red/blue px at +edge", _red(pos).sum(), _blue(pos).sum())
    print("red/blue px at -edge", _red(neg).sum(), _blue(neg).sum())
    assert _red(pos).sum() >= 10 and _blue(pos).sum() == 0
    assert _blue(neg).sum() >= 10 and _red(neg).sum() == 0


def test_results_animate_network_renders(res_htfa, tmp_path):
    r = res_htfa
    E = _signed_edges(4)
    out = r.animate_network(
        tmp_path / "s1.gif", subject=1, connectivity=np.stack([E, -E]), mesh=None,
        frames_dir=tmp_path / "f",
    )  # fmt: skip
    with Image.open(out) as g:
        assert g.n_frames == 2 and g.size == (1600, 1000)
    frames = []
    for k in (1, 2):
        with Image.open(tmp_path / "f" / f"{k}.png") as im:
            assert im.size == (1600, 1000) and im.mode == "RGB"
            frames.append(_rgb(im))
    drawn = r.display_nodes(1)[0]  # nodes are snapped by default
    first = _midpoint_window(frames[0], drawn, 0, 1)
    second = _midpoint_window(frames[1], drawn, 0, 1)
    print("edge 0-1 red/blue px, frame 1:", _red(first).sum(), _blue(first).sum(),
          "frame 2:", _red(second).sum(), _blue(second).sum())  # fmt: skip
    assert _red(first).sum() >= 10 and _blue(first).sum() == 0
    assert _blue(second).sum() >= 10 and _red(second).sum() == 0


def test_results_animate_network_duration_and_spin(res_htfa, tmp_path):
    """HTFAResults.animate_network passes duration, spin and fps through."""
    out = res_htfa.animate_network(
        tmp_path / "s.gif",
        subject=1,
        connectivity=_signed_edges(4),
        mesh=None,
        size=(200, 125),
        fps=5,
        duration=0.8,
        spin=(0, 0, 1),
    )
    with Image.open(out) as g:
        assert g.n_frames == 4 and g.info["duration"] == 200


# ------------------------------------------------- JPEG output and configs


def test_jpeg_output_matches_matlab_export(tmp_path):
    """A .jpg is written at quality 75 with 4:2:0 chroma unless told otherwise."""
    png = bn.plot_network(CENTERS, _conn(), mesh=None, out=tmp_path / "a.png")
    bn.plot_network(CENTERS, _conn(), mesh=None, out=tmp_path / "a.jpg")
    with Image.open(tmp_path / "a.jpg") as im:
        assert im.format == "JPEG"
        a75 = _rgb(im)
    expect = bn._jpeg(png, 75)  # same round trip in memory
    assert np.abs(a75 - _rgb(expect)).max() == 0
    # jpeg_quality compresses the returned image too; lower quality, larger error
    q30 = bn.plot_network(CENTERS, _conn(), mesh=None, jpeg_quality=30,
                          out=tmp_path / "b.jpg")  # fmt: skip
    q95 = bn.plot_network(CENTERS, _conn(), mesh=None, jpeg_quality=95)
    e30, e95 = bn.image_diff(q30, png, png.size), bn.image_diff(q95, png, png.size)
    print("jpeg error q30 / q95:", e30, e95)
    assert e30 > e95 > 0
    assert (tmp_path / "b.jpg").stat().st_size < (tmp_path / "a.jpg").stat().st_size

    # thin edges get paler: the red edge's most saturated pixel loses saturation
    def sat(a):
        return (a[..., 0] - a[..., 1]).max()

    assert sat(_window(_rgb(q30), -10, 50)) < sat(_window(_rgb(png), -10, 50))


def test_animate_network_jpeg_frames(tmp_path):
    bn.animate_network(CENTERS, _frames(), tmp_path / "a.gif", mesh=None,
                       jpeg_quality=75, frames_dir=tmp_path / "f")  # fmt: skip
    names = sorted(p.name for p in (tmp_path / "f").iterdir())
    assert names == ["1.jpg", "2.jpg", "3.jpg"]
    with Image.open(tmp_path / "f" / "2.jpg") as im:
        assert im.format == "JPEG"


def test_config_roundtrip_and_style(tmp_path):
    import scipy.io

    ec = {
        "bak": {"color": np.array([1, 1, 1], np.uint8)},
        "msh": {"color": np.array([0.95, 0.95, 0.95]), "alpha": 0.2},
        "nod": {"CM": np.vstack([[0.502, 0.502, 0.502], np.tile([1.0, 0, 0], (3, 1))])},
        "edg": {"draw_threshold": 0.25,
                "CM": np.vstack([[1.0, 0, 0], np.full((3, 3), 0.38), [0, 0, 1.0]])},
        "lot": {"view_direction": 1, "view_az": -80},
    }  # fmt: skip
    path = tmp_path / "cfg.mat"
    scipy.io.savemat(path, {"EC": ec})
    cfg = bn.read_brainnet_config(path)
    assert cfg["lot"]["view_az"] == -80 and cfg["edg"]["draw_threshold"] == 0.25
    style = bn.config_style(cfg)
    assert style == {
        "background": (1.0, 1.0, 1.0),
        "mesh_color": (0.95, 0.95, 0.95),
        "mesh_opacity": 0.2,
        "node_color": (0.502, 0.502, 0.502),
        "threshold": 0.25,
        "positive_color": (1.0, 0.0, 0.0),
        "negative_color": (0.0, 0.0, 1.0),
    }
    img = bn.plot_network(CENTERS, _conn(), mesh=None, **style)
    ref = bn.plot_network(CENTERS, _conn(), mesh=None)
    assert bn.image_diff(img, ref) == 0.0
    dark = _rgb(bn.plot_network(CENTERS, mesh=None, background=(0, 0, 0)))
    assert tuple(dark[0, 0]) == (0, 0, 0)
    scipy.io.savemat(tmp_path / "x.mat", {"other": 1})
    with pytest.raises(ValueError, match="no BrainNet option struct"):
        bn.read_brainnet_config(tmp_path / "x.mat")


@pytest.mark.skipif(not GT_2014.is_dir(), reason=f"2014 frames not found at {GT_2014}")
def test_2014_config_jpeg_reproduction(ch2, tmp_path):
    """The 2014 option file plus MATLAB-quality JPEG output match the 2014 frame."""
    style = bn.config_style(bn.read_brainnet_config(GT_2014 / "DF_config.mat"))
    assert style["threshold"] == 0.25 and style["mesh_opacity"] == 0.2
    centers, _, sizes, _ = bn.read_node(GT_2014 / "frames/cue_r/1.node")
    E = bn.read_edge(GT_2014 / "frames/cue_r/28.edge")
    out = tmp_path / "28.jpg"
    bn.plot_network(centers, E, sizes=sizes, mesh=ch2, out=out, **style)
    d = bn.image_diff(out, GT_2014 / "frames/cue_r/28.jpg")
    print("2014 frame 28, JPEG q75 with the 2014 config:", d)
    assert d < 2.75


# ------------------------------------------------------------ effective nodes


def _grid(lo, hi, h):
    """Voxel centers of a cube [lo, hi]^3 at spacing h, as (V, 3) mm."""
    ax = np.arange(lo, hi + h / 2, h, dtype=float)
    return np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)


def _sphere(radius, h, center=(0.0, 0.0, 0.0)):
    center = np.asarray(center, dtype=float)
    g = _grid(-radius - h, radius + h, h) + center
    return g[np.linalg.norm(g - center, axis=1) <= radius]


@pytest.mark.parametrize("h", [2.0, 4.0])
def test_effective_nodes_interior_nodes_unchanged(h):
    """A node far from the mask edge keeps its center and width.

    The RBF's per-axis variance is lambda / 2, so lambda_eff = lambda up to the
    grid discretization; this measures that error at 2 mm and 4 mm.
    """
    mask = _grid(-80, 80, h)
    widths = np.array([20.0, 50.0, 100.0, 200.0, 400.0])
    centers = np.array([[0.3, -1.1, 0.7]] * len(widths))
    c_eff, w_eff, info = bn.effective_nodes(centers, widths, mask)
    shift = np.linalg.norm(c_eff - centers, axis=1)
    ratio = w_eff / widths
    print(f"h={h} mm: shifts (mm)", shift, "lambda_eff/lambda", ratio)
    print("mass fraction", info["mass_fraction"])
    assert shift.max() < 1e-3 * h
    assert np.abs(ratio - 1).max() < 2e-3
    np.testing.assert_allclose(info["width_ratio"], ratio)
    np.testing.assert_allclose(info["sigma"], np.sqrt(w_eff / 2))
    np.testing.assert_allclose(info["distance"], shift)
    assert np.abs(info["mass_fraction"] - 1).max() < 2e-3
    assert not info["snapped"].any() and not info["ambiguous"].any()


def test_effective_nodes_outside_sphere_projects_inside():
    """A node outside a sphere moves inside; the reported distance is the move."""
    h, radius = 2.0, 40.0
    mask = _sphere(radius, h)
    centers = np.array([[60.0, 0.0, 0.0], [0.0, -55.0, 10.0]])
    widths = np.array([150.0, 300.0])
    c_eff, w_eff, info = bn.effective_nodes(centers, widths, mask)
    print("effective centers", c_eff, "widths", w_eff, "info", info)
    assert (np.linalg.norm(c_eff, axis=1) < radius).all()
    np.testing.assert_allclose(
        info["distance"], np.linalg.norm(c_eff - centers, axis=1)
    )
    assert (info["distance"] > 20).all()
    # the first node stays on the x axis (symmetry), inside the sphere
    np.testing.assert_allclose(c_eff[0, 1:], 0.0, atol=1e-9)
    # a convex mask holds every weighted centroid: no snap is needed
    assert not info["snapped"].any()
    assert (info["mass_fraction"] < 0.5).all() and (w_eff < widths).all()


def _hemispheres(h, gap):
    """Two balls of radius 30 mm whose surfaces are ``gap`` mm apart along x."""
    off = 30.0 + gap / 2
    return np.vstack([_sphere(30.0, h, (-off, 0, 0)), _sphere(30.0, h, (off, 0, 0))])


def test_effective_nodes_gap_node_snaps_deterministically():
    """A node in the gap between two hemispheres snaps into one of them.

    Its centroid lies in the gap (outside the mask). When it is symmetric the
    two nearest voxels tie: the lowest voxel index wins and the node is flagged
    ambiguous. Off the midline it goes to the nearer hemisphere, unflagged.
    """
    mask = _hemispheres(2.0, gap=10.0)
    centers = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    widths = np.array([400.0, 400.0])
    c_eff, _, info = bn.effective_nodes(centers, widths, mask)
    print("gap nodes", c_eff, info)
    assert info["snapped"].all()
    assert info["ambiguous"].tolist() == [True, False]
    # every effective center is a mask voxel
    d = np.min(np.linalg.norm(mask[None] - c_eff[:, None], axis=2), axis=1)
    np.testing.assert_allclose(d, 0.0, atol=1e-9)
    assert c_eff[0, 0] < 0  # the left ball holds the lower voxel indices
    assert c_eff[1, 0] > 0  # nearer (right) hemisphere
    again, _, info2 = bn.effective_nodes(centers, widths, mask[::-1].copy())
    print("reversed voxel order", again)
    assert again[0, 0] > 0 and info2["ambiguous"][0]  # lowest index is now right
    np.testing.assert_array_equal(bn.effective_nodes(centers, widths, mask)[0], c_eff)


def test_effective_nodes_far_center_walk_is_bounded():
    """A center 100 km away: the segment walk stops at the mask's extent."""
    import time

    mask = _hemispheres(2.0, gap=10.0)
    t0 = time.perf_counter()
    c_eff, _, info = bn.effective_nodes([[0.0, 0.0, 1e5]], [400.0], mask)
    took = time.perf_counter() - t0
    print("far node", c_eff, info, f"{took:.3f} s")
    assert info["snapped"][0] and info["ambiguous"][0]  # gap centroid, symmetric
    assert not _outside(c_eff, mask).any()
    assert took < 1.0  # measured 11 s when the walk ran the full 1e5 mm


def test_effective_nodes_snaps_along_segment_into_concave_mask():
    """A centroid in a ring's hole snaps along the line back to the node."""
    g = _grid(-40, 40, 2.0)
    rho = np.hypot(g[:, 0], g[:, 1])
    ring = g[np.hypot(rho - 30.0, g[:, 2]) <= 6.0]
    centers = np.array([[30.0, 0.0, 0.0]])
    c_eff, _, info = bn.effective_nodes(centers, np.array([5000.0]), ring)
    print("ring node", c_eff, info)
    assert info["snapped"][0] and not info["ambiguous"][0]
    # the first ring voxel on the way from the hole's centroid to x = 30
    assert 22.0 <= c_eff[0, 0] <= 26.0
    np.testing.assert_allclose(c_eff[0, 1:], 0.0, atol=1e-9)


def test_effective_nodes_flat_boundary_half_mass():
    """A node centered on a flat boundary keeps half its mass.

    Inward shift sigma * sqrt(2 / pi) (half-normal mean) and lambda_eff /
    lambda = 1 - 2 / (3 pi) (one axis has half-normal variance).
    """
    g = _grid(-79, 79, 2.0)
    mask = g[g[:, 0] < 0]
    w = 100.0
    c_eff, w_eff, info = bn.effective_nodes(np.zeros((1, 3)), np.array([w]), mask)
    sigma = np.sqrt(w / 2)
    print("flat boundary", c_eff, w_eff / w, info["mass_fraction"])
    assert info["mass_fraction"][0] == pytest.approx(0.5, abs=2e-3)
    assert c_eff[0, 0] == pytest.approx(-sigma * np.sqrt(2 / np.pi), rel=5e-3)
    assert w_eff[0] / w == pytest.approx(1 - 2 / (3 * np.pi), rel=5e-3)
    assert not info["snapped"][0]


def _box(spacing, half=60.0, offset=(0.0, 0.0, 0.0)):
    """A box grid with per-axis spacing, as (V, 3) coordinates."""
    axes = [
        np.arange(-half, half + s / 2, s) + o
        for s, o in zip(spacing, offset, strict=True)
    ]
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)


def test_effective_nodes_anisotropic_grid():
    """On a 3 x 3 x 4 mm grid an interior node between slices stays, mass ~1."""
    affine = np.diag([3.0, 3.0, 4.0, 1.0])
    affine[:3, 3] = -60.0
    coords = _box((3.0, 3.0, 4.0))
    centers = np.array([[0.0, 0.0, 2.0], [1.5, -1.0, -6.0]])
    widths = np.array([60.0, 150.0])
    vol = np.ones((41, 41, 31), bool)
    for mask in (coords, (vol, affine)):
        c_eff, w_eff, info = bn.effective_nodes(centers, widths, mask)
        print("anisotropic", c_eff, info)
        assert not info["snapped"].any()
        assert info["distance"].max() < 1e-3
        np.testing.assert_allclose(info["mass_fraction"], 1.0, atol=2e-3)
        np.testing.assert_allclose(w_eff / widths, 1.0, atol=2e-3)
    spacing, volume = bn._mask_coords(coords)[1:]
    np.testing.assert_allclose(spacing, [3.0, 3.0, 4.0])
    assert volume == pytest.approx(36.0)


@pytest.mark.parametrize("shift", [(1.0, 1.0, 1.0), (1.0, 0.0, 0.0), (0.7, -1.3, 0.4)])
def test_effective_nodes_union_of_shifted_grids(shift):
    """Two subjects' 2 mm grids offset from each other: an interior node stays
    unsnapped with mass ~1, and a node outside lands inside the union."""
    a = _box((2.0, 2.0, 2.0), half=40.0)
    b = _box((2.0, 2.0, 2.0), half=40.0, offset=shift)
    union = np.unique(np.vstack([a, b]), axis=0)
    centers = np.array([[0.3, 0.9, -1.7], [0.0, 0.0, 70.0]])
    widths = np.array([50.0, 100.0])
    c_eff, w_eff, info = bn.effective_nodes(centers, widths, union)
    print(shift, "union", c_eff, info)
    assert not info["snapped"][0] and info["distance"][0] < 0.05
    assert info["mass_fraction"][0] == pytest.approx(1.0, abs=0.03)
    assert w_eff[0] / widths[0] == pytest.approx(1.0, abs=0.03)
    assert not _outside(c_eff, union).any()
    assert c_eff[1, 2] < 42.0


def test_effective_nodes_mask_forms_agree():
    """Coordinates, a NIfTI image and a (volume, affine) pair give one answer."""
    import nibabel as nib

    affine = np.array(
        [[2.0, 0, 0, -40], [0, 2.0, 0, -40], [0, 0, 2.0, -40], [0, 0, 0, 1]]
    )
    ijk = np.stack(np.meshgrid(*[np.arange(41)] * 3, indexing="ij"), -1)
    xyz = ijk @ affine[:3, :3].T + affine[:3, 3]
    vol = np.linalg.norm(xyz, axis=-1) <= 30.0
    coords = xyz[vol]
    centers = np.array([[45.0, 3.0, -2.0], [0.0, 0.0, 0.0]])
    widths = np.array([120.0, 60.0])
    ref = bn.effective_nodes(centers, widths, coords)
    for mask in (nib.Nifti1Image(vol.astype(np.uint8), affine), (vol, affine)):
        got = bn.effective_nodes(centers, widths, mask)
        np.testing.assert_allclose(got[0], ref[0], atol=1e-9)
        np.testing.assert_allclose(got[1], ref[1], rtol=1e-9)
        for key in ref[2]:
            np.testing.assert_allclose(got[2][key], ref[2][key], rtol=1e-9)


def test_effective_nodes_point_nodes_and_errors():
    """Without widths, inside nodes stay and outside nodes go to the nearest voxel."""
    mask = _sphere(20.0, 2.0)
    centers = np.array([[0.4, 0.2, -0.6], [0.0, 0.0, 35.0]])
    c_eff, w_eff, info = bn.effective_nodes(centers, None, mask)
    print("point nodes", c_eff, info)
    np.testing.assert_array_equal(c_eff[0], centers[0])
    np.testing.assert_allclose(c_eff[1], [0.0, 0.0, 20.0])
    assert info["snapped"].tolist() == [False, True]
    assert info["distance"][1] == pytest.approx(15.0)
    assert np.isnan(w_eff).all() and np.isnan(info["mass_fraction"]).all()
    with pytest.raises(ValueError, match="widths has 1 entries for 2 nodes"):
        bn.effective_nodes(centers, np.array([1.0]), mask)
    with pytest.raises(ValueError, match="positive"):
        bn.effective_nodes(centers, np.array([1.0, 0.0]), mask)
    with pytest.raises(ValueError, match="no voxels"):
        bn.effective_nodes(centers, None, np.zeros((0, 3)))
    with pytest.raises(ValueError, match="dimensions"):
        bn.effective_nodes(centers, None, np.zeros((5, 2)))
    with pytest.raises(ValueError, match="coincide"):
        bn.effective_nodes(centers, None, np.zeros((5, 3)))


def _gray(a, y, z, half=4):
    w = _window(a, y, z, half)
    return bool(np.all(np.abs(w[..., 0] - w[..., 2]) < 10) and w.max() < 200)


def test_plot_network_projects_nodes_into_mask(tmp_path):
    """plot_network draws a node at its in-mask centroid unless snap=False."""
    mask = _sphere(40.0, 2.0)
    centers = np.array([[0.0, 70.0, 0.0], [0.0, -20.0, 0.0]])
    widths = np.array([200.0, 50.0])
    c_eff, w_eff, _ = bn.effective_nodes(centers, widths, mask)
    print("effective centers", c_eff)
    assert c_eff[0, 1] < 40
    snapped = bn.plot_network(
        centers, widths=widths, mask=mask, mesh=None, out=tmp_path / "snapped.png"
    )
    raw = bn.plot_network(
        centers, widths=widths, mask=mask, mesh=None, snap=False,
        out=tmp_path / "raw.png",
    )  # fmt: skip
    a, b = _rgb(snapped), _rgb(raw)
    assert _gray(a, c_eff[0, 1], c_eff[0, 2]) and _window(a, 70, 0).min() == 255
    assert _gray(b, 70, 0) and _window(b, c_eff[0, 1], c_eff[0, 2]).min() == 255
    # both nodes drawn at their effective centers, sized by lambda_eff
    radii = bn.node_radius(w_eff)
    got = [_gray_run(a, c_eff[k, 1], c_eff[k, 2]) for k in range(2)]
    print("radii (mm)", radii, "diameters (px)", got)
    np.testing.assert_allclose(got, 2 * radii * SCALE, atol=3)


def test_plot_network_default_mni_mask_for_bare_centers():
    """Bare centers snap into the MNI152 brain mask; a node above the head moves."""
    centers = np.array([[0.0, 95.0, 0.0], [0.0, -20.0, 20.0]])
    img = _rgb(bn.plot_network(centers, mesh=None))
    raw = _rgb(bn.plot_network(centers, mesh=None, snap=False))
    c_eff, _, info = bn.effective_nodes(centers, None, bn.mni_mask())
    print("MNI point nodes", c_eff, info)
    assert info["snapped"][0] and not info["snapped"][1]
    assert 55 < c_eff[0, 1] < 75 and info["distance"][0] > 20
    np.testing.assert_array_equal(c_eff[1], centers[1])
    assert _gray(img, c_eff[0, 1], c_eff[0, 2]) and _window(img, 95, 0).min() == 255
    assert _gray(raw, 95, 0)


def test_plot_network_mass_options(tmp_path):
    """fade_by_mass lightens a node mostly outside the mask and scale_by_mass
    shrinks it; errors without snap.

    Separate renders are not bitwise reproducible on CI's software OpenGL, so
    the renders are compared through measured properties: the darkness at a
    node's center and the diameter of its sphere.
    """
    mask = _sphere(40.0, 2.0)
    # three nodes so that node_radius's min-max mapping cannot hide the scaling
    centers = np.array([[0.0, 50.0, 0.0], [0.0, -10.0, 0.0], [0.0, 0.0, 40.0]])
    widths = np.array([100.0, 100.0, 100.0])
    c_eff, w_eff, info = bn.effective_nodes(centers, widths, mask)
    mass = info["mass_fraction"]
    print("mass fractions", mass)
    assert mass[0] < 0.2 and mass[1] > 0.99 and 0.3 < mass[2] < 0.7
    # the drawn sizes and opacities follow the mass fractions
    _, sizes, alpha = bn._display_nodes(
        centers, None, widths, "global", True, mask, fade_by_mass=True
    )
    np.testing.assert_allclose(sizes, w_eff)
    np.testing.assert_allclose(alpha, 0.15 + 0.85 * np.clip(mass, 0, 1))
    _, scaled_sizes, _ = bn._display_nodes(
        centers, None, widths, "global", True, mask, scale_by_mass=True
    )
    np.testing.assert_allclose(scaled_sizes, w_eff * mass)
    plain = _rgb(bn.plot_network(centers, widths=widths, mask=mask, mesh=None))
    faded = _rgb(
        bn.plot_network(
            centers, widths=widths, mask=mask, mesh=None, fade_by_mass=True,
            out=tmp_path / "faded.png",
        )
    )  # fmt: skip
    y, z = c_eff[0, 1], c_eff[0, 2]
    print(
        "min gray plain/faded", _window(plain, y, z).min(), _window(faded, y, z).min()
    )
    assert _window(faded, y, z).min() > _window(plain, y, z).min() + 40
    # the interior node keeps (nearly) full opacity: as dark as without fading
    inner_plain, inner_faded = _window(plain, -10, 0), _window(faded, -10, 0)
    print("interior node min gray plain/faded", inner_plain.min(), inner_faded.min())
    assert inner_faded.max() < 200 and abs(inner_faded.min() - inner_plain.min()) <= 5
    # scale_by_mass: each sphere's measured diameter matches its radius. A
    # fixed radius = 0.08 * size (clipped to 1-10 mm) shows the scaling
    # directly; the default min-max mapping would rescale it away.
    radius = {"radius_kb": (0.08, 0.0), "radius_clip": (1.0, 10.0)}
    diameters = []
    for scale, s in ((False, w_eff), (True, w_eff * mass)):
        img = _rgb(
            bn.plot_network(centers, widths=widths, mask=mask, mesh=None,
                            scale_by_mass=scale, **radius)
        )  # fmt: skip
        radii = np.clip(0.08 * s, 1.0, 10.0)
        got = [_gray_run(img, c_eff[k, 1], c_eff[k, 2]) for k in range(3)]
        print("scale_by_mass", scale, "radii (mm)", radii, "diameters (px)", got)
        np.testing.assert_allclose(got, 2 * radii * SCALE, atol=3)
        diameters.append(got)
    # the boundary node (mass ~0.4) is drawn much smaller; the interior one not
    assert diameters[1][2] < diameters[0][2] - 20
    assert abs(diameters[1][1] - diameters[0][1]) <= 3
    with pytest.raises(ValueError, match="snap"):
        bn.plot_network(
            centers, widths=widths, mesh=None, snap=False, fade_by_mass=True
        )
    with pytest.raises(ValueError, match="widths"):
        bn.plot_network(centers, mask=mask, mesh=None, scale_by_mass=True)
    with pytest.raises(ValueError, match="fade_by_mass"):
        bn.plot_network(centers, widths=widths, mask=mask, mesh=None, fade_by_mass=1.5)


def test_results_display_nodes_match_effective_nodes(res_htfa, res_tfa):
    """display_nodes uses the subject's voxels, or all subjects' for the template."""
    r = res_htfa
    union = np.unique(np.vstack([s.coords for s in r.subjects]), axis=0)
    got = r.display_nodes()
    ref = bn.effective_nodes(r.global_centers, r.global_widths, union)
    for g, e in zip(got[:2], ref[:2], strict=True):
        np.testing.assert_array_equal(g, e)
    got1 = r.display_nodes(subject=1)
    ref1 = bn.effective_nodes(r.centers[1], r.widths[1], r.subjects[1].coords)
    np.testing.assert_array_equal(got1[0], ref1[0])
    print("template shifts (mm)", got[2]["distance"], "ratio", got[2]["width_ratio"])
    t = res_tfa
    np.testing.assert_array_equal(
        t.display_nodes()[0],
        bn.effective_nodes(t.centers[0], t.widths[0], t.subjects[0].coords)[0],
    )
    # an HTFAResults or a fitted model passed to htfa.brainnet uses the same
    # default masks as display_nodes / the MNI mask
    img_res = _rgb(bn.plot_network(r, _signed_edges(4), mesh=None))
    img_wrap = _rgb(r.plot_network(connectivity=_signed_edges(4), mesh=None))
    assert _red(_midpoint_window(img_res, got[0], 0, 1)).sum() >= 10
    assert _red(_midpoint_window(img_wrap, got[0], 0, 1)).sum() >= 10


def _outside(points, mask):
    """Which points are not within half a voxel (per axis) of a mask voxel."""
    from scipy.spatial import cKDTree

    coords, spacing, _ = bn._mask_coords(mask)
    tree = cKDTree(coords / spacing)
    return tree.query(np.asarray(points) / spacing, p=np.inf)[0] > 0.5 * (1 + 1e-9)


def _check_display_inside(res):
    """Every node the default plot path draws lies within the mask; returns
    the number of fitted centers outside, per mask."""
    n_out = {}
    subjects = ["global", *range(len(res.subjects))]
    for s in subjects:
        mask = res._display_mask(s)
        fitted = res._fitted_nodes(s)[0]
        shown = res.display_nodes(s)[0]
        drawn = bn._display_nodes(res, None, None, s, True, None)[0]
        np.testing.assert_array_equal(drawn, shown)
        n_out[s] = int(_outside(fitted, mask).sum())
        assert not _outside(shown, mask).any(), (s, shown[_outside(shown, mask)])
    return n_out


def test_display_nodes_inside_mask_for_outside_centers(res_htfa):
    """A fit whose centers lie outside the voxels draws every node inside them."""
    import copy

    model = copy.deepcopy(res_htfa.model)
    # the voxels span about 84 mm per axis: these move every center out of them
    push = np.array([[100.0, 0, 0], [0, -100.0, 0], [0, 0, 100.0], [-90, 90, 0]])
    model.global_centers_ = model.global_centers_ + push
    model.centers_ = [c + push for c in model.centers_]
    r = htfa.HTFAResults(model, res_htfa.subjects)
    n_out = _check_display_inside(r)
    print("fitted centers outside the mask:", n_out)
    assert all(n >= 3 for n in n_out.values())


T10 = Path.home() / ".cache/htfa/tutorials"
T10_FIT = T10 / "10_dynamic_isfc_K50_5a3dab79e829.npz"


@pytest.mark.skipif(
    not T10_FIT.exists(), reason=f"tutorial 10's cached fit not found at {T10_FIT}"
)
def test_display_nodes_inside_mask_tutorial_10_fit():
    """Tutorial 10's real HTFA fit (12 development_fmri adults, K=50): every
    drawn node lies within the subjects' voxels."""
    from htfa.io import Subject

    fit = np.load(T10_FIT)
    files = sorted((T10 / "development_fmri").glob("sub-pixar*.npz"))
    if len(files) != len(fit["centers"]):
        pytest.skip(f"{len(files)} cached subjects for {len(fit['centers'])} fitted")
    subjects = []
    for f in files:
        z = np.load(f)
        subjects.append(Subject(data=z["X"], coords=z["coords"]))
    model = htfa.HTFA(K=50)
    model.global_centers_, model.global_widths_ = (
        fit["global_centers"],
        fit["global_widths"],
    )
    model.centers_, model.widths_ = list(fit["centers"]), list(fit["widths"])
    model.weights_ = list(fit["weights"])
    n_out = _check_display_inside(htfa.HTFAResults(model, subjects))
    print("tutorial 10 fitted centers outside the mask:", n_out)
    assert sum(n_out.values()) > 0  # the fit does have centers outside


def test_results_plot_network_draws_display_nodes(res_htfa):
    """The wrapper's default draws edges between the snapped centers."""
    r = res_htfa
    c_eff = r.display_nodes()[0]
    a = _rgb(r.plot_network(connectivity=_signed_edges(4), mesh=None))
    pos = _midpoint_window(a, c_eff, 0, 1)
    neg = _midpoint_window(a, c_eff, 2, 3)
    print("red/blue px at +edge", _red(pos).sum(), _blue(pos).sum())
    assert _red(pos).sum() >= 10 and _blue(pos).sum() == 0
    assert _blue(neg).sum() >= 10 and _red(neg).sum() == 0


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"duration": float("nan")}, "duration must be a positive, finite"),
        ({"duration": float("inf")}, "duration must be a positive, finite"),
        ({"duration": -1.0}, "duration must be a positive, finite"),
        ({"fps": 0}, "fps must be a positive, finite"),
        ({"fps": -5, "n_frames": 3}, "fps must be a positive, finite"),
        ({"fps": float("nan"), "duration": 1}, "fps must be a positive, finite"),
    ],
)
def test_animate_network_rejects_bad_duration_and_fps(tmp_path, kwargs, match):
    c = np.array([[-40.0, -20, 10], [-40, 20, 10]])
    E = np.array([[0, 1.0], [1.0, 0]])
    with pytest.raises(ValueError, match=match):
        bn.animate_network(c, E, tmp_path / "x.gif", mesh=None, snap=False, **kwargs)
    with pytest.raises(ValueError, match="fps must be a positive, finite"):
        bn.write_animation([], tmp_path / "y.gif", fps=0)


def test_spin_camera_fits_a_large_scene(tmp_path):
    # Nodes 900 mm from the center, two of them on the view axis: a camera at a
    # fixed 500 mm with clipping range (1, 1000) loses both; the camera must be
    # placed from the scene's size.
    c = np.array([[900.0, 0, 0], [-900, 0, 0], [0, 900, 0], [0, -900, 0]])
    bn.animate_network(
        c,
        np.zeros((4, 4)),
        tmp_path / "spin.gif",
        mesh=None,
        snap=False,
        n_frames=4,
        spin=(0, 0, 0.5),
        size=(200, 200),
        radius_kb=(0.0, 80.0),
        frames_dir=tmp_path / "frames",
    )
    first = sorted((tmp_path / "frames").iterdir())[0]
    a = np.asarray(Image.open(first).convert("RGB")).astype(float)
    h, w = a.shape[:2]
    middle = a[h // 2 - 3 : h // 2 + 3, w // 2 - 3 : w // 2 + 3]
    print(first.name, "center mean:", middle.mean(), "image min:", a.min())
    # the nodes on the view axis cover the image center
    assert middle.mean() < 200


# ------------------------------------------------- per-node colors and nodes

# Two and three nodes on the y axis, far apart, all drawn with 10 mm radii.
TWO = np.array([[0.0, -50.0, 0.0], [0.0, 50.0, 0.0]])
THREE = np.array([[0.0, -60.0, 0.0], [0.0, 0.0, 0.0], [0.0, 60.0, 0.0]])
_SMALL = {"size": (320, 200), "mesh": None, "snap": False, "radius_kb": (0.0, 10.0)}
# Column between the two nodes of TWO in a 320-pixel-wide left lateral view;
# node 0 (y = -50) is drawn right of it.
_MID = (_px(-50, 0)[0] + _px(50, 0)[0]) / 2 * 320 / 1600


def _hue(a, channel):
    """Pixels whose ``channel`` is above 150 and whose other channels are below 90."""
    others = np.delete(a, channel, axis=-1)
    return (a[..., channel] > 150) & np.all(others < 90, axis=-1)


def _centroid(mask):
    rows, cols = np.nonzero(mask)
    return np.array([cols.mean(), rows.mean()])


def _drawn(a):
    return a.min(axis=-1) < 235


def _scene(centers, node_opacity=None, **style):
    scene_opts, _ = bn._style({"size": (320, 200), "radius_kb": (0.0, 6.0), **style})
    return bn._Scene(
        centers, np.ones(len(centers)), None, **scene_opts, node_opacity=node_opacity
    )


def test_node_colors_shapes():
    """One color, K colors and (T, K) colors as float RGB; alpha dropped."""
    one = bn._node_colors("red", 3)
    print("one", one)
    np.testing.assert_array_equal(one, [1.0, 0.0, 0.0])
    # a flat sequence of 3 or 4 numbers is one color, even for K = 3 or 4
    # (pyvista stores colors as 8-bit, so values come back to within 1/255)
    np.testing.assert_allclose(
        bn._node_colors([0.1, 0.2, 0.3], 3), [0.1, 0.2, 0.3], atol=1 / 255
    )
    np.testing.assert_allclose(bn._node_colors((0.0, 0.0, 1.0, 0.5), 4), [0, 0, 1])
    names = bn._node_colors(["red", "#00ff00", (0.0, 0.0, 1.0)], 3)
    print("names", names)
    np.testing.assert_array_equal(names, np.eye(3))
    rgba = np.c_[np.eye(3), [0.1, 0.5, 1.0]]
    np.testing.assert_array_equal(bn._node_colors(rgba, 3), np.eye(3))
    ints = bn._node_colors(np.array([[255, 0, 0], [0, 0, 255]]), 2)
    np.testing.assert_array_equal(ints, [[1, 0, 0], [0, 0, 1]])
    # the result is accepted again unchanged
    np.testing.assert_array_equal(bn._node_colors(names, 3), names)
    frames = bn._node_colors([["red", "blue"], ["blue", "red"], ["red", "red"]], 2, 3)
    print("frames", frames.shape)
    assert frames.shape == (3, 2, 3)
    np.testing.assert_array_equal(frames[1], [[0, 0, 1], [1, 0, 0]])
    cube = np.random.default_rng(0).uniform(size=(5, 2, 4))
    np.testing.assert_allclose(bn._node_colors(cube, 2, 5), cube[..., :3], atol=1e-2)
    # K colors stay (K, 3) when T is given
    assert bn._node_colors(["red", "blue"], 2, 5).shape == (2, 3)
    for bad, K, T in [
        (["red", "blue"], 3, None),
        (np.zeros((2, 3)), 4, None),
        (np.zeros((5, 2, 3)), 2, None),
        (np.zeros((4, 2, 3)), 2, 5),
        (["red", "blue", (0.0, 1.0)], 3, None),
        ("not a color", 2, None),
    ]:
        with pytest.raises(ValueError, match=f"K={K}") as err:
            bn._node_colors(bad, K, T)
        print("error:", err.value)


def test_plot_network_per_node_colors():
    """K color names color each node; the red blob is where node 0 is drawn."""
    colored = _rgb(bn.plot_network(TWO, node_color=["red", "blue"], **_SMALL))
    gray = _rgb(bn.plot_network(TWO, **_SMALL))
    red, blue = _hue(colored, 0), _hue(colored, 2)
    # the whole red blob, shaded side included: drawn pixels more red than blue
    red_blob = _drawn(colored) & (colored[..., 0] > colored[..., 2])
    first = _drawn(gray) & (np.arange(gray.shape[1]) > _MID)[None]
    print(
        "red px", red.sum(), "blue px", blue.sum(), "red blob centroid",
        _centroid(red_blob), "node 0 centroid (gray render)", _centroid(first),
        "split column", _MID,
    )  # fmt: skip
    assert red.sum() > 100 and blue.sum() > 100
    assert _centroid(blue)[0] < _MID < _centroid(red)[0]
    np.testing.assert_allclose(_centroid(red_blob), _centroid(first), atol=1.0)


def test_plot_network_color_array_shapes():
    """(K, 3) and (K, 4) arrays color each node; one numeric color colors all."""
    eye = _rgb(bn.plot_network(THREE, node_color=np.eye(3), **_SMALL))
    counts = [int(_hue(eye, c).sum()) for c in range(3)]
    print("np.eye(3): red/green/blue px", counts)
    assert min(counts) > 100
    nested = _rgb(bn.plot_network(THREE, node_color=np.eye(3).tolist(), **_SMALL))
    print("nested list px", [int(_hue(nested, c).sum()) for c in range(3)])
    np.testing.assert_allclose(
        [_hue(nested, c).sum() for c in range(3)], counts, rtol=0.05
    )
    one = _rgb(bn.plot_network(THREE, node_color=(1.0, 0.0, 0.0), **_SMALL))
    one_counts = [int(_hue(one, c).sum()) for c in range(3)]
    print("(1, 0, 0): red/green/blue px", one_counts)
    assert one_counts[1] == one_counts[2] == 0
    assert one_counts[0] > 2.7 * counts[0]
    # alpha is ignored: a node with alpha 0.1 is as saturated as without it
    rgba = np.c_[np.eye(3), [0.1, 0.5, 1.0]]
    alpha = _rgb(bn.plot_network(THREE, node_color=rgba, **_SMALL))
    alpha_counts = [int(_hue(alpha, c).sum()) for c in range(3)]
    print("(K, 4): red/green/blue px", alpha_counts)
    np.testing.assert_allclose(alpha_counts, counts, rtol=0.05)


def test_plot_network_per_node_colors_with_fade(res_htfa):
    """Per-node colors with fade_by_mass (one actor per sphere) show both hues."""
    colors = ["red", "blue", "red", "blue"]
    img = _rgb(
        res_htfa.plot_network(
            mesh=None, fade_by_mass=True, node_color=colors, size=(320, 200)
        )
    )
    red, blue = _red(img), _blue(img)
    print("fade_by_mass per-node colors: red px", red.sum(), "blue px", blue.sum())
    assert red.sum() > 20 and blue.sum() > 20


def test_fade_by_mass_lightens_per_node_colors():
    """With per-node colors, fade_by_mass blends each node toward the white
    background by its opacity a = 0.15 + 0.85 * mass fraction, for nodes of
    known, distinct mass fractions. A translucent sphere is drawn with its
    front and back surfaces, each blended at opacity a, so at a node's center
    the faded render's distance from white is 1 - (1 - a)^2 times the
    unfaded one's."""
    mask = _sphere(40.0, 2.0)
    centers = np.array([[0.0, 50.0, 0.0], [0.0, -10.0, 0.0], [0.0, 0.0, 40.0]])
    widths = np.array([100.0, 100.0, 100.0])
    c_eff, _, info = bn.effective_nodes(centers, widths, mask)
    alpha = 0.15 + 0.85 * np.clip(info["mass_fraction"], 0, 1)
    kw = {"widths": widths, "mask": mask, "mesh": None,
          "node_color": ["red", "blue", "green"]}  # fmt: skip
    plain = _rgb(bn.plot_network(centers, **kw))
    faded = _rgb(bn.plot_network(centers, fade_by_mass=True, **kw))
    expect = 1 - (1 - alpha) ** 2
    for k, (y, z) in enumerate(c_eff[:, 1:]):
        d_plain = (255 - _window(plain, y, z)).mean()
        d_faded = (255 - _window(faded, y, z)).mean()
        print(f"node {k}: mass {info['mass_fraction'][k]:.3f}, opacity "
              f"{alpha[k]:.3f}, distance from white plain {d_plain:.1f}, faded "
              f"{d_faded:.1f}, ratio {d_faded / d_plain:.3f}, expected "
              f"{expect[k]:.3f}")  # fmt: skip
        assert d_plain > 100
        assert abs(d_faded / d_plain - expect[k]) < 0.05
    assert alpha[0] < 0.35 and alpha[1] > 0.99 and 0.4 < alpha[2] < 0.8


def test_node_color_bad_shapes():
    """Color counts or shapes that do not match K raise, naming K."""
    four = CENTERS[:4]
    for bad in (
        ["red", "green", "blue"],
        np.zeros((2, 3)),
        np.zeros((2, 4, 3)),  # per-frame colors are for animate_network only
    ):
        with pytest.raises(ValueError, match="K=4") as err:
            bn.plot_network(
                four, node_color=bad, mesh=None, snap=False, size=(320, 200)
            )
        print("error:", err.value)


def test_per_node_colors_rejected_in_lobe_mode():
    """Component lobes take one node_color; K colors and set_nodes raise."""
    centers = np.array([[0.0, -20.0, 0.0], [0.0, 20.0, 0.0]])
    widths = np.array([60.0, 60.0])
    mask = _sphere(40.0, 4.0)
    kw = {"widths": widths, "mask": mask, "mesh": None, "snap": False,
          "node_mode": "components", "size": (320, 200)}  # fmt: skip
    with pytest.raises(ValueError, match="component lobes") as err:
        bn.plot_network(centers, node_color=["red", "blue"], **kw)
    print("error:", err.value)
    one = _rgb(bn.plot_network(centers, node_color="red", **kw))
    print("one color lobes: red px", _red(one).sum())
    assert _red(one).sum() > 50
    geometry = bn._network_geometry(
        centers, None, widths, "global", False, mask, False, False, "components", 0.05
    )
    scene_opts, _ = bn._style({"size": (320, 200)})
    scene = bn._Scene(
        geometry[0], geometry[1], None, **scene_opts, component_surfaces=geometry[4]
    )
    try:
        with pytest.raises(ValueError, match="component lobes"):
            scene.set_nodes(np.full(len(geometry[0]), 5.0))
    finally:
        scene.close()


def test_scene_set_nodes():
    """set_nodes replaces the spheres: new radii, new or kept colors."""
    scene = _scene(TWO, node_color=["red", "blue"])
    try:
        before = _rgb(scene.image())
        np.testing.assert_allclose(scene.radii, 6.0)
        scene.set_nodes(2 * scene.radii, np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]))
        np.testing.assert_allclose(scene.radii, 12.0)
        after = _rgb(scene.image())
        n0, n1 = int(_drawn(before).sum()), int(_drawn(after).sum())
        print("drawn px before/after doubling the radii", n0, n1)
        assert n1 > 2 * n0
        # the colors were swapped: red moved from node 0 to node 1
        print("red centroid before/after", _centroid(_hue(before, 0)),
              _centroid(_hue(after, 0)), "split column", _MID)  # fmt: skip
        assert _centroid(_hue(before, 0))[0] > _MID > _centroid(_hue(after, 0))[0]
        assert _hue(after, 2).sum() > 2 * _hue(before, 2).sum()
        # colors=None keeps the current colors
        scene.set_nodes(scene.radii / 2)
        again = _rgb(scene.image())
        print("drawn px after halving", _drawn(again).sum())
        assert abs(int(_drawn(again).sum()) - n0) <= 0.05 * n0
        assert _centroid(_hue(again, 0))[0] < _MID
        # edges set afterwards render with the new radii
        scene.set_nodes(2 * scene.radii)
        scene.set_edges(np.array([[0, 0.8], [0.8, 0]]), 0.25, "flat", "red", "blue")
        assert _drawn(_rgb(scene.image())).sum() > n1
        for bad in (np.ones(3), np.array([1.0, 0.0]), np.array([1.0, np.nan])):
            with pytest.raises(ValueError, match="radii") as err:
                scene.set_nodes(bad)
            print("error:", err.value)
        with pytest.raises(ValueError, match="K=2"):
            scene.set_nodes(scene.radii, ["red", "green", "blue"])
    finally:
        scene.close()


@pytest.mark.parametrize("opacity", [None, np.array([0.5, 1.0])])
def test_set_nodes_bad_input_leaves_scene_unchanged(opacity):
    """Bad radii or colors raise before set_nodes touches the scene.

    Actors, radii and colors must be exactly unchanged. The image is compared
    with a tolerance: translucent rendering (depth peeling, here for the
    opacity case) can vary by a few pixels between renders of an unchanged
    scene on Mesa and macOS, so at most 20 pixels may differ. A changed node
    changes far more: each covers about 140 pixels (radius 6 mm, 6.6 px at
    this size), and even a 1 mm larger radius adds a rim of about 50 pixels.
    """
    scene = _scene(TWO, node_opacity=opacity, node_color=["red", "blue"])
    try:
        image = np.asarray(scene.image()).astype(int)
        actors, radii = list(scene._nodes), scene.radii.copy()
        colors = scene.node_colors.copy()
        for bad_radii, bad_colors in [
            (np.array([6.0, np.inf]), None),
            (np.array([np.nan, 6.0]), None),
            (np.array([-np.inf, 6.0]), None),
            (np.full(3, 6.0), None),
            (2 * radii, ["red", "green", "blue"]),
            (2 * radii, ["red", "no such color"]),
        ]:
            with pytest.raises(ValueError) as err:
                scene.set_nodes(bad_radii, bad_colors)
            print("error:", err.value)
            assert scene._nodes == actors
            np.testing.assert_array_equal(scene.radii, radii)
            np.testing.assert_array_equal(scene.node_colors, colors)
            again = np.asarray(scene.image()).astype(int)
            changed = np.abs(again - image).max(axis=-1)
            print("pixels changed:", int((changed > 0).sum()),
                  "largest channel change:", int(changed.max()))  # fmt: skip
            assert (changed > 0).sum() <= 20
    finally:
        scene.close()


def test_supersample_style():
    """supersample sets the render scale; the image keeps the requested size."""
    assert bn._style({})[0]["supersample"] == bn._SUPERSAMPLE == 2
    for s in (1, 2):
        img = bn.plot_network(TWO, supersample=s, **_SMALL)
        print("supersample", s, "image size", img.size)
        assert img.size == (320, 200)
        scene = _scene(TWO, supersample=s)
        try:
            print("window size", scene.pl.window_size)
            assert list(scene.pl.window_size) == [320 * s, 200 * s]
        finally:
            scene.close()
    for bad in (0, -1, 1.5, "2", True):
        with pytest.raises(ValueError, match="supersample") as err:
            bn.plot_network(TWO, supersample=bad, **_SMALL)
        print("error:", err.value)


def test_default_scene_actors_unchanged():
    """One color without opacity is one merged node actor, as in 0.1.0;
    per-node colors are one actor too, and opacity gives one per sphere."""
    for style, opacity, n_actors in [
        ({}, None, 1),
        ({"node_color": ["red", "blue"]}, None, 1),
        ({}, np.array([0.5, 1.0]), 2),
        ({"node_color": ["red", "blue"]}, np.array([0.5, 1.0]), 2),
    ]:
        scene = _scene(TWO, node_opacity=opacity, **style)
        try:
            old = list(scene._nodes)
            print(style, "opacity", opacity, "node actors", len(old))
            assert len(old) == n_actors
            scene.set_nodes(2 * scene.radii)
            assert len(scene._nodes) == n_actors
            actors = list(scene.pl.renderer.actors.values())
            assert not any(a in actors for a in old)
            assert all(a in actors for a in scene._nodes)
        finally:
            scene.close()


# ---------------------------------------------- per-frame sizes and colors

_ANIM = {"size": (320, 200), "mesh": None, "snap": False}
# Node 0 grows over three timepoints; node 1 stays the smallest size.
GROW = [[1.0, 1.0], [2.0, 1.0], [4.0, 1.0]]
# 2 mm and 7 mm radii (the radius_range end points) in a 320-pixel-wide frame.
_PX_PER_MM = SCALE * 320 / 1600


def _movie(tmp_path, name, centers, connectivity, n=3, **kw):
    """Render ``name``.mp4 with frames_dir; return its ``n`` frames as arrays."""
    folder = tmp_path / name
    out = bn.animate_network(
        centers, connectivity, tmp_path / f"{name}.mp4", frames_dir=folder,
        **{**_ANIM, **kw},
    )  # fmt: skip
    assert out.stat().st_size > 0
    names = sorted(p.name for p in folder.iterdir())
    print(name, "frames", names)
    assert names == sorted(f"{k + 1}.png" for k in range(n))
    return _pngs(folder, n)


def _areas(frames):
    """Drawn pixels of node 0 (right of _MID) and of node 1 (left), per frame."""
    right = (np.arange(frames[0].shape[1]) > _MID)[None]
    return [
        (int((_drawn(a) & right).sum()), int((_drawn(a) & ~right).sum()))
        for a in frames
    ]


def test_animate_sizes_over_time(tmp_path):
    """(T, K) sizes: node 0 grows, node 1 keeps 2 mm; radii are scaled jointly."""
    frames = _movie(tmp_path, "s", TWO, np.zeros((3, 2, 2)), sizes=GROW)
    areas = _areas(frames)
    a0, a1 = [a for a, _ in areas], [b for _, b in areas]
    print("node 0 areas", a0, "node 1 areas", a1)
    assert a0[0] < a0[1] < a0[2]
    assert max(a1) - min(a1) <= 0.02 * min(a1)
    # joint scaling: node 1 is the smallest (2 mm) node in every frame, and in
    # frame 2 node 0 is (2 + 5/3) mm, not 7 mm as per-frame scaling would give
    # frame 1: both nodes are 2 mm (equal up to rasterization at another place)
    assert abs(a1[0] - a0[0]) <= 0.1 * a0[0]
    assert a1[1] < a0[1] and a1[2] < a0[2]
    ratio = a0[1] / a0[2]
    expect = ((2 + 5 / 3) / 7) ** 2
    print("frame 2 / frame 3 node-0 area", ratio, "expected about", expect)
    assert abs(ratio - expect) < 0.12
    print("node 1 / largest node 0 area", a1[2] / a0[2], "expected about", (2 / 7) ** 2)
    assert a1[2] / a0[2] < 0.2


def test_animate_sizes_interpolated(tmp_path):
    """Sizes follow interpolation between timepoints, like the edges."""
    E = np.zeros((3, 2, 2))
    lin = [
        a for a, _ in _areas(_movie(tmp_path, "lin", TWO, E, 5, n_frames=5, sizes=GROW))
    ]
    print("linear node-0 areas", lin)
    assert lin[0] < lin[1] < lin[2]
    near = [
        a
        for a, _ in _areas(
            _movie(
                tmp_path,
                "near",
                TWO,
                E,
                5,
                n_frames=5,
                sizes=GROW,
                interpolation="nearest",
            )  # fmt: skip
        )
    ]
    print("nearest node-0 areas", near)
    assert len(set(near)) == 3
    assert near[1] == near[2] and near[3] == near[4]


def test_animate_colors_over_time(tmp_path):
    """(T, K) color names change node colors per frame; a (T, K, 3) array agrees."""
    names = [["red", "blue"], ["blue", "red"], ["green", "blue"]]
    frames = _movie(tmp_path, "names", TWO, np.zeros((3, 2, 2)), node_color=names)
    right = (np.arange(320) > _MID)[None]

    def green(a):
        return (a[..., 1] - np.maximum(a[..., 0], a[..., 2])) > 50

    counts = [
        (int((_hue(a, 0) & right).sum()), int((green(a) & right).sum())) for a in frames
    ]
    print("node 0 red/green px per frame", counts)
    assert counts[0][0] > 20 and counts[0][1] == 0
    assert counts[1][0] == 0  # node 0 is blue in frame 2
    assert counts[2][1] > 20 and counts[2][0] == 0
    rgb = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            [[0.0, 128 / 255, 0.0], [0.0, 0.0, 1.0]],
        ]
    )
    _movie(tmp_path, "rgb", TWO, np.zeros((3, 2, 2)), node_color=rgb)
    for k in range(3):
        d = bn.image_diff(
            tmp_path / "names" / f"{k + 1}.png", tmp_path / "rgb" / f"{k + 1}.png"
        )
        print("frame", k + 1, "image_diff names vs (T, K, 3)", d)
        assert d < 1e-6


def test_animate_without_connectivity(tmp_path):
    """connectivity=None draws nodes only; it needs something that varies."""
    frames = _movie(tmp_path, "none", TWO, None, sizes=GROW)
    for k, a in enumerate(frames):
        print("frame", k + 1, "red/blue px", _red(a).sum(), _blue(a).sum(),
              "drawn px", _drawn(a).sum())  # fmt: skip
        assert _red(a).sum() == 0 and _blue(a).sum() == 0 and _drawn(a).sum() > 0
    # the same sizes with edges do draw edge pixels
    E = np.zeros((3, 2, 2))
    E[:, 0, 1] = E[:, 1, 0] = 0.8
    with_edges = _movie(tmp_path, "edges", TWO, E, sizes=GROW)
    assert all(_red(a).sum() > 10 for a in with_edges)
    for kw in ({}, {"sizes": [1.0, 2.0]}, {"node_color": ["red", "blue"]},
               {"n_frames": 4}):  # fmt: skip
        with pytest.raises(ValueError, match="time axis") as err:
            bn.animate_network(TWO, None, tmp_path / "x.mp4", **{**_ANIM, **kw})
        print(kw, "error:", err.value)


def test_animate_colors_without_connectivity(tmp_path):
    """connectivity=None with only (T, K) colors: T frames, node colors change,
    no edges (the frames equal those of an all-zero connectivity series)."""
    names = [["red", "blue"], ["blue", "red"], ["green", "blue"]]
    frames = _movie(tmp_path, "colors", TWO, None, node_color=names)
    zero = _movie(tmp_path, "zero", TWO, np.zeros((3, 2, 2)), node_color=names)
    right = (np.arange(320) > _MID)[None]
    between = np.abs(np.arange(320) - _MID)[None] < 3

    def green(a):
        return (a[..., 1] - np.maximum(a[..., 0], a[..., 2])) > 50

    for k, (a, b) in enumerate(zip(frames, zero, strict=True)):
        node0 = [int((f(a) & right).sum()) for f in (lambda x: _hue(x, 0),
                 lambda x: _hue(x, 2), green)]  # fmt: skip
        print(
            f"frame {k + 1}: node 0 red/blue/green px {node0}, drawn px between "
            f"the nodes {int((_drawn(a) & between).sum())}, px differing from "
            f"the zero-connectivity frame {int((a != b).any(-1).sum())}"
        )
        assert node0[k] > 20 and sum(node0) == node0[k]  # red, then blue, then green
        assert not (_drawn(a) & between).any()  # no edge crosses between the nodes
        np.testing.assert_array_equal(a, b)


def test_animate_time_axes_must_agree(tmp_path):
    """Every time axis (connectivity, sizes, colors, captions) must have one T."""
    E = np.zeros((3, 3, 3))
    out = tmp_path / "x.mp4"
    cases = [
        ({"sizes": np.ones((4, 3))}, r"connectivity has T=3.*sizes has T=4"),
        (
            {"node_color": [["red"] * 3] * 2},
            r"connectivity has T=3.*node_color has T=2",
        ),
        ({"captions": list("abcd")}, "4 captions for 3 frames"),
    ]
    for kw, match in cases:
        with pytest.raises(ValueError, match=match) as err:
            bn.animate_network(THREE, E, out, **{**_ANIM, **kw})
        print(sorted(kw), "error:", err.value)
    with pytest.raises(ValueError, match=r"sizes has T=4.*node_color has T=2") as err:
        bn.animate_network(
            THREE, None, out, sizes=np.ones((4, 3)), node_color=[["red"] * 3] * 2,
            **_ANIM,
        )  # fmt: skip
    print("error:", err.value)
    assert not out.exists()


def test_animate_edges_start_at_frame_radii(tmp_path):
    """A tube wider than the nodes shows its ends: it starts at each frame's radii.

    Edges built from another frame's radii would not shrink frame by frame.
    """
    E = np.zeros((3, 2, 2))
    E[:, 0, 1] = E[:, 1, 0] = 20.0  # intensity style: a 10 mm tube radius
    sizes = [[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]]  # radii 2, 2 + 5/3 and 7 mm
    frames = _movie(tmp_path, "tube", TWO, E, sizes=sizes)
    spans = []
    for a in frames:
        cols = np.nonzero(_red(a).any(axis=0))[0]
        spans.append(int(cols.max() - cols.min() + 1))
    expect = [(100 - 2 * r) * _PX_PER_MM for r in (2, 2 + 5 / 3, 7)]
    print("red span px per frame", spans, "expected about", np.round(expect, 1))
    assert spans[0] > spans[1] > spans[2]
    np.testing.assert_allclose(spans, expect, atol=2.5)


def test_animate_one_k_sizes_still_work(tmp_path):
    """(1, K) and (K, 1) sizes draw the same frames as the flat (K,) array."""
    E = np.zeros((3, 2, 2))
    flat = _movie(tmp_path, "flat", TWO, E, sizes=[1.0, 3.0])
    for name, sizes in (("row", [[1.0, 3.0]]), ("column", [[1.0], [3.0]])):
        frames = _movie(tmp_path, name, TWO, E, sizes=sizes)
        for k, (a, b) in enumerate(zip(frames, flat, strict=True)):
            print(name, "frame", k + 1, "differing values", np.count_nonzero(a != b))
            np.testing.assert_array_equal(a, b)


def test_animate_gif_colored_nodes(tmp_path):
    """A GIF keeps a red node reddish and a blue node bluish, frame by frame.

    The colors swap every frame: Pillow merges identical consecutive GIF frames.
    """
    colors = [["red", "blue"], ["blue", "red"], ["red", "blue"]]
    out = bn.animate_network(
        TWO, np.zeros((3, 2, 2)), tmp_path / "c.gif", node_color=colors,
        **{**_ANIM, "radius_kb": (0.0, 10.0)},
    )  # fmt: skip
    with Image.open(out) as g:
        assert g.n_frames == 3
        decoded = []
        for k in range(3):
            g.seek(k)
            decoded.append(_rgb(g))
    for k, a in enumerate(decoded):
        reddish = (a[..., 0] > 120) & (a[..., 1] < 110) & (a[..., 2] < 110)
        bluish = (a[..., 2] > 120) & (a[..., 0] < 110) & (a[..., 1] < 110)
        print(
            "gif frame", k + 1, "reddish px", reddish.sum(), "bluish px", bluish.sum()
        )
        assert reddish.sum() > 50 and bluish.sum() > 50
        # node 0 (drawn right of _MID) has the frame's first color
        left, right = (bluish, reddish) if colors[k][0] == "red" else (reddish, bluish)
        assert _centroid(left)[0] < _MID < _centroid(right)[0]


def test_animate_scale_by_mass_with_time_sizes(res_htfa, tmp_path):
    """(T, K) sizes with scale_by_mass on a fit: every row times the mass fraction.

    The frames equal those of the sizes multiplied by the fit's in-mask mass
    fractions (`HTFAResults.display_nodes`) by hand, and differ from the
    frames of the unscaled sizes.
    """
    sizes = np.array([[1.0, 1.0, 1.0, 1.0], [3.0, 1.0, 1.0, 3.0], [6.0, 1.0, 1.0, 6.0]])
    mass = res_htfa.display_nodes()[2]["mass_fraction"]
    print("mass fractions", mass)
    assert np.ptp(mass) > 0.05  # scaling changes the relative sizes
    kw = {"snap": True}
    scaled = _movie(
        tmp_path, "mass", res_htfa, None, sizes=sizes, scale_by_mass=True, **kw
    )
    by_hand = _movie(tmp_path, "hand", res_htfa, None, sizes=sizes * mass, **kw)
    plain = _movie(tmp_path, "plain", res_htfa, None, sizes=sizes, **kw)
    for k in range(3):
        same = np.count_nonzero(scaled[k] != by_hand[k])
        other = np.count_nonzero(scaled[k] != plain[k])
        print("frame", k + 1, "values differing from by-hand scaling", same,
              "from unscaled sizes", other)  # fmt: skip
        assert same == 0
    assert any(np.count_nonzero(s != p) > 0 for s, p in zip(scaled, plain, strict=True))
    d = [
        bn.image_diff(tmp_path / "mass" / "1.png", tmp_path / "mass" / f"{k}.png")
        for k in (2, 3)
    ]
    print("image_diff frame 1 vs 2, vs 3", d)
    assert 0 < d[0] < d[1]


def test_animate_time_sizes_rejected_in_lobe_mode(tmp_path):
    """Component lobes take neither (T, K) sizes nor (T, K) colors."""
    centers = np.array([[0.0, -20.0, 0.0], [0.0, 20.0, 0.0]])
    kw = {"widths": np.array([60.0, 60.0]), "mask": _sphere(40.0, 4.0),
          "node_mode": "components", **_ANIM}  # fmt: skip
    E = np.zeros((3, 2, 2))
    with pytest.raises(ValueError, match="node_mode='summary'") as err:
        bn.animate_network(centers, E, tmp_path / "x.mp4", sizes=GROW, **kw)
    print("error:", err.value)
    with pytest.raises(ValueError, match="component lobes") as err:
        bn.animate_network(
            centers, E, tmp_path / "x.mp4", node_color=[["red", "blue"]] * 3, **kw
        )
    print("error:", err.value)


def test_radii_joint_and_positive():
    """_radii maps (K,) like the scene did and (T, K) in one call; radii > 0."""
    opts, _ = bn._style({})
    np.testing.assert_array_equal(
        bn._radii(np.array([1.0, 2.0, 3.0]), opts), bn.node_radius([1.0, 2.0, 3.0])
    )
    joint = bn._radii(np.array(GROW), opts)
    print("joint radii", joint)
    np.testing.assert_allclose(joint, [[2, 2], [2 + 5 / 3, 2], [7, 2]])
    fixed = bn._radii(np.array(GROW), {**opts, "radius_kb": (2.0, 1.0)})
    np.testing.assert_array_equal(fixed, 2 * np.array(GROW) + 1)
    clipped = bn._radii(np.array(GROW), {**opts, "radius_clip": (3.0, 5.0)})
    assert clipped.min() == 3.0 and clipped.max() == 5.0
    with pytest.raises(ValueError, match="positive") as err:
        bn._radii(np.array(GROW), {**opts, "radius_kb": (1.0, -2.0)})
    print("error:", err.value)


def test_plot_network_rejects_time_sizes():
    """(T, K) sizes are for animate_network; plot_network says so."""
    with pytest.raises(ValueError, match=r"\(3, 2\).*animate_network") as err:
        bn.plot_network(TWO, sizes=GROW, **_ANIM)
    print("error:", err.value)
    with pytest.raises(ValueError, match="sizes has 3 entries for 2 nodes") as err:
        bn.plot_network(TWO, sizes=[1.0, 2.0, 3.0], **_ANIM)
    print("error:", err.value)


def test_sizes_length_error_names_the_shapes_of_the_function(tmp_path):
    """Too many sizes: plot_network names (K,) only, animate_network (T, K) too.

    Checked with and without ``snap`` (two code paths validate the sizes).
    """
    mask = np.array([[0.0, -50.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 0.0]])
    for snap in (False, True):
        kw = {**_ANIM, "snap": snap, "mask": mask}
        with pytest.raises(ValueError, match="sizes has 3 entries for 2 nodes") as err:
            bn.plot_network(TWO, sizes=[1.0, 2.0, 3.0], **kw)
        print(f"plot_network, snap={snap}:", err.value)
        assert "(T, K)" not in str(err.value)
        assert "shape (3,)" in str(err.value)
        with pytest.raises(ValueError, match="sizes has 3 entries for 2 nodes") as err:
            bn.animate_network(
                TWO, None, tmp_path / "a.gif", sizes=[1.0, 2.0, 3.0], **kw
            )
        print(f"animate_network, snap={snap}:", err.value)
        assert "((K,) or (T, K))" in str(err.value)


def test_animate_fixed_edges_with_time_sizes(tmp_path):
    """A (K, K) matrix with (T, K) sizes draws the same edge in all T frames."""
    E = np.array([[0.0, 0.8], [0.8, 0.0]])
    frames = _movie(tmp_path, "fixed", TWO, E, sizes=GROW)
    red = [int(_red(a).sum()) for a in frames]
    print("red px per frame", red, "node-0 areas", [a for a, _ in _areas(frames)])
    assert min(red) > 10


# ------------------------------------------------- views, legend, frame_size


def _head(W, n_sub=0, caption=False):
    """Height of the title (and caption) band of a W-pixel-wide layout frame."""
    t = max(16, W // 32)
    s = int(t * 0.8)
    return int(t * 1.5 + n_sub * s * 1.3 + (s * 1.6 if caption else 0) + t * 0.4)


def _band(W):
    """Legend row height: 64 px per 30 px of font, font max(12, 30 W / 1920)."""
    return round(64 * max(12, round(30 * W / 1920)) / 30)


def _dark(a):
    return a.max(axis=-1) < 100


def _pure_red(a):
    """Red pixels that are not orange (green low)."""
    return _red(a) & (a[..., 1] < 80)


def _orange(a):
    """#E69F00-like pixels (230, 159, 0)."""
    return (a[..., 0] > 200) & (abs(a[..., 1] - 159) < 35) & (a[..., 2] < 60)


PN_LEGEND = [("red", "positive", "line"), ("blue", "negative", "line")]


def test_view_titles():
    """Names and titles from names or (name, title) pairs; defaults; errors."""
    views = ("left", "dorsal", "right", "lateral", "anterior", ("ventral", "Bottom"))
    names, titles = bn._view_titles(views)
    print(names, titles)
    assert names == ["left", "dorsal", "right", "lateral", "anterior", "ventral"]
    assert titles == [
        "Left lateral", "Dorsal", "Right lateral", "Left lateral", "Anterior", "Bottom"
    ]  # fmt: skip
    assert bn._view_titles(("left",)) == (["left"], [""])  # one view: no title
    assert bn._view_titles([("left", "L")]) == (["left"], ["L"])
    for bad in ((), ("top",), (("left",),), (("left", 3),), "left"):
        with pytest.raises(ValueError) as err:
            bn._view_titles(bad)
        print("error:", err.value)


def test_views_object_draws_each_view():
    """_Views draws one image per view, equal to plot_network's single view."""
    scene_opts, edge_opts = bn._style({"size": (320, 200)})
    centers = CENTERS
    views = bn._Views(centers, np.ones(6), None, ["left", "dorsal"], scene_opts)
    try:
        images = views.draw(_conn(), edge_opts)
    finally:
        views.close()
    assert [im.size for im in images] == [(320, 200), (320, 200)]
    for im, view in zip(images, ("left", "dorsal"), strict=True):
        ref = bn.plot_network(CENTERS, _conn(), sizes=np.ones(6), view=view, **_ANIM)
        print(view, "differing px", int((_rgb(im) != _rgb(ref)).any(-1).sum()))
        np.testing.assert_array_equal(_rgb(im), _rgb(ref))


def test_views_failure_closes_built_scenes():
    """A failing view closes the scenes built before it."""
    from pyvista.plotting import _ALL_PLOTTERS

    before = set(_ALL_PLOTTERS)
    scene_opts, _ = bn._style({"size": (320, 200)})
    with pytest.raises(ValueError, match="unknown view"):
        bn._Views(CENTERS, np.ones(6), None, ["left", "top"], scene_opts)
    print("open plotters before/after:", len(before), len(_ALL_PLOTTERS))
    assert set(_ALL_PLOTTERS) == before


_RENDER_LOOP = """
import resource, sys
import numpy as np
from htfa import brainnet as bn

centers, E = np.array({centers}), np.ones((6, 6))
unit = 1 if sys.platform == "darwin" else 1024  # ru_maxrss: bytes, else KiB
for i in range(1, 101):
    bn.plot_network(centers, E, size=(320, 200), snap=False)
    if i % 10 == 0:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit / 2**20
        print(i, peak, flush=True)
"""


_NESTING = """
import sys, tempfile, threading
from pathlib import Path
import numpy as np
from htfa import brainnet as bn

opts, edge_opts = bn._style({{"size": (320, 200)}})
centers = np.array({centers})
for n in (0, 1):
    views = ["left"] * n
    # closed in creation order (oldest first), which crashed with a pool per _Views
    a = bn._Views(centers, np.ones(6), None, views, opts)
    b = bn._Views(centers, np.ones(6), None, views, opts)
    a.close()
    b.close()
    print("creation-order close, views:", n, flush=True)
    # built on one thread, closed on another
    box = []
    t = threading.Thread(
        target=lambda: box.append(bn._Views(centers, np.ones(6), None, views, opts))
    )
    t.start()
    t.join()
    box[0].close()
    print("cross-thread close, views:", n, flush=True)
# a public call on another thread: its pool is made and drained there
shots = []
t = threading.Thread(
    target=lambda: shots.append(bn.plot_network(centers, size=(320, 200), snap=False))
)
t.start()
t.join()
print("plot_network on a thread:", shots[0].size, flush=True)


def compose(images, t):
    # a public call nested inside a movie's pool
    inner = bn.plot_network(centers, size=(160, 100), snap=False, mesh=None)
    images[0].paste(inner, (0, 0))
    return images[0]


with tempfile.TemporaryDirectory() as tmp:
    out = bn.animate_network(
        centers, np.ones((2, 6, 6)), Path(tmp) / "n.mp4", compose=compose,
        size=(320, 200), snap=False,
    )
    print("nested plot_network in compose:", out.stat().st_size > 0, flush=True)
print("finished")
"""


def test_autorelease_pools_nest_and_cross_threads():
    """No object owns an autorelease pool, so _Views close in any order and on
    any thread, and public calls nest (a compose calling plot_network).

    With a pool per `_Views`, closing two in creation order aborted the
    process (SIGABRT, "Invalid autorelease pools are a fatal error") and
    closing one on another thread crashed it (SIGSEGV), so this runs in its own
    process.
    """
    import subprocess

    run = subprocess.run(
        [sys.executable, "-c", _NESTING.format(centers=CENTERS.tolist())],
        capture_output=True,
        text=True,
        timeout=300,
    )
    print("exit code", run.returncode, "stdout:", run.stdout, "stderr:", run.stderr,
          sep="\n")  # fmt: skip
    assert run.returncode == 0
    assert run.stdout.splitlines()[-1] == "finished"
    assert "nested plot_network in compose: True" in run.stdout


def test_autorelease_without_pool_class():
    """With no pool class (other systems, or VTK < 9.5) scenes render as with one.

    `_autorelease(None)` is the path a VTK without ``vtkCocoaAutoreleasePool``
    takes. `_autorelease_pool_class` must return the class exactly when this
    VTK provides it on macOS (found here by importing it), else None; which
    branch runs depends on the installed VTK.
    """
    import vtkmodules

    try:
        from vtkmodules.vtkRenderingUI import vtkCocoaAutoreleasePool as provided
    except ImportError:  # VTK < 9.5, or a build without the class
        provided = None
    cls = bn._autorelease_pool_class()
    print("VTK", vtkmodules.__version__, sys.platform, "VTK provides:", provided,
          "pool class:", cls)  # fmt: skip
    if sys.platform == "darwin" and provided is not None:
        assert cls is provided
    else:
        assert cls is None  # rendering then takes the no-pool path below
    scene_opts, edge_opts = bn._style({"size": (320, 200)})
    with bn._autorelease(None):
        views = bn._Views(CENTERS, np.ones(6), None, ["left"], scene_opts)
        try:
            (image,) = views.draw(None, edge_opts)
        finally:
            views.close()
    ref = bn.plot_network(CENTERS, sizes=np.ones(6), view="left", **_ANIM)
    print("differing px", int((_rgb(image) != _rgb(ref)).any(-1).sum()))
    np.testing.assert_array_equal(_rgb(image), _rgb(ref))


@pytest.mark.slow
@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="tests the release of macOS (Cocoa) render windows; other systems "
    "have no autorelease pool",
)
def test_repeated_renders_release_memory(ch2):
    """100 renders in one process: peak memory stops growing, no context leak.

    Each render opens and closes VTK render windows. Before they were released
    (on macOS: the autorelease pool of `_autorelease`), the peak resident memory
    grew by 5 to 9 MB per render with the ch2 mesh at this size, without
    leveling off, and macOS printed "Context leak detected" after about 40 renders.
    With them released it still rises over the first 60 or so renders (the
    allocator keeps freed memory for reuse), then stays flat, so the growth
    is measured over renders 61-100. The loop runs in its own process so
    that nothing else is measured.
    """
    import subprocess

    run = subprocess.run(
        [sys.executable, "-c", _RENDER_LOOP.format(centers=CENTERS.tolist())],
        capture_output=True,
        text=True,
        timeout=600,
    )
    print("stdout (render, peak MB):", run.stdout, "stderr:", run.stderr, sep="\n")
    assert run.returncode == 0
    peaks = dict(np.array(run.stdout.split(), float).reshape(-1, 2))
    for i in range(20, 101, 10):
        growth = (peaks[i] - peaks[i - 10]) / 10
        print(f"renders {i - 9}-{i}: peak growth {growth:.2f} MB per render")
    per_render = (peaks[100] - peaks[60]) / 40
    print(f"peak growth over renders 61-100: {per_render:.2f} MB per render")
    assert per_render < 3.0
    assert "Context leak detected" not in run.stderr


def test_plot_network_views():
    """Two views side by side under their titles, split by a divider."""
    two = bn.plot_network(CENTERS, _conn(), views=("left", "dorsal"), **_ANIM)
    one = bn.plot_network(CENTERS, _conn(), views=("left",), **_ANIM)
    print("two views", two.size, "one view", one.size)
    assert two.size[0] > one.size[0]
    assert two.size[0] / two.size[1] > one.size[0] / one.size[1]
    assert two.size[0] % 2 == 0 and two.size[1] % 2 == 0
    a = _rgb(two)
    W = two.size[0]
    head, mid = _head(W), W // 2
    halves = [a[:, : mid - 2], a[:, mid + 2 :]]
    for k, half in enumerate(halves):
        content, title = int(_drawn(half[head:]).sum()), int(_dark(half[:head]).sum())
        print(f"half {k}: brain px {content}, dark title px {title}")
        assert content > 100 and title > 20
    # the divider: light gray between the panes
    print("divider columns", a[head + 5, mid - 1 : mid + 3].tolist())
    assert np.all(abs(a[head + 2 : -2, mid : mid + 2] - 200) <= 1)
    with pytest.raises(ValueError, match="view or views") as err:
        bn.plot_network(CENTERS, view="left", views=("left",), **_ANIM)
    print("error:", err.value)
    with pytest.raises(ValueError, match="unknown view"):
        bn.plot_network(CENTERS, views=("left", "sideways"), **_ANIM)


def test_views_titles():
    """An empty title leaves its band blank; default titles are drawn."""
    img = bn.plot_network(CENTERS, views=("left", ("dorsal", "")), **_ANIM)
    a = _rgb(img)
    W = img.size[0]
    head, mid = _head(W), W // 2
    left, right = a[:head, :mid], a[:head, mid:]
    print("dark px in left/right title bands", _dark(left).sum(), _dark(right).sum())
    assert _dark(left).sum() > 20
    assert np.all(right == 255)


def test_plot_network_frame_size():
    """frame_size gives exactly that size with content, odd sizes included
    (a still has no H.264 constraint); malformed sizes raise."""
    for size in ((960, 540), (961, 540), (960, 541), (641, 401)):
        for views in (None, ("left", "dorsal")):
            img = bn.plot_network(
                CENTERS, _conn(), views=views, frame_size=size, **_ANIM
            )
            a = _rgb(img)
            print(size, views, img.size, "drawn px", int(_drawn(a).sum()),
                  "red", _red(a).sum())  # fmt: skip
            assert img.size == size
            assert _drawn(a).sum() > 500 and _red(a).sum() > 10
    for bad in ((0, 540), (960, -2), (960,), "big", (960.0, 540), (True, 540)):
        with pytest.raises(ValueError, match="frame_size") as err:
            bn.plot_network(CENTERS, frame_size=bad, **_ANIM)
        print("error:", err.value)
        assert repr(bad) in str(err.value)


def test_animate_odd_frame_size(tmp_path):
    """An odd frame_size raises for an mp4 (saying why) and is kept in a GIF."""
    E = np.stack([_conn(), -_conn()])
    with pytest.raises(ValueError, match="mp4 needs an even frame_size") as err:
        bn.animate_network(
            CENTERS, E, tmp_path / "o.mp4", frame_size=(641, 401), **_ANIM
        )
    print("error:", err.value)
    assert "(641, 401)" in str(err.value) and "H.264" in str(err.value)
    assert not (tmp_path / "o.mp4").exists()
    out = bn.animate_network(
        CENTERS, E, tmp_path / "o.gif", frame_size=(641, 401),
        views=("left", "dorsal"), frames_dir=tmp_path / "odd", **_ANIM,
    )  # fmt: skip
    frames = _pngs(tmp_path / "odd", 2)
    with Image.open(out) as gif:
        print("png frames", [a.shape for a in frames], "gif", gif.size, gif.n_frames)
        assert gif.size == (641, 401) and gif.n_frames == 2
    assert all(a.shape == (401, 641, 3) for a in frames)


def test_plot_network_legend():
    """Legend rows are bands at the bottom with their marks and dark text."""
    fs = (960, 540)
    band = _band(960)
    img = bn.plot_network(CENTERS, legend=PN_LEGEND, frame_size=fs, **_ANIM)
    a = _rgb(img)
    low, above = a[-band:], a[:-band]
    counts = [int(f(x).sum()) for x in (low, above) for f in (_red, _blue, _dark)]
    print("legend band red/blue/dark px", counts[:3], "above", counts[3:])
    assert _red(low).sum() > 20 and _blue(low).sum() > 20 and _dark(low).sum() > 20
    assert _red(above).sum() == 0 and _blue(above).sum() == 0
    rows = [PN_LEGEND, [("#E69F00", "community 1 (4 nodes)", "dot")]]
    two = _rgb(bn.plot_network(CENTERS, legend=rows, frame_size=fs, **_ANIM))
    first, second = two[-2 * band : -band], two[-band:]
    counts = [int(f(x).sum()) for x in (first, second) for f in (_pure_red, _orange)]
    print("red/orange px in row 1", counts[:2], "in row 2", counts[2:])
    assert _pure_red(first).sum() > 20 and _orange(first).sum() == 0
    assert _orange(second).sum() > 20 and _pure_red(second).sum() == 0
    assert _orange(two[:-band]).sum() == 0
    # the views sit above both rows
    print("drawn px above the legend", _drawn(two[: -2 * band]).sum())
    assert _drawn(two[: -2 * band]).sum() > 200  # six gray nodes


def test_legend_marks_dots_and_lines():
    """A dot is a filled circle of radius 12 and a line a 34 x 6 bar, per 30 px
    of font: measured at fonts 30 (W = 1920) and 15 (W = 960)."""
    row = [("red", "dot", "dot"), ("blue", "line", "line")]
    for W, H in ((1920, 1080), (960, 540)):
        size = max(12, round(30 * W / 1920))
        _, gap, mark, radius, lw, pad = bn._legend_geometry(size)
        img = bn.plot_network(CENTERS, legend=row, frame_size=(W, H), **_ANIM)
        low = _rgb(img)[-_band(W) :]
        marks = {}
        for name, mask in (("dot", _pure_red(low)), ("line", _blue(low))):
            rows, cols = np.nonzero(mask)
            w, h = cols.max() - cols.min() + 1, rows.max() - rows.min() + 1
            marks[name] = (int(cols.min()), int(w), int(h), mask.sum() / (w * h))
        print(f"W={W} font {size}: mark {mark}, radius {radius}, line width {lw}; "
              "measured (left, width, height, fill):", marks)  # fmt: skip
        dot, line = marks["dot"], marks["line"]
        # the dot: a disc 2 * radius + 1 px across, filling about pi / 4 of its box
        assert abs(dot[1] - (2 * radius + 1)) <= 1
        assert abs(dot[2] - (2 * radius + 1)) <= 1
        assert 0.7 < dot[3] < 0.85
        # the line: mark wide (+1, ends inclusive), lw high, a full bar
        assert abs(line[1] - (mark + 1)) <= 1 and abs(line[2] - lw) <= 1
        assert line[3] > 0.9
        # the dot is centered in its mark; the line starts one entry later
        label = bn._font(size).getlength("dot")
        x_dot = dot[0] + radius - mark / 2  # where the dot entry starts
        print("dot entry starts at", x_dot, "line at", line[0])
        assert abs(line[0] - (x_dot + mark + pad + label + gap)) <= 2


def test_legend_validation():
    """_legend_rows normalizes rows; bad entries and too-wide rows raise."""
    one = bn._legend_rows([("red", "a", "dot"), ((0, 0, 1.0), "b", "line")])
    assert one == [[((255, 0, 0), "a", "dot"), ((0, 0, 255), "b", "line")]]
    two = bn._legend_rows([[("red", "a", "dot")], [("#E69F00", "c", "line")]])
    assert two == [[((255, 0, 0), "a", "dot")], [((230, 159, 0), "c", "line")]]
    for bad in (
        [("red", "a", "square")], [("red", "a")], [], [[]], [("notacolor", "a", "dot")]
    ):  # fmt: skip
        with pytest.raises(ValueError) as err:
            bn._legend_rows(bad)
        print("error:", err.value)
    wide = [PN_LEGEND, [("red", "x" * 200, "line")]]
    with pytest.raises(ValueError, match="legend row 2") as err:
        bn.plot_network(CENTERS, legend=wide, frame_size=(320, 200), **_ANIM)
    print("error:", err.value)


def test_single_view_layout_path():
    """legend with the default view: a cropped view and no title band."""
    img = bn.plot_network(CENTERS, _conn(), legend=PN_LEGEND, **_ANIM)
    a = _rgb(img)
    band = _band(img.size[0])
    pane = a[:-band]
    rows = np.flatnonzero(_drawn(pane).any(axis=1))
    cols = np.flatnonzero(_drawn(pane).any(axis=0))
    print("frame", img.size, "content rows", rows[[0, -1]], "cols", cols[[0, -1]])
    assert img.size[0] < 320 and img.size[1] < 200 + band
    assert rows[0] <= 10  # no title band above the view
    assert rows[-1] >= pane.shape[0] - 12
    # the view is centered (the frame may be wider, for the legend)
    left, right = cols[0], pane.shape[1] - 1 - cols[-1]
    assert left >= 10 and right >= 10 and abs(left - right) <= 2


def test_natural_frame_widens_for_legend():
    """A narrow view with a wide legend: the frame is as wide as the legend."""
    legend = [("red", "positive correlation", "line"), ("blue", "negative", "line")]
    narrow = bn.plot_network(CENTERS, _conn(), legend=[("red", "a", "dot")], **_ANIM)
    img = bn.plot_network(CENTERS, _conn(), legend=legend, **_ANIM)
    views_w = narrow.size[0]  # the cropped view's width (a one-letter legend fits)
    rows = bn._legend_rows(legend)
    size = max(12, round(30 * views_w / 1920))
    gap = bn._legend_geometry(size)[1]
    need = int(np.ceil(bn._legend_row_widths(rows, size)[0])) + 2 * gap
    print("view width", views_w, "legend needs", need, "frame", img.size)
    assert need > views_w
    assert img.size[0] == need - need % 2
    assert img.size[1] == narrow.size[1]
    a = _rgb(img)
    view = a[: -_band(views_w)]
    cols = np.flatnonzero(_drawn(view).any(axis=0))
    left, right = cols[0], view.shape[1] - 1 - cols[-1]
    print("view content cols", cols[[0, -1]], "margins", left, right)
    assert abs(left - right) <= 2 and left > 10
    # the view's pixels are those of the narrow frame, shifted to the center
    x0 = (need - views_w) // 2
    np.testing.assert_array_equal(
        view[:, x0 : x0 + views_w - 2], _rgb(narrow)[: view.shape[0], : views_w - 2]
    )
    lows = a[-_band(views_w) :]
    assert _red(lows).sum() > 10 and _blue(lows).sum() > 10


def test_views_dark_background():
    """A black background: cropped to the brain, black bands, white text."""
    img = bn.plot_network(
        CENTERS, views=("left", "dorsal"), legend=PN_LEGEND, background="black",
        **_ANIM,
    )  # fmt: skip
    a = _rgb(img)
    W, H = img.size
    head, band = _head(W), _band(W)
    print("frame", img.size, "head", head, "band", band)
    assert W < 2 * 320 and H < head + 200 + band  # cropped
    for name, region in (("titles", a[:head]), ("legend", a[-band:])):
        flat = region.reshape(-1, 3)
        black = (flat == 0).all(axis=1).mean()
        # light text pixels (the legend font is small at this width)
        white = int((flat.min(axis=1) > 100).sum())
        print(name, "black fraction", round(black, 3), "light px", white)
        assert black > 0.8 and white > 10


def test_animate_views_caption_legend(tmp_path):
    """Two views, captions, legend, frame_size: fixed crop, changing caption."""
    frames = _movie(
        tmp_path, "views", TWO, None, sizes=GROW, node_color=["red", "blue"],
        views=("left", "dorsal"), captions=["a", "b", "c"], legend=PN_LEGEND,
        frame_size=(640, 360),
    )  # fmt: skip
    assert all(a.shape == (360, 640, 3) for a in frames)
    # Each brain panel alone: below the header, above the legend, either side
    # of the divider (at the middle, as the two panes are centered).
    rows = slice(_head(640, caption=True), 360 - _band(640))
    panels = {"left": slice(0, 316), "dorsal": slice(324, 640)}
    for name, cols in panels.items():
        # node 1 (blue) keeps its size, so with a fixed crop it does not move
        blue = [_centroid(_blue(a[rows, cols])) for a in frames]
        red = [int(_red(a[rows, cols]).sum()) for a in frames]
        print(name, "blue centroids", [c.round(2).tolist() for c in blue], "red", red)
        assert red[0] < red[1] < red[2]
        for c in blue[1:]:
            assert np.abs(c - blue[0]).max() < 0.5
    t = max(16, 640 // 32)
    titles, cap = slice(0, int(t * 1.5)), slice(int(t * 1.5), _head(640, caption=True))
    for k in (1, 2):
        assert np.array_equal(frames[k][titles], frames[0][titles])
        diff = int((frames[k][cap] != frames[0][cap]).any(-1).sum())
        print(f"caption row differs from frame 1 in {diff} px (frame {k + 1})")
        assert diff > 5


def test_animate_views_spin(tmp_path):
    """views with spin: every view turns; the frames differ.

    Each panel changes on its own and equals the single-view spin movie of
    its view (the 3 columns next to the divider are left out).
    """
    frames = _movie(
        tmp_path, "vspin", CENTERS, _conn(), views=("left", "dorsal"),
        spin=(0, 0, 1), n_frames=3,
    )  # fmt: skip
    sizes = {a.shape for a in frames}
    diffs = [int((frames[k] != frames[0]).any(-1).sum()) for k in (1, 2)]
    print("frame shapes", sizes, "px differing from frame 1", diffs)
    assert len(sizes) == 1 and min(diffs) > 100
    # not cropped: two full 320 x 200 views side by side under the titles
    head = _head(640)
    assert frames[0].shape == (200 + head, 640, 3)
    panels = {"left": slice(0, 317), "dorsal": slice(323, 640)}
    singles = {"left": slice(0, 317), "dorsal": slice(3, 320)}
    for view, cols in panels.items():
        single = _movie(
            tmp_path, f"spin_{view}", CENTERS, _conn(), view=view, spin=(0, 0, 1),
            n_frames=3,
        )  # fmt: skip
        panel = [a[head:, cols] for a in frames]
        moved = [int((panel[k] != panel[0]).any(-1).sum()) for k in (1, 2)]
        same = [int((p != s[:, singles[view]]).any(-1).sum())
                for p, s in zip(panel, single, strict=True)]  # fmt: skip
        print(view, "panel px differing from frame 1", moved,
              "px differing from the single-view spin", same)  # fmt: skip
        assert min(moved) > 100
        assert same == [0, 0, 0]


def test_legacy_caption_path_unchanged(tmp_path):
    """captions without views/legend/frame_size: _with_header frames, as in 0.1.0."""
    plain = _movie(tmp_path, "plain", CENTERS, np.stack([_conn(), -_conn()]), n=2)
    capped = _movie(
        tmp_path, "capped", CENTERS, np.stack([_conn(), -_conn()]), n=2,
        captions=["first", "second"],
    )  # fmt: skip
    head = int(max(16, 320 // 32) * (0.5 + 1.3))
    for a, b, text in zip(plain, capped, ("first", "second"), strict=True):
        print("caption frame", b.shape, "plain", a.shape, "head", head)
        assert b.shape == (200 + head, 320, 3)
        expect = bn._with_header(Image.fromarray(a.astype(np.uint8)), [text])
        np.testing.assert_array_equal(b, _rgb(expect))


@pytest.mark.parametrize(
    "layout",
    [
        {},
        {"legend": PN_LEGEND},
        {"views": ("left", "dorsal"), "legend": PN_LEGEND, "frame_size": (640, 360)},
    ],
    ids=["single-view", "legend", "views-frame_size"],
)
def test_plot_network_caption_is_a_one_frame_movie(tmp_path, layout):
    """plot_network(caption=text) equals the frame of a one-window movie with
    captions=[text], in the single-view path (`_with_header` band) and in the
    layout path (a line under the titles).

    The scene has no edges (zero connectivity), so the renders are exact, and
    black nodes, so the movie's crop reference (nodes in black) and the
    still's crop render are the same image.
    """
    text = "t = 12.5 s"
    style = {**_ANIM, "node_color": "black", **layout}
    still = bn.plot_network(CENTERS, caption=text, **style)
    plain = bn.plot_network(CENTERS, **style)
    frame = _movie(tmp_path, "one", CENTERS, np.zeros((1, 6, 6)), n=1,
                   captions=[text], **style)[0]  # fmt: skip
    print(layout, "still", still.size, "plain", plain.size, "movie", frame.shape)
    np.testing.assert_array_equal(_rgb(still), frame)
    if layout:
        # same frame width; a caption row is added under the (possible) titles
        W = still.size[0]
        assert still.size[0] == plain.size[0]
        has_titles = "views" in layout
        if "frame_size" in layout:
            assert still.size == plain.size == layout["frame_size"]
        else:
            grow = _head(W, caption=True) - _head(W) if has_titles else None
            print("height", plain.size[1], "->", still.size[1], "expected +", grow)
            assert still.size[1] > plain.size[1]
            if grow is not None:
                assert still.size[1] - plain.size[1] in (grow - 1, grow, grow + 1)
    else:
        expect = bn._with_header(plain, [text])
        np.testing.assert_array_equal(_rgb(still), _rgb(expect))
    assert _dark(_rgb(still)[:20]).sum() > 5  # the caption's text is at the top


def test_plot_network_caption_none_is_default():
    """caption=None draws exactly what a call without it draws."""
    for layout in ({}, {"views": ("left", "dorsal"), "legend": PN_LEGEND}):
        a = bn.plot_network(CENTERS, **_ANIM, **layout)
        b = bn.plot_network(CENTERS, caption=None, **_ANIM, **layout)
        print(layout, a.size, b.size)
        np.testing.assert_array_equal(_rgb(a), _rgb(b))
    with pytest.raises(ValueError, match="42") as err:
        bn.plot_network(CENTERS, caption=42, **_ANIM)
    print("error:", err.value)


def test_results_plot_network_forwards_caption(res_tfa):
    """HTFAResults.plot_network passes caption on: the image gains the band."""
    kw = {"mesh": None, "size": (320, 200), "connectivity": np.zeros((3, 3))}
    plain = res_tfa.plot_network(**kw)
    capped = res_tfa.plot_network(caption="subject 0", **kw)
    head = int(max(16, 320 // 32) * (0.5 + 1.3))
    print("plain", plain.size, "captioned", capped.size, "header", head)
    assert capped.size == (320, 200 + head)
    np.testing.assert_array_equal(_rgb(capped)[head:], _rgb(plain))


def test_annotate_pair_three_way_equivalence():
    """annotate_pair is _layout of two images; panes are crop-and-resize."""
    from PIL import ImageDraw

    a = Image.new("RGB", (400, 250), "white")
    ImageDraw.Draw(a).ellipse((80, 60, 300, 200), fill=(200, 30, 30))
    b = Image.new("RGB", (400, 250), "white")
    ImageDraw.Draw(b).rectangle((120, 40, 340, 220), fill=(30, 30, 200))
    titles = ("Original\nsaved files", "Reconstruction")
    for caption, W in (("TR 3", 1200), (None, 777), ("", 640)):
        pair = bn.annotate_pair(a, b, titles, caption, width=W)
        lay = bn._layout([a, b], titles, caption or None, None, None, W, None)
        print(caption, W, pair.size, lay.size)
        np.testing.assert_array_equal(_rgb(pair), _rgb(lay))
        box = bn.content_box([a, b])
        pane_w = W // 2
        pane_h = round(pane_w * (box[3] - box[1]) / (box[2] - box[0]))
        head = _head(W, n_sub=1, caption=bool(caption))
        assert pair.size == (W, head + pane_h)
        expect = b.crop(box).resize((pane_w, pane_h), Image.LANCZOS)
        np.testing.assert_array_equal(
            _rgb(pair)[head:, pane_w + 2 : 2 * pane_w], _rgb(expect)[:, 2:]
        )
        assert _dark(_rgb(pair)[:head]).sum() > 100


def test_animate_colors_resampled_nearest(tmp_path):
    """With n_frames > T, a fractional t takes the nearest timepoint's colors.

    Times 0, 0.5, 1, 1.5, 2: halves round up, so frames 2-3 and 4-5 are the
    same images, and node 0 is pure blue at t = 0.5 (no red-blue blend).
    """
    names = [["red", "blue"], ["blue", "red"], ["green", "blue"]]
    frames = _movie(
        tmp_path, "near", TWO, np.zeros((3, 2, 2)), 5, n_frames=5, node_color=names
    )
    right = (np.arange(320) > _MID)[None]
    for a, b in ((1, 2), (3, 4)):
        print("frames", a + 1, b + 1, "differing values",
              np.count_nonzero(frames[a] != frames[b]))  # fmt: skip
        np.testing.assert_array_equal(frames[a], frames[b])
    half = frames[1]
    blend = (half[..., 0] > 60) & (half[..., 2] > 60) & (half[..., 1] < 60)
    print("t=0.5 node 0: blue px", (_hue(half, 2) & right).sum(),
          "red px", (_hue(half, 0) & right).sum(),
          "purple (blended) px", (blend & right).sum())  # fmt: skip
    assert (_hue(half, 2) & right).sum() > 20
    assert (_hue(half, 0) & right).sum() == 0 and (blend & right).sum() == 0
    green = frames[3][..., 1] - np.maximum(frames[3][..., 0], frames[3][..., 2]) > 50
    print("t=1.5 node 0 green px", (green & right).sum())
    assert (green & right).sum() > 20
    # frames at integer times match the T-frame movie
    exact = _movie(tmp_path, "exact", TWO, np.zeros((3, 2, 2)), node_color=names)
    for k, j in ((0, 0), (2, 1), (4, 2)):
        np.testing.assert_array_equal(frames[k], exact[j])


def test_animate_spin_with_growing_sizes_stays_in_frame(tmp_path):
    """With spin, the zoom fits the largest radius of every node.

    Node 0 grows from 2 to 7 mm; every frame keeps a blank border, so no
    sphere is cut off at the image edge.
    """
    sizes = [[1.0, 1.0], [2.0, 1.0], [4.0, 1.0], [4.0, 1.0]]
    frames = _movie(
        tmp_path, "spin", TWO, None, 8, sizes=sizes, n_frames=8,
        spin=(0, 0, 1), interpolation="nearest",
    )  # fmt: skip
    for k, a in enumerate(frames):
        d = _drawn(a)
        rows, cols = np.nonzero(d)
        border = d[0].sum() + d[-1].sum() + d[:, 0].sum() + d[:, -1].sum()
        print("frame", k + 1, "drawn px", d.sum(), "rows", rows.min(), rows.max(),
              "cols", cols.min(), cols.max(), "border px", border)  # fmt: skip
        assert d.sum() > 0 and border == 0


def test_animate_empty_time_axes_rejected(tmp_path):
    """T = 0 sizes, colors or connectivity raise ValueError naming the input."""
    out = tmp_path / "x.mp4"
    cases = [
        ({"sizes": np.empty((0, 2))}, None, r"sizes .*shape \(0, 2\)"),
        ({"node_color": np.empty((0, 2, 3))}, None, r"node_color.*\(0, 2, 3\)"),
        ({"node_color": []}, None, r"node_color.*\(0,\)"),
        ({}, np.empty((0, 2, 2)), r"connectivity .*\(0, 2, 2\)"),
        ({"sizes": GROW}, np.empty((0, 2, 2)), r"connectivity .*\(0, 2, 2\)"),
    ]
    for kw, E, match in cases:
        with pytest.raises(ValueError, match=match) as err:
            bn.animate_network(TWO, E, out, **{**_ANIM, **kw})
        print(sorted(kw), None if E is None else E.shape, "error:", err.value)
    assert not out.exists()


def test_movie_crop_keeps_nodes_hidden_in_the_first_frame(tmp_path):
    """The crop does not depend on node colors: a node drawn in the background
    color in frame 1 (invisible) and blue in frame 2 is not cropped away."""
    kw = {"views": ["dorsal"], "background": "black"}
    colors = [["red", "black"], ["red", "blue"]]
    still = _rgb(
        bn.plot_network(TWO, node_color=["red", "blue"], view="dorsal",
                        background="black", **_ANIM)
    )  # fmt: skip
    want = int(_hue(still, 2).sum())
    for jobs in (1, 2):
        frames = _movie(
            tmp_path, f"hidden{jobs}", TWO, None, 2, node_color=colors, n_jobs=jobs,
            **kw,
        )  # fmt: skip
        got = [(int(_hue(a, 0).sum()), int(_hue(a, 2).sum())) for a in frames]
        print(f"n_jobs={jobs}: frame shape {frames[0].shape}, red/blue px per frame "
              f"{got}; blue px in an uncropped still {want}")  # fmt: skip
        assert got[0][1] == 0 and got[0][0] > 20  # node 1 is black on black
        assert got[1][0] > 20 and got[1][1] == want > 20  # all of it is in the crop


def test_plot_network_crop_ignores_edges():
    """The still-image crop comes from a render without edges, as in movies."""
    E = np.zeros((6, 6))
    E[0, 1] = E[1, 0] = 20.0  # a 10 mm tube along the top row of nodes
    views = ("left", "dorsal")
    with_e = bn.plot_network(CENTERS, E, views=views, legend=PN_LEGEND, **_ANIM)
    without = bn.plot_network(CENTERS, views=views, legend=PN_LEGEND, **_ANIM)
    # the edge does reach past the nodes, so a crop around it would differ
    boxes = [
        bn.content_box(bn.plot_network(CENTERS, c, view="left", **_ANIM))
        for c in (E, None)
    ]
    print("sizes with/without edges", with_e.size, without.size, "boxes", boxes)
    assert boxes[0] != boxes[1]
    assert with_e.size == without.size
    a, b = _rgb(with_e), _rgb(without)
    gray = _drawn(b) & ~_red(a)  # node pixels the edge does not cover
    print("node px compared", int(gray.sum()))
    assert gray.sum() > 100
    assert np.abs(a[gray] - b[gray]).max() <= 2


def test_frame_size_scales_views_up():
    """A small render in a large frame_size is scaled up to fill the space."""
    views = ("left", "dorsal")
    natural = bn.plot_network(CENTERS, views=views, **_ANIM)
    bw = natural.size[0] / 2
    bh = natural.size[1] - _head(natural.size[0])  # within 1 px (even trim)
    big = _rgb(bn.plot_network(CENTERS, views=views, frame_size=(960, 540), **_ANIM))
    space = 540 - _head(960)
    # the divider spans the pane height
    ph = int((np.abs(big[:, 480] - 200) <= 1).all(axis=1).sum())
    print(f"natural pane {bw} x {bh} (+0/1), divider {ph} px, space 960 x {space}")
    assert ph > 1.5 * bh  # scaled up
    # the scale that fits both ways, for the cropped height bh or bh + 1
    fits = []
    for h in (bh, bh + 1):
        scale = min(960 / (2 * bw), space / h)
        pw, ph_fit = int(bw * scale), int(h * scale)
        print(f"height {h}: scale {scale:.4f}, pane {pw} x {ph_fit}")
        if abs(ph - ph_fit) <= 1:  # the divider spans ph_fit or ph_fit + 1 rows
            fits.append(abs(2 * pw - 960) <= 2 or abs(ph_fit - space) <= 2)
    assert fits and all(fits)


def test_legend_font_floor():
    """Legend text is at least a 12 px font; a 1920-wide frame uses 30 px."""
    from PIL import ImageDraw

    def text_height(img):
        a = _rgb(img)
        band = _band(img.size[0])
        rows = np.flatnonzero(_dark(a[-band:]).any(axis=1))
        return int(rows[-1] - rows[0] + 1), band

    def font_height(size):
        draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
        t = draw.textbbox((0, 0), "Hg", font=bn._font(size))
        return t[3] - t[1]

    legend = [("red", "Hg", "line")]
    small = bn.plot_network(CENTERS, legend=legend, **_ANIM)
    large = bn.plot_network(CENTERS, legend=legend, frame_size=(1920, 1080), **_ANIM)
    for img, size in ((small, 12), (large, 30)):
        height, band = text_height(img)
        print(f"frame {img.size}: text {height} px, font {size} -> {font_height(size)}"
              f" px, band {band}")  # fmt: skip
        assert img.size[0] < 320 or size == 30
        assert abs(height - font_height(size)) <= 2
    assert _band(small.size[0]) == 26 and _band(1920) == 64


def test_annotate_pair_empty_titles_keep_header():
    """annotate_pair reserves the title rows even when both titles are empty."""
    from PIL import ImageDraw

    a = Image.new("RGB", (400, 250), "white")
    ImageDraw.Draw(a).ellipse((80, 60, 300, 200), fill=(200, 30, 30))
    box = bn.content_box([a, a])
    pane_h = round(600 * (box[3] - box[1]) / (box[2] - box[0]))
    for caption in (None, "TR 3"):
        pair = bn.annotate_pair(a, a, ("", ""), caption)
        expect = (1200, _head(1200, caption=bool(caption)) + pane_h)
        print(caption, pair.size, expect)
        assert pair.size == expect


# ------------------------------------------------------------------ compose

_COMPOSE_LOG = {"path": None}


def _stack_compose(images, t):
    """Record what compose receives, paste the images top to bottom."""
    import json

    with open(_COMPOSE_LOG["path"], "a") as f:
        f.write(json.dumps([len(images), list(images[0].size), t]) + "\n")
    out = Image.new("RGB", (images[0].width, sum(im.height for im in images)), "white")
    y = 0
    for im in images:
        out.paste(im, (0, y))
        y += im.height
    return out


def _odd_compose(images, t):
    return Image.new("RGB", (321, 201), "white")


def _growing_compose(images, t):
    return Image.new("RGB", (320 + 2 * int(t), 200), "white")


def _none_compose(images, t):
    return None


def _int_compose(images, t):
    return 42


def _compose_log(tmp_path):
    import json

    _COMPOSE_LOG["path"] = tmp_path / "compose.jsonl"
    path = _COMPOSE_LOG["path"]
    return lambda: [json.loads(s) for s in path.read_text().splitlines()]


def test_compose_receives_views_and_time(tmp_path):
    read = _compose_log(tmp_path)
    bn.animate_network(
        CENTERS, _frames(), tmp_path / "c.gif", views=("left", "dorsal"),
        compose=_stack_compose, frames_dir=tmp_path / "f", **_ANIM,
    )  # fmt: skip
    log = read()
    print(log)
    assert log == [[2, [320, 200], float(t)] for t in range(3)]
    assert isinstance(log[0][2], float)
    for k in (1, 2, 3):
        with Image.open(tmp_path / "f" / f"{k}.png") as png:
            assert png.size == (320, 400)


def test_compose_without_views(tmp_path):
    read = _compose_log(tmp_path)
    bn.animate_network(
        CENTERS, _frames(), tmp_path / "c.gif", compose=_stack_compose, **_ANIM
    )
    log = read()
    print(log)
    assert [entry[0] for entry in log] == [1, 1, 1]


def test_compose_frame_is_returned_as_is(tmp_path):
    """No caption, title, legend or crop is added to what compose returns."""
    _compose_log(tmp_path)
    bn.animate_network(
        CENTERS, _frames(), tmp_path / "c.gif", compose=_stack_compose,
        frames_dir=tmp_path / "f", **_ANIM,
    )  # fmt: skip
    bn.animate_network(
        CENTERS, _frames(), tmp_path / "p.gif", frames_dir=tmp_path / "g", **_ANIM
    )
    for k in (1, 2, 3):
        a = _rgb(Image.open(tmp_path / "f" / f"{k}.png"))
        b = _rgb(Image.open(tmp_path / "g" / f"{k}.png"))
        assert np.array_equal(a, b), k


def test_compose_stacked_views_match_single_view_renders(tmp_path):
    """Top half is the left view's render, bottom half the dorsal view's."""
    _compose_log(tmp_path)
    bn.animate_network(
        CENTERS, _frames(), tmp_path / "s.gif", views=("left", "dorsal"),
        compose=_stack_compose, frames_dir=tmp_path / "s", **_ANIM,
    )  # fmt: skip
    for view, rows in (("left", slice(0, 200)), ("dorsal", slice(200, 400))):
        bn.animate_network(
            CENTERS, _frames(), tmp_path / f"{view}.gif", view=view,
            frames_dir=tmp_path / view, **_ANIM,
        )  # fmt: skip
        for k in (1, 2, 3):
            stacked = _rgb(Image.open(tmp_path / "s" / f"{k}.png"))[rows]
            single = _rgb(Image.open(tmp_path / view / f"{k}.png"))
            print(view, k, stacked.shape, single.shape)
            assert np.array_equal(stacked, single), (view, k)


def test_compose_must_be_callable(tmp_path):
    with pytest.raises(ValueError, match="compose must be a callable") as err:
        bn.animate_network(CENTERS, _frames(), tmp_path / "c.gif", compose=42, **_ANIM)
    print(err.value)
    assert "42" in str(err.value)
    assert not (tmp_path / "c.gif").exists()


@pytest.mark.parametrize(
    "extra",
    [
        {"legend": [("red", "a", "dot")]},
        {"captions": ["a", "b", "c"]},
        {"captions": str},
        {"frame_size": (320, 200)},
    ],
)
def test_compose_excludes_layout_arguments(tmp_path, extra):
    _compose_log(tmp_path)
    with pytest.raises(ValueError, match="compose") as err:
        bn.animate_network(
            CENTERS, _frames(), tmp_path / "c.gif", compose=_stack_compose,
            **extra, **_ANIM,
        )  # fmt: skip
    print(err.value)
    assert not (tmp_path / "c.gif").exists()


def test_compose_bad_return(tmp_path):
    for suffix in (".gif", ".mp4"):
        with pytest.raises(ValueError, match="same size") as err:
            bn.animate_network(
                CENTERS, _frames(), tmp_path / f"a{suffix}",
                compose=_growing_compose, **_ANIM,
            )  # fmt: skip
        print(suffix, err.value)
        assert "(322, 200)" in str(err.value) and "(320, 200)" in str(err.value)
    for bad, text in ((_none_compose, "None"), (_int_compose, "42")):
        with pytest.raises(ValueError, match="compose must return a PIL image") as err:
            bn.animate_network(
                CENTERS, _frames(), tmp_path / "n.gif", compose=bad, **_ANIM
            )
        print(err.value)
        assert f"got {text}" in str(err.value)


def test_compose_odd_size_mp4(tmp_path):
    import imageio.v2 as iio

    out = bn.animate_network(
        CENTERS, _frames(), tmp_path / "o.mp4", compose=_odd_compose, **_ANIM
    )
    reader = iio.get_reader(out, "ffmpeg")
    try:
        shape = reader.get_data(0).shape
    finally:
        reader.close()
    print(shape)
    assert shape[:2] == (200, 320)


# ------------------------------------------------------- parallel rendering
# Worker-side callables reach the worker processes pickled (cloudpickle).
# Module-level ones carry their paths through functools.partial (a global set
# in this process does not reach them); closures defined inside a function
# are pickled by value, as functions defined in a notebook are.
#
# The cases that look at htfa's temporary frame folders run as scenarios in a
# fresh process (`_isolated`) whose TMPDIR, inherited by its workers, is a
# folder of their own, so other tests or htfa processes using the system
# temporary directory can neither disturb them nor lose files to them.


def _frame_dirs():
    """htfa frame folders now in the temporary directory."""
    import tempfile

    return sorted(p.name for p in Path(tempfile.gettempdir()).glob("htfa-frames-*"))


def _alive(pid):
    """Whether process ``pid`` exists and has not exited (on POSIX: a worker
    the pool has not reaped)."""
    if sys.platform == "win32":
        # os.kill(pid, 0) is no probe on Windows (WinError 87; other signals
        # terminate the process): open the process and see whether it exited.
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        synchronize, query_limited = 0x00100000, 0x1000
        handle = kernel32.OpenProcess(synchronize | query_limited, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5  # access denied: it exists
        try:
            return kernel32.WaitForSingleObject(handle, 0) == 0x102  # WAIT_TIMEOUT
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _record(log, text):
    """Add ``text`` to ``log``, a folder holding one file per record.

    Processes appending to one file can overwrite each other's lines on
    Windows (append there seeks to the end, then writes), so each record is a
    file of its own, named so that sorting the names gives the order of writing.
    The file is written under a ``.tmp`` name and then renamed: a worker the
    pool terminates in the middle of writing leaves no record, not an empty one
    (seen on Windows CI: a record file with no text).
    """
    import tempfile
    import time

    Path(log).mkdir(exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f"{time.time_ns():020d}-", suffix=".tmp", dir=log)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, tmp[: -len(".tmp")])


def _records(log):
    """The finished records added to ``log`` by `_record`, in order (none if
    there is no folder)."""
    log = Path(log)
    if not log.exists():
        return []
    return [p.read_text() for p in sorted(log.iterdir()) if p.suffix != ".tmp"]


def _pids(log):
    """Process ids recorded by `_write_pid` in ``log``."""
    return [int(s) for s in _records(log)]


def _write_pid(log):
    _record(log, str(os.getpid()))


_ISOLATED = """
import sys
sys.path.insert(0, {tests!r})
import test_brainnet

test_brainnet.{name}({files!r})
print("finished")
"""


def _isolated(tmp_path, name):
    """Run scenario ``name(files)`` of this module in a fresh process whose
    TMPDIR (inherited by its workers) is a new folder under ``tmp_path``.

    The scenario asserts and prints for itself (its output is printed here);
    this checks that it finished and left no htfa frame folder in its TMPDIR.
    """
    import subprocess
    import time

    tmp, files = tmp_path / "tmpdir", tmp_path / "files"
    tmp.mkdir()
    files.mkdir()
    tests = str(Path(__file__).parent)
    code = _ISOLATED.format(tests=tests, name=name, files=str(files))
    start = time.perf_counter()
    run = subprocess.run(
        [sys.executable, "-c", code], env={**os.environ, "TMPDIR": str(tmp)},
        capture_output=True, text=True, timeout=900,
    )  # fmt: skip
    print(f"{name}: exit code {run.returncode} after "
          f"{time.perf_counter() - start:.1f} s", "stdout:", run.stdout, "stderr:",
          run.stderr, sep="\n")  # fmt: skip
    assert run.returncode == 0
    assert run.stdout.splitlines()[-1] == "finished"
    left = sorted(p.name for p in tmp.glob("htfa-frames-*"))
    print("htfa frame folders left in the scenario's TMPDIR:", left)
    assert not left


def _hold(flag, timeout=300.0):
    """A joblib task that keeps a shared-pool worker busy until ``flag`` exists."""
    import time

    end = time.monotonic() + timeout
    while not os.path.exists(flag):
        if time.monotonic() > end:
            raise TimeoutError(f"{flag} did not appear in {timeout} s")
        time.sleep(0.05)
    return os.getpid()


def _unpickled_in(log, refuse):
    """Record the pid of the process unpickling a `_PidCaptions` or
    `_UnpicklableCompose` (a worker, once per task); refuse if ``refuse``."""
    _write_pid(log)
    if refuse:
        raise RuntimeError(f"refusing to be unpickled in process {os.getpid()}")
    return _PidCaptions(log)


class _PidCaptions:
    """Empty captions; each worker task that unpickles it writes its pid to ``log``."""

    def __init__(self, log):
        self.log = log

    def __call__(self, t):
        return ""

    def __reduce__(self):
        return _unpickled_in, (self.log, False)


class _UnpicklableCompose:
    """A compose that pickles here, but whose unpickling in a worker raises
    (after writing the worker's pid to ``log``): a deterministic failure."""

    def __init__(self, log):
        self.log = log

    def __call__(self, images, t):
        return images[0]

    def __reduce__(self):
        return _unpickled_in, (self.log, True)


def _locking_compose(fail, images, t):
    """At t == 0, leave a folder that cannot be emptied in the call's frame
    folder (and raise ValueError if ``fail``); else the first view."""
    import tempfile

    if t == 0:
        (work,) = Path(tempfile.gettempdir()).glob("htfa-frames-*")
        locked = work / "locked"
        locked.mkdir()
        (locked / "file").write_text("x")
        locked.chmod(0o500)
        if fail:
            raise ValueError("compose refuses t=0.0 after locking")
    return images[0]


def _unlock_frame_dirs(names):
    import shutil
    import tempfile

    for name in names:
        work = Path(tempfile.gettempdir()) / name
        for d in (work, *work.rglob("*")):
            if d.is_dir():
                d.chmod(0o700)
        shutil.rmtree(work)


def _die_compose(pids, images, t):
    """Write this worker's pid to ``pids``, then end the worker process."""
    _write_pid(pids)
    os._exit(1)


def _mp4_count(path):
    import imageio.v2 as iio

    reader = iio.get_reader(path, "ffmpeg")
    try:
        return reader.count_frames()
    finally:
        reader.close()


def _jobs_pair(tmp_path, name, connectivity, n, suffix=".mp4", **kw):
    """The same movie with n_jobs=1 and n_jobs=2; returns both frame folders."""
    import time

    folders = []
    for jobs in (1, 2):
        folder = tmp_path / f"{name}{jobs}"
        start = time.perf_counter()
        out = bn.animate_network(
            CENTERS, connectivity, tmp_path / f"{name}{jobs}{suffix}",
            frames_dir=folder, n_jobs=jobs, **{**_ANIM, **kw},
        )  # fmt: skip
        names = sorted(p.name for p in folder.iterdir())
        print(f"{name} n_jobs={jobs}: {time.perf_counter() - start:.1f} s, {names}")
        assert out.stat().st_size > 0 and len(names) == n
        folders.append(folder)
    return folders


def _assert_same_frames(folders, names):
    for folder in folders:
        assert sorted(p.name for p in folder.iterdir()) == sorted(names)
    for name in names:
        d = bn.image_diff(folders[0] / name, folders[1] / name)
        print(name, "image_diff n_jobs=1 vs 2:", d)
        assert d < 1e-3


def test_n_jobs_matches_sequential(tmp_path):
    """Two views, legend, captions, (T, K) sizes: n_jobs=2 frames equal n_jobs=1."""
    T = 6
    E = np.stack([_conn(0.6 * np.cos(t)) * (-1) ** t for t in range(T)])
    sizes = np.random.default_rng(1).random((T, len(CENTERS))) + 0.5
    folders = _jobs_pair(
        tmp_path, "par", E, T, sizes=sizes, views=("left", "dorsal"),
        legend=PN_LEGEND, captions=[f"window {t}" for t in range(T)],
    )  # fmt: skip
    _assert_same_frames(folders, [f"{k}.png" for k in range(1, T + 1)])
    for jobs in (1, 2):
        count = _mp4_count(tmp_path / f"par{jobs}.mp4")
        print(f"n_jobs={jobs} movie frames:", count)
        assert count == T


def _scenario_temp_folder_removed(files):
    """Run by test_n_jobs_temp_folder_removed in its own process and TMPDIR."""
    import json
    import threading

    from joblib import Parallel, delayed

    files = Path(files)
    before = _frame_dirs()
    print("htfa frame folders before:", before)
    assert before == []
    # A joblib workload in the shared reusable pool, kept running (each task
    # waits for `flag`) across both animate_network calls.
    flag, held = files / "release", []
    workload = threading.Thread(
        target=lambda: held.extend(
            Parallel(n_jobs=2)(delayed(_hold)(str(flag)) for _ in range(2))
        )
    )
    workload.start()
    # compose is a closure, pickled by value like a notebook function; it
    # captures the log path and a stamp of image data
    log, stamp = files / "seen", np.full((8, 8, 3), (12, 34, 56), np.uint8)

    def compose(images, t):
        _record(log, json.dumps([os.getpid(), t, _frame_dirs()]))
        img = images[0].copy()
        img.paste(Image.fromarray(stamp), (0, 0))
        return img

    out = bn.animate_network(
        CENTERS, _frames(), files / "ok.mp4", n_jobs=2, compose=compose,
        frames_dir=files / "ok", **_ANIM,
    )  # fmt: skip
    seen = [json.loads(s) for s in _records(log)]
    print("worker calls (pid, t, folders seen):", seen)
    assert len(seen) == 3 and all(len(set(s[2]) - set(before)) == 1 for s in seen)
    assert sorted(s[1] for s in seen) == [0.0, 1.0, 2.0]
    workers = {s[0] for s in seen}
    assert os.getpid() not in workers  # drawn in workers
    corners = [_rgb(Image.open(files / "ok" / f"{k}.png"))[:8, :8] for k in (1, 2, 3)]
    print("frame corners (the captured stamp):", [c[0, 0].tolist() for c in corners])
    assert all((c == (12, 34, 56)).all() for c in corners)
    assert _mp4_count(out) == 3
    alive = [pid for pid in workers if _alive(pid)]
    print("worker pids", workers, "still alive after the call:", alive)
    assert not alive
    after = _frame_dirs()
    print("after success:", after)
    assert after == before
    errors, called = files / "errors", files / "called"

    def refuse(images, t):
        _write_pid(called)  # every compose call, in every worker, first of all
        if t == 2:
            _write_pid(errors)
            raise ValueError(f"compose refuses t={t}")
        return images[0]

    with pytest.raises(ValueError, match="compose refuses t=2.0") as err:
        bn.animate_network(
            CENTERS, _frames(), files / "bad.mp4", n_jobs=2, compose=refuse, **_ANIM
        )
    print("raised:", type(err.value).__name__, err.value)
    calls = _pids(errors)
    print("compose calls at t=2 (pids):", calls, "alive:", [_alive(p) for p in calls])
    assert len(calls) == 1 and not _alive(calls[0])  # raised once, not retried
    # every worker that composed a frame, also the one that did not raise
    # (it may have been stopped before its first frame, then it is not listed)
    erring = set(_pids(called))
    print("workers that composed in the failing call:", sorted(erring),
          "alive:", [p for p in erring if _alive(p)])  # fmt: skip
    assert calls[0] in erring and not any(_alive(p) for p in erring)
    assert os.getpid() not in erring
    assert not (files / "bad.mp4").exists()
    after = _frame_dirs()
    print("after error:", after)
    assert after == before
    # the shared pool's tasks ran on through both calls and still finish
    print("joblib workload still running after both calls:", workload.is_alive())
    assert workload.is_alive()
    flag.write_text("go")
    workload.join(timeout=120)
    again = Parallel(n_jobs=2)(delayed(os.getpid)() for _ in range(4))
    print("joblib shared-pool workers", held, "afterwards", again)
    assert len(held) == 2 and all(_alive(p) for p in held)
    assert not set(held) & (workers | erring)
    assert set(again) <= set(held)  # the same shared pool, not restarted


def test_n_jobs_temp_folder_removed(tmp_path):
    """The call's frame folder exists while workers draw, and is removed after
    success and after a compose ValueError (raised once, not retried); worker
    processes are gone after both. compose is a closure capturing data, and a
    joblib workload in joblib's shared pool keeps running across both calls."""
    _isolated(tmp_path, "_scenario_temp_folder_removed")


def _scenario_cleanup_failure_surfaces(files):
    """Run by test_n_jobs_cleanup_failure_surfaces in its own process and TMPDIR."""
    import traceback
    from functools import partial

    files = Path(files)
    before = _frame_dirs()
    assert before == []
    left = []
    try:
        with pytest.raises(OSError, match="was not removed") as err:
            bn.animate_network(
                CENTERS, _frames(), files / "ok.mp4", n_jobs=2,
                compose=partial(_locking_compose, False), **_ANIM,
            )  # fmt: skip
        print("after success:", type(err.value).__name__, err.value)
        assert (files / "ok.mp4").stat().st_size > 0
        left = sorted(set(_frame_dirs()) - set(before))
        print("left behind:", left)
        assert len(left) == 1
        _unlock_frame_dirs(left)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError) as err:
                bn.animate_network(
                    CENTERS, _frames(), files / "bad.mp4", n_jobs=2,
                    compose=partial(_locking_compose, True), **_ANIM,
                )  # fmt: skip
        print("rendering error:", type(err.value).__name__, err.value)
        assert type(err.value) is ValueError
        assert str(err.value) == "compose refuses t=0.0 after locking"
        chain, e = [], err.value
        while e is not None and e not in chain:
            chain.append(e)
            e = e.__cause__ or (None if e.__suppress_context__ else e.__context__)
        print("chain:", [f"{type(x).__name__}: {x}" for x in chain])
        found = [x for x in chain if isinstance(x, OSError)]
        assert found and "was not removed" in str(found[0])
        assert isinstance(found[0].__cause__, PermissionError)
        shown = "".join(traceback.format_exception(err.value))
        assert "was not removed" in shown and "after locking" in shown
        left = sorted(set(_frame_dirs()) - set(before))
        assert len(left) == 1
    finally:
        _unlock_frame_dirs(sorted(set(_frame_dirs()) - set(before)))
    assert _frame_dirs() == before


@pytest.mark.skipif(
    os.name != "posix" or os.geteuid() == 0,
    reason="needs POSIX permissions that bind this user (not root)",
)
def test_n_jobs_cleanup_failure_surfaces(tmp_path):
    """A frame folder that cannot be removed is reported, not ignored: OSError
    after a written movie; on a rendering error that error still propagates
    (also with warnings as errors) with the cleanup failure in its chain."""
    _isolated(tmp_path, "_scenario_cleanup_failure_surfaces")


def test_n_jobs_more_workers_than_frames(tmp_path):
    """T=1 with n_jobs=2 and T=2 with n_jobs=8 write complete movies."""
    for T, jobs in ((1, 2), (2, 8)):
        E = np.stack([_conn(), -_conn()])[:T]
        out = bn.animate_network(
            CENTERS, E, tmp_path / f"m{T}.mp4", n_jobs=jobs,
            frames_dir=tmp_path / f"f{T}", **_ANIM,
        )  # fmt: skip
        count = _mp4_count(out)
        names = sorted(p.name for p in (tmp_path / f"f{T}").iterdir())
        print(f"T={T} n_jobs={jobs}: {count} movie frames, {names}")
        assert count == T and names == [f"{k}.png" for k in range(1, T + 1)]


def test_n_jobs_jpeg_and_gif(tmp_path):
    """jpeg_quality gives .jpg frames as in the sequential path; GIF output works."""
    folders = _jobs_pair(tmp_path, "jpg", _frames(), 3, jpeg_quality=80)
    _assert_same_frames(folders, [f"{k}.jpg" for k in (1, 2, 3)])
    folders = _jobs_pair(tmp_path, "gif", _frames(), 3, suffix=".gif")
    _assert_same_frames(folders, [f"{k}.png" for k in (1, 2, 3)])
    for jobs in (1, 2):
        with Image.open(tmp_path / f"gif{jobs}.gif") as g:
            print(f"n_jobs={jobs} gif frames:", g.n_frames, g.size)
            assert g.n_frames == 3 and g.size == (320, 200)


def test_n_jobs_spin(tmp_path):
    """Spin frames drawn by two workers equal the sequential ones."""
    folders = _jobs_pair(tmp_path, "spin", _conn(), 4, n_frames=4, spin=(0, 0, 1))
    _assert_same_frames(folders, [f"{k}.png" for k in (1, 2, 3, 4)])


def test_is_blank():
    black = Image.new("RGB", (320, 200), "black")
    assert bn._is_blank(black, "white")
    assert bn._is_blank(black, (0.9, 0.9, 0.9))
    # a dark background is not checked
    assert not bn._is_blank(black, "black")
    assert not bn._is_blank(black, (0.15, 0.1, 0.1))  # max channel 38 < 40
    render = bn.plot_network(CENTERS, _conn(), **_ANIM)
    a = np.asarray(render)
    print("render near-black fraction:", (a.max(axis=2) < 10).mean())
    assert not bn._is_blank(render, "white")
    # A black rectangle on white: blank when it covers over 5 percent of the
    # pixels and reaches a corner; drawn content that stays inside is not.
    for name, rows, cols, want in (
        ("10% top rows (two corners)", slice(0, 20), slice(None), True),
        ("10% in the bottom-right corner", slice(140, 200), slice(214, 320), True),
        ("4% top rows (two corners)", slice(0, 8), slice(None), False),
        ("10% interior", slice(70, 130), slice(107, 214), False),
        ("98% interior, white border", slice(1, 199), slice(1, 319), False),
    ):
        a = np.full((200, 320, 3), 255, np.uint8)
        a[rows, cols] = 0
        frac = float((a.max(axis=2) < 10).mean())
        blank = bn._is_blank(Image.fromarray(a), "white")
        print(f"{name}: black fraction {frac:.3f} -> blank={blank}")
        assert blank is want


def _scenario_restart_after_worker_death(files):
    """Run by test_n_jobs_restart_after_worker_death in its own process and TMPDIR.

    Frames 0-1 go to one worker and 2-3 to the other. The second worker, at
    frame 2, waits until the first has written both of its PNGs, then dies
    (once: a marker file records it). Every composed frame is logged with its
    worker's pid, so the log shows which frames each pool drew.
    """
    import json
    import tempfile
    import time

    files = Path(files)
    marker, log = files / "died.txt", files / "drawn"

    def compose(images, t):
        if t == 2 and not marker.exists():
            (work,) = Path(tempfile.gettempdir()).glob("htfa-frames-*")
            end = time.monotonic() + 120
            while not all((work / f"{k:05d}.png").exists() for k in (0, 1)):
                assert time.monotonic() < end, "frames 0 and 1 were not written"
                time.sleep(0.05)
            marker.write_text(str(os.getpid()))
            os._exit(1)
        _record(log, json.dumps([os.getpid(), t]))
        return images[0]

    E = np.stack([_conn(), -_conn(), _conn(0.6), -_conn(0.6)])
    out = bn.animate_network(
        CENTERS, E, files / "r.mp4", n_jobs=2, frames_dir=files / "f",
        compose=compose, **_ANIM,
    )  # fmt: skip
    count = _mp4_count(out)
    names = sorted(p.name for p in (files / "f").iterdir())
    died = int(marker.read_text())
    drawn = [json.loads(s) for s in _records(log)]
    print("worker that died:", died, "frames:", count, names)
    print("frames composed (pid, t), in order:", drawn)
    assert marker.exists() and count == 4
    assert names == [f"{k}.png" for k in (1, 2, 3, 4)]
    # each frame composed once: frames 0 and 1, finished before the death,
    # were not redrawn; 2 and 3 were drawn by the new pool's workers
    assert sorted(t for _, t in drawn) == [0.0, 1.0, 2.0, 3.0]
    first = {pid for pid, t in drawn if t < 2}
    again = {pid for pid, t in drawn if t >= 2}
    assert len(first) == 1 and not again & (first | {died})
    pids = first | again | {died}
    print("worker pids", sorted(pids), "alive:", [p for p in pids if _alive(p)])
    assert not any(_alive(p) for p in pids)
    # the recovered movie equals one drawn in this process
    bn.animate_network(CENTERS, E, files / "s.mp4", frames_dir=files / "s", **_ANIM)
    _assert_same_frames([files / "s", files / "f"], names)


def test_n_jobs_restart_after_worker_death(tmp_path):
    """A worker that dies on one frame is replaced; only the missing frames are
    redrawn, by new workers, and the movie equals the sequential one."""
    _isolated(tmp_path, "_scenario_restart_after_worker_death")


def _scenario_gives_up(files):
    """Run by test_n_jobs_gives_up in its own process and TMPDIR."""
    import time
    from functools import partial

    from joblib.externals.loky.process_executor import TerminatedWorkerError

    files = Path(files)
    pids = files / "pids"
    before = _frame_dirs()
    start = time.perf_counter()
    # one frame: each pool gets one task and one worker, which records its pid
    with pytest.raises(RuntimeError, match="5 restarts") as err:
        bn.animate_network(
            CENTERS, _frames()[:1], files / "g.mp4", n_jobs=2,
            compose=partial(_die_compose, str(pids)), **_ANIM,
        )  # fmt: skip
    elapsed = time.perf_counter() - start
    workers = _pids(pids)
    print(f"gave up after {elapsed:.1f} s:", err.value)
    print("worker pids, one per pool:", workers,
          "alive:", [p for p in workers if _alive(p)])  # fmt: skip
    assert isinstance(err.value.__cause__, TerminatedWorkerError)
    # one first pool and 5 restarts, each with a new worker; all gone
    assert len(workers) == 6 and len(set(workers)) == 6
    assert not any(_alive(p) for p in workers)
    assert _frame_dirs() == before
    assert not (files / "g.mp4").exists()


def test_n_jobs_gives_up(tmp_path):
    """Workers that always die: RuntimeError after exactly 5 restarts (6 pools,
    counted by the pids their workers record), workers gone, folder removed."""
    _isolated(tmp_path, "_scenario_gives_up")


def _scenario_unpicklable_compose(files):
    """Run by test_n_jobs_unpicklable_compose_not_retried in its own process."""
    from joblib.externals.loky.process_executor import (
        BrokenProcessPool,
        TerminatedWorkerError,
    )

    files = Path(files)
    log = files / "unpickled"
    with pytest.raises(BrokenProcessPool) as err:
        bn.animate_network(
            CENTERS, _frames()[:1], files / "u.mp4", n_jobs=2,
            compose=_UnpicklableCompose(str(log)), **_ANIM,
        )  # fmt: skip
    workers = _pids(log)
    print("raised:", type(err.value).__name__, err.value)
    print("cause:", repr(err.value.__cause__))
    print("workers that unpickled the task:", workers,
          "alive:", [p for p in workers if _alive(p)])  # fmt: skip
    assert not isinstance(err.value, TerminatedWorkerError)
    assert "un-serialize" in str(err.value)
    assert "refusing to be unpickled" in str(err.value.__cause__)
    assert len(workers) == 1  # one pool: no restart
    assert not any(_alive(p) for p in workers)
    assert _frame_dirs() == [] and not (files / "u.mp4").exists()


def test_n_jobs_unpicklable_compose_not_retried(tmp_path):
    """A task that cannot be unpickled in the worker (a BrokenProcessPool that
    is not a dead worker) fails at once: one pool, no restart."""
    _isolated(tmp_path, "_scenario_unpicklable_compose")


def test_n_jobs_dark_nodes_not_blank(tmp_path):
    """Large black nodes inside a white frame are content, not a failed render."""
    black = {**_ANIM, "node_color": "black", "radius_kb": (0.0, 25.0)}
    render = bn.plot_network(CENTERS, _conn(), **black)
    dark = np.asarray(render).max(axis=2) < 10
    corners = dark[[0, 0, -1, -1], [0, -1, 0, -1]]
    print("near-black fraction:", round(float(dark.mean()), 3), "corners:", corners)
    assert dark.mean() > 0.05 and not corners.any()
    assert not bn._is_blank(render, "white")
    folders = _jobs_pair(tmp_path, "dark", _frames(), 3, **black)
    _assert_same_frames(folders, [f"{k}.png" for k in (1, 2, 3)])


def _scenario_blank_render(files):
    """Run by test_n_jobs_blank_render in its own process and TMPDIR."""
    files = Path(files)
    black = {**_ANIM, "node_color": "black", "radius_kb": (0.0, 200.0)}
    E = np.stack([_conn()])
    render = bn.plot_network(CENTERS, _conn(), **black)
    dark = np.asarray(render).max(axis=2) < 10
    corners = dark[[0, 0, -1, -1], [0, -1, 0, -1]]
    print("near-black fraction:", round(float(dark.mean()), 3), "corners:", corners)
    assert bn._is_blank(render, "white")
    out = bn.animate_network(CENTERS, E, files / "seq.mp4", **black)
    assert _mp4_count(out) == 1
    before = _frame_dirs()
    # one frame: each pool runs one task, whose worker records its pid when it
    # unpickles the captions (the blank check stops the frame before compose
    # or captions would run)
    log = files / "pids"
    with pytest.raises(RuntimeError, match="5 restarts") as err:
        bn.animate_network(
            CENTERS, E, files / "par.mp4", n_jobs=2, captions=_PidCaptions(str(log)),
            **black,
        )  # fmt: skip
    workers = _pids(log)
    print(err.value)
    print("worker pids, one per pool:", workers,
          "alive:", [p for p in workers if _alive(p)])  # fmt: skip
    assert "_BlankRender" in str(err.value) and "rendered black" in str(err.value)
    assert isinstance(err.value.__cause__, bn._BlankRender)
    assert len(workers) == 6 and len(set(workers)) == 6  # 1 pool + 5 restarts
    assert not any(_alive(p) for p in workers)
    assert _frame_dirs() == before
    assert not (files / "par.mp4").exists()


def test_n_jobs_blank_render(tmp_path):
    """Workers reject a real render that is black to its corners on white.

    Black spheres of 200 mm radius fill the whole view, so every frame meets
    the blank-render rule: the pool is restarted exactly 5 times (6 pools,
    counted by their workers' pids) and the call gives up. n_jobs=1 runs no
    check and writes the movie.
    """
    _isolated(tmp_path, "_scenario_blank_render")


def test_animate_network_forwards_encoding(tmp_path):
    """crf and faststart reach write_animation (sequential and parallel).

    x264 writes its settings, ``crf=<value>`` among them, into the stream, so
    the encoded file shows the crf it was given.
    """
    for jobs in (1, 2):
        sizes = {}
        for crf in (10, 40):
            out = bn.animate_network(
                CENTERS, _frames(), tmp_path / f"c{crf}_{jobs}.mp4", crf=crf,
                n_jobs=jobs, **_ANIM,
            )  # fmt: skip
            data = out.read_bytes()
            sizes[crf] = len(data)
            count = _mp4_count(out)
            print(
                f"n_jobs={jobs} crf={crf}: {sizes[crf]} bytes, {count} frames, "
                f"x264 settings say crf={crf}: {f'crf={crf}.0'.encode() in data}"
            )
            assert count == 3 and f"crf={crf}.0".encode() in data
        assert sizes[40] < sizes[10]
    for jobs in (1, 2):
        data = bn.animate_network(
            CENTERS, _frames(), tmp_path / f"fs{jobs}.mp4", faststart=True,
            n_jobs=jobs, **_ANIM,
        ).read_bytes()  # fmt: skip
        moov, mdat = data.find(b"moov"), data.find(b"mdat")
        print(f"faststart n_jobs={jobs}: moov at {moov}, mdat at {mdat}")
        assert 0 < moov < mdat
    for kw in ({"crf": 23}, {"faststart": True}):
        with pytest.raises(ValueError, match="mp4 only"):
            bn.animate_network(CENTERS, _frames(), tmp_path / "x.gif", **kw, **_ANIM)


@pytest.mark.parametrize("n_jobs", [0, 1.5, True, "2"])
def test_n_jobs_validation(tmp_path, n_jobs):
    with pytest.raises(ValueError, match="n_jobs") as err:
        bn.animate_network(
            CENTERS, _frames(), tmp_path / "v.mp4", n_jobs=n_jobs, **_ANIM
        )
    print(err.value)
    assert repr(n_jobs) in str(err.value)
    assert not (tmp_path / "v.mp4").exists()
