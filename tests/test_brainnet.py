"""Tests for htfa.brainnet: BrainNet Viewer files, meshes, rendering, animation.

Everything renders for real (pyvista off screen) and writes real files. The
mesh tests download BrainNet Viewer 1.43 from NITRC on first use (cached in
``htfa.brainnet.mesh_cache_dir()``) and skip only when that download fails.
The 2014 comparison reads the original BrainNet frames from a local copy of
the directed-forgetting-network-dynamics repository and skips without it.
"""

import os
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


# ------------------------------------------------------------------ panels


def test_image_diff_known_values(tmp_path):
    a = Image.fromarray(np.zeros((10, 16, 3), np.uint8))
    b = Image.fromarray(np.full((10, 16, 3), 30, np.uint8))
    a.save(tmp_path / "a.png")
    assert bn.image_diff(tmp_path / "a.png", a) == 0.0
    assert bn.image_diff(a, b) == pytest.approx(30.0)


def test_annotate_pair_and_montage(tmp_path):
    left = bn.plot_network(CENTERS, _conn(), mesh=None)
    right = bn.plot_network(CENTERS, -_conn(), mesh=None)
    pair = bn.annotate_pair(
        left, right, ("Original\nsaved files", "Reconstruction"), "TR 3", width=1200
    )
    pair.save(tmp_path / "pair.png")
    box = bn._content_box([left.convert("RGB"), right.convert("RGB")])
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
    pos = _midpoint_window(a, r.global_centers, 0, 1)
    neg = _midpoint_window(a, r.global_centers, 2, 3)
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
    first = _midpoint_window(frames[0], r.centers[1], 0, 1)
    second = _midpoint_window(frames[1], r.centers[1], 0, 1)
    print("edge 0-1 red/blue px, frame 1:", _red(first).sum(), _blue(first).sum(),
          "frame 2:", _red(second).sum(), _blue(second).sum())  # fmt: skip
    assert _red(first).sum() >= 10 and _blue(first).sum() == 0
    assert _blue(second).sum() >= 10 and _red(second).sum() == 0


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
    # the same image as the effective nodes drawn unsnapped with their widths
    same = bn.plot_network(c_eff, sizes=w_eff, mesh=None, snap=False)
    assert bn.image_diff(snapped, same) == 0.0


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
    """fade_by_mass lightens a node mostly outside the mask; errors without snap."""
    mask = _sphere(40.0, 2.0)
    # three nodes so that node_radius's min-max mapping cannot hide the scaling
    centers = np.array([[0.0, 50.0, 0.0], [0.0, -10.0, 0.0], [0.0, 0.0, 40.0]])
    widths = np.array([100.0, 100.0, 100.0])
    c_eff, _, info = bn.effective_nodes(centers, widths, mask)
    print("mass fractions", info["mass_fraction"])
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
    # the interior node keeps full opacity
    assert np.array_equal(_window(faded, -10, 0), _window(plain, -10, 0))
    scaled = bn.plot_network(
        centers, widths=widths, mask=mask, mesh=None, scale_by_mass=True
    )
    assert bn.image_diff(scaled, Image.fromarray(plain.astype(np.uint8))) > 0
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
