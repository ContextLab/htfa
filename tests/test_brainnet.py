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
    img = bn.plot_network(CENTERS, _conn(0.3), mesh=None, out=out, threshold=0.35)
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
    a2 = _rgb(bn.plot_network(CENTERS, _conn(0.3), mesh=None, threshold=0.25))
    assert _red(_window(a2, -10, -40)).sum() >= 5
    # strict inequality: |E| == threshold is not drawn
    a3 = _rgb(bn.plot_network(CENTERS, _conn(0.3), mesh=None, threshold=0.3))
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
        frames_dir=tmp_path / "frames",
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


def test_results_plot_network_matches_function(res_htfa, tmp_path):
    img = res_htfa.plot_network(mesh=None, threshold=0.0, out=tmp_path / "g.png")
    ref = htfa.plot_network(
        res_htfa.global_centers, res_htfa.isfc(), sizes=res_htfa.global_widths,
        mesh=None, threshold=0.0,
    )  # fmt: skip
    assert bn.image_diff(img, ref) == 0.0
    sub = res_htfa.plot_network(subject=1, mesh=None, threshold=0.0)
    ref1 = htfa.plot_network(
        res_htfa.centers[1], res_htfa.connectivity(1), sizes=res_htfa.widths[1],
        mesh=None, threshold=0.0,
    )  # fmt: skip
    assert bn.image_diff(sub, ref1) == 0.0
    # estimators are accepted directly
    via_model = bn.plot_network(res_htfa.model, res_htfa.isfc(), mesh=None, threshold=0)
    assert bn.image_diff(via_model, ref) == 0.0
    assert bn.image_diff(img, sub) > 0.05


def test_results_animate_network(res_htfa, tmp_path):
    out = res_htfa.animate_network(tmp_path / "s0.gif", subject=0, window=20, mesh=None)
    with Image.open(out) as g:
        assert g.n_frames == 30 - 20 + 1
    out = res_htfa.animate_network(
        tmp_path / "g.gif", window=25, mesh=None, frames_dir=tmp_path / "g"
    )
    assert len(list((tmp_path / "g").iterdir())) == 30 - 25 + 1
    # Pillow merges identical consecutive GIF frames (here: windows with no
    # edge above threshold) and adds up their durations
    with Image.open(out) as g:
        total = 0
        for k in range(g.n_frames):
            g.seek(k)
            total += g.info["duration"]
    print("gif frames", g.n_frames, "total ms", total)
    assert total == (30 - 25 + 1) * 100


def test_tfa_results_global_falls_back_to_subject_0(tmp_path):
    X, R, _ = make_tfa_data((8, 8, 8), n_samples=20, K=3, random_state=2)
    res = _fit(X, coords=R * 10.0 - 35.0, K=3, max_iter=5, random_state=0)
    img = res.plot_network(mesh=None, threshold=0.0)
    ref = htfa.plot_network(
        res.centers[0], res.connectivity(0), sizes=res.widths[0], mesh=None,
        threshold=0.0,
    )  # fmt: skip
    assert bn.image_diff(img, ref) == 0.0
    with pytest.raises(ValueError, match="no global template"):
        bn.plot_network(res, mesh=None, subject="global")


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
