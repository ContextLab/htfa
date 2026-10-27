"""Render the README and documentation cover image (light and dark variants).

A BrainNet Viewer-style network of the HTFA template fitted to the first 12
adults of nilearn's ``development_fmri`` movie-watching dataset (K = 50; the
saved fit of tutorial 10, ``htfa.datasets.fetch_tutorial_fit``, downloaded once
and cached)::

    .venv/bin/python scripts/make_cover.py

- Nodes: the 50 template factors, drawn at their in-brain centroids
  (`htfa.brainnet.effective_nodes` with nilearn's MNI152 brain mask), with
  radii mapped from the effective widths onto 2.5-7 mm (`node_radius`).
- Edges: static inter-subject functional correlation of the factor weights
  (`htfa.isfc`, the whole 168-volume run, averaged over subjects in Fisher-z
  space). The strongest ``TOP_FRACTION`` of the 1225 factor pairs by ``|ISFC|``
  are drawn; tube radius and opacity grow with ``|ISFC|``. Positive edges are
  warm, negative edges cool.
- Node colours: communities of the positive ISFC graph (spectral clustering
  with ``N_COMMUNITIES`` clusters, ``random_state=0``).
- Views: left lateral, dorsal and right lateral (orthographic), on BrainNet
  Viewer's Ch2 mesh (downloaded from NITRC on first use, see
  `htfa.brainnet.fetch_mesh`), on a transparent background.

Writes ``docs/_static/cover_light.png`` and ``docs/_static/cover_dark.png``
(``WIDTH`` pixels wide) and prints the settings, the edge threshold and the
htfa commit used.
"""

import json
import subprocess
from pathlib import Path

import numpy as np
from sklearn.cluster import SpectralClustering

import htfa
from htfa import brainnet

pv = htfa.require("pyvista")
Image = htfa.require("PIL.Image")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "_static"
FIT = "development_fmri_12adults_K50"
TOP_FRACTION = 0.05  # share of factor pairs drawn as edges, by |ISFC|
N_COMMUNITIES = 5
WIDTH = 2400  # final image width in pixels
VIEW_HEIGHT = 1000  # render height of each view, before cropping and scaling
SUPERSAMPLE = 2
RADII = (2.5, 7.0)  # node radii (mm) of the narrowest and widest factor
EDGE_RADII = (0.35, 1.3)  # tube radii (mm) at the threshold and at max |ISFC|
GAP = 40  # pixels between views in the final image

# Categorical node colours (ContextLab green first; no reds or blues, which
# mark the edge signs), and edge colours.
THEMES = {
    "light": {
        "nodes": ["#007030", "#E8A31B", "#7E57C2", "#0097A7", "#5D6D7E"],
        "mesh": "#9AA3AB",
        "mesh_opacity": 0.16,
        "positive": "#E8202A",
        "negative": "#3A52A4",
    },
    "dark": {
        "nodes": ["#4CAF50", "#FFC24B", "#B39DDB", "#4DD0E1", "#B0BEC5"],
        "mesh": "#8B949E",
        "mesh_opacity": 0.10,
        "positive": "#FF5A5F",
        "negative": "#6F8CFF",
    },
}
# Views, left to right (camera directions from htfa.brainnet.VIEWS).
VIEWS = ("left", "dorsal", "right")


def commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def network() -> dict:
    """Node positions, sizes, communities and the drawn edges of the fit."""
    fit = htfa.datasets.fetch_tutorial_fit(FIT)
    weights = list(fit["weights"])
    K = len(fit["global_centers"])
    C = htfa.isfc(weights)
    centers, widths, _ = brainnet.effective_nodes(
        fit["global_centers"], fit["global_widths"], brainnet.mni_mask()
    )
    i, j = np.triu_indices(K, 1)
    n_edges = int(round(TOP_FRACTION * len(i)))
    order = np.argsort(-np.abs(C[i, j]))[:n_edges]
    i, j = i[order], j[order]
    w = C[i, j]
    affinity = np.clip(C, 0, None)
    np.fill_diagonal(affinity, 0)
    labels = SpectralClustering(
        N_COMMUNITIES, affinity="precomputed", random_state=0
    ).fit_predict(affinity)
    # Number communities by size, largest first, so colours are stable.
    sizes = np.bincount(labels, minlength=N_COMMUNITIES)
    rank = np.argsort(np.argsort(-sizes, kind="stable"), kind="stable")
    return {
        "centers": centers,
        "radii": brainnet.node_radius(widths, *RADII),
        "labels": rank[labels],
        "edges": (i, j, w),
        "threshold": float(np.abs(w).min()),
        "max_abs": float(np.abs(w).max()),
        "fit_commit": str(fit["htfa_commit"]),
        "settings": json.loads(str(fit["settings"])),
        "n_subjects": len(weights),
    }


def render(net: dict, mesh: tuple, view: str, theme: dict) -> "Image.Image":
    """One view as an RGBA image, cropped to its content."""
    verts, faces = mesh
    s = SUPERSAMPLE
    pl = pv.Plotter(
        off_screen=True,
        window_size=[int(VIEW_HEIGHT * 1.6) * s, VIEW_HEIGHT * s],
        lighting="light kit",
    )
    pl.render_window.SetMultiSamples(0)
    cells = np.c_[np.full(len(faces), 3), faces].ravel()
    brain = pv.PolyData(verts, cells).compute_normals()
    pl.add_mesh(
        brain,
        color=theme["mesh"],
        opacity=theme["mesh_opacity"],
        smooth_shading=True,
        ambient=0.35,
        diffuse=0.7,
        specular=0.0,
    )
    centers, radii, labels = net["centers"], net["radii"], net["labels"]
    for k, color in enumerate(theme["nodes"]):
        spheres = [
            pv.Sphere(radius=r, center=c, theta_resolution=48, phi_resolution=48)
            for c, r, lab in zip(centers, radii, labels, strict=True)
            if lab == k
        ]
        if spheres:
            pl.add_mesh(
                pv.merge(spheres),
                color=color,
                smooth_shading=True,
                ambient=0.25,
                diffuse=0.8,
                specular=0.35,
                specular_power=20,
            )
    i, j, w = net["edges"]
    lo, hi = net["threshold"], net["max_abs"]
    strength = (np.abs(w) - lo) / max(hi - lo, 1e-12)  # 0 at threshold, 1 at max
    r0, r1 = EDGE_RADII
    for a, b, v, t in zip(i, j, w, strength, strict=True):
        tube = pv.Tube(pointa=centers[a], pointb=centers[b], radius=r0 + t * (r1 - r0))
        pl.add_mesh(
            tube,
            color=theme["positive"] if v > 0 else theme["negative"],
            opacity=0.45 + 0.5 * t,
            smooth_shading=True,
            ambient=0.3,
            diffuse=0.7,
            specular=0.2,
        )
    pl.enable_depth_peeling(number_of_peels=12)
    pl.enable_parallel_projection()
    direction, up = brainnet.VIEWS[view]
    center = (verts.min(axis=0) + verts.max(axis=0)) / 2
    pl.camera.focal_point = center
    pl.camera.position = center + 500.0 * np.asarray(direction)
    pl.camera.up = up
    pl.camera.parallel_scale = 0.55 * float(np.ptp(verts, axis=0).max())
    pl.camera.clipping_range = (1, 1000)
    shot = pl.screenshot(None, transparent_background=True, return_img=True)
    pl.close()
    img = Image.fromarray(np.ascontiguousarray(shot))
    box = img.getchannel("A").point(lambda a: 255 if a > 4 else 0).getbbox()
    if box is None:
        raise RuntimeError(f"empty render for view {view!r}")
    pad = 6 * s
    left, top, right, bottom = box
    if left < pad or top < pad or right > img.width - pad or bottom > img.height - pad:
        raise RuntimeError(f"view {view!r} touches the image border: {box}")
    return img.crop((left - pad, top - pad, right + pad, bottom + pad))


def compose(views: list) -> "Image.Image":
    """Views side by side, centered vertically, ``WIDTH`` pixels wide."""
    h = max(v.height for v in views)
    w = sum(v.width for v in views) + GAP * SUPERSAMPLE * (len(views) - 1)
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    x = 0
    for v in views:
        canvas.alpha_composite(v, (x, (h - v.height) // 2))
        x += v.width + GAP * SUPERSAMPLE
    scale = WIDTH / canvas.width
    return canvas.resize((WIDTH, round(canvas.height * scale)), Image.LANCZOS)


def save(img: "Image.Image", path: Path) -> None:
    # Quantize to a 256-colour palette with alpha: a much smaller PNG with no
    # visible banding for these flat-shaded colours.
    img.quantize(256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE).save(
        path, optimize=True
    )


def main() -> None:
    net = network()
    mesh = brainnet.load_mesh("ch2")
    i, _, w = net["edges"]
    print(f"fit: {FIT} ({net['n_subjects']} subjects), settings {net['settings']}")
    print(f"fit made at htfa commit {net['fit_commit']}; rendered at {commit()}")
    print(
        f"edges: top {TOP_FRACTION:.0%} of pairs by |ISFC| = {len(i)} edges, "
        f"|ISFC| >= {net['threshold']:.3f} (max {net['max_abs']:.3f}); "
        f"{(w > 0).sum()} positive, {(w < 0).sum()} negative"
    )
    print("community sizes:", np.bincount(net["labels"]).tolist())
    for name, theme in THEMES.items():
        img = compose([render(net, mesh, v, theme) for v in VIEWS])
        path = OUT / f"cover_{name}.png"
        save(img, path)
        print(
            f"{path.relative_to(ROOT)}: {img.width} x {img.height} px, "
            f"{path.stat().st_size / 1e3:.0f} kB"
        )


if __name__ == "__main__":
    main()
