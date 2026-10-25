# htfa API guide

This guide covers the public API of htfa 0.1.0 as it is on `main`. The full
Sphinx site with tutorials is still being written; until then, this page and
the docstrings (`help(htfa.fit)`, `help(htfa.HTFA)`, ...) are the reference.

The API may still change before the first tagged release, so pin a commit:

```bash
pip install "htfa @ git+https://github.com/ContextLab/htfa@<commit-sha>"
```

## Quick start: a folder of fMRI runs

```python
import htfa

# One .nii / .nii.gz file per subject, e.g. sub-01_bold.nii.gz, sub-02_bold.nii.gz, ...
res = htfa.fit("path/to/folder", K=50, random_state=0)

res.global_centers  # (K, 3) template centers, in world (mm) coordinates
res.global_widths  # (K,)   template widths, w = 2 * sigma**2
res.centers[0]  # (K, 3) subject 0's centers
res.widths[0]  # (K,)   subject 0's widths
res.weights[0]  # (K, T) subject 0's factor time courses
res.subjects[0].source  # the file subject 0 was read from
```

`htfa.fit` loads the inputs with `htfa.load` and then fits `htfa.HTFA` when
there are several subjects, or `htfa.TFA` when there is one.

### How a folder is read

- Each `.nii` or `.nii.gz` file is one subject. A 4D file is that subject's
  time series. A 3D file is one subject with a single sample.
- Files are taken in natural filename order (`sub-2` before `sub-10`). The
  extension check ignores case. Dot files and other files (`.json` sidecars,
  `.tsv`, ...) are skipped.
- The folder is not searched recursively. For a BIDS dataset, pass a list of
  paths instead:

  ```python
  from pathlib import Path

  runs = sorted(Path("bids_root").glob("sub-*/func/*_task-rest_bold.nii.gz"))
  res = htfa.fit([str(p) for p in runs], K=50, random_state=0)
  ```

- Voxel coordinates are world coordinates computed from each file's affine,
  so they are in mm for standard NIfTI files.
- With the default `mask=None`, a voxel is kept when it is finite and not zero
  at every time point. Pass `mask=` to choose voxels yourself: a path, a nibabel
  image, a fitted nilearn `NiftiMasker`, a boolean array, or `'epi'` /
  `'background'` (nilearn's automatic masks). If a NaN falls inside an explicit
  mask, `load` raises an error.
- Subjects can have different numbers of voxels and of time points.
  `ValueError` is raised only when they differ in coordinate dimensionality
  (for example, 2D images mixed with NIfTI).
- `stack=True` treats every 3D volume in the folder as one sample of a
  *single* subject's time series.

### Other inputs

`htfa.fit` and `htfa.load` accept these inputs. In a list, each element is
one subject:

| Input | Treated as |
|-|-|
| Path to a `.nii` / `.nii.gz` file | One subject |
| Path to a folder | One subject per NIfTI or image file (as above) |
| nibabel image (e.g. from nilearn) | One subject |
| `(X, Y, Z, T)` / `(X, Y, Z)` / `(H, W)` numpy array, no `coords` | Image grid; coordinates are voxel indices times `spacing` |
| `(V, T)` numpy array plus `coords=(V, D)` | Voxel by sample data at the given coordinates |
| `.png` / `.jpg` / `.tif` image | One subject with one sample (TIFF pages become samples) |
| List of any of the above | One subject per element; `coords` may be a matching list |

You need the `images` extra for 2D images and the `nilearn` extra for nilearn
masks. If they are missing, they are installed with pip on first use. To turn
that off, set `HTFA_AUTO_INSTALL=0` or call `htfa.set_autoinstall(False)`.

## `htfa.fit(data, K=50, coords=None, mask=None, stack=False, channels="gray", spacing=None, **model_kwargs)`

`data`, `coords`, `mask`, `stack`, `channels` and `spacing` are passed to
`htfa.load`. The other keyword arguments go to `HTFA` (several subjects) or
`TFA` (one subject). Unknown keywords raise `TypeError`. HTFA-only keywords,
such as `max_global_iter`, fail when there is only one subject. The function
returns an `HTFAResults`.

Parameters you are most likely to set:

| Parameter | Default | Meaning |
|-|-|-|
| `K` | 50 | Number of factors |
| `standardize` | `'auto'` | `'auto'` z-scores each voxel over time when T > 1. Keep it on for fMRI; without it the fit is dominated by anatomical intensity differences. `False` fits raw values |
| `random_state` | `None` | Seed for the template subject, the initialization and the subsampling. Set it for reproducible fits |
| `max_global_iter` (HTFA) | 10 | Template updates. The fit often warns `ConvergenceWarning` at 10; raising it (e.g. 30) lets the template settle |
| `max_local_iter` (HTFA) | 10 | Iterations of each subject's TFA per template update |
| `max_iter` (TFA) | 100 | Iterations for a single-subject fit |
| `threshold` | 1e-3 | Convergence threshold on the scaled change in centers and widths |
| `voxel_ratio`, `tr_ratio` | 1.0 | Fraction of voxels / time points drawn (anew each iteration) for the center and width solve. Values below 1 speed up large fits; the weights always use all data |
| `max_voxels`, `max_trs` | `None` | Absolute caps on those subsample sizes |
| `init` | `'auto'` | Starting points tried for the template subject: `'ncorr'`, `'power'`, `'kmeans'` (BrainIAK's), or a list. The one with the lowest final objective is kept |
| `n_screen_iter` | 0 | If > 0, run each start for this many iterations and continue only the best one. Faster, but may pick a different start of equal quality |
| `backend` | `'auto'` | `'numpy'`, `'torch'`, `'cuda'`, `'mps'` or `'cpu'`. `'auto'` uses a GPU only for large problems when torch is installed |
| `n_jobs` (HTFA) | `None` | joblib workers for the per-subject fits (numpy backend) |
| `verbose` | `False` | Print per-iteration progress |

Fitting time on real whole-brain data has not been measured yet; that
benchmark is in progress. For a first run on large images, consider a smaller
`K`, a brain mask, or `max_voxels` (for example `max_voxels=5000`) to bound
the cost of each solve.

## `HTFAResults`

The object returned by `htfa.fit`:

| Member | Returns |
|-|-|
| `model` | The fitted `TFA` or `HTFA` estimator |
| `subjects` | List of `htfa.io.Subject` (data, coords, affine, mask, source) |
| `centers`, `widths`, `weights` | Per-subject lists: `(K, D)`, `(K,)`, `(K, T)` |
| `global_centers`, `global_widths` | HTFA template, or `None` for a single-subject TFA fit |
| `factors(subject=0)` | Factor images `(V, K)` at the subject's voxels |
| `reconstruct(subject=0)` | `(V, T)` reconstruction, in the standardized (z-scored) space of the fit |
| `to_nifti(subject=0, what="reconstruction")` | NIfTI image of the reconstruction or (`what="factors"`) of the factor images, NaN outside the mask |
| `to_image(subject=0, what=...)` | The same as a numpy array in image space |
| `connectivity(subject=0)` | `(K, K)` Pearson correlation of the factor time courses |
| `dynamic_connectivity(subject=0, window=20)` | `(T - window + 1, K, K)` sliding-window connectivity |
| `isfc(window=None)` | Inter-subject factor correlation `(K, K)`, or sliding-window with `window`; needs equal `T` across subjects |
| `plot_nodes(subject="global")`, `plot_connectivity(subject=0)`, `plot_reconstruction(subject=0, sample=0)` | Plots (need the `plot` extra) |
| `plot_network(subject="global", connectivity=None, **kw)`, `animate_network(out, subject="global", window=20, connectivity=None, **kw)` | BrainNet Viewer-style 3D network image / mp4 or GIF (need the `brain3d` extra; see below). Connectivity defaults to `isfc` for `"global"` and to `connectivity` / `dynamic_connectivity` for a subject |

Factors are in template order for HTFA fits. Factor `k` of every subject
corresponds to global factor `k`.

## Estimators

`htfa.TFA` and `htfa.HTFA` are scikit-learn style estimators. Use them
directly when you already have arrays:

```python
from htfa import HTFA

# Xs: list of (V_s, T_s) arrays; Rs: list of (V_s, D) coordinate arrays
model = HTFA(K=20, max_global_iter=30, random_state=0).fit(Xs, Rs)
model.global_centers_, model.global_widths_  # template
model.global_centers_cov_  # (K, D, D) posterior covariance
model.global_widths_var_  # (K,) posterior variance
model.centers_, model.widths_, model.weights_  # per-subject lists
model.n_iter_, model.converged_
```

```python
from htfa import TFA

model = TFA(K=10, random_state=0).fit(X, R)  # X: (V, T), R: (V, D)
model.centers_, model.widths_, model.weights_
model.factors(), model.reconstruct()
model.converged_reason_  # 'threshold', 'plateau' or 'max_iter'
model.init_, model.init_objectives_
```

`TFA.fit(X, coords, prior=None, start=None)` also accepts an explicit start,
`start=(centers, widths)`, with shapes `(K, D)` and `(K,)`.

## Connectivity functions

These work on any `(K, T)` weight matrices:

- `htfa.static_connectivity(weights)`: `(K, K)` Pearson correlation.
- `htfa.dynamic_connectivity(weights, window)`: `(T - window + 1, K, K)`.
- `htfa.isfc(list_of_weights, window=None)`: correlation of each subject's
  factor time courses with the mean of the others. The result is symmetrized
  and averaged over subjects in Fisher-z space.

## BrainNet Viewer-style network plots

`htfa.plot_network` and `htfa.animate_network` draw factor centers as gray
spheres inside a translucent brain surface and edges with `|E| > threshold`
as tubes, red for positive and blue for negative weights. The drawing rules
follow BrainNet Viewer 1.43 (Xia, Wang & He, 2013, PLoS ONE 8: e68910):
white background, mesh 0.95 gray at opacity 0.2, node radii scaled linearly
onto 2-7 mm from the node sizes (factor widths for a fit), edge diameter
`|E|` mm at opacity 0.7, orthographic left lateral view. Rendering is off
screen with pyvista (the `brain3d` extra). Centers must be `(K, 3)`
coordinates in MNI mm for the nodes to sit inside the brain surface.

```python
import htfa

res = htfa.fit("path/to/folder", K=50, max_global_iter=30, random_state=0)
img = res.plot_network(threshold=0.25, out="network.png")  # PIL image, 1600 x 1000
res.animate_network("network.mp4", subject=0, window=20, fps=10)

# arbitrary nodes and connectivity
htfa.plot_network(centers, E, sizes=widths, view="dorsal", out="dorsal.png")
htfa.animate_network(
    centers, E_t, "movie.gif", captions=[f"TR {t}" for t in range(len(E_t))]
)
```

| Argument | Default | Meaning |
|-|-|-|
| `nodes` | | `HTFAResults`, fitted `TFA`/`HTFA`, or `(K, 3)` centers |
| `connectivity` | `None` | `(K, K)` for `plot_network` (`None`: nodes only), `(T, K, K)` for `animate_network` |
| `sizes` | fit widths, else equal | Node sizes, mapped onto 2-7 mm radii |
| `subject` | `"global"` | Which centers of a fit to draw |
| `mesh` | `"ch2"` | `"ch2"` (Ch2 with cerebellum), `"icbm152"`, `"icbm152_smoothed"`, a `.nv` path, `(vertices, faces)`, or `None` |
| `threshold` | 0.25 | Edges with `\|E\| > threshold` are drawn |
| `view` | `"lateral"` | `"lateral"`/`"left"`, `"right"`, `"dorsal"`, `"ventral"`, `"anterior"`, `"posterior"` |
| `edge_style` | `"intensity"` | `"intensity"` (diameter `\|E\|` mm, opacity 0.7) or `"flat"` (0.5 mm, opaque) |
| `size` | `(1600, 1000)` | Image size in pixels |
| `positive_color`, `negative_color`, `node_color`, `mesh_color`, `mesh_opacity` | red, blue, 0.502 gray, 0.95 gray, 0.2 | Colors |
| `out` | `None` | `plot_network`: save PNG/JPEG. `animate_network`: required, `.mp4` (H.264) or `.gif` |
| `fps`, `captions`, `frames_dir` | 10, `None`, `None` | `animate_network` only: frame rate, one caption per frame (drawn in a band above the brain), folder for per-frame PNGs |

In a GIF, Pillow merges consecutive identical frames and adds up their
durations, so the playing time is unchanged but `n_frames` can be smaller
than `T`.

The brain meshes are BrainNet Viewer's own. BrainNet Viewer is distributed
under the GNU GPL, so htfa does not include them: on first use htfa downloads
BrainNet Viewer 1.43 (21 MB) from NITRC, checks SHA-256 checksums, and keeps
the three meshes in `$HTFA_DATA_DIR/brainnet`, or by default
`~/.cache/htfa/brainnet` (`%LOCALAPPDATA%\htfa\brainnet` on Windows). Without
network access and without a cached copy, `fetch_mesh` raises
`ConnectionError`.

`htfa.brainnet` also has readers and writers for BrainNet files
(`read_node`/`write_node`, `read_edge`/`write_edge`, `read_nv`/`write_nv`),
`fetch_mesh`, `node_radius`, `write_animation(images, out, fps)`,
`image_diff(a, b)`, and two annotated layouts that keep text off the brain:
`annotate_pair(left, right, titles, caption)` (two frames side by side under
a header band) and `montage(images, row_labels, col_labels, title)` (a grid).

## Synthetic data

`htfa.make_tfa_data(grid_shape, n_samples, K, ...)` and
`htfa.make_htfa_data(n_subjects, grid_shape, n_samples, K, ...)` return
`(X, R, truth)` and `(Xs, Rs, truth)`. The `truth` dict holds the generating
centers, widths and weights; `make_htfa_data` adds `global_centers` and
`global_widths`. The synthetic data are already on a z-scored scale, so fit
them with `standardize=False`. `htfa.synthetic.match_factors(true_centers,
est_centers)` returns the permutation that aligns estimated factors with the
true ones.

```python
import htfa
from htfa.synthetic import match_factors

Xs, Rs, truth = htfa.make_htfa_data(
    n_subjects=3, grid_shape=(12, 12, 12), n_samples=60, K=5, random_state=0
)
res = htfa.fit(
    Xs, K=5, coords=Rs, standardize=False, max_global_iter=30, random_state=0
)
perm = match_factors(truth["centers"][0], res.centers[0])
abs(res.centers[0][perm] - truth["centers"][0]).max()  # about 0.02 grid units
```
