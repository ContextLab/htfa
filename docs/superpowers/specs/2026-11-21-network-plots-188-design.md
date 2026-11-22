# Network plots and animations for issue #188: design

Date: 2026-11-21. Issue: ContextLab/htfa#188. Target version: 0.2.0.
Branch: `epic/network-plots-188`.

## Intent

The timecorr tutorial `docs/tutorial/htfa_network_animations.ipynb` renders
four 1920 x 1080, 30 fps movies of brain networks. It carries about 150 lines
of rendering code and reaches into the private `htfa.brainnet._Scene` and
`_SUPERSAMPLE`, because `plot_network` and `animate_network` in htfa 0.1.0
cannot draw what it needs. After this change the tutorial can make the same
movies through htfa's public functions alone.

## Goals

1. One color per node in `plot_network` and `animate_network`.
2. Node sizes and node colors that change from frame to frame.
3. Several views composed in one frame.
4. A legend of colored dots and lines with labels.
5. Frames rendered in parallel workers.
6. Encoding options (`crf`, `faststart`) for MP4 output.
7. A public `content_box` crop helper.
8. A `compose=` callable on `animate_network` for callers who want their own
   frame layout.
9. Version 0.2.0 prepared: version bump, changelog, API docs, htfa tutorials
   10 and 11, README.

## Non-goals

- Publishing to PyPI. That waits for the repo owner's explicit go.
- Editing the timecorr repository. Its session does that once 0.2.0 exists.
- Node positions that change over time.
- A public scene class. `_Scene` stays private.
- An x264 `preset` option (the tutorial uses `slow`; the issue does not ask
  for it).

## Anti-goals

- Output of existing calls must not change. A call that passes none of the
  new arguments takes the same code path and gives the same pixels as 0.1.0.
- No silent black or blank frames in a movie rendered by workers.
- No leaked plotters, worker processes or temporary frame folders, on
  success or on error.

## Constraints

- Project rules: no mocks, a test for every function, no dead code, no
  duplicated code, minimal change.
- One `pyvista` plotter is one camera. Several views means one scene per view.
- `_headless_window()` probes GL and changes `os.environ`; each worker
  process runs it for itself.
- joblib is already a core dependency. VTK off-screen rendering is not
  assumed thread-safe, so workers are processes.
- imageio's `quality` and an explicit `-crf` must not both be passed.

## Approaches

1. **Everything through keyword arguments** of `plot_network` and
   `animate_network`. One call makes the movie and the public surface stays
   small, but htfa owns the frame layout and callers cannot change it.
2. **Public building blocks** (a scene class with `draw`, a legend function,
   a frame renderer) with `animate_network` as a thin wrapper. Full control
   for callers, at the price of a much larger public API to keep stable.
3. **Approach 1 plus a `compose=` callable** that receives the per-view
   images of a frame and returns the finished frame.

The repo owner picked **approach 3** (2026-11-21): the small surface of
approach 1, with one escape hatch for custom layouts.

## Design

### Node colors (`node_color` style key)

`node_color` accepts:

- one color (name, hex string, or 3 or 4 numbers), as today;
- K colors: a sequence of K color specs, or an array of shape (K, 3) or
  (K, 4);
- in `animate_network` only, per-frame colors: a (T, K) nested sequence of
  color specs or an array of shape (T, K, 3) or (T, K, 4).

Rule for the ambiguous case: a flat sequence of 3 or 4 numbers is always one
color. Three nodes with numeric colors need a (3, 3) array. Anything else
that is not one of the shapes above raises `ValueError` naming the shape
received and K.

Per-frame colors are not blended; a frame uses the colors of the nearest
timepoint. A fourth (alpha) channel is dropped, as `_rgb` does for a single
color today; node opacity stays with `fade_by_mass`.

In component-lobe mode, K colors and per-frame colors or sizes raise
`ValueError`, as `sizes` does there today.

### Node sizes

`sizes` in `animate_network` accepts (K,) as today or (T, K). `plot_network`
accepts (K,) only. (1, K) and (K, 1) arrays, which pass silently today
because of a `ravel`, keep working as (K,).

Radii come from `node_radius` applied to the whole (T, K) array at once, so
sizes are comparable across frames. A caller who wants per-frame scaling
rescales before passing, or passes radii in mm with `radius_kb=(1, 0)`.

Between timepoints, sizes follow the `interpolation` argument ("linear" or
"nearest"), the same as edges. `scale_by_mass` multiplies each frame's sizes.
Edges start at the sphere surfaces of the frame's own radii.

`connectivity` in `htfa.brainnet.animate_network` stays a required
positional argument but may be `None` when `sizes` or `node_color` carries
the time axis; then no edges are drawn. When more than one argument has a
time axis, their T must agree, else `ValueError`; a `captions` sequence must
have that length too. If nothing has a time axis, `ValueError`.
`HTFAResults.animate_network` is unchanged: there `None` still means
"compute ISFC", so a movie without edges goes through the function.

### Views

New keyword `views` on both functions: a sequence of view names from
`VIEWS`, each either a name or a `(name, title)` pair. Default titles:
left and lateral "Left lateral", right "Right lateral", and the capitalized
name for the others. An empty title hides it. With a single view there is
no title unless one is given. Passing both `view` and `views` raises
`ValueError`.

The layout path is taken when `views`, `legend` or `frame_size` is given.
Otherwise the 0.1.0 path runs unchanged, including captions drawn by
`_with_header`. In the layout path each view is rendered at `size` (the
existing style key), then the frame is built:

- one crop box for all views and all frames, from `content_box` over a
  reference render of every view (the scene without edges, nodes at their
  largest radius), so the brain does not move between frames;
- cropped views side by side, laid out as `annotate_pair` does today:
  titles above the views, caption under the titles, a divider between
  views, the same fonts relative to the frame width;
- legend rows, if any, centered below.

`annotate_pair` and the new layout share one private helper for N views;
`annotate_pair` keeps its signature and its output.

New keyword `frame_size=(W, H)`: the composed frame is exactly this size.
Title, caption and legend bands are drawn at width W; the views are scaled
(Lanczos, up or down) to fit the space left and centered on the background
color.
Without `frame_size`, W is the summed width of the cropped views and the
height follows, trimmed to even width and height.

`spin` with `views` rotates every view; the views are then not cropped.

### Legend

New keyword `legend` on both functions: a list of `(color, label, kind)`
with `kind` "dot" or "line" for one row, or a list of such lists for several
rows. Geometry follows the tutorial's `legend_band`: font size
`max(12, round(30 * W / 1920))` for frame width W (the floor keeps text
readable in small frames), one band 64 px high per 30 px of font for each
row, entries centered as a group, black text, mark then
label, sizes proportional to the font size (gap 40, mark width 34, dot
radius 12, line width 6, mark-to-text 10, all per 30 px of font).

### `compose`

New keyword `compose` on `animate_network`: a callable
`compose(images, t) -> image`. `images` is the list of per-view PIL RGB
images of one frame (uncropped, at `size`), `t` the frame's time as a float.
Its return value is the frame as is: htfa adds no caption, titles or legend,
so passing `compose` together with `legend`, `captions` or `frame_size`
raises `ValueError`. Every frame must come back the same size.
With `n_jobs` other than 1 the callable runs in worker processes.

`plot_network` has no `compose`: a caller can render each view and combine
the images directly.

### Parallel rendering

New keyword `n_jobs=1` on `animate_network`. With 1, frames are drawn in
this process and streamed to the encoder exactly as today, with no new
checks. Otherwise (`effective_n_jobs(n_jobs)` worker processes):

- the pool belongs to this call. It is not joblib's shared reusable pool,
  so stopping it cannot disturb a running HTFA fit. Functions defined in a
  notebook must work as `compose`;
- frames are split into contiguous chunks;
- each task builds one scene per view, draws its chunk by replacing only
  spheres and edges, writes finished frames as lossless PNG into a
  temporary folder owned by this call (temporary name, then rename, so a
  file exists only when complete) and closes its scenes in `finally`;
- the parent takes the files in frame order and treats each as the
  sequential path treats a drawn frame: `jpeg_quality`, a copy into
  `frames_dir` under today's names (`1.png`, `2.png`, ... or `.jpg`), then
  the encoder. The temporary folder is removed on success and on error.

Blank-render check, in workers only: a view image in which more than
5 percent of pixels are near black (max channel below 10), while the
background color is not dark, is a failed render (the tutorial's rule).
The task raises a private `RuntimeError` subclass.

Restarts: on a blank render or a dead worker, the pool is stopped with its
workers, and only the missing frames are redrawn by a new pool, up to
5 restarts, then `RuntimeError`. Any other exception (for example from
`compose`, or a `ValueError`) is raised at once, without a retry.

New style key `supersample` (default 2, the present `_SUPERSAMPLE`), so
callers can trade quality for speed without touching a private global.

### Encoding

`write_animation(images, out, fps=10, *, crf=None, faststart=False)`.

- `crf`: integer 0 to 51. When given, imageio's `quality` is set to `None`
  and `-crf` is passed to ffmpeg. When `None`, encoding is as in 0.1.0.
- `faststart`: adds `-movflags +faststart`.
- Either option with a GIF output raises `ValueError`.

`animate_network` gains the same two keywords and forwards them.

### `content_box`

`content_box(images, margin=10, background="white")` returns
`(left, top, right, bottom)`: the bounding box of pixels that differ from
the background color by more than 5 in any channel, unioned over the images,
grown by `margin` and clipped to the image. An image that is all background
gives the whole image, as today. `images` is one image or a
sequence; each is a PIL image or a path. Images of different sizes raise
`ValueError`. Exported from `htfa.brainnet` and from `htfa`. The private
`_content_box` is renamed, not kept alongside.

### `HTFAResults`

`HTFAResults.plot_network` and `.animate_network` already forward
`**kwargs`; their docstrings name the new arguments.

### Version and docs

- `0.2.0` in `pyproject.toml` and `htfa/__init__.py`; the version pinned
  in `tests/test_package.py` follows.
- `docs/changelog.rst` (the existing changelog): a 0.2.0 section.
- `docs/api.rst`: add `content_box` (`docs/generated` is not tracked, so
  nothing else is committed for it). `docs/API.md` is updated if it lists
  these functions.
- Tutorial 11 (plotting and appearance): per-node colors, views, legend,
  `content_box`. Tutorial 10 (dynamic ISFC animation): per-frame sizes and
  colors, `n_jobs`, `crf` and `faststart`, `compose`. Both notebooks are
  re-executed with the data they use now, and their assets under
  `docs/_static/tutorials/` regenerated.
- README: one short example of the new arguments.

## Testing

Real rendering and real encoding, no mocks, in `tests/test_brainnet.py`:

- per-node colors: pixels of each given color are present; a flat 3-number
  color still means one color; bad shapes raise;
- (T, K) sizes: a node's pixel area differs between frames; joint scaling;
  mismatched T raises;
- (T, K) colors change between frames;
- `views`: frame holds one panel per view; `frame_size` gives exactly that
  size; `view` with `views` raises;
- `legend`: legend colors and rows are present below the views;
- `compose`: called with one image per view and the frame time; its return
  value is the frame; forbidden combinations raise;
- `n_jobs=2` frames match `n_jobs=1` frames (`image_diff` under tolerance);
  the temporary folder is gone afterwards, also after an error;
- the blank-render rule on real images: an all-black image fails, a normal
  render passes, a dark background is not checked;
- restart: a `compose` that ends its worker process the first time it sees
  one frame (a marker file on disk records that it did) still gives a
  complete movie; a `compose` that raises `ValueError` fails at once;
- `crf`: a higher value gives a smaller file; `faststart` puts the `moov`
  atom before `mdat`; both raise for GIF;
- `content_box` on synthetic images, including a non-white background.

Checks outside the suite, reported in the PR:

- existing calls give identical pixels on this branch and on `main`, on the
  same machine;
- rendered PNGs and one short movie looked at by eye;
- a clip in the style of the timecorr tutorial (two views, caption, two
  legend rows, per-node colors, per-frame radii, 1920 x 1080) made through
  the public API only;
- `make lint`, `make test`, `make docs`, and the suite in a fresh
  environment built from the committed tree.

## Left to the builder

- How spheres are replaced between frames (one merged mesh per color, or
  per-cell colors), as long as opacity from `fade_by_mass` still works and
  the scene's mesh is drawn once.
- Chunk size for parallel tasks.
- Names and placement of private helpers.
- The cause of black renders is not known here. The check and retry above
  are specified by their behavior; the builder should note any reproduction.
