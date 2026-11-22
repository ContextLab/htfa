# Network plots and animations (#188) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `plot_network` / `animate_network` per-node colors, per-frame sizes and colors, several views per frame, a legend, a `compose` callable, parallel rendering and MP4 encoding options, plus a public `content_box`, released as htfa 0.2.0.

**Architecture:** All rendering stays in `htfa/brainnet.py`. `_Scene` learns to replace its node spheres; a private `_Views` object holds one `_Scene` per view and draws one frame's view images; a private `_layout` function composes view images, titles, caption and legend rows and is shared with `annotate_pair`. `animate_network` either streams frames from one `_Views` (as today) or has worker processes, each with its own `_Views`, write frames to a temporary folder.

**Tech Stack:** Python 3.10+, numpy, pyvista (off-screen VTK), Pillow, imageio + imageio-ffmpeg, joblib (bundled loky), pytest.

**Spec:** `docs/superpowers/specs/2026-11-21-network-plots-188-design.md`. Read it before any task.

## Global Constraints

- A call that passes none of the new arguments takes the 0.1.0 code path and gives the same pixels as before.
- Real rendering and real encoding in tests. No mocks, no monkeypatching of htfa or library functions.
- No dead code, no duplicated code; reuse helpers already in `brainnet.py` (`_rgb`, `_open`, `_font`, `_center_text`, `_at_time`, `_nearest`, `_save`, `_jpeg`, `image_diff`).
- New tests go in `tests/test_brainnet.py`, reusing its fixtures and helpers. Never weaken or delete an existing test; if one fails, fix the code.
- Keep test renders small (`size=(320, 200)` unless a test needs more) so the suite stays fast.
- Before every commit: `make lint` and `.venv/bin/pytest tests/test_brainnet.py tests/test_headless.py -q` both pass. Never `git add -A`; add files by path. Do not push.
- Commit messages: `Issue #188: <description>`, ending with exactly these two lines:
  `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01DcKtyt2ULJKvi7Z1bbZU4G`
- American English in docs and docstrings; numpydoc docstrings as in the file.
- Errors for bad arguments are `ValueError` with the received value or shape in the message.

## Review Focus

1. Three or four nodes with numeric colors: a flat 3- or 4-number `node_color` is one color; a (3, 3) array is three colors (Task 3).
2. Odd widths or heights from the layout or from `compose`: MP4 frames are trimmed to even size, and `frame_size` with an odd number raises (Task 5).
3. GIF output with colored nodes: the movie is written and a node keeps roughly its color (Task 4).
4. More workers than frames, and a one-frame movie, with `n_jobs=2` (Task 7).
5. A legend row wider than the frame raises `ValueError` naming the row; a dark `background` with `views` still crops to the brain (Task 5).

---

### Task 1: public `content_box`

**Files:** Modify `htfa/brainnet.py` (`_content_box` and its callers), `htfa/__init__.py`, `docs/api.rst`, `tests/test_brainnet.py` (the existing `bn._content_box` call follows the rename).

**Interfaces:**
- Produces: `content_box(images: Any, margin: int = 10, background: Any = "white") -> tuple[int, int, int, int]`, in `htfa.brainnet.__all__` (if the module has one) and `htfa.__all__`.

- [ ] **Step 1: failing tests**
  - `test_content_box_white`: a 100 x 60 white RGB image with a (10, 10, 10) rectangle at x 20-39, y 30-44 gives `(10, 20, 50, 55)`; `margin=0` gives `(20, 30, 40, 45)`.
  - `test_content_box_single_image_path_and_union`: one image (not in a list) works; a path to a saved PNG works; two images give the union of their boxes.
  - `test_content_box_background`: a black image with a white rectangle and `background="black"` gives the rectangle's box; with the default it gives the whole image minus nothing, i.e. `(0, 0, w, h)`.
  - `test_content_box_all_background_and_sizes`: all-white gives `(0, 0, w, h)`; images of different sizes raise `ValueError`.
  - `test_content_box_exported`: `htfa.content_box is htfa.brainnet.content_box`; `not hasattr(htfa.brainnet, "_content_box")`.
- [ ] **Step 2:** run them, expect failures (`AttributeError`).
- [ ] **Step 3:** implement. A pixel is content when any channel differs from the background color (`_rgb(background)` scaled to 0-255) by more than 5; for white this equals today's `min < 250` rule. Accept one image or a sequence; open each with `_open`. Rename, do not alias. Add to `docs/api.rst` autosummary next to `annotate_pair`.
- [ ] **Step 4:** tests pass, lint passes. **Step 5:** commit.

### Task 2: `crf` and `faststart` in `write_animation`

**Files:** Modify `htfa/brainnet.py` (`write_animation`), `tests/test_brainnet.py`.

**Interfaces:**
- Produces: `write_animation(images, out, fps=10, *, crf: int | None = None, faststart: bool = False) -> Path`.

- [ ] **Step 1: failing tests** (frames: 12 real 320 x 200 noise-free synthetic PIL images with a moving colored square)
  - `test_write_animation_crf_size`: file size with `crf=40` is smaller than with `crf=10`; both read back with imageio as 12 frames of 320 x 200.
  - `test_write_animation_default_unchanged`: bytes written with no new arguments equal bytes written by a second call with no new arguments, and differ from `crf=23` output (guards that the default path keeps `quality=8`).
  - `test_write_animation_faststart`: in the file bytes, `b"moov"` comes before `b"mdat"` with `faststart=True` and after it without.
  - `test_write_animation_options_validated`: `crf=52`, `crf=-1`, `crf=2.5` raise `ValueError`; `crf=23` or `faststart=True` with a `.gif` raise `ValueError`.
- [ ] **Step 2:** run, expect `TypeError` for the unknown keywords.
- [ ] **Step 3:** implement. With `crf`: `quality=None` and `output_params=["-crf", str(crf)]`; `faststart` appends `["-movflags", "+faststart"]`. With neither, the `get_writer` call is byte-for-byte today's. Add a numpydoc Parameters section.
- [ ] **Step 4:** pass + lint. **Step 5:** commit.

### Task 3: per-node colors and replaceable nodes in `_Scene`

**Files:** Modify `htfa/brainnet.py` (`_Scene`, `_style`, `plot_network` docstring), `tests/test_brainnet.py`.

**Interfaces:**
- Produces:
  - `_node_colors(node_color: Any, K: int, T: int | None = None) -> Array`: float RGB in 0-1, shape `(3,)` for one color, `(K, 3)` for per-node, `(T, K, 3)` for per-frame (only when `T` is given). Alpha channels are dropped.
  - `_Scene.__init__(..., supersample: int = _SUPERSAMPLE)`; style key `supersample` (default `_SUPERSAMPLE`) in `_style`.
  - `_Scene.set_nodes(radii: Array, colors: Array | None = None) -> None`: replaces the node spheres with radii in mm (shape `(K,)`) and, when given, colors of shape `(3,)` or `(K, 3)`; updates `self.radii` so edges set afterwards start at the new surfaces. Raises `ValueError` in component-lobe mode.
  - `_Scene(node_color=...)` accepts what `_node_colors` returns for `(3,)` and `(K, 3)`.

- [ ] **Step 1: failing tests** (two nodes far apart, `snap=False`, `mesh=None`, lateral view)
  - `test_plot_network_per_node_colors`: `node_color=["red", "blue"]` gives an image with reddish pixels (r > 150, g < 90, b < 90) and bluish pixels (b > 150, r < 90, g < 90); the red blob's centroid is where the first node's blob is in a one-color render.
  - `test_plot_network_color_array_shapes`: three nodes with `node_color=np.eye(3)` shows three hues; `node_color=(1.0, 0.0, 0.0)` with three nodes shows only red; `(K, 4)` arrays work and ignore alpha.
  - `test_plot_network_per_node_colors_with_fade`: per-node colors with `fade_by_mass=True` on a real fit fixture render both hues.
  - `test_node_color_bad_shapes`: K=4 with 3 color names, a (2, 3) array for K=4, and a (T, K, 3) array in `plot_network` raise `ValueError` mentioning K.
  - `test_per_node_colors_rejected_in_lobe_mode`: K colors with `node_mode` component lobes raise `ValueError`.
  - `test_scene_set_nodes`: build a `_Scene`, take `image()`, call `set_nodes(2 * scene.radii, colors)`, take `image()`: the count of non-background pixels grows by more than 2x, colors match; `scene.radii` is the new radii; closing works.
  - `test_supersample_style`: `supersample=1` and `supersample=2` both give images of the requested `size`; `supersample=0` raises `ValueError`.
  - `test_default_scene_actors_unchanged`: with one color and no opacity the scene has exactly one node actor (merged mesh), as before.
- [ ] **Step 2:** run, expect failures.
- [ ] **Step 3:** implement. Keep the one-color code path's `add_mesh` call arguments exactly as they are today (pixel identity). Move sphere creation out of `_build` into one private method used by `_build` and `set_nodes`; keep the actors in `self._nodes` so they can be removed. Per-node colors without opacity: one merged mesh with per-cell `rgb` scalars, as `set_edges` does. With opacity: one actor per sphere, each with its color. `set_nodes` passes `render=False` when removing and adding.
- [ ] **Step 4:** pass + lint. **Step 5:** commit.

### Task 4: per-frame sizes and colors in `animate_network`

**Files:** Modify `htfa/brainnet.py` (`animate_network`, `_per_node` or `_display_nodes` as needed), `tests/test_brainnet.py`.

**Interfaces:**
- Consumes: `_node_colors`, `_Scene.set_nodes` (Task 3).
- Produces: `animate_network` accepting `sizes` of shape `(K,)` or `(T, K)`, `node_color` up to `(T, K)` colors, and `connectivity=None`. Private helper `_radii(sizes: Array, scene_opts: dict) -> Array` (the `node_radius` call now in `_Scene.__init__`, usable on `(T, K)`), used by `_Scene` and `animate_network`.

- [ ] **Step 1: failing tests** (K=2 or 3 nodes, T=3, `frames_dir=tmp_path`, mp4 output, `snap=False`, `mesh=None`)
  - `test_animate_sizes_over_time`: `sizes=[[1, 1], [2, 1], [4, 1]]`: node 0's blob area grows frame to frame, node 1's stays within 2 percent; node 1 is the smallest radius (2 mm) in every frame (joint scaling: `radius_range` end points are reached once over the whole array, not per frame).
  - `test_animate_sizes_interpolated`: with `n_frames=5`, frame 2's node-0 area lies strictly between frames 1 and 3; with `interpolation="nearest"` there are only 3 distinct areas.
  - `test_animate_colors_over_time`: `(T, K)` color names: frame 1 has a red node, frame 3 has that node green; a `(T, K, 3)` array gives the same frames (`image_diff` < 1e-6).
  - `test_animate_without_connectivity`: `connectivity=None` with `(T, K)` sizes writes T frames with no edge pixels; `connectivity=None` with nothing time-varying raises `ValueError`.
  - `test_animate_time_axes_must_agree`: connectivity T=3 with sizes T=4, with colors T=2, or with 4 captions each raise `ValueError` naming both lengths.
  - `test_animate_edges_start_at_frame_radii`: one edge, node radius growing: the edge's visible length shrinks as the radii grow.
  - `test_animate_one_k_sizes_still_work`: `(1, K)` and `(K, 1)` sizes behave as `(K,)` (same frames as the flat array).
  - `test_animate_gif_colored_nodes` (Review Focus 3): a `.gif` with a red and a blue node reads back with reddish and bluish pixels (thresholds 120 / 110).
  - `test_animate_scale_by_mass_with_time_sizes`: on a real fit fixture, `(T, K)` sizes with `scale_by_mass=True` runs and frames differ over time.
  - `test_animate_time_sizes_rejected_in_lobe_mode`: raises `ValueError`.
- [ ] **Step 2:** run, expect the 0.1.0 errors (`sizes has 6 entries for 2 nodes`, pyvista color errors).
- [ ] **Step 3:** implement. Radii for all timepoints come from one `_radii` call on the whole array; the frame's radii are interpolated with the same rule as edges (`interpolation`), colors use `_nearest`. When neither sizes nor colors vary over time, `set_nodes` is never called and the frame loop is today's.
- [ ] **Step 4:** pass + lint. **Step 5:** commit.

### Task 5: views, legend, `frame_size` and the shared layout

**Files:** Modify `htfa/brainnet.py` (`annotate_pair`, `plot_network`, `animate_network`, new private helpers), `tests/test_brainnet.py`.

**Interfaces:**
- Consumes: `content_box` (Task 1), `_Scene.set_nodes`, `_radii` (Tasks 3-4).
- Produces:
  - `_Views`: `__init__(self, centers, node_sizes, mesh, views: Sequence[str], scene_opts: dict, **scene_kwargs)` builds one `_Scene` per view (closing those already built if one fails); `draw(self, E: Array | None, edge_opts: dict, radii: Array | None = None, colors: Array | None = None, rotation: Array | None = None) -> list[Image]`; `fit_spin(rotations)`; `close()`. The legacy single-`view` path uses it with one view.
  - `_legend_rows(legend: Any) -> list[list[tuple[tuple[int, int, int], str, str]]]`: validates and normalizes one row or several.
  - `_layout(images: Sequence[Any], titles: Sequence[str], caption: str | None, legend: list | None, crop: tuple[int, int, int, int] | None, width: int | None, frame_size: tuple[int, int] | None, background: Any = "white") -> Image`.
  - `plot_network(..., views=None, legend=None, frame_size=None)` and the same three keywords on `animate_network`.
  - `_view_titles(views) -> tuple[list[str], list[str]]` (names, titles) from names or `(name, title)` pairs.

- [ ] **Step 1:** before changing `annotate_pair`, save its output for two synthetic 400 x 250 images (titles with a subtitle line, a caption, default width and `crop=None`) as a `.npy` in the scratchpad. After the refactor the output must be array-equal. Report the comparison; do not commit the file.
- [ ] **Step 2: failing tests**
  - `test_plot_network_views`: `views=("left", "dorsal")` gives an image wider than tall-ratio of one view; splitting it at the divider, each half has brain content; the title text rows have dark pixels above each panel. `view="left", views=("left",)` raises `ValueError`; an unknown view name raises.
  - `test_views_titles`: `views=("left", ("dorsal", ""))` leaves the right title band white; default titles for `left` and `dorsal` are "Left lateral" and "Dorsal" (`_view_titles`).
  - `test_plot_network_frame_size`: `frame_size=(960, 540)` gives exactly that size, with content, for one view and for two; `frame_size=(961, 540)` raises `ValueError` (Review Focus 2).
  - `test_plot_network_legend`: `legend=[("red", "positive", "line"), ("blue", "negative", "line")]` adds a band at the bottom holding red and blue pixels and dark text; two rows (`[[...], [("#E69F00", "community 1 (4 nodes)", "dot")]]`) make the band twice as high, with orange pixels in the lower row only.
  - `test_legend_validation`: a kind other than "dot"/"line", a 2-tuple entry, and a row wider than the frame (`frame_size=(320, 200)` with a 200-character label) each raise `ValueError`; the last names the row (Review Focus 5).
  - `test_single_view_layout_path`: `legend=...` with the default view gives a cropped view (non-background content touches within `margin` of the panel edge) and no title band.
  - `test_views_dark_background`: `background="black", views=("left", "dorsal")` crops to the brain (panel content box is smaller than the uncropped render); the title and legend bands are the background color, with white text (text is black when the background's mean channel is 128 or more, white otherwise; `annotate_pair` passes white and so keeps black text).
  - `test_animate_views_caption_legend`: T=3, two views, `captions=["a", "b", "c"]`, a legend, `frame_size=(640, 360)`: every frame is 640 x 360; the brain's content box inside the frame is identical in all frames (fixed crop) although node sizes change; the caption row differs between frames.
  - `test_animate_views_spin`: `views` with `spin=(0, 0, 1)` writes the movie; frames differ; no error.
  - `test_legacy_caption_path_unchanged`: `captions` without `views`/`legend`/`frame_size` gives frames of size `(w, h + header)` exactly as `_with_header` makes them.
  - `test_annotate_pair_three_way_equivalence`: `annotate_pair(a, b, titles, caption, width=W)` equals `_layout([a, b], titles, caption, None, None, W, None)` array-wise.
- [ ] **Step 3:** run, expect `TypeError` / `AttributeError`.
- [ ] **Step 4:** implement per the spec's Views and Legend sections. `_layout` owns the pane, title, caption and divider drawing now in `annotate_pair`, for N images; `annotate_pair` becomes a call to it. Legend geometry: the spec's numbers. Crop reference: one extra draw per view with no edges and each node at its largest radius over time, taken once before the first frame. Natural width without `frame_size`: the summed cropped-view widths.
- [ ] **Step 5:** pass + lint; the Step 1 comparison is array-equal. **Step 6:** commit.

### Task 6: `compose`

**Files:** Modify `htfa/brainnet.py` (`animate_network`), `tests/test_brainnet.py`.

**Interfaces:**
- Consumes: `_Views` (Task 5).
- Produces: `animate_network(..., compose: Callable[[list[Any], float], Any] | None = None)`.

- [ ] **Step 1: failing tests**
  - `test_compose_receives_views_and_time`: a module-level function that records `(len(images), images[0].size, t)` to a JSON-lines file and returns the two images pasted top to bottom: with `views=("left", "dorsal")`, `size=(320, 200)`, T=3, the file has 3 lines, each `(2, [320, 200], t)` with t = 0, 1, 2, and the saved frames are 320 x 400.
  - `test_compose_without_views`: `images` has length 1.
  - `test_compose_excludes_layout_arguments`: `compose` with `legend`, with `captions`, or with `frame_size` raises `ValueError`.
  - `test_compose_bad_return`: a compose returning frames of two different sizes raises `ValueError` ("same size"); one returning `None` raises `TypeError` or `ValueError` with "compose" in the message.
  - `test_compose_odd_size_mp4`: a compose returning 321 x 201 frames writes an mp4 of 320 x 200 (Review Focus 2).
- [ ] **Step 2:** run, expect `TypeError`. **Step 3:** implement. **Step 4:** pass + lint. **Step 5:** commit.

### Task 7: parallel rendering with `n_jobs`

**Files:** Modify `htfa/brainnet.py` (`animate_network`, new private helpers), `tests/test_brainnet.py`.

**Interfaces:**
- Consumes: `_Views`, `_layout`, `compose` handling (Tasks 5-6).
- Produces:
  - `animate_network(..., n_jobs: int = 1, crf: int | None = None, faststart: bool = False)`; the last two are forwarded to `write_animation`.
  - `class _BlankRender(RuntimeError)`.
  - `_is_blank(img: Any, background: Any) -> bool`: true when more than 5 percent of pixels have max channel below 10. Always false when the background color's own max channel (0-255) is below 40 (a dark background is not checked).
  - A module-level worker function (picklable) that builds a `_Views`, draws a list of frame indices and writes `NNNNN.png` via a temporary name and `os.replace`.

- [ ] **Step 1: failing tests** (mark the multi-process ones `slow` only if one exceeds 30 s)
  - `test_n_jobs_matches_sequential`: T=6, two views, legend, captions, `(T, K)` sizes: frames from `n_jobs=2` and `n_jobs=1` (both via `frames_dir`) have the same names (`1.png` ... `6.png`) and `image_diff` < 1e-3 pairwise; both movies have 6 frames.
  - `test_n_jobs_temp_folder_removed`: no folder created by the call is left under `tempfile.gettempdir()` after success, and after a `compose` that raises `ValueError` (which surfaces as `ValueError`, raised once, with the movie file absent).
  - `test_n_jobs_more_workers_than_frames` (Review Focus 4): T=1 with `n_jobs=2`, and T=2 with `n_jobs=8`, write complete movies.
  - `test_n_jobs_jpeg_and_gif`: `jpeg_quality=80` with `frames_dir` gives `.jpg` files as in the sequential path; `.gif` output works with `n_jobs=2`.
  - `test_n_jobs_spin`: spin frames from `n_jobs=2` match `n_jobs=1` (`image_diff` < 1e-3).
  - `test_is_blank`: an all-black 320 x 200 image is blank on a white background and not on a black one; a real `plot_network` render is not blank; an image with a 10 percent black rectangle on white is blank.
  - `test_n_jobs_restart_after_worker_death`: a module-level `compose` that calls `os._exit(1)` the first time it is asked for t == 2 (it creates a marker file first; when the marker exists it composes normally): the call returns, the movie has all T=4 frames, the marker exists.
  - `test_n_jobs_gives_up`: a `compose` that always `os._exit(1)`s raises `RuntimeError` mentioning the number of restarts (5); no worker processes of the call are left (check with `psutil` only if it is already a dependency, else by the pool's own shutdown having returned) and the temporary folder is gone.
  - `test_animate_network_forwards_encoding`: `crf=40` gives a smaller file than `crf=10`; `faststart=True` puts `moov` before `mdat`.
  - `test_n_jobs_validation`: `n_jobs=0` raises `ValueError`.
- [ ] **Step 2:** run, expect `TypeError`.
- [ ] **Step 3:** implement per the spec's Parallel rendering section. The pool is this call's own executor from `joblib.externals.loky` (not `get_reusable_executor`, not `joblib.Parallel`'s shared pool), so it pickles notebook functions with cloudpickle and can be shut down with `kill_workers=True` without touching other joblib users; if the bundled loky cannot give a private pool, stop and report instead of using the shared one. Restart only on `_BlankRender` or a terminated worker (`TerminatedWorkerError` / `BrokenProcessPool`); at most 5 restarts; redraw only frames whose PNG is missing. The parent reads the PNGs in order and feeds the same per-frame tail as the sequential path (`_jpeg`, `frames_dir` copy, `write_animation`); that tail is one function shared by both paths. `n_jobs=1` does not create a pool or a temporary folder and runs no blank check.
- [ ] **Step 4:** pass + lint; run the whole `tests/test_brainnet.py` twice to check the worker tests are stable. **Step 5:** commit.

### Task 8: docstrings, version 0.2.0, changelog, API docs, README

**Files:** Modify `htfa/brainnet.py` (docstrings of `plot_network`, `animate_network`), `htfa/results.py` (wrapper docstrings only), `pyproject.toml`, `htfa/__init__.py`, `tests/test_package.py`, `docs/changelog.rst`, `docs/api.rst`, `docs/API.md` (only if it lists these functions), `README.md`.

- [ ] **Step 1:** `tests/test_package.py` expects `"0.2.0"`; run, expect failure; bump both version strings; pass.
- [ ] **Step 2:** document every new argument in numpydoc style, with one doctest-skipped example in `animate_network` showing `sizes=(T, K)`, `node_color`, `views`, `legend`, `n_jobs`, `crf`, `faststart`. State in `HTFAResults.animate_network` that `connectivity=None` there still computes ISFC.
- [ ] **Step 3:** `docs/changelog.rst`: a 0.2.0 section above 0.1.0, in the file's existing style, listing the additions and "no change to existing calls". README: one short example. Factual descriptions only.
- [ ] **Step 4:** `make lint`, `make test`, `make docs` (no new warnings). **Step 5:** commit.

### Task 9: tutorials 10 and 11

**Files:** Modify `docs/tutorials/10_dynamic_isfc_animation.ipynb`, `docs/tutorials/11_plotting_and_appearance.ipynb`, assets under `docs/_static/tutorials/`.

- [ ] **Step 1:** read both notebooks' structure and how their assets are produced (do not guess; follow the existing build path).
- [ ] **Step 2:** tutorial 11: a section on per-node colors, `views`, `legend`, `frame_size` and `content_box`. Tutorial 10: a section on per-frame sizes and colors, `n_jobs`, `crf` / `faststart`, and `compose`. Keep render sizes modest so the docs build time grows by under two minutes.
- [ ] **Step 3:** execute both notebooks with the project venv's kernel and the data they use now; regenerate assets; look at every new figure and one frame of every new movie.
- [ ] **Step 4:** `make docs` with no new warnings; `make lint`; `make test`. **Step 5:** commit.

### Task 10: branch verification

- [ ] **Step 1:** legacy pixel identity: for six existing-style calls (one view each of lateral and dorsal, with and without edges, `fade_by_mass`, a captioned 3-frame animation, a spin), render on this branch and on `main` (a second worktree) on this machine and compare arrays exactly. Report any difference as a defect to fix.
- [ ] **Step 2:** a clip in the timecorr tutorial's style through the public API only: 1920 x 1080, two views, caption, two legend rows, per-node colors, per-frame radii in mm (`snap=False`, `radius_kb=(1, 0)`, `threshold=0.0`), `n_jobs=3`, `crf=23`, `faststart=True`, 30 frames. Look at three frames.
- [ ] **Step 3:** full suite in a fresh environment built from the committed tree; `make lint`; `make docs`.
- [ ] **Step 4:** whole-branch review (codex), fix findings, repeat Steps 1-3 if code changed.
