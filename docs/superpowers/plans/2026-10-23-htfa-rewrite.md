# HTFA Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the placeholder `htfa` package with a correct, fast, easy-to-install
port of BrainIAK TFA/HTFA. It loads NIfTI files, folders, 2D images, nibabel/nilearn
objects and arrays, and ships complete Sphinx docs.

**Architecture:**
- **Fitting:** NumPy/SciPy. Centers and widths are fitted with `scipy.optimize.least_squares`
  (trf method), driven by an analytic, matrix-free Jacobian.
- **Heavy matrix products:** delegated to a small "engine": NumPy, or torch on
  CUDA/MPS/CPU.
- **HTFA:** fits subjects in parallel with joblib and runs BrainIAK's MAP template
  update.
- **Thin layers on top:** `load()` turns any input into per-subject `(X, R)`;
  `fit()` + `HTFAResults` give one-call use, plots and export.

**Tech Stack:**
- Python ≥3.10; numpy, scipy, scikit-learn, nibabel, joblib, threadpoolctl
- torch: required on macOS arm64, on demand elsewhere
- on-demand extras: nilearn, matplotlib, pillow, tifffile
- tooling: pytest, ruff, mypy; Sphinx (furo, numpydoc, sphinx-gallery, nbsphinx)

**Spec:** `docs/superpowers/specs/2026-10-23-htfa-rewrite-design.md`

## Global Constraints

- Work directly on `main`, modifying in place. Commit after every task. Commit messages
  end with the attribution lines from the session system reminder.
- `requires-python = ">=3.10"`. CI: ubuntu-latest, windows-latest, macos-latest ×
  Python 3.10, 3.11, 3.12, 3.13, 3.14.
- Build backend: setuptools (`setuptools>=77`, `setuptools.build_meta`). Version
  `0.1.0`. Pure-Python wheel; no compiled code.
- Dependencies:
  - Required: `numpy>=1.24`, `scipy>=1.10`, `scikit-learn>=1.3`, `nibabel>=5`,
    `joblib>=1.3`, `threadpoolctl>=3`, plus
    `torch>=2.2; sys_platform=='darwin' and platform_machine=='arm64'`.
  - Extras: `plot = [matplotlib>=3.7, nilearn>=0.10]`,
    `images = [pillow>=10, tifffile>=2023.1]`, `nilearn = [nilearn>=0.10]`,
    `gpu = [torch>=2.2]`.
  - `dev` and `docs` extras for tooling.
- All arrays are float64 internally. The only exception is the large matrix products on
  MPS, which run in float32 inside an engine.
- Data orientation: per-subject data `X` is `(V, T)` (voxels × samples). Coordinates
  `R` are `(V, D)`, D ∈ {2, 3}. Widths use BrainIAK units, `w = 2σ²`.
- Defaults that differ from BrainIAK, all user decisions:
  - `standardize='auto'`
  - no subsampling: `voxel_ratio=1.0`, `tr_ratio=1.0`, `max_voxels=None`, `max_trs=None`
  - `backend='auto'` (MPS on Apple Silicon when large enough)
- No mocks, ever. Tests use real arrays, real files and real installs. A test may skip
  only when the hardware isn't there (no CUDA/MPS) or `-m brainiak`/`-m network` was
  not selected. Every skip carries an explicit reason. No `assert True`, no skipping
  inside `except`.
- Tolerances in recovery tests are set from measured error with headroom, written as
  named constants with a comment saying what was measured. They are never loosened to
  make a failing test pass. If a test fails, fix the code.
- Public names (exported from `htfa/__init__.py`): `TFA`, `HTFA`, `fit`, `load`,
  `HTFAResults`, `Subject`, `make_tfa_data`, `make_htfa_data`, `static_connectivity`,
  `dynamic_connectivity`, `isfc`, `set_autoinstall`, `__version__`.
- On-demand installs: `HTFA_AUTO_INSTALL=0` or `htfa.set_autoinstall(False)` disables
  them. The error message then names the exact pip command.
- Docs: furo theme with the ContextLab palette (`#007030`/`#005524` light,
  `#4CAF50`/`#81C784` dark) and the Nunito Sans font, copied from the hypertools
  conf.py snippet in `notes/2026-10-23-session.md`.
- Don't write policy, positioning or "speaking for the lab" text in the README or docs
  without the user's approval. Draft it and ask (user CLAUDE.md rule). Factual
  descriptions of what the code does are fine.
- Keep the repo clean. Scratch scripts go in the session scratchpad or `scripts/`.

## Review Focus

1. **NaN, Inf or all-zero voxels in a NIfTI.** They are masked out by the default mask.
   They never reach the optimizer, and no NaN appears in results. *Task 8:
   `test_load_nifti_masks_nan_and_zero_voxels`.*
2. **K ≥ the number of voxels, or a subject with fewer than K voxels.** A clear
   `ValueError` naming K and the voxel count, not a K-means or least-squares traceback.
   *Task 6: `test_tfa_rejects_k_not_less_than_voxels`.*
3. **HTFA mixing a single-sample subject (T=1) with multi-sample subjects.** It fits,
   standardization treats each subject by its own T, and weights have the right
   shapes. *Task 7: `test_htfa_mixed_sample_counts`.*
4. **Integer or float32 inputs** (uint8 PNGs, int16 NIfTI, float32 arrays) and
   non-contiguous array views. They are converted to float64; results match the
   float64 input. *Task 8: `test_load_uint8_image_is_float64`; Task 6:
   `test_tfa_accepts_float32_and_strided_views`.*
5. **Upper-case or mixed-case file extensions** (`.NII.GZ`, `.TIF`, `.JPEG`) and
   folders containing unrelated files (`.json` sidecars, `.DS_Store`). Images are
   recognized case-insensitively, and unrelated files are ignored with no error.
   *Task 8: `test_load_folder_ignores_sidecars_and_case`.*

---

### Task 1: Clean the repo and set up packaging

**Files:**
- Delete:
  - tracked: `htfa/**`, `tests/**`, `docs/conf.py`, `docs/index.rst`, `.claude.backup/`,
    `docker/`, `.dockerignore`, `assets/images/coverage.svg`,
    `cookiecutter-config-file.yml`, `poetry.lock`, `setup.cfg`, `demo_performance.py`,
    `test_backend_selection.py`, `HTFA_PERFORMANCE.md`, `INFRASTRUCTURE.md`,
    `TECHNICAL_DESIGN.md`, `.github/.stale.yml`, `.github/release-drafter.yml`, and
    `.github/workflows/{build,greetings,monitoring,quality-gates,release-drafter,unified-ci}.yml`
  - untracked: `.ccpm_backup/`, `.coverage`, `coverage.xml`, `htmlcov/`, `.mypy_cache/`,
    `.pytest_cache/`, empty `data/` and `tutorials/`
- Keep: `.claude/`, `CLAUDE.md`, `LICENSE`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`,
  `SECURITY.md`, `.editorconfig`, `.github/ISSUE_TEMPLATE/`,
  `.github/PULL_REQUEST_TEMPLATE.md`, `docs/superpowers/`, `notes/`
- Create:
  - `pyproject.toml` (rewritten)
  - `htfa/__init__.py` (`__version__ = "0.1.0"` only, for now)
  - `tests/conftest.py`, `tests/test_package.py`
  - `.github/workflows/tests.yml`
- Modify:
  - `.github/dependabot.yml` (ecosystem `pip` + `github-actions`)
  - `.pre-commit-config.yaml` (ruff, ruff-format)
  - `Makefile`: targets `install`, `test`, `lint`, `docs`
  - `CLAUDE.md`: the Commands section and architecture tree match the new layout
  - `.gitignore`: add `docs/_build`, `docs/auto_examples`, `docs/gen_modules`,
    `htmlcov`, `.coverage`

**Interfaces:**
- Produces:
  - `pip install -e ".[dev]"` works.
  - pytest markers `brainiak`, `network`, `gpu`, `slow` are registered in pyproject.
  - `-m brainiak` and `-m network` are deselected by default via `addopts`.

- [ ] **Step 1:** Commit any uncommitted work (`git status` must be clean), then delete
  the files listed above. Before deleting each untracked directory, run `ls` on it to
  confirm it is a backup or cache.
- [ ] **Step 2: Write the failing test** `tests/test_package.py::test_version`:
  `import htfa; assert htfa.__version__ == "0.1.0"`, and assert that
  `importlib.metadata.version("htfa") == "0.1.0"`.
- [ ] **Step 3:** Write `pyproject.toml` (Global Constraints) and the ruff config
  (line length 88, rules `E,F,W,I,UP,B,NPY`). The mypy config covers `htfa/` only.
- [ ] **Step 4:** Create a fresh venv:
  `uv venv -p 3.12 .venv && VIRTUAL_ENV=.venv uv pip install -e ".[dev]"`. Then run
  `.venv/bin/pytest -q`. Expected: `1 passed`.
- [ ] **Step 5:** Write `.github/workflows/tests.yml`:
  - matrix OS × Python as in Global Constraints, `fail-fast: false`
  - steps: `pip install -e ".[dev]"`, `ruff check .`, `ruff format --check .`,
    `pytest -q`
- [ ] **Step 6:** Commit: `Clean repository and set up setuptools packaging`.

---

### Task 2: Numerical kernels (factors, weights, residual, analytic Jacobian)

**Files:**
- Create: `htfa/_kernels.py`, `tests/test_kernels.py`

**Interfaces:**
- Produces, all float64 NumPy:
  - `sq_distances(R: (V,D), centers: (K,D)) -> (V,K)`. Computed by summing per-dimension
    squared differences, *not* the `|r|²−2r·c+|c|²` expansion, to avoid cancellation.
  - `factors(R, centers, widths: (K,)) -> (V,K)`: `exp(-sq_distances / widths)`.
  - `ridge_weights(X: (V,T), F: (V,K), method: str = "rr") -> (K,T)`:
    - `"rr"`: solve `(FᵀF + var(X)·I) W = FᵀX` with `scipy.linalg.solve(assume_a="pos")`
    - `"ols"`: same without the ridge term
    - any other value: `ValueError`
  - `max_sigma(R) -> float`: `2.0 * max(std(R, axis=0))**2`.
  - `bounds(R, K, lower_ratio, upper_ratio) -> (lb, ub)`. Parameter layout is
    `x = [centers.ravel() (K·D), widths (K)]`. Center bounds are `nanmin`/`nanmax` of
    R per dimension; width bounds are `ratio * max_sigma(R)`.
  - `init_centers_widths(R, K, random_state) -> (centers (K,D), widths (K,))`:
    `KMeans(K, init="k-means++", n_init=10, random_state=random_state)` on R; widths
    `max_sigma(R)`.
  - `@dataclass Prior`: `centers (K,D)`, `widths (K,)`, `centers_cov (K,D,D)`,
    `widths_var (K,)`, `scale: float`.
  - `class ResidualProblem`:
    - `__init__(self, X (V,T), W (K,T), R (V,D), engine, prior: Prior | None = None)`.
      Data scale `s = 1/(sqrt(2)*std(X))`; if `std(X) == 0`, `s = 1`.
    - `.residual(x) -> (V·T + 2K if prior else V·T,)`
    - `.jacobian(x) -> scipy.sparse.linalg.LinearOperator` (`matvec`, `rmatvec`,
      `matmat`, `rmatmat`)
    - `.column_norms(x) -> (K·(D+1),)`, used as `x_scale = 1/norms` (zero norms map to 1)
  - `def solve_centers_widths(problem, x0, bounds, method="trf", loss="soft_l1", x_scale="jac") -> (x, cost)`:
    - calls `least_squares(..., jac=problem.jacobian, tr_solver="lsmr")`
    - `x_scale="jac"` means `1/problem.column_norms(x0)`
  - `ResidualProblem(..., compress: bool = False)`. When true, which callers use only
    for `loss="linear"`:
    - replace `(X, W)` by `(X Q, Rqᵀ)`, where `Wᵀ = Q Rq` is the thin QR
    - this is exact up to the constant `‖X(I − QQᵀ)‖²`
    - `.cost_offset` holds that constant, so reported costs match the uncompressed ones
- Consumes: the engine interface from Task 3. For this task, implement
  `NumpyEngine` inline in `htfa/_engines.py` (Task 3 adds torch), with:
  - `set(X, W)`
  - `residual(F) -> (V·T,)`: `X − F@W`
  - `times_w(A) -> (V·T,)`: `A@W`
  - `times_wt(u) -> (V,K)`: `u.reshape(V,T) @ Wᵀ`

  The engine returns NumPy float64. `ResidualProblem` applies `s` and signs.

Residual and Jacobian maths (the algorithm the signatures don't determine).
Notation: `C` (K,D), `w` (K,), `F = factors`, `d = sq_distances`, `Fw = F/w`.

```
data residual          r_d = s·(X − F W)                          (raveled V·T)
prior center residual  r_c[k] = sqrt(scale · Δc_kᵀ Σ_k⁻¹ Δc_k),   Δc = C − prior.centers
prior width residual   r_w[k] = sqrt(scale / var_k) · (w_k − prior.widths_k)   (signed; same square as BrainIAK)
J·p   (p = [pc (K,D), pw (K)]):
  A = Fw ∘ (2·(R pcᵀ − rowsum(C∘pc)) + d ∘ (pw/w))                (V,K)
  data block   = −s · times_w(A)
  center block = scale · rowsum((Σ⁻¹Δc) ∘ pc) / r_c   (0 where r_c == 0)
  width block  = sqrt(scale/var) ∘ pw
Jᵀ·u  (u = [u_d, u_c, u_w]):
  G = Fw ∘ times_wt(u_d)                                          (V,K)
  g_c = −2s·(GᵀR − C ∘ colsum(G)) + (scale·Σ⁻¹Δc / r_c) ∘ u_c     (0 term where r_c == 0)
  g_w = −s·colsum(G ∘ d)/w + sqrt(scale/var) ∘ u_w
column norms² = data part s²·colsum(A_j²)·‖W_k‖² per parameter j (exact, since each
  column is the outer product a_j ⊗ W_k), plus the prior parts squared.
```

- [ ] **Step 1: Write failing tests** in `tests/test_kernels.py`:
  - `test_factors_match_bruteforce[D=2,3]`: compare against a double loop over v, k,
    `atol=1e-12`.
  - `test_sq_distances_no_cancellation`: coordinates near 1e6 with spacing 1 still give
    exact integers.
  - `test_ridge_weights_matches_normal_equations` (rr and ols), and that `"xx"` raises
    `ValueError`.
  - `test_bounds_values`: hand-computed lb/ub for a 3-point R.
  - `test_jacobian_matches_finite_differences[prior=None|Prior; D=2,3]`: build the
    dense J by applying `matvec` to identity columns. Compare with central differences
    of `residual` (step 1e-6): `max abs err < 1e-6 · max|J|`. Check
    `rmatvec(u) == J_denseᵀ u` for random u.
  - `test_column_norms_exact`: equal to `np.linalg.norm(J_dense, axis=0)`.
  - `test_compressed_problem_same_solution`:
    - with linear loss, `compress=True` and `False` give the same x (atol 1e-6)
    - the costs are equal once `cost_offset` is added
  - `test_solve_recovers_centers_widths_given_true_weights`:
    - V=2000 on a 3D grid, K=5, T=100, noise 0.1
    - start from truth + N(0, 2 mm)
    - recovered centers within 0.5 mm; widths within 10% relative
- [ ] **Step 2:** Run `pytest tests/test_kernels.py -v`. Expected: FAIL (ImportError).
- [ ] **Step 3:** Implement `htfa/_kernels.py` and `NumpyEngine` in `htfa/_engines.py`.
- [ ] **Step 4:** Run `pytest tests/test_kernels.py -v`. Expected: all PASS.
- [ ] **Step 5:** Commit: `Add analytic-Jacobian TFA kernels`.

---

### Task 3: Engines and backend selection (torch: CUDA, MPS, CPU)

**Files:**
- Modify: `htfa/_engines.py`
- Test: `tests/test_engines.py`

**Interfaces:**
- Produces:
  - `TorchEngine(device: str)`, with the same methods as `NumpyEngine`:
    - `X` and `W` are stored on the device: float32 on `mps`, float64 otherwise
    - results come back as NumPy float64
  - `select_engine(backend: str, V: int, T: int, K: int) -> NumpyEngine | TorchEngine`:
    - `backend` is one of `'auto','numpy','cpu','torch','cuda','mps'`; anything else
      raises `ValueError`
    - `'cpu'`/`'numpy'` → NumpyEngine
    - `'torch'` → `TorchEngine` on the best available torch device
    - `'cuda'`/`'mps'` → that device. If torch is missing, call
      `htfa._lazy.ensure_torch()` (Task 4). If the device isn't available, raise
      `RuntimeError`.
    - `'auto'` picks a `TorchEngine` when **all** of these hold, otherwise NumpyEngine:
      - torch is importable without installing
      - `cuda` or `mps` is available
      - `V*T*K >= GPU_MIN_WORK`
  - `GPU_MIN_WORK: int`, a module constant. Its initial value is `2e8`; Task 11 tunes it
    from measurements.
  - `engine.name -> str`: `"numpy"`, `"torch-cuda"`, `"torch-mps"` or `"torch-cpu"`.

- [ ] **Step 1: Write failing tests:**
  - `test_select_engine_cpu` → `name == "numpy"`.
  - `test_select_engine_rejects_unknown`.
  - `test_auto_small_problem_uses_numpy` (V=100, T=10, K=5).
  - `test_torch_engine_matches_numpy[device]`, parametrized over the available devices
    among `cpu/cuda/mps`. Skip with reason `"no <device>"` when one is absent.
    `residual`, `times_w` and `times_wt` match NumpyEngine to rtol 1e-10 (float64) or
    1e-5 (mps).
  - `test_auto_large_problem_uses_gpu_when_available`, marked `gpu`. Skip with reason
    when there is no cuda/mps.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement. Torch is imported lazily inside `TorchEngine` and
  `select_engine`; `import htfa` must not import torch.
- [ ] **Step 4:** Run `pytest tests/test_engines.py -v`. Expected: PASS. On this M2 Max
  the mps cases run, not skip.
- [ ] **Step 5:** Commit: `Add NumPy and torch (CUDA/MPS) engines`.

---

### Task 4: On-demand installation

**Files:**
- Create: `htfa/_lazy.py`, `tests/test_lazy.py`

**Interfaces:**
- Produces:
  - `set_autoinstall(enabled: bool) -> None` and `autoinstall_enabled() -> bool`. The
    env var `HTFA_AUTO_INSTALL`: `"0"/"false"/"no"` (any case) disables.
  - `lazy_import(module: str, extra: str, purpose: str | None = None)`:
    - imports the module if present
    - otherwise reads the requirement specs for `extra` from
      `importlib.metadata.requires("htfa")`
    - if autoinstall is on, pip-installs them via
      `subprocess.run([sys.executable, "-m", "pip", "install", "-q", *reqs], check=True)`,
      then `importlib.invalidate_caches()` and imports
    - if it's off, or the install fails, raises `ImportError` whose message contains
      `pip install "htfa[<extra>]"`
  - `torch_index_url(nvidia_smi_output: str | None, system: str) -> str | None`:
    - `None` (PyPI default) when system is `Darwin` or there's no NVIDIA output
    - otherwise parses `CUDA Version: X.Y` and returns
      `https://download.pytorch.org/whl/<tag>`, where tag is the highest of
      `CUDA_WHEEL_TAGS = {(13, 2): "cu132", (13, 0): "cu130", (12, 6): "cu126"}` that
      is ≤ the driver version. These three were verified on 2026-10-23 to carry
      torch 2.14 wheels.
    - below 12.6: returns `https://download.pytorch.org/whl/cpu` and the caller warns
  - `ensure_torch() -> module`:
    - if torch is importable, returns it
    - else runs `nvidia-smi` (when on PATH) and passes its output to `torch_index_url`
    - prints `"htfa: installing torch (GPU backend; this can be a large download)..."`
      to stderr, then pip-installs `torch>=2.2`, with `--index-url` when there is one
    - respects `autoinstall_enabled()`
- Consumes: the extras declared in Task 1.

- [ ] **Step 1: Write failing tests:**
  - `test_autoinstall_env_var`: monkeypatch the env var (environment manipulation, not
    a mock).
  - `test_lazy_import_present_module` (`json`).
  - `test_lazy_import_disabled_message`: set autoinstall off, call
    `lazy_import("htfa_nonexistent_pkg", extra="images")`, assert `ImportError`
    containing `pip install "htfa[images]"`.
  - `test_torch_index_url_parsing`: real `nvidia-smi` header strings for driver CUDA
    13.2, 12.8 and 11.8 map to cu132, cu126 and cpu; `Darwin` maps to None.
  - `test_lazy_import_installs_into_fresh_venv`, marked `network`:
    - `python -m venv` a temp env and `pip install -e <repo>` into it, without the
      `images` extra
    - run a subprocess that calls `htfa._lazy.lazy_import("tifffile", extra="images")`
    - assert it prints the tifffile version and exits 0
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `pytest tests/test_lazy.py -v -m "not network or network"`.
  Expected: PASS, including the network test, run locally.
- [ ] **Step 5:** Commit: `Add on-demand dependency installation`.

---

### Task 5: Synthetic data with ground truth

**Files:**
- Create: `htfa/synthetic.py`, `tests/test_synthetic.py`

**Interfaces:**
- Produces:
  - `make_tfa_data(grid_shape=(20, 20, 20), n_samples=100, K=10, noise=0.1, spacing=1.0, random_state=None) -> tuple[X (V,T), R (V,D), dict]`:
    - D = `len(grid_shape)`, so 2 or 3
    - R holds the grid coordinates × spacing
    - truth dict keys: `centers (K,D)`, `widths (K,)`, `weights (K,T)`
    - centers are drawn uniformly within the central 80% of each axis, re-drawn until
      every pair is ≥ `min_sep = 0.5·sqrt(mean width)` apart, so they stay identifiable
    - widths `w = 2σ²`, with σ uniform in [0.06, 0.12] × the largest axis extent
    - weights N(0,1); `X = F W + noise·N(0,1)`
  - `make_htfa_data(n_subjects=5, grid_shape=(20,20,20), n_samples=100, K=10, noise=0.1, center_jitter=1.0, width_jitter=0.05, spacing=1.0, random_state=None) -> tuple[list[X], list[R], dict]`:
    - truth keys: `global_centers`, `global_widths`, `centers` (list), `widths` (list),
      `weights` (list)
    - subject centers = global + N(0, center_jitter²·I), clipped to the grid
    - subject widths = global · exp(N(0, width_jitter)), the HTFA-communications
      generator design
  - `match_factors(true_centers, est_centers) -> np.ndarray`: the Hungarian
    permutation `perm` such that `est_centers[perm]` aligns with the true centers.
    Used by tests and docs.

- [ ] **Step 1: Write failing tests:**
  - shapes, determinism with `random_state`, center separation
  - `X − F W` has std ≈ noise (within 5%)
  - `match_factors` recovers a known shuffle
- [ ] **Step 2:** Run them; they fail. **Step 3:** Implement. **Step 4:** Run; they
  pass.
- [ ] **Step 5:** Commit: `Add synthetic data generators with ground truth`.

---

### Task 6: TFA estimator

**Files:**
- Create: `htfa/tfa.py`, `tests/test_tfa.py`

**Interfaces:**
- Consumes: Task 2 kernels, `select_engine` (Task 3), `make_tfa_data` and
  `match_factors` (Task 5).
- Produces:
  - `standardize_data(X, mode: str | bool) -> X`:
    - `'auto'`: z-score each row (voxel) across columns when T > 1; leave as-is when
      T == 1
    - `True`: per row when T > 1, across all voxels when T == 1
    - `False`: unchanged
    - zero-variance rows become 0
    - Task 7 imports this function.
  - `class TFA(sklearn.base.BaseEstimator)`:
    - `__init__(self, K=50, max_iter=10, threshold=1.0, nlss_method="trf", nlss_loss="soft_l1", x_scale="jac", weight_method="rr", upper_ratio=1.8, lower_ratio=0.02, voxel_ratio=1.0, tr_ratio=1.0, max_voxels=None, max_trs=None, standardize="auto", backend="auto", random_state=None, verbose=False)`
    - `fit(self, X, coords, prior: Prior | None = None) -> TFA`
    - after fitting:
      - `centers_ (K,D)`, `widths_ (K,)`, `weights_ (K,T)`
      - `n_iter_ (int)`, `converged_ (bool)`, `cost_ (float)`, `engine_ (str)`
    - `factors(self, coords=None) -> (V,K)`: at the fitted coords by default
    - `reconstruct(self) -> (V,T)`: in the standardized space, `F @ weights_`
- Algorithm (spec "The model"):
  - init from `init_centers_widths`, or `prior.centers/widths` when a prior is given
  - each iteration:
    - draw the subsample with `rng = np.random.default_rng(random_state)`, sizes
      `min(n, max_x or n, max(1, int(ratio·n)))`
    - F, then W = `ridge_weights`
    - solve (with `Prior.scale = 0.5·(Vs·Ts)/(V·T)`), passing
      `compress=(nlss_loss == "linear")`
    - Hungarian-reorder the new centers and widths to the previous ones
    - converged when `max|Δx| ≤ threshold`
  - the final `weights_` are computed on the full data
  - validation messages:
    - X not 2D; coords shape mismatch; D ∉ {2,3}; NaN in X or coords
    - `K >= V` gives `ValueError(f"K={K} must be smaller than the number of voxels ({V})")`

- [ ] **Step 1: Write failing tests** (tolerances are named constants, set from
  measured runs):
  - `test_tfa_recovers_known_parameters_3d`:
    - `make_tfa_data((20,20,20), 100, K=8, noise=0.1, random_state=0)`
    - after `match_factors`: max center error < `CENTER_TOL_MM`, median relative width
      error < `WIDTH_RTOL`
    - each matched weight row correlates with the truth at r > 0.95
  - `test_tfa_recovers_known_parameters_2d`: grid (40,40), K=6; same assertions.
  - `test_tfa_subsampling_still_recovers`: voxel_ratio=0.25, tr_ratio=0.5; looser
    named tolerances.
  - `test_tfa_rejects_k_not_less_than_voxels` (Review Focus 2).
  - `test_tfa_accepts_float32_and_strided_views` (Review Focus 4): the fit on
    `X.astype(float32)[:, ::1]` and a Fortran-ordered copy equals the float64 fit,
    `atol=1e-6`.
  - `test_tfa_deterministic_with_random_state`.
  - `test_standardize_modes`: all three modes, including T==1.
  - `test_brainiak_bug_sampling_clip`: `max_voxels=10**6` on V=500 doesn't crash.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement `htfa/tfa.py`.
- [ ] **Step 4:** Run `pytest tests/test_tfa.py -v`. Expected: PASS. Record the
  measured errors in the constants' comments.
- [ ] **Step 5:** Commit: `Add TFA estimator with parameter-recovery tests`.

---

### Task 7: HTFA estimator

**Files:**
- Create: `htfa/htfa.py`, `tests/test_htfa.py`

**Interfaces:**
- Consumes: `TFA`, `standardize_data`, `Prior`, kernels, `make_htfa_data`,
  `match_factors`.
- Produces:
  - `map_update(prior_mean, prior_cov, global_cov_scaled, observations (n_subj, D)) -> (post_mean, post_cov)`:
    the formula from BrainIAK `_map_update`, using `np.linalg.solve` in place of `inv`.
  - `class HTFA(BaseEstimator)`:
    - `__init__(self, K=50, max_global_iter=10, max_local_iter=10, threshold=0.01, nlss_method="trf", nlss_loss="soft_l1", x_scale="jac", weight_method="rr", upper_ratio=1.8, lower_ratio=0.02, voxel_ratio=1.0, tr_ratio=1.0, max_voxels=None, max_trs=None, standardize="auto", backend="auto", n_jobs=None, random_state=None, verbose=False)`
    - `fit(self, X: list, coords: list) -> HTFA`
    - after fitting:
      - `global_centers_ (K,D)`, `global_widths_ (K,)`, `global_centers_cov_ (K,D,D)`,
        `global_widths_var_ (K,)`
      - `centers_` (list of (K,D)), `widths_` (list of (K,)), `weights_` (list of (K,T_s))
      - `n_iter_`, `converged_`, `engine_`
- Algorithm (spec "HTFA"):
  - template from subject `rng.integers(n_subj)`:
    - centers and widths from `init_centers_widths`
    - `centers_cov = cov(R.T)·K^(−2/3)` per k
    - `widths_var = max_sigma(R)` per k
    - `global_*_scaled = /n_subj`
  - each global iteration:
    1. Fit each subject's TFA with `max_iter=max_local_iter`, against the current
       `Prior`. Run in parallel with `joblib.Parallel(n_jobs, backend="loky")` when the
       engine is numpy, and sequentially for torch engines. Cap BLAS threads with
       `threadpoolctl.threadpool_limits(max(1, cpu_count // n_workers))` inside each
       worker.
    2. Hungarian-match each local posterior to the template.
    3. Run `map_update` for centers per k, and the scalar form for widths.
    4. Hungarian-match the global posterior to the global prior, applying the
       permutation to centers, widths, centers_cov and widths_var together. This is
       BrainIAK bug #4 fixed.
    5. Converged when `max|prior − posterior|` over centers and widths ≤ `threshold`.
  - final per-subject weights on the full data.

- [ ] **Step 1: Write failing tests:**
  - `test_map_update_matches_closed_form`: scalar Gaussian conjugate update, computed by
    hand.
  - `test_global_assignment_permutes_all_fields`: build a posterior that is a known
    permutation of the prior and assert that all four fields come back aligned.
  - `test_htfa_recovers_global_template`:
    - `make_htfa_data(n_subjects=4, (15,15,15), 80, K=6, random_state=0)`
    - global centers within `GLOBAL_CENTER_TOL_MM`
    - each subject's centers within `LOCAL_CENTER_TOL_MM`
    - matched subject weights at r > 0.9
  - `test_htfa_2d`: the same on (30,30) grids.
  - `test_htfa_mixed_sample_counts` (Review Focus 3): subjects with T = 80, 80 and 1.
  - `test_htfa_parallel_equals_serial`: `n_jobs=1` vs `n_jobs=2`, identical results.
  - `test_htfa_different_voxel_counts_per_subject`: crop the grids differently.
- [ ] **Step 2:** Run them; they fail. **Step 3:** Implement. **Step 4:** Run
  `pytest tests/test_htfa.py -v`; they pass.
- [ ] **Step 5:** Commit: `Add HTFA estimator with MAP template updates`.

---

### Task 8: Loading inputs

**Files:**
- Create: `htfa/io.py`, `tests/test_io.py`

**Interfaces:**
- Produces:
  - `@dataclass Subject`:
    - `data (V,T) float64`, `coords (V,D) float64`
    - `affine: (4,4) | None`, `mask: np.ndarray[bool] | None` (image-space shape)
    - `image_shape: tuple | None`, `source: str | None`
  - `load(data, coords=None, mask=None, stack=False, channels="gray", spacing=None) -> list[Subject]`:
    - `data` may be:
      - a path to a file or folder (`str`/`os.PathLike`)
      - a nibabel `SpatialImage`
      - a NumPy array
      - a list mixing any of these (one subject per element, unless `stack=True`)
    - recognized extensions (case-insensitive):
      - NIfTI: `.nii`, `.nii.gz`
      - images: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`, `.gif`
    - a folder is sorted by natural filename order; other files are ignored
    - images load through `lazy_import("PIL", extra="images")`; tiff through
      `tifffile`, which handles 16-bit and multi-page files
    - grayscale uses ITU-R 601 luma, `0.299R+0.587G+0.114B`; `channels="separate"`
      makes each channel a sample
    - NIfTI coords: `ijk @ affine[:3,:3].T + affine[:3,3]`
    - `mask`:
      - `None`: finite and not all-zero across samples
      - a path or nibabel image: nonzero
      - a fitted `nilearn.maskers.NiftiMasker`: its `mask_img_`
      - `'epi'` or `'background'`: nilearn `compute_epi_mask` / `compute_background_mask`
    - arrays:
      - `(V,T)` with `coords`
      - `(H,W)`, `(H,W,T)`, `(X,Y,Z)` or `(X,Y,Z,T)` without coords are image grids.
        A 2D array is one image. A 3D array without coords is treated as a 3D volume.
        `(H,W,T)` stacks are passed as a list of 2D arrays with `stack=True`.
      - a 2D array with `coords` given is `(V,T)`
    - `spacing` (scalar or per-axis) scales pixel coordinates for images and bare
      arrays
    - all subjects in one call must share D; otherwise `ValueError`
  - `Subject.to_image(values (V,) or (V,T)) -> nibabel.Nifti1Image | np.ndarray`:
    places values back into image space (NIfTI when there's an affine, otherwise an
    array of `image_shape`), with NaN outside the mask.
- Consumes: `lazy_import` (Task 4).

- [ ] **Step 1: Write failing tests**, all writing real files to `tmp_path`:
  - `test_load_4d_nifti_coords_from_affine`: a non-identity affine with rotation and
    shift; check coords against `nibabel.affines.apply_affine`.
  - `test_load_3d_nifti_is_one_sample`.
  - `test_load_nifti_gz_and_folder_of_subjects`: three .nii.gz files make 3 subjects.
  - `test_load_folder_stack_true`: ten 3D volumes make 1 subject with T=10.
  - `test_load_png_jpg_tiff`: grayscale and RGB; luma values checked against a hand
    calculation; `channels="separate"` gives T=3; 16-bit tiff values kept.
  - `test_load_uint8_image_is_float64` (Review Focus 4).
  - `test_load_nibabel_object_and_mixed_list`: a list of
    `[path, Nifti1Image, ndarray+coords]` makes 3 subjects.
  - `test_load_niftimasker_mask`: requires nilearn (installed by the `dev` extra).
  - `test_load_nifti_masks_nan_and_zero_voxels` (Review Focus 1).
  - `test_load_folder_ignores_sidecars_and_case` (Review Focus 5): `.NII.GZ`, `.TIF`,
    `.json`, `.DS_Store`.
  - `test_to_image_roundtrip`: `load` → `to_image(data)` reproduces the input inside
    the mask.
  - `test_load_rejects_mixed_dimensionality`.
- [ ] **Step 2:** Run them; they fail. **Step 3:** Implement. **Step 4:** Run; they
  pass.
- [ ] **Step 5:** Commit: `Add input loading for NIfTI, images, nibabel objects and arrays`.

---

### Task 9: Connectivity, results object and `fit()`

**Files:**
- Create: `htfa/connectivity.py`, `htfa/results.py`, `htfa/fit.py`
- Modify: `htfa/__init__.py`, to export the public names in Global Constraints
- Test: `tests/test_connectivity.py`, `tests/test_results.py`, `tests/test_fit.py`

**Interfaces:**
- Produces:
  - `static_connectivity(weights (K,T)) -> (K,K)`: Pearson correlation of the factor
    time courses.
  - `dynamic_connectivity(weights (K,T), window: int) -> (T−window+1, K, K)`.
  - `isfc(weights: list[(K,T)], window: int | None = None) -> (K,K) or (T−w+1,K,K)`:
    - for each subject, correlate its weights with the mean of the others
    - symmetrize as `(C + Cᵀ)/2`
    - Fisher-z average across subjects (`arctanh` with `clip ±(1−1e-7)`), then `tanh`
    - every subject must share T
  - `class HTFAResults`:
    - `__init__(self, model: TFA | HTFA, subjects: list[Subject])`
    - attributes:
      - `model`, `subjects`
      - `centers`, `widths`, `weights`: lists per subject
      - `global_centers`, `global_widths`: `None` for TFA
    - `factors(subject=0) -> (V,K)`; `reconstruct(subject=0) -> (V,T)`
    - `to_nifti(subject=0, what="reconstruction"|"factors") -> Nifti1Image`
      (`ValueError` for 2D image subjects)
    - `to_image(subject=0, what=...) -> np.ndarray` (NaN outside the mask)
    - `connectivity(subject=0)`, `dynamic_connectivity(subject=0, window=20)`,
      `isfc(window=None)`
    - `plot_nodes(subject="global"|int, ax=None, **kwargs)`:
      - 3D: `nilearn.plotting.plot_connectome(np.eye(K)·0, centers, ...)`; for
        `"global"`, global nodes in black, size 50, with each subject's nodes in a
        Spectral color, size 20
      - 2D: matplotlib scatter over the mean image
    - `plot_connectivity(subject=0, threshold="75%", ax=None)`: 3D through
      `plot_connectome`; 2D draws lines between centers
    - `plot_reconstruction(subject=0, sample=0)`: original vs reconstruction, side by
      side (`plot_glass_brain` for 3D, `imshow` for 2D)
    - all plot functions return the figure or display object, and use
      `lazy_import("matplotlib", "plot")` and `lazy_import("nilearn", "plot")`
  - `fit(data, K=50, coords=None, mask=None, stack=False, channels="gray", spacing=None, **model_kwargs) -> HTFAResults`:
    - `load(...)`, then TFA for 1 subject or HTFA for more
    - `model_kwargs` go to the estimator; an unknown key raises `TypeError` naming it
- Consumes: Tasks 6–8.

- [ ] **Step 1: Write failing tests:**
  - `test_static_connectivity_equals_corrcoef`.
  - `test_dynamic_connectivity_window`: shapes, and window = T equals the static result.
  - `test_isfc_identical_subjects_is_static`: identical subjects give the static
    correlation.
  - `test_isfc_requires_equal_T`.
  - `test_fit_numpy_single_subject_returns_tfa`, and the HTFA case for a list.
  - `test_fit_nifti_folder_end_to_end`:
    - write synthetic `make_htfa_data` subjects as .nii.gz with an affine
    - `fit(folder, K=6)`
    - recovered global centers in mm, matched to the truth mapped through the affine,
      within tolerance
  - `test_fit_png_stack_end_to_end`: the 2D synthetic data written as PNGs, with
    `stack=True`.
  - `test_fit_unknown_kwarg_raises`.
  - `test_to_nifti_shapes_and_affine`.
  - `test_plots_render_png[plot_nodes|plot_connectivity|plot_reconstruction; 2D|3D]`:
    `savefig` to `tmp_path`; the file exists and is > 5 KB. During development, view
    each PNG once by eye and say so in the commit message.
- [ ] **Step 2:** Run them; they fail. **Step 3:** Implement. **Step 4:** Run the full
  `pytest -q`; it passes.
- [ ] **Step 5:** Commit: `Add fit(), HTFAResults, connectivity and plotting`.

---

### Task 10: BrainIAK cross-check

**Files:**
- Create: `tests/test_brainiak_crosscheck.py`, `.github/workflows/brainiak.yml`

**Interfaces:**
- Consumes: `ResidualProblem`, `TFA`, `HTFA`, `make_tfa_data`, `make_htfa_data`.
  `brainiak` is imported with `pytest.importorskip("brainiak", reason=...)` under marker
  `brainiak`.

- [ ] **Step 1: Write the tests:**
  - `test_residual_matches_brainiak_on_zscored_data`:
    - z-score so std(X)=1 exactly, at which point BrainIAK's multiply-by-σ equals our
      divide (spec)
    - feed identical centers, widths, W and template to
      `brainiak.factoranalysis.tfa.TFA()._residual_multivariate` and to
      `ResidualProblem.residual`
    - equal to 1e-5. BrainIAK's float accumulator limits precision; the comment says so.
  - `test_factors_match_brainiak`: `TFA.get_factors` vs `factors`, 1e-12.
  - `test_tfa_recovery_not_worse_than_brainiak`:
    - same synthetic data, BrainIAK with its default 2-point Jacobian
    - our center error ≤ 1.1 × BrainIAK's
    - record both runtimes in the test output (`print`; run with `-s` in CI)
  - `test_htfa_global_recovery_not_worse_than_brainiak`: 3 subjects. BrainIAK's HTFA
    needs mpi4py, which installs from wheels.
- [ ] **Step 2:** Run locally:
  - `uv venv -p 3.11 .venv-brainiak && VIRTUAL_ENV=.venv-brainiak uv pip install -e ".[dev]" brainiak`
  - `.venv-brainiak/bin/pytest -m brainiak -v -s`
  - Expected: PASS.

  If a cross-check fails, find out which side is wrong before changing anything. Our
  code is wrong unless the difference is one of the seven documented BrainIAK bugs.
- [ ] **Step 3:** Write `.github/workflows/brainiak.yml`: ubuntu-latest, Python 3.11,
  install as in Step 2, run `pytest -m brainiak -v -s`.
- [ ] **Step 4:** Add `.venv-brainiak/` to `.gitignore`, then commit:
  `Add BrainIAK cross-check tests`.

---

### Task 11: Benchmarks, GPU accuracy and the auto threshold

**Files:**
- Create: `scripts/benchmark.py`, `tests/test_gpu_accuracy.py`
- Modify: `htfa/_engines.py` (`GPU_MIN_WORK`)

**Interfaces:**
- Produces:
  - `scripts/benchmark.py --sizes small,medium,large --backends numpy,mps,cuda,brainiak`:
    - prints a table of wall time per TFA fit and the center error
    - sizes are (V,T,K) = (5k,200,20), (20k,300,100), (50k,1000,200)
    - brainiak only runs for small and medium, and only when it is installed
  - a tuned `GPU_MIN_WORK`

- [ ] **Step 1: Write the failing test** `test_mps_fit_matches_cpu_accuracy` (marker
  `gpu`; skips with a reason when there is no MPS or CUDA):
  - TFA on `make_tfa_data((30,30,30), 200, K=20)`, with `backend="mps"` (or `"cuda"`)
    and with `backend="cpu"`
  - matched center error on the GPU ≤ CPU error + 0.05 mm
  - weights correlate > 0.999 between the two
- [ ] **Step 2:** Run it. If MPS float32 falls short, add a float64 CPU refinement to
  `TFA.fit`: one final `solve_centers_widths` call on a NumpyEngine, started from the
  GPU solution, used only for float32 engines. Re-run until it passes.
- [ ] **Step 3:** Run `scripts/benchmark.py` on this machine. Set `GPU_MIN_WORK` to the
  smallest measured V·T·K at which mps beats numpy by more than 1.2×. Paste the table
  into `notes/` and into the docs performance page (Task 12).
- [ ] **Step 4:** Run the full `pytest -q` and `pytest -m gpu -v`. Both pass.
- [ ] **Step 5:** Commit: `Add benchmarks, GPU accuracy test, tuned auto threshold`.

---

### Task 12: Documentation and README

**Files:**
- Create:
  - `docs/conf.py`, `docs/index.rst`, `docs/installation.rst`, `docs/api.rst`,
    `docs/inputs.rst`, `docs/performance.rst`, `docs/optional_dependencies.rst`,
    `docs/tutorials.rst`
  - `docs/tutorials/*.ipynb`:
    - `01_quickstart`
    - `02_tfa_single_subject`
    - `03_htfa_multi_subject`
    - `04_inputs_nifti_folders_nilearn_arrays`
    - `05_2d_images`
    - `06_connectivity_and_dynamics` (on nilearn `fetch_development_fmri(n_subjects=5)`:
      the MIND 2017 analyses in the spec's tutorial scope)
    - `07_parameter_recovery`
    - `08_gpu_and_performance`
  - `docs/_static/custom.css`, `docs/_static/favicon.ico`
  - `docs/requirements.txt`, `.readthedocs.yaml`
  - `examples/README.txt` and `examples/plot_*.py`, at least one per public function:
    - `plot_tfa_nodes`, `plot_htfa_global_template`, `plot_reconstruction`,
      `plot_connectivity`, `plot_dynamic_connectivity`, `plot_isfc`, `plot_load_inputs`,
      `plot_2d_image_factors`, `plot_synthetic_recovery`
  - `.github/workflows/docs.yml`
- Modify: `README.md`, rewritten

**Interfaces:**
- Consumes: the whole public API.
- Requirements:
  - numpydoc docstrings on every public function and class, each with an Examples
    section
  - the `api.rst` autosummary lists every public name
  - notebooks are executed locally, with outputs saved (`nbsphinx_execute = "never"`,
    as in hypertools)
  - gallery scripts execute at build time
  - the CI docs workflow runs `sphinx-build -W -b html docs docs/_build/html`
  - the RTD config is ubuntu-22.04, Python 3.11, `pip install .[plot,images,nilearn]`
    plus `-r docs/requirements.txt`
  - the README follows the hypertools order: logo/badges, Overview, Quick tour,
    Installation ("optional extras install themselves on demand"), Documentation,
    Citing, Contributing, Testing
  - the Citing section lists Manning et al. 2014 (PLoS ONE), Manning et al. 2014 (PRNI),
    Manning et al. 2018 (NeuroImage) and Anderson et al. 2016 (IEEE BigData), with
    DOIs taken from BrainIAK's docstrings. Check each URL by fetching it.
  - any non-factual wording (motivation, recommendations on behalf of the lab) is shown
    to the user before committing

- [ ] **Step 1:** Write `docs/conf.py` with furo, the palette, the extensions (autodoc,
  numpydoc, autosummary, viewcode, sphinx_gallery.gen_gallery, nbsphinx) and the
  footer GitHub icon.
- [ ] **Step 2:** Write the docstrings, `.rst` pages, gallery scripts and notebooks.
  Execute the notebooks with
  `jupyter nbconvert --execute --to notebook --inplace`.
- [ ] **Step 3:** Run `sphinx-build -W -b html docs docs/_build/html`. Expected: exit 0,
  no warnings. Screenshot the index, one tutorial, one gallery page and the API page in
  both light and dark themes, then inspect them and fix anything visibly wrong.
- [ ] **Step 4:** Check every external link with `sphinx-build -b linkcheck docs docs/_build/linkcheck`.
  Expected: no broken links.
- [ ] **Step 5:** Commit: `Rewrite documentation, tutorials and README`.

---

### Task 13: Release workflow and the final full check

**Files:**
- Create: `.github/workflows/release.yml`
- Modify: `notes/` (final session notes)

**Interfaces:**
- Produces:
  - a release workflow:
    - triggered on GitHub release `published`
    - builds with `python -m build`
    - checks with `twine check`
    - publishes with `pypa/gh-action-pypi-publish` through trusted publishing, in a
      `pypi` environment
    - nothing is published until the user configures PyPI and approves

- [ ] **Step 1:** Write the workflow. Build locally with `python -m build`, run
  `twine check dist/*`, then install the wheel into a fresh venv and run
  `python -c "import htfa; print(htfa.__version__)"`. Expected output: `0.1.0`.
- [ ] **Step 2:** Run every check in sequence: `ruff check .`, `ruff format --check .`,
  `mypy htfa`, `pytest -q`, `pytest -m "gpu or network" -v`, `pytest -m brainiak -v`
  (brainiak venv), and the docs build with `-W`. If any fix is needed, re-run *all* of
  them.
- [ ] **Step 3:** Scan for secrets: `git grep -nIiE "(api[_-]?key|secret|password|token)"`.
  Expected: no credentials.
- [ ] **Step 4:** Commit and push to `origin main`. Verify the push with
  `git ls-remote origin main`. Watch the CI runs on all OS × Python jobs and fix any
  failures, keeping the tests as written.
- [ ] **Step 5:** Ask the user for approval to configure PyPI and publish 0.1.0.
