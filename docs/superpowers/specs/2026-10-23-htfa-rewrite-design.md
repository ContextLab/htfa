# HTFA rewrite — design

Date: 2026-10-23. Status: draft for approval.

## Intent

`htfa` is a standalone, fast, easy-to-install port of BrainIAK's Topographic Factor
Analysis (TFA) and Hierarchical TFA (HTFA)
(`brainiak/factoranalysis/{tfa,htfa}.py`, `tfa_extension.cpp`). It exists because
BrainIAK has become hard to install, maintain and use. Priorities, in order:

1. **Correctness.** The model and fitting procedure match BrainIAK's published
   algorithm, with BrainIAK's bugs fixed.
2. **Performance.** Fast enough for real whole-brain, multi-subject datasets, using the
   hardware that is available: multithreaded BLAS, multiple processes, CUDA and Apple
   Metal GPUs.
3. **Ease of installation.** `pip install htfa` works on Windows, Linux and macOS with
   no compiler. Optional dependencies install themselves on first use, as in hypertools.
4. **Ease of use.** One call takes NIfTI files, folders, 2D images, nilearn/nibabel
   images, NumPy arrays, or lists of any of these.

The current repository is almost entirely placeholder code: a fake MAP update, a
factor formula whose units don't match its own bounds, backends that are never called,
and tests that never check parameter recovery. It is rewritten in place on `main`. The
user decided this.

## Goals

- `htfa.TFA` and `htfa.HTFA`: scikit-learn-style estimators.
- `htfa.fit(data, K=..., ...)`: a one-call helper. It loads data, builds coordinates,
  fits TFA (one subject) or HTFA (several subjects), and returns an `HTFAResults`
  object with plotting and NIfTI export.
- Supported inputs:
  - .nii and .nii.gz files, 3D or 4D
  - folders of NIfTI files or 2D images
  - 2D images: jpg, png, tiff, bmp and similar
  - nibabel/nilearn image objects
  - NumPy arrays, with or without coordinates
  - lists mixing any of the above
- Unit tests that recover known parameters from synthetic data, with no mocks.
- Sphinx docs written from scratch in the hypertools style, with tutorials for every
  user-facing function.
- CI on ubuntu, windows and macos × Python 3.10–3.14.

## Non-goals

- MPI. Multiprocessing on one machine replaces it.
- Exact numerical reproduction of BrainIAK's bugs. There is no compatibility mode
  (user decision).
- BrainIAK-compatible attribute names such as `local_posterior_` or `get_centers()`
  (user decision).
- BIDS-specific querying. A BIDS `func/` folder is just a folder of NIfTI files; pybids
  is dropped.
- Surface (mesh) data.

## Anti-goals (failures even if it "works")

- Tests that pass without checking recovery of known ground truth, or that hide
  failures with `skip` or `assert True`.
- A required install step that needs a compiler, MPI or a Rust toolchain.
- GPU (float32) fits that are measurably less accurate than float64 CPU fits.
- Docs that describe features the code doesn't have.

## The model (what "correct" means)

For subject *s* with voxel coordinates `R` (V×D, D=2 or 3) and data `X` (V×T):

- **Standardization (user decision):** `standardize='auto'` by default.
  - A subject with more than one sample has each voxel's time series z-scored before
    fitting. Voxels with zero variance are left as zeros.
  - A subject with a single sample (one 3D volume or one image) is not standardized.
  - `standardize=True` forces it: per voxel, or across voxels when there is only one
    sample.
  - `standardize=False` fits raw values. Docs and docstrings must explain that the
    corrected likelihood then makes the prior relatively stronger on high-variance data.
- **Subsampling (user decision):** off by default. `voxel_ratio=1.0`, `tr_ratio=1.0`,
  and no caps, so `scale = 0.5`. BrainIAK-style subsampling is opt-in through
  `voxel_ratio`, `tr_ratio`, `max_voxels` and `max_trs`.

- Factors: `F[v,k] = exp(-||r_v - c_k||² / w_k)`, where `w_k = 2σ_k²`. This is
  BrainIAK's convention. The kernel is generalized to any D, so 2D images work.
- Weights, with F fixed: ridge regression `W = (FᵀF + β I)⁻¹ FᵀX`, β = var(X).
  `weight_method='ols'` drops the ridge term.
- Centers and widths, with W fixed: bounded nonlinear least squares, using scipy
  `least_squares` (trf method, `soft_l1` loss by default; both are configurable as in
  BrainIAK). Bounds come from BrainIAK's `get_bounds`: centers within the coordinate
  range, widths in `[lower_ratio, upper_ratio] × 2·max(std(R))²`.
- Residual vector (the fixed version):
  - Data term: `(X − FW) / (√2·std(X))`. BrainIAK *multiplies* by `std(X)/√2`. The two
    agree when std(X)=1, which gives an exact cross-check on z-scored data.
  - When an HTFA template is present, prior terms:
    - centers: `sqrt(scale · Δcᵀ Σ_k⁻¹ Δc)`
    - widths: `sqrt(scale · Δw² / var_k)`
    - `scale = 0.5 · (V_sub·T_sub)/(V·T)`, as in BrainIAK.
- TFA outer loop:
  - Initialize centers by K-means++ on R. Initialize widths to `2·max(std(R))²`.
  - Repeat up to `max_iter` times:
    1. Subsample voxels and TRs. The sample is redrawn each iteration and seeded.
    2. Compute F, then W, then solve for centers and widths.
    3. Hungarian-match the new centers to the previous ones.
    4. Stop when the largest change is ≤ `threshold`.
  - Then compute final F and W on the full data.
- HTFA:
  - Global template initialized from a randomly chosen subject, as in BrainIAK, but
    seeded:
    - center covariance `cov(R)·K^(−2/3)`
    - width variance `2·max(std(R))²`
    - both scaled by `1/n_subjects`
  - Each global iteration:
    1. Fit each subject's TFA against the template (in parallel).
    2. Update the template means and (co)variances with BrainIAK's conjugate MAP update.
    3. Hungarian-match the posterior to the prior, permuting *all* fields.
    4. Check convergence.
  - Then compute final per-subject weights on the full data.

### BrainIAK bugs fixed (each gets a regression test)

1. `tfa_extension.cpp::factor`: `tmp1..3` are shared across OpenMP threads, a data
   race. Gone: there is no C++.
2. `recon`: the accumulator is a `float`. Everything is float64 unless the user
   chooses float32 on the GPU.
3. The data residual is multiplied by σ instead of divided (above).
4. `HTFA._assign_posterior`: computes the Hungarian permutation but writes centers and
   widths back unpermuted, so they fall out of step with their (co)variances.
5. The template subject is chosen with an unseeded `np.random`. It now uses
   `random_state`.
6. `np.random.choice(n, max_num_voxel, replace=False)` crashes when there are fewer
   voxels than the cap. Sample sizes are now clipped to what is available.
7. The factor kernel is hardcoded to 3D. It now works for any D.

## Performance design

Benchmarks were run on an M2 Max with Accelerate BLAS (`notes/2026-10-23-session.md`).
The gains come from the algorithm, not from the language:

- **Analytic, matrix-free Jacobian.** BrainIAK uses 2-point finite differences: K·(D+1)
  full residual evaluations per Jacobian. The analytic operator gives the same solution
  45× faster (0.9 s vs 41 s at V=3000, T=200, K=20). It is written so that all the
  heavy work is BLAS matrix products:
  - `J p = −s·A(p)W`, where `A = F/w ∘ (2(R pcᵀ − c·pc) + d·pw/w)`
  - `Jᵀu` is built from `G = F/w ∘ (U Wᵀ)`, `GᵀR` and column sums
  - It is checked against finite differences in tests.
- **No forced subsampling.** On full data with no subsampling (V=20k, T=300, K=100),
  one solve takes 2.5 s (linear loss) or 3.5 s (`soft_l1`) in pure NumPy. That is why
  full-data fitting is the default. Subsampling is opt-in for very large datasets.
- **QR compression for `loss='linear'`.** Because `‖X − FW‖² = ‖XQ − F Rᵀ‖² + const`
  (with `Wᵀ = QR`), the residual shrinks from V×T to V×K entries. This is exact.
- **Parallelism.**
  - Subjects are fitted in parallel with joblib (`n_jobs`).
  - BLAS threads are capped per worker with threadpoolctl, to avoid oversubscription.
  - Within one fit, multithreaded BLAS does the work.
- **GPU, optional** (`backend='torch'`, installed on demand). The same operator runs on
  CUDA (float64 by default) or Apple MPS (float32 only, because MPS has no float64).
  The matrix products ran about 7× faster on MPS at V=50k, T=1000, K=200.
  - **Apple GPU by default (user decision).** On Apple Silicon macOS, torch is a
    regular dependency (pip environment marker
    `sys_platform=='darwin' and platform_machine=='arm64'`; the macOS wheel is small).
    `backend='auto'` then uses MPS when the problem is large enough to pay for the
    transfer, with no flag needed.
  - MPS has no float64, so the GPU does only the large matrix products in float32.
    The optimizer state, bounds and convergence checks stay float64 on the CPU (my
    call).
  - Tests must show that MPS fits recover synthetic parameters to the same tolerance
    as float64 CPU fits. If they don't, the MPS path refines on the CPU in float64
    before returning. Either way, `backend='cpu'` turns the GPU off.
  - `backend='auto'` also picks CUDA when torch is already installed and the problem is
    large enough. Elsewhere it never triggers an install.
  - `backend='cuda'` or `'mps'` with torch missing auto-installs torch (user decision),
    behind the same on-demand switch as the other extras. It prints a notice about the
    download size, and chooses the CUDA wheel index URL when an NVIDIA GPU is detected
    (`nvidia-smi`) on Linux or Windows.
- **Rejected options:**
  - Rust or C++: the hot path is BLAS-bound, so a compiled kernel buys little and adds
    wheel builds.
  - numba: speeds up only the V×K exponential, which is a small share of the time.
  - JAX: weak Windows GPU story. The torch backend covers GPUs on all three OSes.

## Package layout

```
htfa/
  __init__.py        # TFA, HTFA, fit, HTFAResults, load, set_autoinstall, __version__
  _kernels.py        # factors, residual, analytic Jacobian operator (NumPy)
  _torch_backend.py  # same operator in torch (lazy)
  tfa.py             # TFA estimator
  htfa.py            # HTFA estimator (template, MAP update, parallel subjects)
  io.py              # load(): paths/folders/images/nibabel/arrays -> (X, R, meta) per subject
  fit.py             # fit(): one-call helper
  results.py         # HTFAResults: plots (nilearn/matplotlib), to_nifti, reconstruct
  synthetic.py       # make_tfa_data / make_htfa_data with ground truth
  _lazy.py           # on-demand install (hypertools pattern)
tests/               # pytest; real data, real files, no mocks
docs/                # Sphinx: furo + ContextLab look, numpydoc, sphinx-gallery, nbsphinx
examples/            # sphinx-gallery scripts (all executed in the docs build)
```

## Inputs and coordinates

- **Mapping (user decision):** each file or list element is a subject.
  - A 4D NIfTI is one subject's time series.
  - A 3D NIfTI or a 2D image is one subject with one sample.
  - `stack=True` treats a folder (or list) of 3D volumes or 2D images as one subject's
    time series instead.
  - Color images are converted to grayscale unless `channels='separate'`, in which case
    each channel is a sample (my call).
- **NIfTI coordinates:** voxel indices mapped through the affine, `R = ijk @ A[:3,:3]ᵀ
  + A[:3,3]`, in mm.
- **Mask:** `mask=None` keeps voxels that are finite and not constant-zero across
  samples. This is my call: deterministic, and needs no nilearn. A mask can also be
  given as a path, a nibabel image, a fitted `NiftiMasker`, or `'epi'`/`'background'`
  (which use nilearn).
- **2D images:** coordinates are (row, col) pixel positions times an optional `spacing`.
  Same masking rule.
- **NumPy arrays:** a `(V, T)` array needs `coords=(V, D)`. A 2D or 3D array without
  `coords` is treated as an image grid (pixel or voxel indices). Lists of arrays mean
  several subjects.
- **Different voxel counts:** subjects may have different voxel counts and coordinates,
  as HTFA allows.

## Dependencies

- **Required:** numpy, scipy, scikit-learn (K-means), nibabel, joblib, threadpoolctl.
- **On demand** (as pyproject extras, installed by `_lazy.lazy_import`):
  - nilearn: masking strategies, glass-brain plots
  - matplotlib: plotting
  - pillow and tifffile: 2D images
  - torch: GPU backend
- **Opting out:** `HTFA_AUTO_INSTALL=0` or `htfa.set_autoinstall(False)` turns
  auto-install off.
- **Build:** setuptools.

## Testing

- **Synthetic recovery (core):**
  - Generate data with known centers, widths and weights, using the same generator
    design as HTFA-communications (template plus subject-level jitter).
  - Assert that matched centers fall within a tolerance stated in mm, that widths fall
    within a relative tolerance, and that reconstructed weights correlate above a
    threshold.
  - Cover 2D and 3D, TFA and HTFA, and the global template.
- **Jacobian:** analytic vs finite differences, for random parameters.
- **BrainIAK cross-check** (`pytest -m brainiak`, its own CI job on Linux):
  - On z-scored data, residuals and costs should match BrainIAK's.
  - Recovered parameters on the same synthetic data should be no worse.
- **Regression tests** for each fixed BrainIAK bug.
- **I/O tests:** write real .nii, .nii.gz, png, tiff and jpg files, plus folders, then
  load them and check shapes, coordinates and affines.
- **Plots:** rendered to PNG in tests and inspected manually during development.
- **Benchmarks:** a script under `scripts/`, not part of CI pass/fail.

## Documentation

- **Look:** the hypertools style:
  - furo theme with the ContextLab palette (#007030 light, #4CAF50 dark)
  - Nunito Sans font
  - numpydoc and autosummary for the API reference
- **Content:**
  - sphinx-gallery `examples/plot_*.py`, executed during the docs build
  - nbsphinx tutorials under `docs/tutorials/`, covering:
    - quick start
    - TFA on one subject
    - HTFA on several subjects
    - inputs: NIfTI, folders, images, nilearn objects, arrays
    - 2D images
    - connectivity and dynamic connectivity, reproducing the MIND 2017 analyses on
      nilearn `development_fmri`
    - performance and GPU
    - parameter recovery
- **Hosting:** Read the Docs, configured by `.readthedocs.yaml`.
- **Connectivity tutorial scope (my call):**
  - glass-brain hub plots, global and per-subject
  - static connectome, thresholded at 75%
  - sliding-window dynamic connectivity
  - ISFC with Fisher-z averaging
  - one short animation
  - hypertools trajectory plots, with hypertools as a docs-only dependency

## Release

A GitHub Actions release workflow with PyPI trusted publishing builds the sdist and a
pure-Python wheel. Nothing is published until the user approves and configures the
PyPI project (user decision). Version scheme: semantic versioning, starting at 0.1.0
(user decision).
- **README:** rewritten in the hypertools order. I'll draft the citation and policy
  wording for approval rather than publish it.

## Left to the builder

Internal function boundaries, exact test tolerances (set from measured recovery error,
never loosened to pass), the gallery's figure styling, and the threshold `backend='auto'`
uses.
