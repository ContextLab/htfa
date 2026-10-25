# htfa

A from-scratch, standalone Python implementation of Hierarchical Topographic Factor Analysis (HTFA), ported from the algorithm in BrainIAK.

[![tests](https://github.com/ContextLab/htfa/actions/workflows/tests.yml/badge.svg)](https://github.com/ContextLab/htfa/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **Pre-release.** The TFA and HTFA estimators, `htfa.fit`, input loading (NIfTI files and folders, nilearn/nibabel objects, numpy arrays, 2D images), connectivity and the results object are implemented and tested. The API may change before the first tagged release. The Sphinx documentation is not written yet; see [docs/API.md](docs/API.md) for the API guide. See [Status and roadmap](#status-and-roadmap).

## Overview

Topographic factor analysis (TFA) models a data matrix `X` (voxels by time points) as `X ≈ F W + noise`. Each of the `K` factors is a radial basis function with a center and a width in coordinate space, and `W` holds the factors' weights over time [1]. Hierarchical TFA (HTFA) fits TFA to several subjects at once, with a global template that the per-subject factors are tied to [2, 3, 4].

This package re-implements these algorithms from scratch. It does not import BrainIAK. The design is described in `docs/superpowers/specs/2026-10-23-htfa-rewrite-design.md`.

Differences from BrainIAK, and whether each exists yet:

| Point | Status |
|-|-|
| No MPI and no compiler needed; the core dependencies are pure Python wheels (numpy, scipy, scikit-learn, nibabel, joblib) | Implemented (packaging) |
| Analytic Jacobian for the center/width least-squares problem | Implemented (`htfa/_kernels.py`) |
| NumPy engine, plus a torch engine for CUDA, Apple Metal (MPS) and CPU, with an `auto` backend that uses a GPU only for large problems when torch is already installed | Implemented (`htfa/_engines.py`); selected with the estimators' `backend` argument |
| Optional dependencies install themselves on first use | Implemented (`htfa/_lazy.py`) |
| Synthetic data with ground truth, for 2D and 3D coordinate grids | Implemented (`htfa.synthetic`) |
| 2D images as input (pillow/tifffile extras) | Implemented (`htfa/io.py`) |
| NIfTI files, folders of NIfTI files, and nilearn/nibabel objects as input | Implemented (`htfa/io.py`) |
| `TFA` and `HTFA` estimators and `htfa.fit` | Implemented (`htfa/tfa.py`, `htfa/htfa.py`, `htfa/fit.py`) |

## Status and roadmap

Implemented (plan Tasks 1 to 10):

- Packaging and repository cleanup
- Numerical kernels: factors, ridge weights, residual and analytic Jacobian
- NumPy and torch engines and backend selection
- On-demand installation of optional dependencies
- Synthetic data generators and factor matching
- TFA estimator
- HTFA estimator
- Loading inputs: NIfTI files and folders, images, nilearn/nibabel objects, arrays
- Connectivity, results object and `htfa.fit`
- Cross-check against BrainIAK (`pytest -m brainiak`, run in CI)

Not yet done (Tasks 11 to 13):

- Benchmarks, GPU accuracy checks and tuning of the `auto` backend threshold (Task 11; in progress)
- Sphinx documentation and tutorials, including a check on real fMRI data (Task 12)
- Release workflow and PyPI publication (Task 13)

The full plan is in `docs/superpowers/plans/2026-10-23-htfa-rewrite.md`.

## Installation

The package is not on PyPI yet. Install from GitHub:

```bash
pip install git+https://github.com/ContextLab/htfa.git
```

For development, using [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/ContextLab/htfa.git
cd htfa
uv venv
uv pip install -e ".[dev]"
```

Requires Python 3.10 or newer. On Apple Silicon Macs, torch is a core dependency.

### Optional extras

These extras are declared in `pyproject.toml`:

| Extra | Contents |
|-|-|
| `plot` | matplotlib, nilearn |
| `images` | pillow, tifffile |
| `nilearn` | nilearn |
| `gpu` | torch |
| `brain3d` | pyvista, imageio, imageio-ffmpeg, pillow (BrainNet Viewer-style network plots and animations, `htfa.brainnet`) |
| `dev` | pytest, pytest-cov, ruff, mypy, build, twine, plus the optional libraries above |
| `docs` | sphinx, furo, numpydoc, sphinx-gallery, nbsphinx, and others |

Optional dependencies are installed with pip the first time they are needed, using the versions declared for the extra. To turn this off, set the environment variable `HTFA_AUTO_INSTALL=0` or call:

```python
import htfa

htfa.set_autoinstall(False)
```

## Example

Fit HTFA to a folder with one `.nii` / `.nii.gz` file per subject:

```python
import htfa

res = htfa.fit("path/to/folder", K=50, max_global_iter=30, random_state=0)
res.global_centers  # (K, 3) template centers, in world (mm) coordinates
res.weights[0]  # (K, T) factor time courses of the first subject (file order)
res.connectivity(0)  # (K, K) factor connectivity of the first subject
```

On synthetic data with known parameters:

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
print(abs(res.centers[0][perm] - truth["centers"][0]).max())
```

Output:

```
0.01688680355447225
```

See [docs/API.md](docs/API.md) for the inputs `htfa.fit` accepts, its parameters and the results object.

## Testing

```bash
pytest
```

By default, tests marked `brainiak` or `network` are skipped (`addopts = "--strict-markers -m 'not brainiak and not network'"`). Markers:

- `brainiak`: compares against the reference BrainIAK implementation (requires brainiak)
- `network`: requires network access (downloads data)
- `gpu`: requires a GPU backend (torch with CUDA or MPS)
- `slow`: long-running test

For example, `pytest -m brainiak` runs the BrainIAK comparison tests.

## Citing

1. J. R. Manning, R. Ranganath, K. A. Norman, and D. M. Blei (2014). Topographic factor analysis: a Bayesian model for inferring brain networks from neural data. *PLoS ONE* 9(5). [doi:10.1371/journal.pone.0094914](https://doi.org/10.1371/journal.pone.0094914)
2. J. R. Manning, R. Ranganath, W. Keung, N. B. Turk-Browne, J. D. Cohen, K. A. Norman, and D. M. Blei (2014). Hierarchical topographic factor analysis. *2014 International Workshop on Pattern Recognition in Neuroimaging (PRNI)*. [doi:10.1109/PRNI.2014.6858530](https://doi.org/10.1109/prni.2014.6858530)
3. J. R. Manning, X. Zhu, T. L. Willke, R. Ranganath, K. Stachenfeld, U. Hasson, D. M. Blei, and K. A. Norman (2018). A probabilistic approach to discovering dynamic full-brain functional connectivity patterns. *NeuroImage*. [doi:10.1016/j.neuroimage.2018.01.071](https://doi.org/10.1016/j.neuroimage.2018.01.071)
4. M. J. Anderson, M. Capotă, J. S. Turek, X. Zhu, T. L. Willke, Y. Wang, P.-H. Chen, J. R. Manning, P. J. Ramadge, and K. A. Norman (2016). Enabling factor analysis on thousand-subject neuroimaging datasets. *IEEE International Conference on Big Data*. [doi:10.1109/BigData.2016.7840719](https://doi.org/10.1109/BigData.2016.7840719)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT; see [LICENSE](LICENSE). `htfa/brainnet.py` and `htfa/timecourse.py` adapt code from ContextLab/directed-forgetting-network-dynamics (commit 31a2b66; MIT License, Copyright (c) 2019 Contextual Dynamics Laboratory). The BrainNet Viewer brain meshes used by `htfa.brainnet` are not included; they are downloaded from NITRC on first use (BrainNet Viewer is distributed under the GNU GPL).
