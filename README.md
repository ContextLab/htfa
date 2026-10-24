# htfa

A from-scratch, standalone Python implementation of Hierarchical Topographic Factor Analysis (HTFA), ported from the algorithm in BrainIAK.

[![tests](https://github.com/ContextLab/htfa/actions/workflows/tests.yml/badge.svg)](https://github.com/ContextLab/htfa/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **Work in progress.** `main` is in the middle of a rewrite. The TFA and HTFA estimators, `htfa.fit`, input loading and plotting are not implemented yet. See [Status and roadmap](#status-and-roadmap) for what exists today.

## Overview

Topographic factor analysis (TFA) models a data matrix `X` (voxels by time points) as `X ≈ F W + noise`. Each of the `K` factors is a radial basis function with a center and a width in coordinate space, and `W` holds the factors' weights over time [1]. Hierarchical TFA (HTFA) fits TFA to several subjects at once, with a global template that the per-subject factors are tied to [2, 3, 4].

This package re-implements these algorithms from scratch. It does not import BrainIAK. The design is described in `docs/superpowers/specs/2026-10-23-htfa-rewrite-design.md`.

Differences from BrainIAK, and whether each exists yet:

| Point | Status |
|-|-|
| No MPI and no compiler needed; the core dependencies are pure Python wheels (numpy, scipy, scikit-learn, nibabel, joblib) | Implemented (packaging) |
| Analytic Jacobian for the center/width least-squares problem | Implemented (`htfa/_kernels.py`) |
| NumPy engine, plus a torch engine for CUDA, Apple Metal (MPS) and CPU, with an `auto` backend that uses a GPU only for large problems when torch is already installed | Implemented (`htfa/_engines.py`); not yet reachable through an estimator |
| Optional dependencies install themselves on first use | Implemented (`htfa/_lazy.py`) |
| Synthetic data with ground truth, for 2D and 3D coordinate grids | Implemented (`htfa.synthetic`) |
| 2D images as input (pillow/tifffile extras) | Planned |
| NIfTI files and nilearn objects as input | Planned |
| `TFA` and `HTFA` estimators and `htfa.fit` | In progress / planned |

## Status and roadmap

Implemented:

- Packaging and repository cleanup (plan Task 1)
- Numerical kernels: factors, ridge weights, residual and analytic Jacobian (Task 2)
- NumPy and torch engines and backend selection (Task 3)
- On-demand installation of optional dependencies (Task 4)
- Synthetic data generators and factor matching (Task 5)

Not yet available (plan Tasks 6 to 13):

- TFA estimator (Task 6; in progress)
- HTFA estimator (Task 7)
- Loading inputs: NIfTI, images, nilearn objects (Task 8)
- Connectivity, results object and `htfa.fit` (Task 9)
- Cross-check against BrainIAK (Task 10)
- Benchmarks, GPU accuracy checks and the `auto` threshold (Task 11)
- Plotting and Sphinx documentation (Task 12)
- Release workflow (Task 13)

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
| `dev` | pytest, pytest-cov, ruff, mypy, build, twine, plus the optional libraries above |
| `docs` | sphinx, furo, numpydoc, sphinx-gallery, nbsphinx, and others |

Optional dependencies are installed with pip the first time they are needed, using the versions declared for the extra. To turn this off, set the environment variable `HTFA_AUTO_INSTALL=0` or call:

```python
import htfa
htfa.set_autoinstall(False)
```

## Example

Only the synthetic-data helpers exist so far. This simulates single-subject TFA data on a 2D grid and recovers the ordering of shuffled factor centers (it does not fit a model):

```python
import numpy as np
from htfa.synthetic import make_tfa_data, match_factors

# Simulate X = F @ W + noise: 20 x 20 pixel grid, 50 time points, 4 factors
X, R, truth = make_tfa_data(grid_shape=(20, 20), n_samples=50, K=4, random_state=0)
print(X.shape, R.shape)                      # (400, 50) (400, 2)
print(truth["centers"].shape, truth["widths"].shape, truth["weights"].shape)

# Match a shuffled copy of the true centers back to the originals
rng = np.random.default_rng(1)
shuffled = truth["centers"][rng.permutation(4)]
perm = match_factors(truth["centers"], shuffled)
print(np.allclose(shuffled[perm], truth["centers"]))
```

Output:

```
(400, 50) (400, 2)
(4, 2) (4,) (4, 50)
True
```

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

MIT; see [LICENSE](LICENSE).
