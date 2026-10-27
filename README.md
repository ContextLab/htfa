# htfa

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/_static/cover_dark.png">
  <img alt="HTFA factor network: 50 template factors drawn as spheres inside a translucent brain, with the strongest inter-subject correlations as edges" src="docs/_static/cover_light.png" width="100%">
</picture>

<sub>HTFA template (K = 50) fitted to 12 adults watching a movie (nilearn's <code>development_fmri</code>), from the left, above and right. Edges: the strongest 5% of factor pairs by inter-subject functional correlation (|ISFC| &ge; 0.137; red positive, blue negative). Node size: factor width; colour: ISFC community. Made with <code>scripts/make_cover.py</code>.</sub>

[![tests](https://github.com/ContextLab/htfa/actions/workflows/tests.yml/badge.svg)](https://github.com/ContextLab/htfa/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/htfa/badge/?version=latest)](https://htfa.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **Pre-release.** htfa is not on PyPI and has no tagged release yet, so the API may change. See [Status](#status).

## Overview

Topographic factor analysis (TFA) models a data matrix `X` (voxels by time points) as `X ≈ F W + noise`. Each of the `K` factors is a radial basis function with a center and a width in coordinate space, and `W` holds the factors' weights over time [1]. Hierarchical TFA (HTFA) fits TFA to several subjects at once, with a global template that the per-subject factors are tied to [2, 3, 4].

htfa is a from-scratch Python implementation of these algorithms, ported from the algorithm in [BrainIAK](https://brainiak.org). It does not import BrainIAK.

- No MPI and no compiler: the core dependencies are pure Python wheels (numpy, scipy, scikit-learn, nibabel, joblib).
- Analytic Jacobian for the center/width least-squares problem (`htfa/_kernels.py`).
- A NumPy engine and a torch engine (CUDA, Apple Metal, CPU), chosen with the estimators' `backend` argument; `auto` uses a GPU only for large problems when torch is already installed.
- Two solver modes: `mode='fast'` (the default) and `mode='exact'`, the reference that the equivalence tests and the BrainIAK cross-checks use. On real fMRI the fast mode fitted TFA 2.8 (K = 50) and 5.3 (K = 100) times faster and can end at a different local minimum of the same objective; its HTFA template differs from the exact one by 15.8 mm (median center distance) on 12 adults (see the Performance and Validation pages of the docs).
- Inputs: NIfTI files and folders, nilearn/nibabel objects, numpy arrays and 2D images.
- Synthetic data with ground truth (`htfa.synthetic`), factor-level connectivity, a results object, and BrainNet Viewer-style 3D network plots (`htfa.brainnet`).

## Quick tour

Generate synthetic data for three subjects with known factor parameters, fit HTFA, and compare the recovered centers with the truth:

```python
import htfa
from htfa.synthetic import match_factors

Xs, Rs, truth = htfa.make_htfa_data(
    n_subjects=3, grid_shape=(12, 12, 12), n_samples=60, K=5, random_state=0
)
res = htfa.fit(
    Xs, K=5, coords=Rs, standardize=False, max_global_iter=30, random_state=0
)
print(res.global_centers.shape)  # template centers
print(res.weights[0].shape)  # factor time courses of the first subject
print(res.connectivity(0).shape)  # factor connectivity of the first subject
perm = match_factors(truth["centers"][0], res.centers[0])
print(abs(res.centers[0][perm] - truth["centers"][0]).max())
```

Output:

```
(5, 3)
(5, 60)
(5, 5)
0.01688680355447225
```

For real data, pass a folder with one `.nii` / `.nii.gz` file per subject:

```python
res = htfa.fit("path/to/folder", K=50, random_state=0)
```

The inputs `htfa.fit` accepts, its parameters and the results object are described in the [documentation](https://htfa.readthedocs.io/en/latest/) and in [docs/API.md](docs/API.md).

## Installation

htfa is not on PyPI yet. Install from GitHub:

```bash
pip install git+https://github.com/ContextLab/htfa.git
```

For development, using [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/ContextLab/htfa.git
cd htfa
make install
```

`make install` creates `.venv` (Python 3.12) and installs the package with the `dev` and `docs` extras. The package requires Python 3.10 or newer. On Apple Silicon Macs, torch is a core dependency.

### Optional extras

| Extra | Contents |
|-|-|
| `plot` | matplotlib, nilearn |
| `images` | pillow, tifffile |
| `nilearn` | nilearn |
| `gpu` | torch |
| `brain3d` | pyvista, imageio, imageio-ffmpeg, pillow (BrainNet Viewer-style network plots and animations, `htfa.brainnet`) |
| `tutorials` | hypertools, sentence-transformers (only to re-execute the word-decoding tutorial) |
| `dev` | pytest, pytest-cov, ruff, mypy, build, twine, plus the optional libraries above |
| `docs` | sphinx, furo, numpydoc, sphinx-gallery, nbsphinx, and others |

Optional dependencies are installed with pip the first time they are needed, using the versions declared for the extra. To turn this off, set the environment variable `HTFA_AUTO_INSTALL=0` or call:

```python
import htfa

htfa.set_autoinstall(False)
```

## Documentation

The Sphinx site is built from `docs/` with `make docs` (output in `docs/_build/html`). Its pages:

- [Installation](https://htfa.readthedocs.io/en/latest/installation.html)
- [Inputs](https://htfa.readthedocs.io/en/latest/inputs.html)
- [Optional dependencies](https://htfa.readthedocs.io/en/latest/optional_dependencies.html)
- [Tutorials](https://htfa.readthedocs.io/en/latest/tutorials.html)
- [Example gallery](https://htfa.readthedocs.io/en/latest/auto_examples/index.html)
- [API reference](https://htfa.readthedocs.io/en/latest/api.html)

The hosted site at htfa.readthedocs.io is not live yet (Read the Docs is not configured); until it is, build the docs locally.

## Status

Done:

- Tasks 1 to 11 of the [rewrite plan](docs/superpowers/plans/2026-10-23-htfa-rewrite.md): packaging, numerical kernels, NumPy and torch engines, on-demand dependencies, synthetic data, the `TFA` and `HTFA` estimators, input loading, connectivity, results and `htfa.fit`, the BrainIAK cross-check (`pytest -m brainiak`, run in CI), and benchmarks.
- Task 14: BrainNet Viewer-style network plots (`htfa.brainnet`).
- The Sphinx documentation site and example gallery (Tasks 12a and 12b).

In progress:

- Tutorials.
- A faster solver.

Not done:

- Release workflow and PyPI publication (Task 13).

## Testing

```bash
make test
```

By default, tests marked `brainiak` or `network` are skipped. Markers:

- `brainiak`: compares against the reference BrainIAK implementation (requires brainiak)
- `network`: requires network access (downloads data)
- `gpu`: requires a GPU backend (torch with CUDA or MPS)
- `slow`: long-running test

For example, `.venv/bin/pytest -m brainiak` runs the BrainIAK comparison tests.

## Citing

1. J. R. Manning, R. Ranganath, K. A. Norman, and D. M. Blei (2014). Topographic factor analysis: a Bayesian model for inferring brain networks from neural data. *PLoS ONE* 9(5). [doi:10.1371/journal.pone.0094914](https://doi.org/10.1371/journal.pone.0094914)
2. J. R. Manning, R. Ranganath, W. Keung, N. B. Turk-Browne, J. D. Cohen, K. A. Norman, and D. M. Blei (2014). Hierarchical topographic factor analysis. *2014 International Workshop on Pattern Recognition in Neuroimaging (PRNI)*. [doi:10.1109/PRNI.2014.6858530](https://doi.org/10.1109/prni.2014.6858530)
3. J. R. Manning, X. Zhu, T. L. Willke, R. Ranganath, K. Stachenfeld, U. Hasson, D. M. Blei, and K. A. Norman (2018). A probabilistic approach to discovering dynamic full-brain functional connectivity patterns. *NeuroImage*. [doi:10.1016/j.neuroimage.2018.01.071](https://doi.org/10.1016/j.neuroimage.2018.01.071)
4. M. J. Anderson, M. Capotă, J. S. Turek, X. Zhu, T. L. Willke, Y. Wang, P.-H. Chen, J. R. Manning, P. J. Ramadge, and K. A. Norman (2016). Enabling factor analysis on thousand-subject neuroimaging datasets. *IEEE International Conference on Big Data*. [doi:10.1109/BigData.2016.7840719](https://doi.org/10.1109/BigData.2016.7840719)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT; see [LICENSE](LICENSE). `htfa/brainnet.py` and `htfa/timecourse.py` adapt code from ContextLab/directed-forgetting-network-dynamics (commit 31a2b66; MIT License, Copyright (c) 2019 Contextual Dynamics Laboratory). The BrainNet Viewer brain meshes used by `htfa.brainnet` are not included; they are downloaded from NITRC on first use (BrainNet Viewer is distributed under the GNU GPL).
