# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hierarchical Topographic Factor Analysis (HTFA) - A lightweight Python implementation for neuroimaging data analysis based on the BrainIAK algorithm. Packaged with setuptools (`pyproject.toml`); the dev environment is managed with uv (`make install`). Dependencies are kept minimal.

## Commands

```bash
make install   # uv venv (.venv, Python 3.12) + pip install -e ".[dev,docs]"
make test      # .venv/bin/pytest -q
make lint      # ruff check, ruff format --check, mypy
make docs      # sphinx-build -> docs/_build/html

# Run a specific test
.venv/bin/pytest tests/test_package.py::test_version -xvs
```

Markers: `brainiak`, `network` (deselected by default), `gpu`, `slow`.
Run `make lint` and `make test` before committing.

## Architecture

Package layout:

```
htfa/
├── __init__.py       # public API, __version__
├── _kernels.py       # numerical kernels, factor mixtures, weights, objective
├── _engines.py       # compute engines (NumPy / torch)
├── _lazy.py          # lazy optional-dependency imports
├── tfa.py            # single-subject TFA
├── htfa.py           # hierarchical multi-subject HTFA
├── io.py             # NIfTI / image loading
├── fit.py            # main API entry point
├── results.py        # results container, plotting, export
├── connectivity.py   # factor-level connectivity
├── groups.py         # nested templates and permutation comparisons
├── stats.py          # reconstruction statistics and nominal parameter counts
└── synthetic.py      # synthetic data generation
tests/                # pytest suite (real implementations, no mocks)
docs/                 # Sphinx documentation
examples/             # example scripts / gallery
```

Dependencies: numpy, scipy, scikit-learn, nibabel, joblib, threadpoolctl (torch on
Apple Silicon). Extras: `plot`, `images`, `nilearn`, `gpu`, `dev`, `docs`.

## Testing Strategy

- Flat `tests/test_*.py` files; shared fixtures in `tests/conftest.py`.
- Real implementations only: no mocks.
- Markers (registered, `--strict-markers`):
  - `brainiak`: compares against the reference BrainIAK implementation (needs brainiak); deselected by default
  - `network`: needs network access / downloads data; deselected by default
  - `gpu`: needs a GPU backend (torch with CUDA or MPS)
  - `slow`: long-running test
