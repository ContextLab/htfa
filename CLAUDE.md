# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hierarchical Topographic Factor Analysis (HTFA) - A lightweight Python implementation for neuroimaging data analysis based on the BrainIAK algorithm. Uses Poetry for dependency management and provides minimal dependencies for easy installation.

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

```
htfa/
├── __init__.py       # public API, __version__
├── _kernels.py       # numerical kernels (RBF factors, weights, objective)
├── _engines.py       # compute engines (NumPy / torch)
├── _lazy.py          # lazy optional-dependency imports
├── tfa.py            # single-subject TFA
├── htfa.py           # hierarchical multi-subject HTFA
├── io.py             # NIfTI / image loading
├── fit.py            # main API entry point
├── results.py        # results container, plotting, export
├── connectivity.py   # factor-level connectivity
└── synthetic.py      # synthetic data generation
tests/                # pytest suite (real implementations, no mocks)
docs/                 # Sphinx documentation
examples/             # example scripts / gallery
```

Dependencies: numpy, scipy, scikit-learn, nibabel, joblib, threadpoolctl (torch on
Apple Silicon). Extras: `plot`, `images`, `nilearn`, `gpu`, `dev`, `docs`.

## Testing Strategy

- Unit tests in `tests/unit/` - Component isolation
- Integration tests in `tests/integration/` - End-to-end flows
- Optimization tests in `tests/test_optimization.py` - Algorithm validation
- No mocks for core functionality - test real implementations
- Coverage target: >90%