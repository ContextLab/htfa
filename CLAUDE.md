# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python package for Hierarchical Topographic Factor Analysis (HTFA) built using Poetry for dependency management. The project aims to create a lightweight, standalone implementation based on the BrainIAK HTFA algorithm but with minimal dependencies for easy installation and use.

**Current Status**: Early development phase with basic infrastructure complete.

## Package Structure

- `htfa/core/` - Core HTFA and TFA implementations
- `htfa/utils/` - Utility functions and helpers
- `tests/unit/` - Unit tests
- `tests/integration/` - Integration tests (to be added)
- `docs/` - Documentation (to be added)
- `tutorials/` - Jupyter notebook tutorials (to be added)
- `examples/` - Example scripts (to be added)
- Package configuration in `pyproject.toml`
- Development tooling configured via Makefile

## Current Implementation Status

### Completed
- ✅ Basic package structure and dependencies
- ✅ Core TFA and HTFA class scaffolding with scikit-learn interface
- ✅ Basic test suite with synthetic data
- ✅ GitHub issues created for development roadmap
- ✅ Technical design document

### Core Dependencies
- `numpy` - Numerical computations
- `scipy` - Optimization algorithms  
- `scikit-learn` - Base estimator interface and clustering
- `nilearn` - Neuroimaging analysis and visualization
- `nibabel` - NIfTI file handling
- `pybids` - BIDS dataset parsing
- `pandas` - Tabular data handling
- `matplotlib` - Basic visualization
- `seaborn`, `plotly` - Advanced visualization (dev dependencies)

### Development Roadmap
See GitHub issues:
- Issue #60: Core TFA optimization algorithm
- Issue #61: HTFA hierarchical optimization
- Issue #62: Comprehensive test suite
- Issue #63: Documentation and tutorials
- Issue #64: Performance optimization
- Issue #65: Advanced HTFA variants
- Issue #66: BIDS integration and intuitive user interface

### Design Goals
- **Simple API**: Single function call `htfa.fit_bids()` for complete analysis
- **BIDS Native**: Direct support for BIDS neuroimaging datasets
- **Sensible Defaults**: Automatic parameter inference from data
- **Rich Results**: HTFAResults class with built-in plotting and NIfTI export
- **Minimal Dependencies**: Focused on essential neuroimaging packages

## Common Development Commands

### Setup and Installation
- `make install` - Install dependencies and setup development environment
- `make pre-commit-install` - Install pre-commit hooks (after `git init`)

### Code Quality and Testing
- `make test` - Run pytest with coverage reporting
- `make lint` - Run all linters (equivalent to `make test && make check-codestyle && make mypy && make check-safety`)
- `make codestyle` or `make formatting` - Auto-format code using pyupgrade, isort, and black
- `make check-codestyle` - Check code style without modifying files
- `make mypy` - Run static type checking
- `make check-safety` - Security checks with safety and bandit

### Single Test Execution
To run a specific test file:
```bash
poetry run pytest tests/test_example/test_hello.py -v
```

To run a specific test function:
```bash
poetry run pytest tests/test_example/test_hello.py::test_hello -v
```

### Development Tools

The project uses:
- **Poetry** for dependency management
- **Black** for code formatting (88 character line length)
- **isort** for import sorting
- **mypy** for type checking with strict settings
- **pytest** for testing with coverage reporting
- **pre-commit** hooks for automated quality checks
- **darglint** for docstring validation (Google style)

### Before Committing

Always run the full lint suite to ensure code quality:
```bash
make lint
```

This runs tests, code style checks, type checking, and security scans.

### Architecture Notes

- Package follows standard Python package structure with `__init__.py` containing version management
- Uses importlib_metadata for Python <3.8 compatibility
- Test coverage reporting generates SVG badge in `assets/images/coverage.svg`
- Supports Python 3.7+ with type hints throughout