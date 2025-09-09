# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hierarchical Topographic Factor Analysis (HTFA) - A lightweight Python implementation for neuroimaging data analysis based on the BrainIAK algorithm. Uses Poetry for dependency management and provides minimal dependencies for easy installation.

## Commands

### Development
```bash
# Install dependencies and setup environment
make install

# Run full test suite with coverage
make test

# Run specific test
poetry run pytest tests/unit/test_basic.py::test_tfa_initialization -xvs

# Format code (auto-fix)
make codestyle  # or make formatting

# Lint and check code quality (no auto-fix)
make lint  # Runs: test, check-codestyle, mypy, check-safety

# Individual checks
make check-codestyle  # Style checking only
make mypy            # Type checking
make check-safety     # Security scan
```

### Before Committing
Always run: `make lint`

## Architecture

### Core Algorithm Structure
```
htfa/
├── core/
│   ├── tfa.py     # Base TFA with K-means init, factor estimation
│   └── htfa.py    # Hierarchical extension, multi-subject analysis
├── htfa/          # Advanced implementations
│   ├── optimization.py  # Adam, ALS, mini-batch optimizers
│   ├── core.py         # Abstract backend interface
│   └── backends/       # JAX/PyTorch acceleration
├── bids.py        # BIDS dataset integration
├── fit.py         # Main API entry point (htfa.fit_bids)
├── results.py     # HTFAResults with plotting/NIfTI export
└── validation.py  # Input validation framework
```

### Key Design Patterns

1. **Scikit-learn Interface**: All estimators inherit from `BaseEstimator`
2. **Factory Pattern**: `fit.py` auto-detects input type (BIDS vs numpy)
3. **Strategy Pattern**: Swappable optimization backends (JAX/PyTorch/NumPy)
4. **Builder Pattern**: HTFAResults incrementally builds visualization/export

### Algorithm Flow

1. **TFA** (single subject):
   - K-means initialization on spatial coordinates
   - Iterative factor/weight estimation
   - Ridge regression for weights

2. **HTFA** (multi-subject):
   - Initialize per-subject TFA models
   - Compute global template (average factors)
   - Hierarchical optimization with MAP estimation
   - Factor matching across subjects

### Key Dependencies
- Core: `numpy`, `scipy`, `scikit-learn`
- Neuroimaging: `nilearn`, `nibabel`, `pybids`
- Visualization: `matplotlib` (required), `seaborn`, `plotly` (dev)

## Testing Strategy

- Unit tests in `tests/unit/` - Component isolation
- Integration tests in `tests/integration/` - End-to-end flows
- Optimization tests in `tests/test_optimization.py` - Algorithm validation
- No mocks for core functionality - test real implementations
- Coverage target: >90%