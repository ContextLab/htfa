# Contributing

## Setup

The project uses [uv](https://docs.astral.sh/uv/) and a `Makefile`:

```bash
git clone https://github.com/ContextLab/htfa.git
cd htfa
make install
```

`make install` creates `.venv` (Python 3.12) and installs the package in editable mode with the `dev` and `docs` extras.

## Commands

| Command | What it does |
|-|-|
| `make test` | `pytest -q` in `.venv` |
| `make lint` | `ruff check .`, `ruff format --check .` and `mypy` |
| `make docs` | builds the Sphinx site into `docs/_build/html` |

Run a single test with, for example, `.venv/bin/pytest tests/test_package.py::test_version -xvs`. To fix formatting, run `.venv/bin/ruff format .` (it also formats Python code blocks in Markdown files).

## Tests

- Tests live in flat `tests/test_*.py` files; shared fixtures are in `tests/conftest.py`.
- Tests use real implementations and real data (no mocks).
- Markers (`--strict-markers` is on): `brainiak` (needs brainiak) and `network` (needs network access) are deselected by default; `gpu` needs a CUDA or MPS torch backend; `slow` marks long-running tests.

## Before submitting

1. Add tests for your changes.
2. Update the documentation (`docs/`, docstrings, README) if behavior or the API changed.
3. Run `make lint`, `make test` and `make docs`. If a fix changes anything, run all three again.
