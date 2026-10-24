VENV := .venv
BIN := $(VENV)/bin

.PHONY: install
install:
	uv venv -p 3.12 $(VENV)
	VIRTUAL_ENV=$(VENV) uv pip install -e ".[dev,docs]"

.PHONY: test
test:
	$(BIN)/pytest -q

.PHONY: lint
lint:
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .
	$(BIN)/mypy

.PHONY: docs
docs:
	$(BIN)/sphinx-build -b html docs docs/_build/html
