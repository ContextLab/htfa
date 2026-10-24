"""Packaging smoke test."""

import importlib.metadata

import htfa


def test_version():
    assert htfa.__version__ == "0.1.0"
    assert importlib.metadata.version("htfa") == "0.1.0"
