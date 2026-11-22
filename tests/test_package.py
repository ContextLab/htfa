"""Packaging smoke test."""

import importlib.metadata

import htfa


def test_version():
    assert htfa.__version__ == "0.2.0"
    assert importlib.metadata.version("htfa") == "0.2.0"
