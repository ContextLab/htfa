# mypy: disable-error-code=attr-defined
"""Hierarchical Topographic Factor Analysis

A lightweight, standalone implementation of Hierarchical Topographic Factor Analysis (HTFA)
for neuroimaging data, based on the BrainIAK implementation but with minimal dependencies.
"""

from importlib import metadata as importlib_metadata

from htfa.core.htfa import HTFA
from htfa.core.tfa import TFA
from htfa.fit import fit
from htfa.results import HTFAResults

# Try to import optional BIDS functionality
try:
    from htfa.bids import fit_bids
    __all__ = ["HTFA", "TFA", "HTFAResults", "fit", "fit_bids", "get_version", "version"]
except ImportError:
    __all__ = ["HTFA", "TFA", "HTFAResults", "fit", "get_version", "version"]


def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: str = get_version()
