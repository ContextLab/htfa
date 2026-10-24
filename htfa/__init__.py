"""Hierarchical Topographic Factor Analysis."""

from ._lazy import set_autoinstall
from .connectivity import dynamic_connectivity, isfc, static_connectivity
from .fit import fit
from .htfa import HTFA
from .io import load
from .results import HTFAResults
from .tfa import TFA

__version__ = "0.1.0"

__all__ = [
    "HTFA",
    "HTFAResults",
    "TFA",
    "dynamic_connectivity",
    "fit",
    "isfc",
    "load",
    "set_autoinstall",
    "static_connectivity",
]
