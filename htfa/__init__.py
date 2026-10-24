"""Hierarchical Topographic Factor Analysis."""

from ._lazy import set_autoinstall
from .connectivity import dynamic_connectivity, isfc, static_connectivity
from .fit import fit
from .htfa import HTFA
from .io import Subject, load
from .results import HTFAResults
from .synthetic import make_htfa_data, make_tfa_data
from .tfa import TFA

__version__ = "0.1.0"

__all__ = [
    "HTFA",
    "HTFAResults",
    "Subject",
    "TFA",
    "dynamic_connectivity",
    "fit",
    "isfc",
    "load",
    "make_htfa_data",
    "make_tfa_data",
    "set_autoinstall",
    "static_connectivity",
]
