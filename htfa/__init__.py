"""Hierarchical Topographic Factor Analysis."""

from . import datasets as datasets
from ._lazy import require, set_autoinstall
from .brainnet import animate_network, plot_network
from .connectivity import dynamic_connectivity, isfc, static_connectivity
from .fit import fit
from .groups import group_permutation_test
from .htfa import HTFA
from .io import Subject, load
from .results import HTFAResults
from .synthetic import make_group_data, make_htfa_data, make_tfa_data
from .tfa import TFA
from .timecourse import plot_timecourse

__version__ = "0.1.0"

__all__ = [
    "HTFA",
    "HTFAResults",
    "Subject",
    "TFA",
    "animate_network",
    "dynamic_connectivity",
    "fit",
    "group_permutation_test",
    "isfc",
    "load",
    "make_group_data",
    "make_htfa_data",
    "make_tfa_data",
    "plot_network",
    "plot_timecourse",
    "require",
    "set_autoinstall",
    "static_connectivity",
]
