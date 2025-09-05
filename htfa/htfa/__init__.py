# type: ignore[attr-defined]
"""Hierarchical Topographic Factor Analysis"""

from importlib import metadata as importlib_metadata

from .core import HTFA, HTFABackend, NumPyBackend
from .benchmark import HTFABenchmark, BenchmarkResult, benchmark_function
from .optimization import (
    Optimizer, AlternatingLeastSquares, AdamOptimizer, 
    MiniBatchOptimizer, SparseOptimizer, create_optimizer,
    OptimizationResult, OptimizationScheduler
)

# Try to import optional backends
JAXBackend = None
try:
    from .backends.jax_backend import JAXBackend
except ImportError:
    pass

PyTorchBackend = None
try:
    from .backends.pytorch_backend import PyTorchBackend  
except ImportError:
    pass


def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: str = get_version()

__all__ = [
    'HTFA',
    'HTFABackend',
    'NumPyBackend',
    'JAXBackend',
    'PyTorchBackend',
    'HTFABenchmark',
    'BenchmarkResult', 
    'benchmark_function',
    'Optimizer',
    'AlternatingLeastSquares',
    'AdamOptimizer',
    'MiniBatchOptimizer',
    'SparseOptimizer',
    'create_optimizer',
    'OptimizationResult',
    'OptimizationScheduler',
    'version',
    'get_version'
]
