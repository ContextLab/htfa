"""Utility functions for HTFA."""

# Hardware detection
from .hardware_detection import (
    detect_cuda,
    detect_metal,
    detect_rocm,
    detect_tpu,
    get_cpu_info,
    get_memory_info,
    has_gpu,
    get_gpu_memory_mb,
    detect_all_hardware,
    clear_hardware_cache,
    get_recommended_backend
)

# Library detection
from .library_detection import (
    detect_numpy,
    detect_jax,
    detect_pytorch,
    detect_tensorflow,
    test_import,
    check_version_compatibility,
    detect_all_libraries,
    get_optimal_backend,
    clear_detection_cache,
    benchmark_backend,
    estimate_memory_usage
)

__all__ = [
    # Hardware detection
    'detect_cuda',
    'detect_metal',
    'detect_rocm',
    'detect_tpu',
    'get_cpu_info',
    'get_memory_info',
    'has_gpu',
    'get_gpu_memory_mb',
    'detect_all_hardware',
    'clear_hardware_cache',
    'get_recommended_backend',
    # Library detection
    'detect_numpy',
    'detect_jax',
    'detect_pytorch',
    'detect_tensorflow',
    'test_import',
    'check_version_compatibility',
    'detect_all_libraries',
    'get_optimal_backend',
    'clear_detection_cache',
    'benchmark_backend',
    'estimate_memory_usage'
]
