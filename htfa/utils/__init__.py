"""Utility functions for HTFA."""

# Hardware detection
from .hardware_detection import (
    clear_hardware_cache,
    detect_all_hardware,
    detect_cuda,
    detect_metal,
    detect_rocm,
    detect_tpu,
    get_cpu_info,
    get_gpu_memory_mb,
    get_memory_info,
    get_recommended_backend,
    has_gpu,
)

# Library detection
from .library_detection import (
    benchmark_backend,
    check_version_compatibility,
    clear_detection_cache,
    detect_all_libraries,
    detect_jax,
    detect_numpy,
    detect_pytorch,
    detect_tensorflow,
    estimate_memory_usage,
    get_optimal_backend,
    test_import,
)

__all__ = [
    # Hardware detection
    "detect_cuda",
    "detect_metal",
    "detect_rocm",
    "detect_tpu",
    "get_cpu_info",
    "get_memory_info",
    "has_gpu",
    "get_gpu_memory_mb",
    "detect_all_hardware",
    "clear_hardware_cache",
    "get_recommended_backend",
    # Library detection
    "detect_numpy",
    "detect_jax",
    "detect_pytorch",
    "detect_tensorflow",
    "test_import",
    "check_version_compatibility",
    "detect_all_libraries",
    "get_optimal_backend",
    "clear_detection_cache",
    "benchmark_backend",
    "estimate_memory_usage",
]
