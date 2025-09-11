"""Library detection module for automatic backend selection."""

import importlib
import warnings
from functools import lru_cache
from typing import Dict, Any, Optional
from packaging import version


@lru_cache(maxsize=1)
def detect_numpy() -> Dict[str, Any]:
    """Detect NumPy availability and version.
    
    Returns:
        Dictionary with NumPy information:
        - available: bool (always True as it's a core dependency)
        - version: str
    """
    try:
        import numpy as np
        return {
            'available': True,
            'version': np.__version__
        }
    except ImportError:
        # Should never happen as numpy is a core dependency
        return {
            'available': False,
            'version': None
        }


@lru_cache(maxsize=1)
def detect_jax() -> Dict[str, Any]:
    """Detect JAX availability and capabilities.
    
    Returns:
        Dictionary with JAX information:
        - available: bool
        - version: str or None
        - gpu_available: bool
        - tpu_available: bool
        - backend: str (cpu, gpu, or tpu)
    """
    result = {
        'available': False,
        'version': None,
        'gpu_available': False,
        'tpu_available': False,
        'backend': 'cpu'
    }
    
    try:
        import jax
        result['available'] = True
        result['version'] = jax.__version__
        
        # Check for GPU/TPU support
        try:
            devices = jax.devices()
            for device in devices:
                if 'gpu' in str(device).lower():
                    result['gpu_available'] = True
                    result['backend'] = 'gpu'
                elif 'tpu' in str(device).lower():
                    result['tpu_available'] = True
                    result['backend'] = 'tpu'
        except Exception:
            pass
            
        # Also check jaxlib version
        try:
            import jaxlib
            result['jaxlib_version'] = jaxlib.__version__
        except ImportError:
            pass
            
    except ImportError:
        pass
        
    return result


@lru_cache(maxsize=1)
def detect_pytorch() -> Dict[str, Any]:
    """Detect PyTorch availability and capabilities.
    
    Returns:
        Dictionary with PyTorch information:
        - available: bool
        - version: str or None
        - cuda_available: bool
        - mps_available: bool (Metal Performance Shaders on macOS)
        - device: str (cpu, cuda, or mps)
    """
    result = {
        'available': False,
        'version': None,
        'cuda_available': False,
        'mps_available': False,
        'device': 'cpu'
    }
    
    try:
        import torch
        result['available'] = True
        result['version'] = torch.__version__
        
        # Check CUDA availability
        if torch.cuda.is_available():
            result['cuda_available'] = True
            result['device'] = 'cuda'
            result['cuda_device_count'] = torch.cuda.device_count()
            
        # Check MPS availability (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            result['mps_available'] = True
            if not result['cuda_available']:  # Prefer CUDA over MPS
                result['device'] = 'mps'
                
    except ImportError:
        pass
        
    return result


@lru_cache(maxsize=1)
def detect_tensorflow() -> Dict[str, Any]:
    """Detect TensorFlow availability (for future use).
    
    Returns:
        Dictionary with TensorFlow information:
        - available: bool
        - version: str or None
        - gpu_available: bool
    """
    result = {
        'available': False,
        'version': None,
        'gpu_available': False
    }
    
    try:
        import tensorflow as tf
        result['available'] = True
        result['version'] = tf.__version__
        
        # Check GPU availability
        result['gpu_available'] = len(tf.config.list_physical_devices('GPU')) > 0
        
    except ImportError:
        pass
        
    return result


def test_import(module_name: str) -> bool:
    """Safely test if a module can be imported.
    
    Args:
        module_name: Name of the module to test.
        
    Returns:
        True if module can be imported, False otherwise.
    """
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def check_version_compatibility(
    library: str,
    min_version: Optional[str] = None,
    max_version: Optional[str] = None
) -> bool:
    """Check if library version is within specified range.
    
    Args:
        library: Library name to check.
        min_version: Minimum required version (inclusive).
        max_version: Maximum allowed version (exclusive).
        
    Returns:
        True if version is compatible, False otherwise.
    """
    detection_funcs = {
        'numpy': detect_numpy,
        'jax': detect_jax,
        'pytorch': detect_pytorch,
        'torch': detect_pytorch,  # Alias
        'tensorflow': detect_tensorflow,
        'tf': detect_tensorflow  # Alias
    }
    
    if library not in detection_funcs:
        return False
        
    info = detection_funcs[library]()
    if not info['available'] or not info['version']:
        return False
        
    try:
        current = version.parse(info['version'])
        
        if min_version and current < version.parse(min_version):
            return False
            
        if max_version and current >= version.parse(max_version):
            return False
            
        return True
        
    except Exception:
        return False


def detect_all_libraries() -> Dict[str, Any]:
    """Detect all available libraries.
    
    Returns:
        Dictionary with all library information.
    """
    return {
        'numpy': detect_numpy(),
        'jax': detect_jax(),
        'pytorch': detect_pytorch(),
        'tensorflow': detect_tensorflow()
    }


def get_optimal_backend() -> str:
    """Determine optimal backend based on available libraries.
    
    Returns:
        Optimal backend name: 'jax', 'pytorch', or 'numpy'
    """
    libs = detect_all_libraries()
    
    # Priority 1: JAX with GPU/TPU
    if libs['jax']['available']:
        if libs['jax']['gpu_available'] or libs['jax']['tpu_available']:
            return 'jax'
    
    # Priority 2: PyTorch with GPU
    if libs['pytorch']['available']:
        if libs['pytorch']['cuda_available'] or libs['pytorch']['mps_available']:
            return 'pytorch'
    
    # Priority 3: JAX with CPU (JIT compilation advantage)
    if libs['jax']['available']:
        return 'jax'
    
    # Priority 4: PyTorch with CPU
    if libs['pytorch']['available']:
        return 'pytorch'
    
    # Priority 5: NumPy (always available)
    return 'numpy'


def clear_detection_cache():
    """Clear all cached library detection results."""
    detect_numpy.cache_clear()
    detect_jax.cache_clear()
    detect_pytorch.cache_clear()
    detect_tensorflow.cache_clear()


# Performance testing stubs for future use
def benchmark_backend(backend: str, test_size: int = 1000) -> float:
    """Benchmark a backend's performance (stub for future implementation).
    
    Args:
        backend: Backend name to test.
        test_size: Size of test array.
        
    Returns:
        Time in seconds for test operation.
    """
    # This would run a standard operation and time it
    # For now, return placeholder
    return 0.0


def estimate_memory_usage(backend: str, array_shape: tuple) -> int:
    """Estimate memory usage for given backend and array shape (stub).
    
    Args:
        backend: Backend name.
        array_shape: Shape of array to estimate.
        
    Returns:
        Estimated memory usage in MB.
    """
    # Placeholder for memory estimation
    import numpy as np
    elements = np.prod(array_shape)
    bytes_per_element = 8  # Assuming float64
    return (elements * bytes_per_element) // (1024 * 1024)


if __name__ == '__main__':
    # Test library detection
    import json
    print("Library Detection Results:")
    print(json.dumps(detect_all_libraries(), indent=2))
    print(f"\nOptimal backend: {get_optimal_backend()}")