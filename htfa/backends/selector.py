"""Backend selector for automatic backend selection."""

import logging
import warnings
from typing import Optional, Dict, Any, Tuple
from functools import lru_cache

from ..utils import (
    detect_all_hardware,
    detect_all_libraries,
    has_gpu,
    get_gpu_memory_mb
)

logger = logging.getLogger(__name__)


class BackendSelector:
    """Intelligent backend selector based on hardware and library availability."""
    
    # Backend priority scores (higher is better)
    BACKEND_SCORES = {
        'jax_tpu': 100,
        'jax_gpu': 90,
        'pytorch_cuda': 85,
        'pytorch_mps': 80,
        'jax_cpu': 60,
        'pytorch_cpu': 50,
        'numpy': 30
    }
    
    def __init__(self):
        """Initialize the backend selector."""
        self._hardware = None
        self._libraries = None
        self._selection_cache = {}
        
    @property
    def hardware(self) -> Dict[str, Any]:
        """Get hardware information (cached)."""
        if self._hardware is None:
            self._hardware = detect_all_hardware()
        return self._hardware
        
    @property
    def libraries(self) -> Dict[str, Any]:
        """Get library information (cached)."""
        if self._libraries is None:
            self._libraries = detect_all_libraries()
        return self._libraries
        
    def _score_backends(self) -> Dict[str, Tuple[int, str]]:
        """Score each available backend.
        
        Returns:
            Dictionary mapping backend names to (score, reason) tuples.
        """
        scores = {}
        
        # Check JAX availability
        if self.libraries['jax']['available']:
            if self.libraries['jax']['tpu_available']:
                scores['jax'] = (self.BACKEND_SCORES['jax_tpu'], 
                               "JAX with TPU acceleration")
            elif self.libraries['jax']['gpu_available']:
                scores['jax'] = (self.BACKEND_SCORES['jax_gpu'], 
                               "JAX with GPU acceleration")
            else:
                # JAX with CPU - check memory for JIT benefit
                if self.hardware['memory']['total_mb'] >= 8192:  # 8GB+
                    scores['jax'] = (self.BACKEND_SCORES['jax_cpu'], 
                                   "JAX with JIT compilation")
                else:
                    scores['jax'] = (self.BACKEND_SCORES['jax_cpu'] - 10, 
                                   "JAX on CPU (limited memory)")
        
        # Check PyTorch availability
        if self.libraries['pytorch']['available']:
            if self.libraries['pytorch']['cuda_available']:
                scores['pytorch'] = (self.BACKEND_SCORES['pytorch_cuda'], 
                                   f"PyTorch with CUDA ({self.libraries['pytorch'].get('cuda_device_count', 1)} GPU(s))")
            elif self.libraries['pytorch']['mps_available']:
                scores['pytorch'] = (self.BACKEND_SCORES['pytorch_mps'], 
                                   "PyTorch with Metal Performance Shaders")
            else:
                scores['pytorch'] = (self.BACKEND_SCORES['pytorch_cpu'], 
                                   "PyTorch on CPU")
        
        # NumPy is always available
        scores['numpy'] = (self.BACKEND_SCORES['numpy'], 
                         "NumPy (fallback backend)")
        
        return scores
        
    def select_backend(self, backend: Optional[str] = None) -> str:
        """Select the optimal backend.
        
        Args:
            backend: Explicitly requested backend. If None, auto-select.
            
        Returns:
            Selected backend name ('jax', 'pytorch', or 'numpy').
        """
        # Use explicit backend if provided
        if backend is not None:
            # Validate the backend exists
            valid_backends = ['numpy', 'jax', 'pytorch']
            if backend.lower() in valid_backends:
                logger.info(f"Using explicitly requested backend: {backend}")
                return backend.lower()
            else:
                warnings.warn(f"Invalid backend '{backend}', auto-selecting instead")
                backend = None
        
        # Check cache
        cache_key = 'auto'
        if cache_key in self._selection_cache:
            backend, reason = self._selection_cache[cache_key]
            logger.info(f"Using cached backend selection: {backend} ({reason})")
            return backend
        
        # Score all backends
        scores = self._score_backends()
        
        # Select highest scoring backend
        if not scores:
            backend = 'numpy'
            reason = "No backends available, using NumPy"
        else:
            # Sort by score (descending)
            sorted_backends = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
            backend, (score, reason) = sorted_backends[0]
            
            # Log all available backends
            logger.debug("Backend scores:")
            for b, (s, r) in sorted_backends:
                logger.debug(f"  {b}: {s} - {r}")
        
        # Cache the selection
        self._selection_cache[cache_key] = (backend, reason)
        
        # Log selection
        logger.info(f"Auto-selected backend: {backend} ({reason})")
        
        # Log additional context
        if backend == 'jax' and not self.hardware['has_gpu']:
            logger.info("JAX selected for CPU - will use JIT compilation for performance")
        elif backend == 'pytorch' and not self.hardware['has_gpu']:
            logger.info("PyTorch selected for CPU - consider JAX for better CPU performance")
        elif backend == 'numpy':
            if not self.libraries['jax']['available'] and not self.libraries['pytorch']['available']:
                logger.info("NumPy selected - install JAX or PyTorch for better performance")
            else:
                logger.warning("NumPy selected despite other backends being available - check configuration")
        
        return backend
        
    def get_device(self, backend: str) -> Optional[str]:
        """Get the device string for the selected backend.
        
        Args:
            backend: The backend name.
            
        Returns:
            Device string (e.g., 'cuda:0', 'mps', 'cpu') or None.
        """
        if backend == 'pytorch':
            if self.libraries['pytorch']['cuda_available']:
                return 'cuda:0'
            elif self.libraries['pytorch']['mps_available']:
                return 'mps'
            else:
                return 'cpu'
        elif backend == 'jax':
            return self.libraries['jax'].get('backend', 'cpu')
        else:
            return None
            
    def estimate_memory_requirement(self, n_voxels: int, n_times: int, k_components: int) -> int:
        """Estimate memory requirement for HTFA.
        
        Args:
            n_voxels: Number of voxels.
            n_times: Number of time points.
            k_components: Number of components.
            
        Returns:
            Estimated memory requirement in MB.
        """
        # Rough estimation: main data + factors + weights
        data_size = n_voxels * n_times * 8  # float64
        factors_size = k_components * n_times * 8
        weights_size = k_components * n_voxels * 8
        
        # Add overhead (temporary arrays, etc.)
        overhead = 2.0
        
        total_bytes = (data_size + factors_size + weights_size) * overhead
        return int(total_bytes / (1024 * 1024))
        
    def validate_selection(self, backend: str, n_voxels: int, n_times: int, k_components: int) -> bool:
        """Validate if selected backend can handle the problem size.
        
        Args:
            backend: Selected backend.
            n_voxels: Number of voxels.
            n_times: Number of time points.
            k_components: Number of components.
            
        Returns:
            True if backend can handle the problem size.
        """
        required_mb = self.estimate_memory_requirement(n_voxels, n_times, k_components)
        
        # Check GPU memory if using GPU backend
        if backend in ['jax', 'pytorch'] and self.hardware['has_gpu']:
            gpu_memory = get_gpu_memory_mb()
            if gpu_memory and required_mb > gpu_memory * 0.8:  # Leave 20% buffer
                logger.warning(f"Problem size ({required_mb}MB) may exceed GPU memory ({gpu_memory}MB)")
                return False
        
        # Check system memory
        available_mb = self.hardware['memory']['available_mb']
        if required_mb > available_mb * 0.8:  # Leave 20% buffer
            logger.warning(f"Problem size ({required_mb}MB) may exceed available memory ({available_mb}MB)")
            return False
            
        return True
        
    def clear_cache(self):
        """Clear selection cache."""
        self._selection_cache.clear()
        self._hardware = None
        self._libraries = None


# Global selector instance
_selector = BackendSelector()


def select_backend(backend: Optional[str] = None) -> str:
    """Select the optimal backend.
    
    Args:
        backend: Explicitly requested backend. If None, auto-select.
        
    Returns:
        Selected backend name ('jax', 'pytorch', or 'numpy').
    """
    return _selector.select_backend(backend)


def get_backend_device(backend: str) -> Optional[str]:
    """Get the device string for a backend.
    
    Args:
        backend: The backend name.
        
    Returns:
        Device string or None.
    """
    return _selector.get_device(backend)


def validate_backend_selection(backend: str, n_voxels: int, n_times: int, k_components: int) -> bool:
    """Validate if backend can handle problem size.
    
    Args:
        backend: Selected backend.
        n_voxels: Number of voxels.
        n_times: Number of time points.
        k_components: Number of components.
        
    Returns:
        True if backend can handle the problem size.
    """
    return _selector.validate_selection(backend, n_voxels, n_times, k_components)


def clear_backend_cache():
    """Clear all backend selection caches."""
    _selector.clear_cache()
    from ..utils import clear_hardware_cache, clear_detection_cache
    clear_hardware_cache()
    clear_detection_cache()