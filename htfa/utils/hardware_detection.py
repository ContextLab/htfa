"""Hardware detection module for automatic backend selection."""

import os
import platform
import subprocess
import warnings
from functools import lru_cache
from typing import Dict, Optional, Any, List
import psutil


@lru_cache(maxsize=1)
def detect_cuda() -> Dict[str, Any]:
    """Detect CUDA availability and GPU information.
    
    Returns:
        Dictionary with CUDA information:
        - available: bool
        - version: str or None
        - device_count: int
        - devices: list of device info dicts
    """
    result = {
        'available': False,
        'version': None,
        'device_count': 0,
        'devices': []
    }
    
    # Check CUDA environment variables
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home and os.path.exists(cuda_home):
        result['available'] = True
        
    # Try nvidia-smi for detailed GPU info
    try:
        output = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,compute_cap', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if output.returncode == 0:
            result['available'] = True
            devices = []
            for line in output.stdout.strip().split('\n'):
                if line:
                    parts = line.split(', ')
                    if len(parts) >= 2:
                        devices.append({
                            'name': parts[0],
                            'memory_mb': int(parts[1].replace(' MiB', '')) if ' MiB' in parts[1] else 0,
                            'compute_capability': parts[2] if len(parts) > 2 else 'unknown'
                        })
            result['devices'] = devices
            result['device_count'] = len(devices)
            
        # Get CUDA version
        version_output = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if version_output.returncode == 0:
            result['version'] = version_output.stdout.strip().split('\n')[0]
            
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
        
    return result


@lru_cache(maxsize=1)
def detect_metal() -> Dict[str, bool]:
    """Detect Metal availability on macOS.
    
    Returns:
        Dictionary with Metal information:
        - available: bool
    """
    result = {'available': False}
    
    if platform.system() == 'Darwin':
        # Check if we're on macOS 10.11+ (Metal requirement)
        try:
            mac_version = platform.mac_ver()[0]
            if mac_version:
                major, minor = map(int, mac_version.split('.')[:2])
                if major > 10 or (major == 10 and minor >= 11):
                    result['available'] = True
        except (ValueError, AttributeError):
            pass
            
    return result


@lru_cache(maxsize=1)
def detect_rocm() -> Dict[str, Any]:
    """Detect ROCm availability for AMD GPUs.
    
    Returns:
        Dictionary with ROCm information:
        - available: bool
        - version: str or None
    """
    result = {
        'available': False,
        'version': None
    }
    
    # Check for ROCm installation
    rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')
    if os.path.exists(rocm_path):
        result['available'] = True
        # Try to get version
        version_file = os.path.join(rocm_path, '.info', 'version')
        if os.path.exists(version_file):
            try:
                with open(version_file, 'r') as f:
                    result['version'] = f.read().strip()
            except IOError:
                pass
                
    # Try rocm-smi command
    try:
        output = subprocess.run(
            ['rocm-smi', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if output.returncode == 0:
            result['available'] = True
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
        
    return result


@lru_cache(maxsize=1)
def detect_tpu() -> Dict[str, bool]:
    """Detect TPU availability.
    
    Returns:
        Dictionary with TPU information:
        - available: bool
    """
    result = {'available': False}
    
    # Check for TPU via environment variable (Google Cloud TPU)
    if os.environ.get('TPU_NAME'):
        result['available'] = True
        
    # Check for TPU device files (Coral Edge TPU)
    if os.path.exists('/dev/apex_0') or os.path.exists('/dev/accel0'):
        result['available'] = True
        
    return result


@lru_cache(maxsize=1)
def get_cpu_info() -> Dict[str, Any]:
    """Get CPU information.
    
    Returns:
        Dictionary with CPU information:
        - count: int (logical cores)
        - physical_count: int (physical cores)
        - frequency_mhz: float
    """
    result = {
        'count': psutil.cpu_count(logical=True),
        'physical_count': psutil.cpu_count(logical=False),
        'frequency_mhz': 0.0
    }
    
    try:
        freq = psutil.cpu_freq()
        if freq:
            result['frequency_mhz'] = freq.current
    except Exception:
        pass
        
    return result


@lru_cache(maxsize=1)
def get_memory_info() -> Dict[str, int]:
    """Get system memory information.
    
    Returns:
        Dictionary with memory information:
        - total_mb: int
        - available_mb: int
    """
    mem = psutil.virtual_memory()
    return {
        'total_mb': mem.total // (1024 * 1024),
        'available_mb': mem.available // (1024 * 1024)
    }


def has_gpu() -> bool:
    """Quick check if any GPU is available.
    
    Returns:
        True if any GPU (CUDA, Metal, ROCm, TPU) is available.
    """
    return (
        detect_cuda()['available'] or
        detect_metal()['available'] or
        detect_rocm()['available'] or
        detect_tpu()['available']
    )


def get_gpu_memory_mb() -> Optional[int]:
    """Get total GPU memory if available.
    
    Returns:
        Total GPU memory in MB, or None if no GPU.
    """
    cuda_info = detect_cuda()
    if cuda_info['available'] and cuda_info['devices']:
        return sum(d.get('memory_mb', 0) for d in cuda_info['devices'])
    return None


def detect_all_hardware() -> Dict[str, Any]:
    """Detect all hardware capabilities.
    
    Returns:
        Dictionary with all hardware information.
    """
    return {
        'cuda': detect_cuda(),
        'metal': detect_metal(),
        'rocm': detect_rocm(),
        'tpu': detect_tpu(),
        'cpu': get_cpu_info(),
        'memory': get_memory_info(),
        'has_gpu': has_gpu(),
        'gpu_memory_mb': get_gpu_memory_mb()
    }


def clear_hardware_cache():
    """Clear all cached hardware detection results."""
    detect_cuda.cache_clear()
    detect_metal.cache_clear()
    detect_rocm.cache_clear()
    detect_tpu.cache_clear()
    get_cpu_info.cache_clear()
    get_memory_info.cache_clear()


def get_recommended_backend() -> str:
    """Get recommended backend based on hardware.
    
    Returns:
        Recommended backend name: 'jax', 'pytorch', or 'numpy'
    """
    hardware = detect_all_hardware()
    
    # Prefer JAX with GPU/TPU
    if hardware['cuda']['available'] or hardware['tpu']['available']:
        return 'jax'
    
    # PyTorch with GPU
    if hardware['metal']['available'] or hardware['rocm']['available']:
        return 'pytorch'
    
    # JAX for CPU if enough memory (JIT compilation advantage)
    if hardware['memory']['total_mb'] >= 16384:  # 16GB+
        return 'jax'
    
    # Default to numpy for compatibility
    return 'numpy'


if __name__ == '__main__':
    # Test hardware detection
    import json
    print("Hardware Detection Results:")
    print(json.dumps(detect_all_hardware(), indent=2))
    print(f"\nRecommended backend: {get_recommended_backend()}")