#!/usr/bin/env python
"""Test script for automatic backend selection."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from htfa.utils import detect_all_hardware, detect_all_libraries, get_optimal_backend
from htfa.backends.selector import BackendSelector, select_backend
from htfa.core.htfa import HTFA
from htfa.core.tfa import TFA
import numpy as np


def test_detection():
    """Test hardware and library detection."""
    print("=" * 60)
    print("TESTING DETECTION MODULES")
    print("=" * 60)
    
    # Test hardware detection
    print("\n1. Hardware Detection:")
    hardware = detect_all_hardware()
    print(f"   - Has GPU: {hardware['has_gpu']}")
    print(f"   - CUDA available: {hardware['cuda']['available']}")
    print(f"   - Metal available: {hardware['metal']['available']}")
    print(f"   - CPU cores: {hardware['cpu']['count']}")
    print(f"   - Memory: {hardware['memory']['total_mb']}MB total")
    
    # Test library detection
    print("\n2. Library Detection:")
    libraries = detect_all_libraries()
    print(f"   - NumPy: {libraries['numpy']['available']} (v{libraries['numpy'].get('version', 'N/A')})")
    print(f"   - JAX: {libraries['jax']['available']} (v{libraries['jax'].get('version', 'N/A')})")
    print(f"   - PyTorch: {libraries['pytorch']['available']} (v{libraries['pytorch'].get('version', 'N/A')})")
    
    # Test optimal backend selection
    print("\n3. Optimal Backend:")
    optimal = get_optimal_backend()
    print(f"   - Recommended: {optimal}")
    

def test_selector():
    """Test BackendSelector class."""
    print("\n" + "=" * 60)
    print("TESTING BACKEND SELECTOR")
    print("=" * 60)
    
    selector = BackendSelector()
    
    # Test auto-selection
    print("\n1. Auto-selection:")
    selected = selector.select_backend(None)
    print(f"   - Selected: {selected}")
    
    # Test explicit selection
    print("\n2. Explicit selection:")
    for backend in ['numpy', 'jax', 'pytorch']:
        try:
            selected = selector.select_backend(backend)
            print(f"   - {backend}: OK")
        except Exception as e:
            print(f"   - {backend}: Failed ({e})")
    
    # Test device detection
    print("\n3. Device detection:")
    for backend in ['numpy', 'jax', 'pytorch']:
        device = selector.get_device(backend)
        print(f"   - {backend}: {device}")
    
    # Test memory estimation
    print("\n4. Memory estimation:")
    mem_req = selector.estimate_memory_requirement(n_voxels=10000, n_times=500, k_components=10)
    print(f"   - Required: {mem_req}MB")
    

def test_models():
    """Test HTFA and TFA with auto-selection."""
    print("\n" + "=" * 60)
    print("TESTING MODEL INTEGRATION")
    print("=" * 60)
    
    # Create small test data
    np.random.seed(42)
    data = np.random.randn(100, 50)  # 100 voxels, 50 time points
    
    print("\n1. TFA with auto-selection:")
    try:
        tfa = TFA(K=5, backend=None, verbose=True)
        print(f"   - Backend: {tfa.backend.__class__.__name__}")
        # Test fit
        tfa.fit(data)
        print("   - Fit: OK")
    except Exception as e:
        print(f"   - Failed: {e}")
    
    print("\n2. HTFA with auto-selection:")
    try:
        htfa = HTFA(K=5, backend=None, verbose=True)
        print(f"   - Backend: {htfa.backend.__class__.__name__}")
        # Test fit with multi-subject data
        data_list = [data, data + 0.1 * np.random.randn(100, 50)]
        htfa.fit(data_list)
        print("   - Fit: OK")
    except Exception as e:
        print(f"   - Failed: {e}")
    
    print("\n3. Explicit backend selection:")
    for backend_name in ['numpy', 'jax', 'pytorch']:
        try:
            tfa = TFA(K=5, backend=backend_name, verbose=False)
            print(f"   - {backend_name}: {tfa.backend.__class__.__name__}")
        except Exception as e:
            print(f"   - {backend_name}: Not available ({e})")


def test_module_function():
    """Test module-level select_backend function."""
    print("\n" + "=" * 60)
    print("TESTING MODULE FUNCTION")
    print("=" * 60)
    
    print("\n1. Module-level select_backend:")
    selected = select_backend(None)
    print(f"   - Auto-selected: {selected}")
    
    print("\n2. With explicit backend:")
    selected = select_backend('numpy')
    print(f"   - Explicit numpy: {selected}")


if __name__ == "__main__":
    print("AUTOMATIC BACKEND SELECTION TEST SUITE")
    print("=" * 60)
    
    try:
        test_detection()
        test_selector()
        test_module_function()
        test_models()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)