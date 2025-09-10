"""Backend interface for HTFA numerical operations.

This package provides an abstract interface for different numerical computing
backends (NumPy, JAX, PyTorch), allowing HTFA to be backend-agnostic while
enabling performance optimizations and specialized hardware acceleration.

Usage
-----
The backend system is designed for dependency injection into HTFA classes:

    from htfa.backends import Backend, get_default_backend, create_backend
    
    # Use default backend (typically NumPy)
    backend_name = get_default_backend()
    backend = create_backend(backend_name)
    
    # Or specify a particular backend
    backend = create_backend('jax')  # Once JAX backend is implemented
    
    # Pass to HTFA
    htfa = HTFA(K=10, backend=backend)

Extension
---------
To add a new backend:

1. Implement the Backend abstract base class:

    from htfa.backends.base import Backend, register_backend
    
    class MyBackend(Backend):
        @property
        def name(self) -> str:
            return "mybackend"
        
        # Implement all abstract methods...
    
2. Register the backend:

    register_backend("mybackend", MyBackend)

3. The backend will then be available via create_backend("mybackend")

Design Principles
-----------------
- Minimal interface covering only operations actually used by TFA/HTFA
- Type-agnostic - backends handle their own array types  
- Interoperability through to_numpy/from_numpy conversion methods
- Registry pattern for dynamic backend selection
- No performance overhead for NumPy backend (direct pass-through)
"""

from htfa.backends.base import (
    Backend,
    register_backend,
    get_available_backends,
    create_backend,
    get_default_backend,
)

__all__ = [
    "Backend",
    "register_backend", 
    "get_available_backends",
    "create_backend",
    "get_default_backend",
]