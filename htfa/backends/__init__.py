"""Backend interface for HTFA numerical operations.

This package provides an abstract interface for different numerical computing
backends (NumPy, JAX, PyTorch), allowing HTFA to be backend-agnostic while
enabling performance optimizations and specialized hardware acceleration.

Usage
-----
The backend system is designed for dependency injection into HTFA classes:

    from htfa.backends import Backend, get_default_backend, create_backend

    # Use default backend (auto-selected based on hardware/libraries)
    backend_name = get_default_backend()
    backend = create_backend(backend_name)

    # Or let HTFA auto-select by passing backend=None
    htfa = HTFA(K=10, backend=None)  # Auto-selects optimal backend

    # Or specify a particular backend
    backend = create_backend('jax')  # Force JAX backend

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
    create_backend,
    get_available_backends,
    get_default_backend,
    register_backend,
)

# Import automatic backend selection
from htfa.backends.selector import (
    BackendSelector,
    clear_backend_cache,
    get_backend_device,
    select_backend,
    validate_backend_selection,
)

__all__ = [
    "Backend",
    "register_backend",
    "get_available_backends",
    "create_backend",
    "get_default_backend",
    # Automatic selection
    "select_backend",
    "get_backend_device",
    "validate_backend_selection",
    "clear_backend_cache",
    "BackendSelector",
]
