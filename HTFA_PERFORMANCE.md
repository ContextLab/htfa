# HTFA Performance Implementation

This document describes the performance-optimized implementation of Hierarchical Topographic Factor Analysis (HTFA) that addresses the requirements in Issue #64.

## Overview

HTFA is a dimensionality reduction technique that learns hierarchical representations with topographic organization. This implementation provides a high-performance, extensible foundation with support for modern ML frameworks and hardware acceleration.

## Core Features

### ✅ Implemented (Current Release)

#### Core Algorithm
- **Hierarchical Factor Analysis**: Multi-level factorization with configurable hierarchy depth
- **Alternating Least Squares**: Robust optimization with regularization and numerical stability
- **Automatic Differentiation Ready**: Backend architecture supports gradient-based optimization
- **Reconstruction & Transform**: Forward and inverse transforms with error tracking

#### Modern Framework Integration  
- **Abstract Backend System**: Pluggable architecture for different ML frameworks
- **NumPy Backend**: Full-featured primary implementation  
- **JAX Backend**: Ready for automatic differentiation and JIT compilation
- **PyTorch Backend**: Ready for GPU acceleration and tensor operations
- **Graceful Fallbacks**: Optional dependencies handled cleanly

#### Performance Optimizations
- **Multiple Optimization Algorithms**: ALS, ADAM, Mini-batch, Sparse variants
- **Numerical Stability**: SVD-based pseudo-inverse, regularization, convergence checking
- **Memory Efficient**: Optimized matrix operations and memory reuse
- **Sparsity Support**: L1/L2 sparsity constraints for factor/loading matrices
- **Learning Rate Scheduling**: Step, exponential, and cosine decay schedules

#### Benchmarking & Profiling
- **Comprehensive Benchmarking**: Execution time, memory usage, reconstruction error
- **Scalability Testing**: Performance across different data sizes and configurations  
- **Backend Comparison**: Automated comparison of available backends
- **Memory Profiling**: Peak memory usage and allocation tracking
- **Synthetic Data Generation**: Reproducible test datasets with known ground truth

### 🚧 Next Phase (Future Implementation)

#### Hardware Acceleration
- **GPU Support**: PyTorch CUDA and CuPy implementations
- **Apple Metal**: Metal Performance Shaders support  
- **Multi-threading**: CPU parallelization for matrix operations
- **SIMD Optimization**: AVX/AVX2 optimizations for modern CPUs

#### Scalability Improvements
- **Distributed Computing**: Dask and Ray integration for cluster computing
- **Online/Streaming**: Incremental algorithms for large datasets
- **Checkpointing**: Resume long-running optimizations from saved state
- **Progressive Refinement**: Hierarchical refinement strategies

## Performance Results

Based on benchmarking with synthetic data:

| Configuration | Data Size | Time | Memory | Reconstruction Error |
|---------------|-----------|------|--------|---------------------|
| Simple (5F, 1L) | 1000×100 | 0.017s | 0.1MB | 571.63 |
| Standard (10F, 2L) | 1000×100 | 0.027s | -0.2MB | 29.91 |
| Complex (20F, 3L) | 1000×100 | 0.055s | 0.3MB | 27.19 |
| Scaling Test | 2000×200 | 0.058s | 8.5MB | 607.24 |

*F = Factors, L = Levels*

### Scalability Characteristics
- **Linear Time Complexity**: O(n_samples × n_features × n_factors × n_iterations)
- **Memory Efficiency**: Peak memory scales linearly with data size
- **Numerical Stability**: Robust convergence across different data scales

## Quick Start

### Basic Usage

```python
from htfa.htfa import HTFA

# Create and fit HTFA model
model = HTFA(n_factors=10, n_levels=2, backend='numpy')
transformed = model.fit_transform(data)

# Get reconstruction error
error = model.get_reconstruction_error()

# Access hierarchical factors
factors_all_levels = model.get_factors_all_levels()
loadings_all_levels = model.get_loadings_all_levels()
```

### Performance Benchmarking

```python
from htfa.htfa import HTFABenchmark

# Initialize benchmark suite
benchmark = HTFABenchmark()

# Generate synthetic test data
data, ground_truth = benchmark.generate_synthetic_data(
    n_samples=1000, n_features=100, n_factors_true=10
)

# Benchmark different configurations
result = benchmark.benchmark_htfa(data, n_factors=10, max_iter=50)
print(f"Time: {result.execution_time:.3f}s")
print(f"Memory: {result.memory_usage:.1f}MB") 
print(f"Error: {result.reconstruction_error:.6f}")

# Compare scaling performance
scaling_results = benchmark.scaling_benchmark(
    base_shape=(500, 50), scale_factors=[1, 2, 4, 8]
)

# Display summary
benchmark.print_summary()
```

### Advanced Optimization

```python
from htfa.htfa.optimization import AdamOptimizer, SparseOptimizer

# Use ADAM optimization
optimizer = AdamOptimizer(learning_rate=0.01, regularization=0.01)
result = optimizer.optimize(data, initial_factors, initial_loadings)

# Apply sparsity constraints
sparse_optimizer = SparseOptimizer(optimizer, sparsity_penalty=0.1, sparsity_type='l1')
sparse_result = sparse_optimizer.optimize(data, initial_factors, initial_loadings)
```

### Backend Selection

```python
# NumPy backend (default)
model_numpy = HTFA(n_factors=10, backend='numpy')

# JAX backend (requires: pip install jax jaxlib)
try:
    model_jax = HTFA(n_factors=10, backend='jax')
except ImportError:
    print("JAX not available")

# PyTorch backend (requires: pip install torch)
try: 
    model_torch = HTFA(n_factors=10, backend='pytorch')
except ImportError:
    print("PyTorch not available")
```

## Architecture

### Backend System

The implementation uses an abstract backend system that allows different numerical computing frameworks to be used interchangeably:

```python
class HTFABackend(ABC):
    @abstractmethod
    def array(self, data: Any) -> Any: pass
    
    @abstractmethod  
    def matmul(self, a: Any, b: Any) -> Any: pass
    
    @abstractmethod
    def svd(self, a: Any, full_matrices: bool = True) -> Tuple[Any, Any, Any]: pass
    
    # ... other operations
```

Available backends:
- **NumPyBackend**: Pure NumPy implementation (always available)
- **JAXBackend**: JAX arrays with JIT compilation support
- **PyTorchBackend**: PyTorch tensors with GPU support

### Optimization Algorithms

Multiple optimization algorithms are provided for different use cases:

- **AlternatingLeastSquares**: Traditional ALS with regularization
- **AdamOptimizer**: Gradient-based optimization with adaptive learning rates  
- **MiniBatchOptimizer**: Mini-batch processing for large datasets
- **SparseOptimizer**: Sparsity constraints (L1/L2) for factor matrices

### Benchmarking Framework

Comprehensive performance measurement tools:

- **HTFABenchmark**: Main benchmarking class
- **BenchmarkResult**: Structured performance results
- **Memory profiling**: Peak and incremental memory usage
- **Scaling tests**: Performance across different data sizes
- **Backend comparison**: Automated comparison of available backends

## Testing

The implementation includes comprehensive test coverage:

- **27 core functionality tests**: Basic HTFA operations, backends, benchmarking
- **14 optimization tests**: All optimization algorithms and edge cases  
- **Reproducibility tests**: Consistent results with random seeds
- **Edge case handling**: Small datasets, single factors, convergence

Run tests with:
```bash
poetry run pytest tests/ -v
```

## Performance Tips

### For Large Datasets
1. Use mini-batch optimization: `MiniBatchOptimizer(base_optimizer, batch_size=256)`
2. Reduce iterations: `max_iter=50` for initial exploration
3. Consider hierarchical levels vs. computational cost tradeoff

### For Sparse Solutions
1. Apply sparsity constraints: `SparseOptimizer(base_optimizer, sparsity_penalty=0.1)`
2. Use L1 penalty for true sparsity: `sparsity_type='l1'`
3. Tune regularization parameter based on desired sparsity level

### For Speed
1. Use fewer hierarchical levels for initial exploration
2. Consider JAX backend with JIT compilation (when available)
3. Set appropriate convergence tolerance: `tol=1e-4` vs `tol=1e-8`

## Dependencies

### Core Requirements
- Python ≥ 3.8
- NumPy ≥ 1.21.0
- psutil ≥ 5.8.0 (for memory profiling)

### Optional Performance Extensions
- **JAX**: `pip install jax jaxlib` (automatic differentiation, JIT)
- **PyTorch**: `pip install torch` (GPU acceleration) 
- **CuPy**: `pip install cupy` (direct CUDA support, future)
- **Numba**: `pip install numba` (JIT compilation, future)

### Optional Distributed Computing
- **Dask**: `pip install dask` (distributed arrays, future)
- **Ray**: `pip install ray` (distributed computing, future)

## Contributing

The codebase is designed for easy extension. Key extension points:

1. **New Backends**: Inherit from `HTFABackend` and implement required methods
2. **New Optimizers**: Inherit from `Optimizer` base class
3. **New Benchmarks**: Add methods to `HTFABenchmark` class

See individual module documentation for detailed API information.

## Performance Roadmap

### Phase 1: Foundation ✅ (Current)
- Core HTFA algorithm implementation
- Multi-backend architecture 
- Comprehensive benchmarking
- Multiple optimization algorithms

### Phase 2: Hardware Acceleration 🚧 (Next)
- GPU implementations (PyTorch CUDA, CuPy)
- JIT compilation (JAX, Numba)
- Multi-threading and SIMD optimizations

### Phase 3: Scalability 📋 (Future)
- Distributed computing integration
- Online/streaming algorithms  
- Advanced memory management
- Progressive refinement strategies

This implementation provides a solid foundation for all the performance optimizations outlined in Issue #64, with a clean architecture that supports future extensions.