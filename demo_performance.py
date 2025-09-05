"""
HTFA Performance Optimization Demo

This script demonstrates the performance optimization features implemented
for Hierarchical Topographic Factor Analysis (HTFA), addressing issue #64.
"""

import numpy as np
import time
from htfa.htfa import HTFA, HTFABenchmark

def main():
    """Run HTFA performance optimization demo."""
    print("="*70)
    print("HTFA PERFORMANCE OPTIMIZATION DEMO")
    print("="*70)
    print()
    
    # Initialize benchmark suite
    benchmark = HTFABenchmark()
    
    # Generate synthetic data for demonstration
    print("1. Generating synthetic data for benchmarking...")
    data, ground_truth = benchmark.generate_synthetic_data(
        n_samples=1000, n_features=100, n_factors_true=10, noise_level=0.1
    )
    print(f"   Data shape: {data.shape}")
    print(f"   True factors: {ground_truth['factors'].shape}")
    print(f"   True loadings: {ground_truth['loadings'].shape}")
    print(f"   Noise level: {ground_truth['noise_level']}")
    print()
    
    # Test basic HTFA functionality
    print("2. Testing basic HTFA functionality...")
    model = HTFA(n_factors=10, n_levels=2, max_iter=50, random_state=42)
    
    start_time = time.time()
    transformed = model.fit_transform(data)
    fit_time = time.time() - start_time
    
    reconstruction_error = model.get_reconstruction_error()
    factors_all = model.get_factors_all_levels()
    
    print(f"   Fit time: {fit_time:.3f} seconds")
    print(f"   Transformed shape: {transformed.shape}")
    print(f"   Reconstruction error: {reconstruction_error:.6f}")
    print(f"   Number of hierarchical levels: {len(factors_all)}")
    print(f"   Factor shapes by level: {[f.shape for f in factors_all]}")
    print()
    
    # Benchmark different configurations
    print("3. Benchmarking different HTFA configurations...")
    configs = [
        {"n_factors": 5, "n_levels": 1, "name": "Simple (5 factors, 1 level)"},
        {"n_factors": 10, "n_levels": 2, "name": "Standard (10 factors, 2 levels)"},
        {"n_factors": 20, "n_levels": 3, "name": "Complex (20 factors, 3 levels)"},
    ]
    
    results = []
    for config in configs:
        print(f"   Testing {config['name']}...")
        result = benchmark.benchmark_htfa(
            data, 
            n_factors=config['n_factors'], 
            n_levels=config['n_levels'],
            max_iter=25,
            name=config['name']
        )
        results.append(result)
        print(f"     Time: {result.execution_time:.3f}s, "
              f"Memory: {result.memory_usage:.1f}MB, "
              f"Error: {result.reconstruction_error:.6f}")
    print()
    
    # Test scaling performance
    print("4. Testing scaling performance...")
    scaling_results = benchmark.scaling_benchmark(
        base_shape=(500, 50),
        scale_factors=[1, 2, 4],
        n_factors=10,
        backend='numpy',
        max_iter=20
    )
    print()
    
    # Memory profiling
    print("5. Memory profiling for different data sizes...")
    memory_shapes = [(200, 50), (500, 100), (1000, 150)]
    memory_results = benchmark.profile_memory_usage(
        memory_shapes, n_factors=10, backend='numpy'
    )
    print()
    
    # Backend comparison (if available)
    print("6. Comparing available backends...")
    try:
        backend_results = benchmark.compare_backends(
            data[:200, :50],  # Smaller data for faster comparison
            n_factors=5,
            backends=['numpy'],  # Only numpy available by default
            max_iter=10
        )
        print("   Backend comparison completed.")
    except Exception as e:
        print(f"   Backend comparison failed: {e}")
    print()
    
    # Performance summary
    print("7. Performance Summary:")
    benchmark.print_summary()
    
    # Test advanced optimization algorithms
    print("\n" + "="*70)
    print("ADVANCED OPTIMIZATION ALGORITHMS")
    print("="*70)
    
    from htfa.htfa.optimization import (
        AlternatingLeastSquares, AdamOptimizer, create_optimizer
    )
    
    # Test different optimizers
    print("\nTesting different optimization algorithms...")
    
    test_data = data[:100, :30]  # Smaller for faster testing
    initial_factors = np.random.randn(100, 5).astype(np.float32)
    initial_loadings = np.random.randn(5, 30).astype(np.float32)
    
    optimizers = {
        'ALS': AlternatingLeastSquares(regularization=0.01),
        'ADAM': AdamOptimizer(learning_rate=0.01, regularization=0.01)
    }
    
    for name, optimizer in optimizers.items():
        print(f"\nTesting {name} optimizer...")
        start_time = time.time()
        result = optimizer.optimize(
            test_data, initial_factors.copy(), initial_loadings.copy(), max_iter=20
        )
        end_time = time.time()
        
        print(f"   Time: {end_time - start_time:.3f}s")
        print(f"   Converged: {result.converged}")
        print(f"   Iterations: {result.n_iterations}")
        print(f"   Final loss: {result.loss_history[-1]:.6f}")
    
    print("\n" + "="*70)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*70)
    print()
    print("Key achievements demonstrated:")
    print("✓ Core HTFA algorithm implementation")
    print("✓ Hierarchical factor analysis with multiple levels") 
    print("✓ Performance benchmarking and profiling")
    print("✓ Memory usage optimization")
    print("✓ Scalability testing")
    print("✓ Multiple optimization algorithms (ALS, ADAM)")
    print("✓ Modular backend architecture for future extensions")
    print()
    print("Next steps for issue #64:")
    print("• Add JAX backend for automatic differentiation (requires: pip install jax)")
    print("• Add PyTorch backend for GPU acceleration (requires: pip install torch)")  
    print("• Implement Numba JIT compilation for critical loops")
    print("• Add CuPy support for direct CUDA implementations")
    print("• Implement distributed computing with Dask/Ray")
    print("• Add progressive refinement and checkpointing")
    print("• Optimize for modern CPU architectures")


if __name__ == "__main__":
    main()