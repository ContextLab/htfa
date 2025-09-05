"""Performance benchmarking utilities for HTFA."""

import time
import psutil
import functools
import numpy as np
from typing import Dict, List, Callable, Any, Optional, Tuple
from dataclasses import dataclass
from contextlib import contextmanager

from .core import HTFA


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    name: str
    execution_time: float
    memory_usage: float
    peak_memory: float
    reconstruction_error: float
    data_shape: Tuple[int, ...]
    backend: str
    parameters: Dict[str, Any]


class HTFABenchmark:
    """Benchmarking suite for HTFA performance."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
    
    @contextmanager
    def _monitor_memory(self):
        """Context manager to monitor memory usage."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        peak_memory = initial_memory
        
        def update_peak():
            nonlocal peak_memory
            current = process.memory_info().rss / 1024 / 1024
            peak_memory = max(peak_memory, current)
        
        try:
            yield update_peak
        finally:
            final_memory = process.memory_info().rss / 1024 / 1024
            self._current_memory_usage = final_memory - initial_memory
            self._current_peak_memory = peak_memory - initial_memory
    
    def benchmark_htfa(
        self,
        data: np.ndarray,
        n_factors: int,
        backend: str = 'numpy',
        n_levels: int = 2,
        max_iter: int = 100,
        name: Optional[str] = None
    ) -> BenchmarkResult:
        """Benchmark HTFA performance on given data.
        
        Parameters
        ----------
        data : ndarray
            Input data for benchmarking.
        n_factors : int
            Number of factors to extract.
        backend : str
            Backend to use ('numpy', 'jax', 'pytorch').
        n_levels : int
            Number of hierarchical levels.
        max_iter : int
            Maximum iterations for optimization.
        name : str, optional
            Name for this benchmark run.
        
        Returns
        -------
        result : BenchmarkResult
            Benchmark results.
        """
        if name is None:
            name = f"HTFA_{backend}_{data.shape}_{n_factors}f_{n_levels}l"
        
        # Create HTFA model
        model = HTFA(
            n_factors=n_factors,
            n_levels=n_levels,
            backend=backend,
            max_iter=max_iter,
            random_state=42
        )
        
        # Monitor performance
        with self._monitor_memory() as update_peak:
            start_time = time.perf_counter()
            
            # Fit model
            model.fit(data)
            update_peak()
            
            # Transform data
            transformed = model.transform(data)
            update_peak()
            
            end_time = time.perf_counter()
        
        # Create benchmark result
        result = BenchmarkResult(
            name=name,
            execution_time=end_time - start_time,
            memory_usage=self._current_memory_usage,
            peak_memory=self._current_peak_memory,
            reconstruction_error=model.get_reconstruction_error(),
            data_shape=data.shape,
            backend=backend,
            parameters={
                'n_factors': n_factors,
                'n_levels': n_levels,
                'max_iter': max_iter
            }
        )
        
        self.results.append(result)
        return result
    
    def compare_backends(
        self,
        data: np.ndarray,
        n_factors: int,
        backends: List[str] = ['numpy', 'jax', 'pytorch'],
        **kwargs
    ) -> List[BenchmarkResult]:
        """Compare performance across different backends.
        
        Parameters
        ----------
        data : ndarray
            Input data for benchmarking.
        n_factors : int
            Number of factors to extract.
        backends : List[str]
            List of backends to compare.
        **kwargs
            Additional arguments for HTFA.
        
        Returns
        -------
        results : List[BenchmarkResult]
            Benchmark results for each backend.
        """
        results = []
        for backend in backends:
            try:
                result = self.benchmark_htfa(
                    data, n_factors, backend=backend, 
                    name=f"Backend_Comparison_{backend}", **kwargs
                )
                results.append(result)
                print(f"✓ {backend}: {result.execution_time:.3f}s, "
                      f"Error: {result.reconstruction_error:.6f}")
            except Exception as e:
                print(f"✗ {backend}: Failed - {str(e)}")
        
        return results
    
    def scaling_benchmark(
        self,
        base_shape: Tuple[int, int] = (1000, 100),
        scale_factors: List[float] = [1, 2, 4, 8],
        n_factors: int = 10,
        backend: str = 'numpy',
        **kwargs
    ) -> List[BenchmarkResult]:
        """Benchmark scaling performance with different data sizes.
        
        Parameters
        ----------
        base_shape : Tuple[int, int]
            Base data shape (n_samples, n_features).
        scale_factors : List[float]
            Factors to scale the data size.
        n_factors : int
            Number of factors to extract.
        backend : str
            Backend to use.
        **kwargs
            Additional arguments for HTFA.
        
        Returns
        -------
        results : List[BenchmarkResult]
            Benchmark results for each scale.
        """
        results = []
        
        for scale in scale_factors:
            shape = (int(base_shape[0] * scale), int(base_shape[1] * scale))
            data = np.random.randn(*shape).astype(np.float32)
            
            result = self.benchmark_htfa(
                data, n_factors, backend=backend,
                name=f"Scaling_{backend}_{scale}x", **kwargs
            )
            results.append(result)
            
            print(f"Scale {scale}x: {shape} -> {result.execution_time:.3f}s, "
                  f"{result.memory_usage:.1f}MB")
        
        return results
    
    def generate_synthetic_data(
        self, 
        n_samples: int = 1000, 
        n_features: int = 100,
        n_factors_true: int = 5,
        noise_level: float = 0.1,
        random_state: int = 42
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Generate synthetic data for benchmarking.
        
        Parameters
        ----------
        n_samples : int
            Number of samples.
        n_features : int
            Number of features.
        n_factors_true : int
            True number of underlying factors.
        noise_level : float
            Level of noise to add.
        random_state : int
            Random state for reproducibility.
        
        Returns
        -------
        data : ndarray
            Generated data.
        ground_truth : dict
            Dictionary containing ground truth factors and loadings.
        """
        np.random.seed(random_state)
        
        # Generate true factors and loadings
        true_factors = np.random.randn(n_samples, n_factors_true)
        true_loadings = np.random.randn(n_factors_true, n_features)
        
        # Generate data
        data = np.dot(true_factors, true_loadings)
        
        # Add noise
        noise = np.random.randn(*data.shape) * noise_level
        data += noise
        
        ground_truth = {
            'factors': true_factors,
            'loadings': true_loadings,
            'noise_level': noise_level
        }
        
        return data.astype(np.float32), ground_truth
    
    def profile_memory_usage(
        self,
        data_shapes: List[Tuple[int, int]],
        n_factors: int = 10,
        backend: str = 'numpy'
    ) -> List[BenchmarkResult]:
        """Profile memory usage for different data sizes.
        
        Parameters
        ----------
        data_shapes : List[Tuple[int, int]]
            List of data shapes to test.
        n_factors : int
            Number of factors.
        backend : str
            Backend to use.
        
        Returns
        -------
        results : List[BenchmarkResult]
            Memory profiling results.
        """
        results = []
        
        for shape in data_shapes:
            data = np.random.randn(*shape).astype(np.float32)
            
            result = self.benchmark_htfa(
                data, n_factors, backend=backend,
                name=f"Memory_Profile_{shape[0]}x{shape[1]}"
            )
            results.append(result)
            
            print(f"Shape {shape}: Peak memory {result.peak_memory:.1f}MB, "
                  f"Usage {result.memory_usage:.1f}MB")
        
        return results
    
    def print_summary(self) -> None:
        """Print summary of all benchmark results."""
        if not self.results:
            print("No benchmark results available.")
            return
        
        print("\n" + "="*80)
        print("HTFA BENCHMARK SUMMARY")
        print("="*80)
        
        print(f"{'Name':<30} {'Time(s)':<10} {'Memory(MB)':<12} {'Error':<12} {'Shape':<15}")
        print("-" * 80)
        
        for result in self.results:
            print(f"{result.name:<30} {result.execution_time:<10.3f} "
                  f"{result.memory_usage:<12.1f} {result.reconstruction_error:<12.6f} "
                  f"{str(result.data_shape):<15}")
        
        # Best performance summary
        fastest = min(self.results, key=lambda r: r.execution_time)
        lowest_memory = min(self.results, key=lambda r: r.memory_usage)
        lowest_error = min(self.results, key=lambda r: r.reconstruction_error)
        
        print("\n" + "="*80)
        print("BEST PERFORMANCE:")
        print(f"Fastest: {fastest.name} ({fastest.execution_time:.3f}s)")
        print(f"Lowest Memory: {lowest_memory.name} ({lowest_memory.memory_usage:.1f}MB)")
        print(f"Lowest Error: {lowest_error.name} ({lowest_error.reconstruction_error:.6f})")
        print("="*80)
    
    def clear_results(self) -> None:
        """Clear all benchmark results."""
        self.results.clear()


def benchmark_function(func: Callable) -> Callable:
    """Decorator to benchmark function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper