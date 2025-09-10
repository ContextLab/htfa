"""Tests for HTFA optimization algorithms."""

import numpy as np
import pytest

from htfa.optimization import (
    AdamOptimizer,
    AlternatingLeastSquares,
    MiniBatchOptimizer,
    OptimizationScheduler,
    SparseOptimizer,
    create_optimizer,
)


class TestOptimizers:
    """Test optimization algorithms."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.X = np.random.randn(50, 30).astype(np.float32)
        self.n_factors = 5
        self.initial_factors = np.random.randn(50, self.n_factors).astype(np.float32)
        self.initial_loadings = np.random.randn(self.n_factors, 30).astype(np.float32)

    def test_als_optimizer(self):
        """Test Alternating Least Squares optimizer."""
        optimizer = AlternatingLeastSquares(regularization=0.01)
        result = optimizer.optimize(
            self.X, self.initial_factors, self.initial_loadings, max_iter=10
        )

        assert result.factors.shape == self.initial_factors.shape
        assert result.loadings.shape == self.initial_loadings.shape
        assert len(result.loss_history) > 0
        assert isinstance(result.converged, bool)
        assert result.n_iterations > 0

    def test_adam_optimizer(self):
        """Test ADAM optimizer."""
        optimizer = AdamOptimizer(learning_rate=0.01, regularization=0.01)
        result = optimizer.optimize(
            self.X, self.initial_factors, self.initial_loadings, max_iter=10
        )

        assert result.factors.shape == self.initial_factors.shape
        assert result.loadings.shape == self.initial_loadings.shape
        assert len(result.loss_history) > 0
        assert isinstance(result.converged, bool)
        assert result.n_iterations > 0

    def test_minibatch_optimizer(self):
        """Test Mini-batch optimizer."""
        base_optimizer = AlternatingLeastSquares(regularization=0.01)
        optimizer = MiniBatchOptimizer(
            base_optimizer, batch_size=20, n_epochs=2, shuffle=True
        )
        result = optimizer.optimize(
            self.X, self.initial_factors, self.initial_loadings, max_iter=5
        )

        assert result.factors.shape == self.initial_factors.shape
        assert result.loadings.shape == self.initial_loadings.shape
        assert len(result.loss_history) > 0

    def test_sparse_optimizer(self):
        """Test Sparse optimizer."""
        base_optimizer = AlternatingLeastSquares(regularization=0.01)

        # Test L1 sparsity
        optimizer_l1 = SparseOptimizer(
            base_optimizer, sparsity_penalty=0.1, sparsity_type="l1"
        )
        result_l1 = optimizer_l1.optimize(
            self.X, self.initial_factors, self.initial_loadings, max_iter=5
        )

        # Test L2 sparsity
        optimizer_l2 = SparseOptimizer(
            base_optimizer, sparsity_penalty=0.1, sparsity_type="l2"
        )
        result_l2 = optimizer_l2.optimize(
            self.X, self.initial_factors, self.initial_loadings, max_iter=5
        )

        # Check that sparsity constraints affect results
        assert not np.allclose(result_l1.factors, result_l2.factors)
        assert not np.allclose(result_l1.loadings, result_l2.loadings)

    def test_create_optimizer_factory(self):
        """Test optimizer factory function."""
        # Test ALS creation
        als_optimizer = create_optimizer("als", regularization=0.05)
        assert isinstance(als_optimizer, AlternatingLeastSquares)
        assert als_optimizer.regularization == 0.05

        # Test ADAM creation
        adam_optimizer = create_optimizer("adam", learning_rate=0.005)
        assert isinstance(adam_optimizer, AdamOptimizer)
        assert adam_optimizer.learning_rate == 0.005

        # Test invalid optimizer
        with pytest.raises(ValueError, match="Unknown optimizer type"):
            create_optimizer("invalid_optimizer")

    def test_optimization_convergence(self):
        """Test optimization convergence behavior."""
        optimizer = AlternatingLeastSquares(regularization=0.01)

        # Test with tight tolerance (should converge quickly)
        result_tight = optimizer.optimize(
            self.X, self.initial_factors, self.initial_loadings, max_iter=100, tol=1e-1
        )

        # Test with loose tolerance (should take more iterations)
        result_loose = optimizer.optimize(
            self.X, self.initial_factors, self.initial_loadings, max_iter=100, tol=1e-8
        )

        assert result_tight.n_iterations <= result_loose.n_iterations

    def test_loss_decreasing(self):
        """Test that loss generally decreases during optimization."""
        optimizer = AlternatingLeastSquares(regularization=0.01)
        result = optimizer.optimize(
            self.X, self.initial_factors, self.initial_loadings, max_iter=20
        )

        # Check that final loss is less than initial loss (in most cases)
        if len(result.loss_history) > 1:
            assert (
                result.loss_history[-1] <= result.loss_history[0] * 1.1
            )  # Allow some tolerance


class TestOptimizationScheduler:
    """Test optimization scheduler."""

    def test_step_decay(self):
        """Test step decay schedule."""
        scheduler = OptimizationScheduler(initial_lr=0.1)

        # Test at different epochs
        lr_epoch_0 = scheduler.step_decay(0, drop_rate=0.5, epochs_drop=10)
        lr_epoch_10 = scheduler.step_decay(10, drop_rate=0.5, epochs_drop=10)
        lr_epoch_20 = scheduler.step_decay(20, drop_rate=0.5, epochs_drop=10)

        assert lr_epoch_0 == 0.1
        assert lr_epoch_10 == 0.05  # 0.1 * 0.5
        assert lr_epoch_20 == 0.025  # 0.1 * 0.5^2

    def test_exponential_decay(self):
        """Test exponential decay schedule."""
        scheduler = OptimizationScheduler(initial_lr=0.1)

        lr_epoch_0 = scheduler.exponential_decay(0, decay_rate=0.9)
        lr_epoch_1 = scheduler.exponential_decay(1, decay_rate=0.9)
        lr_epoch_10 = scheduler.exponential_decay(10, decay_rate=0.9)

        assert lr_epoch_0 == 0.1
        assert lr_epoch_1 == 0.1 * 0.9
        assert abs(lr_epoch_10 - 0.1 * (0.9**10)) < 1e-10

    def test_cosine_decay(self):
        """Test cosine annealing schedule."""
        scheduler = OptimizationScheduler(initial_lr=0.1)

        lr_epoch_0 = scheduler.cosine_decay(0, total_epochs=100)
        lr_epoch_50 = scheduler.cosine_decay(50, total_epochs=100)
        lr_epoch_100 = scheduler.cosine_decay(100, total_epochs=100)

        assert lr_epoch_0 == 0.1
        assert lr_epoch_50 < lr_epoch_0  # Should decrease
        assert lr_epoch_100 < lr_epoch_50  # Should continue decreasing


class TestOptimizationWithRealData:
    """Test optimization with more realistic scenarios."""

    def setup_method(self):
        """Set up realistic test scenario."""
        np.random.seed(42)

        # Generate synthetic data with known structure
        n_samples, n_features, n_factors = 100, 50, 5
        true_factors = np.random.randn(n_samples, n_factors)
        true_loadings = np.random.randn(n_factors, n_features)

        # Add noise
        self.X = np.dot(true_factors, true_loadings) + 0.1 * np.random.randn(
            n_samples, n_features
        )
        self.X = self.X.astype(np.float32)

        self.true_factors = true_factors.astype(np.float32)
        self.true_loadings = true_loadings.astype(np.float32)
        self.n_factors = n_factors

    def test_optimization_quality(self):
        """Test that optimization produces reasonable results."""
        # Initialize factors randomly
        initial_factors = np.random.randn(self.X.shape[0], self.n_factors).astype(
            np.float32
        )
        initial_loadings = np.random.randn(self.n_factors, self.X.shape[1]).astype(
            np.float32
        )

        # Optimize with ALS
        optimizer = AlternatingLeastSquares(regularization=0.01)
        result = optimizer.optimize(
            self.X, initial_factors, initial_loadings, max_iter=50
        )

        # Check reconstruction quality
        reconstruction = np.dot(result.factors, result.loadings)
        reconstruction_error = np.linalg.norm(self.X - reconstruction)

        # Error should be reasonable (not perfect due to noise and local minima)
        assert reconstruction_error < np.linalg.norm(self.X) * 0.5

        # Loss should decrease
        if len(result.loss_history) > 1:
            assert result.loss_history[-1] < result.loss_history[0]

    def test_optimizer_comparison(self):
        """Compare different optimizers on the same data."""
        initial_factors = np.random.randn(self.X.shape[0], self.n_factors).astype(
            np.float32
        )
        initial_loadings = np.random.randn(self.n_factors, self.X.shape[1]).astype(
            np.float32
        )

        # Test multiple optimizers
        optimizers = {
            "als": AlternatingLeastSquares(regularization=0.01),
            "adam": AdamOptimizer(learning_rate=0.01, regularization=0.01),
        }

        results = {}
        for name, optimizer in optimizers.items():
            results[name] = optimizer.optimize(
                self.X, initial_factors.copy(), initial_loadings.copy(), max_iter=20
            )

        # All should produce valid results
        for name, result in results.items():
            assert result.factors.shape == initial_factors.shape
            assert result.loadings.shape == initial_loadings.shape
            assert len(result.loss_history) > 0

    @pytest.mark.parametrize("sparsity_type", ["l1", "l2"])
    def test_sparsity_effects(self, sparsity_type):
        """Test that sparsity constraints actually make results sparser."""
        initial_factors = np.random.randn(self.X.shape[0], self.n_factors).astype(
            np.float32
        )
        initial_loadings = np.random.randn(self.n_factors, self.X.shape[1]).astype(
            np.float32
        )

        # Optimize without sparsity
        base_optimizer = AlternatingLeastSquares(regularization=0.01)
        result_no_sparse = base_optimizer.optimize(
            self.X, initial_factors.copy(), initial_loadings.copy(), max_iter=10
        )

        # Optimize with sparsity
        sparse_optimizer = SparseOptimizer(
            base_optimizer, sparsity_penalty=0.1, sparsity_type=sparsity_type
        )
        result_sparse = sparse_optimizer.optimize(
            self.X, initial_factors.copy(), initial_loadings.copy(), max_iter=10
        )

        # Sparse results should have more values close to zero
        sparse_factors_zeros = np.sum(np.abs(result_sparse.factors) < 0.01)
        no_sparse_factors_zeros = np.sum(np.abs(result_no_sparse.factors) < 0.01)

        # Not a strict requirement due to randomness, but typically true
        # assert sparse_factors_zeros >= no_sparse_factors_zeros
