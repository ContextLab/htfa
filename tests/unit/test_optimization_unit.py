"""Tests for htfa.optimization module."""

import numpy as np
import pytest

from htfa.optimization import (
    AdamOptimizer,
    AlternatingLeastSquares,
    MiniBatchOptimizer,
    OptimizationResult,
    OptimizationScheduler,
    SparseOptimizer,
    create_optimizer,
)


class TestOptimizationResult:
    """Test OptimizationResult dataclass."""

    def test_optimization_result_creation(self):
        """Test creating an OptimizationResult."""
        factors = np.random.randn(10, 5)
        loadings = np.random.randn(5, 20)
        loss_history = [100, 90, 80]
        
        result = OptimizationResult(
            factors=factors,
            loadings=loadings,
            loss_history=loss_history,
            converged=True,
            n_iterations=3
        )
        
        assert np.array_equal(result.factors, factors)
        assert np.array_equal(result.loadings, loadings)
        assert result.loss_history == loss_history
        assert result.converged is True
        assert result.n_iterations == 3


class TestAlternatingLeastSquares:
    """Test AlternatingLeastSquares optimizer."""

    def test_als_initialization(self):
        """Test ALS optimizer initialization."""
        opt = AlternatingLeastSquares(regularization=0.05)
        assert opt.regularization == 0.05

    def test_als_optimize_convergence(self):
        """Test ALS optimization with convergence."""
        # Create a simple factorization problem
        true_factors = np.random.randn(20, 3)
        true_loadings = np.random.randn(3, 30)
        X = np.dot(true_factors, true_loadings)
        
        # Add small noise
        X += 0.01 * np.random.randn(*X.shape)
        
        # Initialize with random values
        init_factors = np.random.randn(20, 3)
        init_loadings = np.random.randn(3, 30)
        
        opt = AlternatingLeastSquares(regularization=0.01)
        result = opt.optimize(X, init_factors, init_loadings, max_iter=100, tol=1e-4)
        
        assert isinstance(result, OptimizationResult)
        assert result.factors.shape == (20, 3)
        assert result.loadings.shape == (3, 30)
        assert len(result.loss_history) > 0
        assert result.n_iterations > 0
        assert result.n_iterations <= 100

    def test_als_optimize_max_iterations(self):
        """Test ALS hitting max iterations."""
        X = np.random.randn(20, 30)
        init_factors = np.random.randn(20, 5)
        init_loadings = np.random.randn(5, 30)
        
        opt = AlternatingLeastSquares(regularization=0.01)
        result = opt.optimize(X, init_factors, init_loadings, max_iter=2, tol=1e-10)
        
        assert result.n_iterations == 2
        assert not result.converged

    def test_als_update_factors(self):
        """Test factor update step."""
        X = np.random.randn(10, 20)
        loadings = np.random.randn(3, 20)
        
        opt = AlternatingLeastSquares(regularization=0.1)
        factors = opt._update_factors(X, loadings)
        
        assert factors.shape == (10, 3)

    def test_als_update_loadings(self):
        """Test loadings update step."""
        X = np.random.randn(10, 20)
        factors = np.random.randn(10, 3)
        
        opt = AlternatingLeastSquares(regularization=0.1)
        loadings = opt._update_loadings(X, factors)
        
        assert loadings.shape == (3, 20)


class TestAdamOptimizer:
    """Test AdamOptimizer."""

    def test_adam_initialization(self):
        """Test ADAM optimizer initialization."""
        opt = AdamOptimizer(
            learning_rate=0.01,
            beta1=0.95,
            beta2=0.99,
            epsilon=1e-7,
            regularization=0.05
        )
        
        assert opt.learning_rate == 0.01
        assert opt.beta1 == 0.95
        assert opt.beta2 == 0.99
        assert opt.epsilon == 1e-7
        assert opt.regularization == 0.05

    def test_adam_optimize(self):
        """Test ADAM optimization."""
        X = np.random.randn(15, 25)
        init_factors = np.random.randn(15, 4)
        init_loadings = np.random.randn(4, 25)
        
        opt = AdamOptimizer(learning_rate=0.001, regularization=0.01)
        result = opt.optimize(X, init_factors, init_loadings, max_iter=50, tol=1e-4)
        
        assert isinstance(result, OptimizationResult)
        assert result.factors.shape == (15, 4)
        assert result.loadings.shape == (4, 25)
        assert len(result.loss_history) > 0
        assert result.n_iterations > 0

    def test_adam_compute_gradients(self):
        """Test gradient computation."""
        X = np.random.randn(10, 20)
        factors = np.random.randn(10, 3)
        loadings = np.random.randn(3, 20)
        
        opt = AdamOptimizer(regularization=0.1)
        grad_factors, grad_loadings = opt._compute_gradients(X, factors, loadings)
        
        assert grad_factors.shape == factors.shape
        assert grad_loadings.shape == loadings.shape

    def test_adam_update(self):
        """Test ADAM parameter update."""
        params = np.random.randn(10, 5)
        gradients = np.random.randn(10, 5)
        m = np.zeros_like(params)
        v = np.zeros_like(params)
        
        opt = AdamOptimizer()
        new_params, new_m, new_v = opt._adam_update(params, gradients, m, v, t=1)
        
        assert new_params.shape == params.shape
        assert new_m.shape == m.shape
        assert new_v.shape == v.shape
        assert not np.array_equal(new_params, params)

    def test_adam_compute_loss(self):
        """Test loss computation."""
        X = np.random.randn(10, 20)
        factors = np.random.randn(10, 3)
        loadings = np.random.randn(3, 20)
        
        opt = AdamOptimizer(regularization=0.1)
        loss = opt._compute_loss(X, factors, loadings)
        
        assert isinstance(loss, float)
        assert loss > 0


class TestMiniBatchOptimizer:
    """Test MiniBatchOptimizer."""

    def test_minibatch_initialization(self):
        """Test mini-batch optimizer initialization."""
        base_opt = AlternatingLeastSquares()
        opt = MiniBatchOptimizer(
            base_optimizer=base_opt,
            batch_size=128,
            n_epochs=5,
            shuffle=False
        )
        
        assert opt.base_optimizer is base_opt
        assert opt.batch_size == 128
        assert opt.n_epochs == 5
        assert opt.shuffle is False

    def test_minibatch_optimize(self):
        """Test mini-batch optimization."""
        X = np.random.randn(100, 50)
        init_factors = np.random.randn(100, 5)
        init_loadings = np.random.randn(5, 50)
        
        base_opt = AlternatingLeastSquares(regularization=0.01)
        opt = MiniBatchOptimizer(
            base_optimizer=base_opt,
            batch_size=25,
            n_epochs=2,
            shuffle=True
        )
        
        result = opt.optimize(X, init_factors, init_loadings, max_iter=10, tol=1e-4)
        
        assert isinstance(result, OptimizationResult)
        assert result.factors.shape == (100, 5)
        assert result.loadings.shape == (5, 50)

    def test_minibatch_no_shuffle(self):
        """Test mini-batch without shuffling."""
        X = np.random.randn(50, 30)
        init_factors = np.random.randn(50, 3)
        init_loadings = np.random.randn(3, 30)
        
        base_opt = AlternatingLeastSquares()
        opt = MiniBatchOptimizer(
            base_optimizer=base_opt,
            batch_size=10,
            n_epochs=1,
            shuffle=False
        )
        
        result = opt.optimize(X, init_factors, init_loadings, max_iter=5)
        
        assert result.factors.shape == (50, 3)

    def test_minibatch_early_convergence(self):
        """Test early convergence in mini-batch."""
        # Create easy problem that converges quickly
        true_factors = np.random.randn(40, 2)
        true_loadings = np.random.randn(2, 20)
        X = np.dot(true_factors, true_loadings)
        
        base_opt = AlternatingLeastSquares(regularization=0.001)
        opt = MiniBatchOptimizer(
            base_optimizer=base_opt,
            batch_size=20,
            n_epochs=10,
            shuffle=False
        )
        
        result = opt.optimize(X, true_factors, true_loadings, max_iter=100, tol=1e-8)
        
        # Should converge early
        assert result.converged


class TestSparseOptimizer:
    """Test SparseOptimizer."""

    def test_sparse_initialization(self):
        """Test sparse optimizer initialization."""
        base_opt = AlternatingLeastSquares()
        opt = SparseOptimizer(
            base_optimizer=base_opt,
            sparsity_penalty=0.2,
            sparsity_type='l1'
        )
        
        assert opt.base_optimizer is base_opt
        assert opt.sparsity_penalty == 0.2
        assert opt.sparsity_type == 'l1'

    def test_sparse_optimize_l1(self):
        """Test sparse optimization with L1 penalty."""
        X = np.random.randn(20, 30)
        init_factors = np.random.randn(20, 4)
        init_loadings = np.random.randn(4, 30)
        
        base_opt = AlternatingLeastSquares(regularization=0.01)
        opt = SparseOptimizer(
            base_optimizer=base_opt,
            sparsity_penalty=0.1,
            sparsity_type='l1'
        )
        
        result = opt.optimize(X, init_factors, init_loadings, max_iter=10)
        
        assert result.factors.shape == (20, 4)
        assert result.loadings.shape == (4, 30)
        
        # Check that sparsity was applied (should have some zeros)
        assert np.sum(result.factors == 0) > 0 or np.sum(result.loadings == 0) > 0

    def test_sparse_optimize_l2(self):
        """Test sparse optimization with L2 penalty."""
        X = np.random.randn(15, 25)
        init_factors = np.random.randn(15, 3)
        init_loadings = np.random.randn(3, 25)
        
        base_opt = AlternatingLeastSquares()
        opt = SparseOptimizer(
            base_optimizer=base_opt,
            sparsity_penalty=0.5,
            sparsity_type='l2'
        )
        
        result = opt.optimize(X, init_factors, init_loadings, max_iter=5)
        
        assert result.factors.shape == (15, 3)
        
        # L2 shrinkage should reduce magnitude but not create zeros
        assert np.all(np.abs(result.factors) <= np.abs(init_factors))

    def test_sparse_apply_sparsity_l1(self):
        """Test L1 sparsity application."""
        matrix = np.array([[0.5, -0.3, 1.0], [0.05, -0.15, 0.8]])
        
        opt = SparseOptimizer(
            base_optimizer=AlternatingLeastSquares(),
            sparsity_penalty=0.1,
            sparsity_type='l1'
        )
        
        sparse_matrix = opt._apply_sparsity(matrix)
        
        # Values below threshold should be zero
        assert sparse_matrix[1, 0] == 0  # 0.05 < 0.1
        assert sparse_matrix[0, 0] == 0.4  # 0.5 - 0.1
        assert sparse_matrix[0, 2] == 0.9  # 1.0 - 0.1

    def test_sparse_apply_sparsity_l2(self):
        """Test L2 sparsity application."""
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        opt = SparseOptimizer(
            base_optimizer=AlternatingLeastSquares(),
            sparsity_penalty=1.0,
            sparsity_type='l2'
        )
        
        sparse_matrix = opt._apply_sparsity(matrix)
        
        # Should shrink by factor of 1/(1+1) = 0.5
        np.testing.assert_array_almost_equal(sparse_matrix, matrix * 0.5)

    def test_sparse_invalid_type(self):
        """Test invalid sparsity type."""
        opt = SparseOptimizer(
            base_optimizer=AlternatingLeastSquares(),
            sparsity_type='invalid'
        )
        
        matrix = np.random.randn(5, 5)
        with pytest.raises(ValueError, match="Unknown sparsity type"):
            opt._apply_sparsity(matrix)


class TestCreateOptimizer:
    """Test create_optimizer factory function."""

    def test_create_als_optimizer(self):
        """Test creating ALS optimizer."""
        opt = create_optimizer('als', regularization=0.05)
        
        assert isinstance(opt, AlternatingLeastSquares)
        assert opt.regularization == 0.05

    def test_create_adam_optimizer(self):
        """Test creating ADAM optimizer."""
        opt = create_optimizer('adam', learning_rate=0.01)
        
        assert isinstance(opt, AdamOptimizer)
        assert opt.learning_rate == 0.01

    def test_create_minibatch_optimizer(self):
        """Test creating mini-batch optimizer."""
        base_opt = AdamOptimizer()
        opt = create_optimizer('minibatch', base_optimizer=base_opt, batch_size=64)
        
        assert isinstance(opt, MiniBatchOptimizer)
        assert opt.batch_size == 64
        assert isinstance(opt.base_optimizer, AdamOptimizer)

    def test_create_minibatch_default_base(self):
        """Test mini-batch with default base optimizer."""
        opt = create_optimizer('minibatch', batch_size=32)
        
        assert isinstance(opt, MiniBatchOptimizer)
        assert isinstance(opt.base_optimizer, AlternatingLeastSquares)

    def test_create_sparse_optimizer(self):
        """Test creating sparse optimizer."""
        opt = create_optimizer('sparse', sparsity_penalty=0.2, sparsity_type='l1')
        
        assert isinstance(opt, SparseOptimizer)
        assert opt.sparsity_penalty == 0.2
        assert opt.sparsity_type == 'l1'

    def test_create_invalid_optimizer(self):
        """Test invalid optimizer type."""
        with pytest.raises(ValueError, match="Unknown optimizer type"):
            create_optimizer('invalid')


class TestOptimizationScheduler:
    """Test OptimizationScheduler."""

    def test_scheduler_initialization(self):
        """Test scheduler initialization."""
        scheduler = OptimizationScheduler(initial_lr=0.01)
        
        assert scheduler.initial_lr == 0.01
        assert scheduler.current_lr == 0.01

    def test_step_decay(self):
        """Test step decay schedule."""
        scheduler = OptimizationScheduler(initial_lr=1.0)
        
        # First 10 epochs should have same LR
        lr = scheduler.step_decay(5, drop_rate=0.5, epochs_drop=10)
        assert lr == 1.0
        
        # After 10 epochs, should drop by 0.5
        lr = scheduler.step_decay(10, drop_rate=0.5, epochs_drop=10)
        assert lr == 0.5
        
        # After 20 epochs, should drop again
        lr = scheduler.step_decay(20, drop_rate=0.5, epochs_drop=10)
        assert lr == 0.25

    def test_exponential_decay(self):
        """Test exponential decay schedule."""
        scheduler = OptimizationScheduler(initial_lr=1.0)
        
        lr = scheduler.exponential_decay(0, decay_rate=0.9)
        assert lr == 1.0
        
        lr = scheduler.exponential_decay(1, decay_rate=0.9)
        assert lr == 0.9
        
        lr = scheduler.exponential_decay(2, decay_rate=0.9)
        assert lr == 0.81

    def test_cosine_decay(self):
        """Test cosine annealing schedule."""
        scheduler = OptimizationScheduler(initial_lr=1.0)
        
        # At epoch 0, should be initial LR
        lr = scheduler.cosine_decay(0, total_epochs=10)
        assert lr == 1.0
        
        # At half epochs, should be around 0.5
        lr = scheduler.cosine_decay(5, total_epochs=10)
        np.testing.assert_almost_equal(lr, 0.5, decimal=2)
        
        # At final epoch, should be close to 0
        lr = scheduler.cosine_decay(10, total_epochs=10)
        np.testing.assert_almost_equal(lr, 0.0, decimal=2)