"""Optimization utilities and advanced algorithms for HTFA."""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass


@dataclass
class OptimizationResult:
    """Result from optimization algorithm."""
    factors: np.ndarray
    loadings: np.ndarray
    loss_history: list
    converged: bool
    n_iterations: int


class Optimizer(ABC):
    """Abstract base class for optimization algorithms."""
    
    @abstractmethod
    def optimize(
        self,
        X: np.ndarray,
        initial_factors: np.ndarray,
        initial_loadings: np.ndarray,
        max_iter: int = 1000,
        tol: float = 1e-6
    ) -> OptimizationResult:
        """Optimize factors and loadings."""
        pass


class AlternatingLeastSquares(Optimizer):
    """Traditional Alternating Least Squares optimizer."""
    
    def __init__(self, regularization: float = 0.01):
        self.regularization = regularization
    
    def optimize(
        self,
        X: np.ndarray,
        initial_factors: np.ndarray,
        initial_loadings: np.ndarray,
        max_iter: int = 1000,
        tol: float = 1e-6
    ) -> OptimizationResult:
        """Optimize using alternating least squares."""
        factors = initial_factors.copy()
        loadings = initial_loadings.copy()
        loss_history = []
        
        prev_loss = float('inf')
        
        for iteration in range(max_iter):
            # Update factors
            factors = self._update_factors(X, loadings)
            
            # Update loadings  
            loadings = self._update_loadings(X, factors)
            
            # Compute loss
            reconstruction = np.dot(factors, loadings)
            loss = np.linalg.norm(X - reconstruction) ** 2
            loss += self.regularization * (np.linalg.norm(factors) ** 2 + 
                                         np.linalg.norm(loadings) ** 2)
            loss_history.append(loss)
            
            # Check convergence
            if abs(prev_loss - loss) < tol:
                break
                
            prev_loss = loss
        
        converged = iteration < max_iter - 1
        
        return OptimizationResult(
            factors=factors,
            loadings=loadings,
            loss_history=loss_history,
            converged=converged,
            n_iterations=iteration + 1
        )
    
    def _update_factors(self, X: np.ndarray, loadings: np.ndarray) -> np.ndarray:
        """Update factors using least squares."""
        LLT = np.dot(loadings, loadings.T)
        reg_term = self.regularization * np.eye(LLT.shape[0])
        LLT_reg = LLT + reg_term
        
        XLT = np.dot(X, loadings.T)
        factors = np.dot(XLT, np.linalg.pinv(LLT_reg))
        return factors
    
    def _update_loadings(self, X: np.ndarray, factors: np.ndarray) -> np.ndarray:
        """Update loadings using least squares."""
        FTF = np.dot(factors.T, factors)
        reg_term = self.regularization * np.eye(FTF.shape[0])
        FTF_reg = FTF + reg_term
        
        FTX = np.dot(factors.T, X)
        loadings = np.dot(np.linalg.pinv(FTF_reg), FTX)
        return loadings


class AdamOptimizer(Optimizer):
    """ADAM optimizer for HTFA."""
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        regularization: float = 0.01
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.regularization = regularization
    
    def optimize(
        self,
        X: np.ndarray,
        initial_factors: np.ndarray,
        initial_loadings: np.ndarray,
        max_iter: int = 1000,
        tol: float = 1e-6
    ) -> OptimizationResult:
        """Optimize using ADAM algorithm."""
        factors = initial_factors.copy()
        loadings = initial_loadings.copy()
        
        # Initialize ADAM parameters
        m_factors, v_factors = np.zeros_like(factors), np.zeros_like(factors)
        m_loadings, v_loadings = np.zeros_like(loadings), np.zeros_like(loadings)
        
        loss_history = []
        prev_loss = float('inf')
        
        for iteration in range(max_iter):
            # Compute gradients
            grad_factors, grad_loadings = self._compute_gradients(X, factors, loadings)
            
            # Update ADAM parameters and variables
            factors, m_factors, v_factors = self._adam_update(
                factors, grad_factors, m_factors, v_factors, iteration + 1
            )
            loadings, m_loadings, v_loadings = self._adam_update(
                loadings, grad_loadings, m_loadings, v_loadings, iteration + 1
            )
            
            # Compute loss
            loss = self._compute_loss(X, factors, loadings)
            loss_history.append(loss)
            
            # Check convergence
            if abs(prev_loss - loss) < tol:
                break
                
            prev_loss = loss
        
        converged = iteration < max_iter - 1
        
        return OptimizationResult(
            factors=factors,
            loadings=loadings,
            loss_history=loss_history,
            converged=converged,
            n_iterations=iteration + 1
        )
    
    def _compute_gradients(
        self, X: np.ndarray, factors: np.ndarray, loadings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute gradients of the loss function."""
        reconstruction = np.dot(factors, loadings)
        residual = X - reconstruction
        
        # Gradient w.r.t factors
        grad_factors = -2 * np.dot(residual, loadings.T)
        grad_factors += 2 * self.regularization * factors
        
        # Gradient w.r.t loadings
        grad_loadings = -2 * np.dot(factors.T, residual)
        grad_loadings += 2 * self.regularization * loadings
        
        return grad_factors, grad_loadings
    
    def _adam_update(
        self, params: np.ndarray, gradients: np.ndarray, m: np.ndarray, 
        v: np.ndarray, t: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform ADAM parameter update."""
        # Update biased first moment estimate
        m = self.beta1 * m + (1 - self.beta1) * gradients
        
        # Update biased second raw moment estimate
        v = self.beta2 * v + (1 - self.beta2) * (gradients ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat = m / (1 - self.beta1 ** t)
        
        # Compute bias-corrected second raw moment estimate
        v_hat = v / (1 - self.beta2 ** t)
        
        # Update parameters
        params = params - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return params, m, v
    
    def _compute_loss(self, X: np.ndarray, factors: np.ndarray, loadings: np.ndarray) -> float:
        """Compute loss function."""
        reconstruction = np.dot(factors, loadings)
        mse_loss = np.linalg.norm(X - reconstruction) ** 2
        reg_loss = self.regularization * (np.linalg.norm(factors) ** 2 + 
                                        np.linalg.norm(loadings) ** 2)
        return mse_loss + reg_loss


class MiniBatchOptimizer(Optimizer):
    """Mini-batch optimizer for large-scale HTFA."""
    
    def __init__(
        self,
        base_optimizer: Optimizer,
        batch_size: int = 256,
        n_epochs: int = 10,
        shuffle: bool = True
    ):
        self.base_optimizer = base_optimizer
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.shuffle = shuffle
    
    def optimize(
        self,
        X: np.ndarray,
        initial_factors: np.ndarray,
        initial_loadings: np.ndarray,
        max_iter: int = 1000,
        tol: float = 1e-6
    ) -> OptimizationResult:
        """Optimize using mini-batches."""
        n_samples = X.shape[0]
        factors = initial_factors.copy()
        loadings = initial_loadings.copy()
        loss_history = []
        
        total_iterations = 0
        prev_loss = float('inf')
        converged = False
        
        for epoch in range(self.n_epochs):
            # Shuffle data if requested
            if self.shuffle:
                indices = np.random.permutation(n_samples)
                X_shuffled = X[indices]
                factors_shuffled = factors[indices]
            else:
                X_shuffled = X
                factors_shuffled = factors
            
            # Process mini-batches
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                factors_batch = factors_shuffled[start_idx:end_idx]
                
                # Optimize on batch
                result = self.base_optimizer.optimize(
                    X_batch, factors_batch, loadings,
                    max_iter=max_iter // self.n_epochs,
                    tol=tol
                )
                
                # Update parameters
                factors[start_idx:end_idx] = result.factors
                loadings = result.loadings
                
                total_iterations += result.n_iterations
                loss_history.extend(result.loss_history)
                
                # Check global convergence
                if result.loss_history:
                    current_loss = result.loss_history[-1]
                    if abs(prev_loss - current_loss) < tol:
                        converged = True
                        break
                    prev_loss = current_loss
            
            if converged:
                break
        
        return OptimizationResult(
            factors=factors,
            loadings=loadings,
            loss_history=loss_history,
            converged=converged,
            n_iterations=total_iterations
        )


class SparseOptimizer(Optimizer):
    """Optimizer with sparsity constraints for HTFA."""
    
    def __init__(
        self,
        base_optimizer: Optimizer,
        sparsity_penalty: float = 0.1,
        sparsity_type: str = 'l1'  # 'l1' or 'l2'
    ):
        self.base_optimizer = base_optimizer
        self.sparsity_penalty = sparsity_penalty
        self.sparsity_type = sparsity_type
    
    def optimize(
        self,
        X: np.ndarray,
        initial_factors: np.ndarray,
        initial_loadings: np.ndarray,
        max_iter: int = 1000,
        tol: float = 1e-6
    ) -> OptimizationResult:
        """Optimize with sparsity constraints."""
        # Get base optimization result
        result = self.base_optimizer.optimize(
            X, initial_factors, initial_loadings, max_iter, tol
        )
        
        # Apply sparsity constraints
        factors_sparse = self._apply_sparsity(result.factors)
        loadings_sparse = self._apply_sparsity(result.loadings)
        
        return OptimizationResult(
            factors=factors_sparse,
            loadings=loadings_sparse,
            loss_history=result.loss_history,
            converged=result.converged,
            n_iterations=result.n_iterations
        )
    
    def _apply_sparsity(self, matrix: np.ndarray) -> np.ndarray:
        """Apply sparsity constraint to matrix."""
        if self.sparsity_type == 'l1':
            # Soft thresholding for L1 penalty
            threshold = self.sparsity_penalty
            return np.sign(matrix) * np.maximum(0, np.abs(matrix) - threshold)
        elif self.sparsity_type == 'l2':
            # Shrinkage for L2 penalty  
            shrinkage_factor = 1 / (1 + self.sparsity_penalty)
            return matrix * shrinkage_factor
        else:
            raise ValueError(f"Unknown sparsity type: {self.sparsity_type}")


def create_optimizer(optimizer_type: str, **kwargs) -> Optimizer:
    """Factory function to create optimizers.
    
    Parameters
    ----------
    optimizer_type : str
        Type of optimizer ('als', 'adam', 'minibatch', 'sparse').
    **kwargs
        Optimizer-specific parameters.
    
    Returns
    -------
    optimizer : Optimizer
        Configured optimizer instance.
    """
    if optimizer_type == 'als':
        return AlternatingLeastSquares(**kwargs)
    elif optimizer_type == 'adam':
        return AdamOptimizer(**kwargs)
    elif optimizer_type == 'minibatch':
        base_optimizer = kwargs.pop('base_optimizer', AlternatingLeastSquares())
        return MiniBatchOptimizer(base_optimizer, **kwargs)
    elif optimizer_type == 'sparse':
        base_optimizer = kwargs.pop('base_optimizer', AlternatingLeastSquares())
        return SparseOptimizer(base_optimizer, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


class OptimizationScheduler:
    """Learning rate scheduler for optimization algorithms."""
    
    def __init__(self, initial_lr: float = 0.001):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
    
    def step_decay(self, epoch: int, drop_rate: float = 0.5, epochs_drop: int = 10) -> float:
        """Step decay schedule."""
        self.current_lr = self.initial_lr * (drop_rate ** (epoch // epochs_drop))
        return self.current_lr
    
    def exponential_decay(self, epoch: int, decay_rate: float = 0.95) -> float:
        """Exponential decay schedule."""
        self.current_lr = self.initial_lr * (decay_rate ** epoch)
        return self.current_lr
    
    def cosine_decay(self, epoch: int, total_epochs: int) -> float:
        """Cosine annealing schedule."""
        self.current_lr = self.initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))
        return self.current_lr