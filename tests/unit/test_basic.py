"""Basic tests for HTFA package structure."""

import numpy as np
import pytest

import htfa
from htfa.core.htfa import HTFA
from htfa.core.tfa import TFA


def test_package_imports():
    """Test that package imports work correctly."""
    assert hasattr(htfa, "TFA")
    assert hasattr(htfa, "HTFA")
    assert hasattr(htfa, "version")


def test_tfa_initialization():
    """Test TFA class initialization."""
    tfa = TFA(K=5, verbose=True)
    assert tfa.K == 5
    assert tfa.verbose is True
    assert tfa.factors_ is None  # Not fitted yet


def test_htfa_initialization():
    """Test HTFA class initialization."""
    htfa_model = HTFA(K=3, max_global_iter=5)
    assert htfa_model.K == 3
    assert htfa_model.max_global_iter == 5
    assert htfa_model.global_template_ is None  # Not fitted yet


def test_tfa_input_validation():
    """Test TFA input validation."""
    tfa = TFA(K=2)

    # Test with invalid input dimensions
    with pytest.raises(ValueError, match="X must be a 2D array"):
        tfa.fit(np.random.randn(10))  # 1D array should fail


def test_htfa_input_validation():
    """Test HTFA input validation."""
    htfa_model = HTFA(K=2)

    # Test with non-list input
    with pytest.raises(ValueError, match="X must be a list of arrays"):
        htfa_model.fit(np.random.randn(10, 20))

    # Test with empty list
    with pytest.raises(ValueError, match="X cannot be empty"):
        htfa_model.fit([])


def test_tfa_basic_fitting():
    """Test basic TFA fitting functionality."""
    # Create synthetic data
    np.random.seed(42)
    n_voxels = 50
    X = np.random.randn(n_voxels, 100)  # 50 voxels, 100 timepoints
    coords = np.random.randn(n_voxels, 3)  # 3D coordinates for each voxel

    tfa = TFA(K=3, max_iter=10, verbose=True)
    tfa.fit(X, coords)

    # Check that model was fitted
    assert tfa.factors_ is not None
    assert tfa.weights_ is not None
    assert tfa.factors_.shape[0] == 3  # K factors
    assert tfa.weights_.shape[1] == 3  # K factors


def test_htfa_basic_fitting():
    """Test basic HTFA fitting functionality."""
    # Create synthetic multi-subject data
    np.random.seed(42)
    n_voxels = 30
    X = [
        np.random.randn(n_voxels, 50),  # Subject 1: 30 voxels, 50 timepoints
        np.random.randn(n_voxels, 60),  # Subject 2: 30 voxels, 60 timepoints
    ]
    coords = [
        np.random.randn(n_voxels, 3),  # 3D coordinates for subject 1
        np.random.randn(n_voxels, 3),  # 3D coordinates for subject 2
    ]

    htfa_model = HTFA(K=2, max_global_iter=2, max_local_iter=5, verbose=True)
    htfa_model.fit(X, coords)

    # Check that model was fitted
    assert htfa_model.global_template_ is not None
    assert htfa_model.factors_ is not None
    assert htfa_model.weights_ is not None
    assert len(htfa_model.factors_) == 2  # Two subjects
    assert len(htfa_model.weights_) == 2  # Two subjects
