"""Tests for convergence control features in TFA and HTFA.

This module tests convergence tracking, warning emission, and results integration
for both TFA and HTFA models.
"""

import warnings
import numpy as np
import pytest

from htfa.core.tfa import TFA
from htfa.core.htfa import HTFA
from htfa.results import HTFAResults


class TestTFAConvergence:
    """Test convergence tracking for TFA."""
    
    def test_tfa_convergence_tracking(self):
        """Test that TFA tracks convergence information."""
        # Generate synthetic data
        np.random.seed(42)
        n_voxels, n_timepoints = 100, 50
        X = np.random.randn(n_voxels, n_timepoints)
        
        # Fit model with high tolerance for quick convergence
        tfa = TFA(K=5, max_iter=100, tol=1.0, verbose=False)
        tfa.fit(X)
        
        # Check convergence_info_ is populated
        assert hasattr(tfa, 'convergence_info_')
        assert tfa.convergence_info_ is not None
        assert 'converged' in tfa.convergence_info_
        assert 'n_iterations' in tfa.convergence_info_
        assert 'final_tolerance' in tfa.convergence_info_
        
        # Should converge with high tolerance
        assert tfa.convergence_info_['converged'] is True
        assert tfa.convergence_info_['n_iterations'] < tfa.max_iter
    
    def test_tfa_non_convergence_warning(self):
        """Test that TFA emits warning when not converging."""
        # Generate synthetic data
        np.random.seed(42)
        n_voxels, n_timepoints = 100, 50
        X = np.random.randn(n_voxels, n_timepoints)
        
        # Fit model with very low tolerance and few iterations
        tfa = TFA(K=5, max_iter=2, tol=1e-20, verbose=False)
        
        # Should emit warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tfa.fit(X)
            
            # Check warning was raised
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "did not converge" in str(w[0].message)
            assert "max_iter" in str(w[0].message)
        
        # Check convergence_info_ indicates non-convergence
        assert tfa.convergence_info_['converged'] is False
        assert tfa.convergence_info_['n_iterations'] == tfa.max_iter
    
    def test_tfa_convergence_with_different_tolerances(self):
        """Test TFA convergence with different tolerance values."""
        np.random.seed(42)
        n_voxels, n_timepoints = 100, 50
        X = np.random.randn(n_voxels, n_timepoints)
        
        # Test with different tolerances
        tolerances = [1.0, 0.1, 0.01]
        iterations_list = []
        
        for tol in tolerances:
            tfa = TFA(K=5, max_iter=500, tol=tol, verbose=False)
            tfa.fit(X)
            iterations_list.append(tfa.convergence_info_['n_iterations'])
        
        # More relaxed tolerance should converge faster
        assert iterations_list[0] <= iterations_list[1]
        assert iterations_list[1] <= iterations_list[2]


class TestHTFAConvergence:
    """Test convergence tracking for HTFA."""
    
    def test_htfa_convergence_tracking(self):
        """Test that HTFA tracks convergence information."""
        # Generate synthetic multi-subject data
        np.random.seed(42)
        n_subjects = 3
        n_voxels, n_timepoints = 50, 30
        X = [np.random.randn(n_voxels, n_timepoints) for _ in range(n_subjects)]
        
        # Fit model with relaxed settings for convergence
        htfa = HTFA(K=3, max_global_iter=10, max_local_iter=10, tol=1.0, verbose=False)
        htfa.fit(X)
        
        # Check convergence_info_ is populated
        assert hasattr(htfa, 'convergence_info_')
        assert htfa.convergence_info_ is not None
        assert 'converged' in htfa.convergence_info_
        assert 'n_iterations' in htfa.convergence_info_
        assert 'subject_convergence' in htfa.convergence_info_
        
        # Check subject convergence info
        subject_conv = htfa.convergence_info_['subject_convergence']
        assert len(subject_conv) == n_subjects
        for subj_info in subject_conv:
            assert 'converged' in subj_info
            assert 'n_iterations' in subj_info
    
    def test_htfa_non_convergence_warning(self):
        """Test that HTFA emits warning when not converging."""
        # Generate synthetic multi-subject data
        np.random.seed(42)
        n_subjects = 3
        n_voxels, n_timepoints = 50, 30
        X = [np.random.randn(n_voxels, n_timepoints) for _ in range(n_subjects)]
        
        # Fit model with very few iterations
        htfa = HTFA(K=3, max_global_iter=1, max_local_iter=2, tol=1e-20, verbose=False)
        
        # Should emit warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            htfa.fit(X)
            
            # Check at least one warning was raised (HTFA and possibly TFA warnings)
            assert len(w) >= 1
            # Check for HTFA-specific warning
            htfa_warnings = [warn for warn in w if "HTFA" in str(warn.message)]
            assert len(htfa_warnings) == 1
            assert "did not converge" in str(htfa_warnings[0].message)
        
        # Check convergence_info_ indicates non-convergence
        assert htfa.convergence_info_['converged'] is False
        assert htfa.convergence_info_['n_iterations'] == htfa.max_global_iter
    
    def test_htfa_subject_convergence_tracking(self):
        """Test that HTFA tracks per-subject convergence."""
        # Generate synthetic multi-subject data with different noise levels
        np.random.seed(42)
        n_subjects = 3
        n_voxels, n_timepoints = 50, 30
        
        # Create data with different noise levels per subject
        X = []
        for i in range(n_subjects):
            noise_level = 0.1 * (i + 1)  # Increasing noise
            data = np.random.randn(n_voxels, n_timepoints) * noise_level
            X.append(data)
        
        # Fit model
        htfa = HTFA(K=3, max_global_iter=20, max_local_iter=50, tol=0.1, verbose=False)
        htfa.fit(X)
        
        # Check that subject convergence is tracked
        subject_conv = htfa.convergence_info_['subject_convergence']
        assert len(subject_conv) == n_subjects
        
        # Each subject should have convergence info
        for i, subj_info in enumerate(subject_conv):
            assert isinstance(subj_info, dict)
            assert 'converged' in subj_info
            assert 'n_iterations' in subj_info
            assert 'final_tolerance' in subj_info


class TestHTFAResultsConvergence:
    """Test convergence integration in HTFAResults."""
    
    def test_results_convergence_properties(self):
        """Test HTFAResults convergence properties."""
        # Create mock data for HTFAResults
        n_subjects = 3
        n_factors = 5
        n_voxels = 100
        n_timepoints = 50
        
        global_template = np.random.randn(n_factors, n_voxels)
        subject_factors = [np.random.randn(n_factors, n_voxels) for _ in range(n_subjects)]
        subject_weights = [np.random.randn(n_timepoints, n_factors) for _ in range(n_subjects)]
        
        # Create results with convergence info
        fit_info = {
            'convergence_info': {
                'converged': True,
                'n_iterations': 8,
                'subject_convergence': [
                    {'converged': True, 'n_iterations': 45},
                    {'converged': True, 'n_iterations': 50},
                    {'converged': False, 'n_iterations': 100},
                ]
            }
        }
        
        results = HTFAResults(
            global_template=global_template,
            subject_factors=subject_factors,
            subject_weights=subject_weights,
            bids_info={'dataset_name': 'test'},
            preprocessing={},
            model_params={'K': n_factors},
            fit_info=fit_info,
        )
        
        # Test properties
        assert results.is_converged is True
        assert results.n_iterations == 8
        assert isinstance(results.convergence_info, dict)
        assert results.convergence_info['converged'] is True
    
    def test_results_convergence_summary(self):
        """Test HTFAResults convergence summary method."""
        # Create mock data
        n_subjects = 3
        n_factors = 5
        n_voxels = 100
        n_timepoints = 50
        
        global_template = np.random.randn(n_factors, n_voxels)
        subject_factors = [np.random.randn(n_factors, n_voxels) for _ in range(n_subjects)]
        subject_weights = [np.random.randn(n_timepoints, n_factors) for _ in range(n_subjects)]
        
        # Create results with mixed convergence
        fit_info = {
            'convergence_info': {
                'converged': False,
                'n_iterations': 10,
                'subject_convergence': [
                    {'converged': True, 'n_iterations': 45},
                    {'converged': False, 'n_iterations': 100},
                    {'converged': True, 'n_iterations': 67},
                ]
            }
        }
        
        results = HTFAResults(
            global_template=global_template,
            subject_factors=subject_factors,
            subject_weights=subject_weights,
            bids_info={'dataset_name': 'test'},
            preprocessing={},
            model_params={'K': n_factors},
            fit_info=fit_info,
        )
        
        # Get summary
        summary = results.get_convergence_summary()
        
        # Check summary contents
        assert summary['converged'] is False
        assert summary['n_iterations'] == 10
        assert summary['n_subjects_converged'] == 2
        assert summary['total_subjects'] == 3
        assert 'warnings' in summary
        assert len(summary['warnings']) > 0
        assert 'subject_details' in summary
    
    def test_results_repr_with_convergence(self):
        """Test HTFAResults __repr__ includes convergence status."""
        # Create mock data
        n_subjects = 2
        n_factors = 3
        n_voxels = 50
        n_timepoints = 25
        
        global_template = np.random.randn(n_factors, n_voxels)
        subject_factors = [np.random.randn(n_factors, n_voxels) for _ in range(n_subjects)]
        subject_weights = [np.random.randn(n_timepoints, n_factors) for _ in range(n_subjects)]
        
        # Test with converged model
        fit_info_converged = {
            'convergence_info': {'converged': True, 'n_iterations': 5}
        }
        
        results_converged = HTFAResults(
            global_template=global_template,
            subject_factors=subject_factors,
            subject_weights=subject_weights,
            bids_info={'dataset_name': 'test_dataset'},
            preprocessing={},
            model_params={'K': n_factors},
            fit_info=fit_info_converged,
        )
        
        repr_str = repr(results_converged)
        assert "converged=Yes" in repr_str
        assert "n_subjects=2" in repr_str
        assert "n_factors=3" in repr_str
        
        # Test with non-converged model
        fit_info_not_converged = {
            'convergence_info': {'converged': False, 'n_iterations': 100}
        }
        
        results_not_converged = HTFAResults(
            global_template=global_template,
            subject_factors=subject_factors,
            subject_weights=subject_weights,
            bids_info={'dataset_name': 'test_dataset'},
            preprocessing={},
            model_params={'K': n_factors},
            fit_info=fit_info_not_converged,
        )
        
        repr_str = repr(results_not_converged)
        assert "converged=No" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])