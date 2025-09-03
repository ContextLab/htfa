# HTFA Toolbox Technical Design Document

## Overview

This document outlines the technical design and architecture for the standalone Hierarchical Topographic Factor Analysis (HTFA) toolbox. The goal is to create a lightweight, performant, and easy-to-install implementation based on the BrainIAK HTFA algorithm but with minimal dependencies.

## Project Goals

### Primary Objectives
1. **Standalone Implementation**: Remove dependencies on the complex BrainIAK ecosystem
2. **Minimal Dependencies**: Use only essential packages (NumPy, SciPy, scikit-learn)
3. **Performance**: Optimize for speed and memory efficiency
4. **Usability**: Provide clear APIs, documentation, and tutorials
5. **Extensibility**: Design for future enhancements and research

### Success Criteria
- Installation requires < 5 dependencies
- 10x easier installation than BrainIAK
- Performance comparable to or better than BrainIAK implementation
- Complete documentation with tutorials
- Active research and development roadmap

## Architecture Overview

### Package Structure
```
htfa/
├── htfa/                    # Main package
│   ├── __init__.py         # Package exports
│   ├── core/               # Core algorithms
│   │   ├── __init__.py
│   │   ├── tfa.py          # Base TFA implementation
│   │   └── htfa.py         # Hierarchical TFA
│   └── utils/              # Utilities
│       ├── __init__.py
│       ├── data.py         # Data handling
│       ├── visualization.py # Plotting utilities
│       └── validation.py   # Input validation
├── tests/                  # Test suite
├── docs/                   # Documentation
├── tutorials/              # Jupyter tutorials
└── examples/               # Example scripts
```

### Dependency Strategy
- **Core Dependencies**: NumPy, SciPy, scikit-learn, nilearn, nibabel
- **BIDS Support**: pybids for dataset parsing and validation
- **Visualization**: matplotlib, seaborn, nilearn plotting utilities
- **Optional Dependencies**: Jupyter (tutorials), plotly (interactive plots)
- **Development Dependencies**: pytest, black, mypy, etc.

## Algorithm Design

### Topographic Factor Analysis (TFA)

#### Mathematical Formulation
TFA decomposes neural data X into spatial factors F and weights W:
```
X ≈ W · F
```

Where:
- X: (n_voxels, n_timepoints) data matrix
- F: (K, n_voxels) spatial factors  
- W: (n_timepoints, K) weight matrix
- K: number of factors

#### Implementation Strategy
```python
class TFA(BaseEstimator):
    def __init__(self, K, max_iter=500, tol=1e-6, ...):
        # Initialize parameters
        
    def fit(self, X, coords=None):
        # 1. Initialize factors using k-means clustering
        # 2. Iterative optimization:
        #    - Update factors given weights
        #    - Update weights given factors  
        #    - Check convergence
        # 3. Return fitted model
        
    def _initialize_parameters(self, X, coords):
        # K-means clustering for spatial initialization
        
    def _optimize(self, X, coords):
        # Main optimization loop
        # - Non-linear least squares for factors
        # - Ridge/OLS regression for weights
```

#### Key Components
1. **Initialization**: K-means clustering of spatial coordinates
2. **Factor Estimation**: Non-linear least squares optimization
3. **Weight Estimation**: Ridge regression or ordinary least squares
4. **Convergence**: Track likelihood or parameter changes

### Hierarchical TFA (HTFA)

#### Mathematical Formulation
HTFA extends TFA to multi-subject analysis:
```
X_i ≈ W_i · F_global + noise
```

Where:
- X_i: Data for subject i
- W_i: Subject-specific weights
- F_global: Shared spatial factors across subjects

#### Implementation Strategy
```python
class HTFA(BaseEstimator):
    def __init__(self, K, max_global_iter=10, max_local_iter=50, ...):
        # Initialize parameters
        
    def fit(self, X_list, coords_list=None):
        # 1. Initialize subject-specific TFA models
        # 2. Hierarchical optimization:
        #    - Compute global template
        #    - Update subject models
        #    - Check convergence
        # 3. Extract final parameters
```

#### Hierarchical Optimization Algorithm
1. **Initialization**: Fit individual TFA models for each subject
2. **Global Template**: Average spatial factors across subjects
3. **Iterative Updates**:
   - Update each subject's model using global template information
   - Recompute global template
   - Check convergence of global template
4. **Factor Matching**: Use linear sum assignment to align factors across subjects

## User Interface Design

### BIDS Dataset Integration

#### Simple, Intuitive API
```python
import htfa

# Simplest possible usage - sensible defaults inferred from data
results = htfa.fit_bids('/path/to/bids/dataset')

# With basic customization
results = htfa.fit_bids(
    '/path/to/bids/dataset',
    K=10,                    # number of factors (auto-inferred if not specified)
    task='rest',             # BIDS task filter
    space='MNI152NLin2009cAsym'  # standardized space
)

# Advanced usage with custom parameters
results = htfa.fit_bids(
    '/path/to/bids/dataset', 
    K=15,
    subjects=['sub-01', 'sub-02'],  # subset of subjects
    sessions=['ses-1'],             # specific sessions
    mask='/path/to/mask.nii.gz',    # custom brain mask
    smoothing_fwhm=6,               # preprocessing options
    standardize=True,
    detrend=True
)
```

#### Automatic Data Inference
- **Factor Count (K)**: Estimated using information criteria (AIC/BIC) or explained variance
- **Brain Mask**: Automatically detected from BIDS derivatives or created from data
- **Preprocessing**: Sensible defaults (smoothing, detrending, standardization) based on data properties
- **Coordinate System**: Extracted from NIfTI headers and template files
- **Quality Control**: Automatic detection of motion artifacts, outliers, and data quality issues

### Results Class Design

#### Comprehensive Results Object
```python
class HTFAResults:
    """Comprehensive results from HTFA analysis with built-in visualization and export."""
    
    # Core fitted parameters
    global_template: np.ndarray           # (K, n_voxels) global spatial factors
    subject_factors: List[np.ndarray]     # Per-subject spatial factors
    subject_weights: List[np.ndarray]     # Per-subject temporal weights
    
    # Metadata and data provenance
    bids_info: dict                       # Original BIDS dataset information
    preprocessing: dict                   # Applied preprocessing steps
    model_params: dict                    # Model hyperparameters used
    fit_info: dict                        # Convergence and optimization details
    
    # Spatial information
    template_img: nibabel.Nifti1Image     # Template image for reconstruction
    brain_mask: np.ndarray                # Brain mask used
    coordinates: np.ndarray               # Voxel coordinates (x,y,z)
    
    def plot_global_factors(self, **kwargs) -> None:
        """Plot global spatial factors as brain maps."""
        
    def plot_subject_factors(self, subject_id, **kwargs) -> None:
        """Plot subject-specific factors as brain maps."""
        
    def plot_temporal_weights(self, subject_id, **kwargs) -> None:
        """Plot temporal weight timeseries."""
        
    def plot_network_summary(self, **kwargs) -> None:
        """Summary visualization of all factors and networks."""
        
    def to_nifti(self, factor_idx=None, subject_id=None) -> nibabel.Nifti1Image:
        """Reconstruct NIfTI images from HTFA factors."""
        
    def save_results(self, output_dir: str) -> None:
        """Save all results in BIDS-derivatives format."""
        
    def get_network_timeseries(self, subject_id: str) -> pd.DataFrame:
        """Extract network timeseries for further analysis."""
```

#### Built-in Visualization Methods
```python
# Global factor visualization
results.plot_global_factors(
    display_mode='mosaic',
    colorbar=True,
    threshold=0.3
)

# Subject-specific analysis
results.plot_subject_factors('sub-01', factors=[0, 1, 2])
results.plot_temporal_weights('sub-01', networks=['DMN', 'Attention'])

# Network summary dashboard
results.plot_network_summary(
    include_timeseries=True,
    include_connectivity=True,
    save_path='/path/to/figures/'
)

# Interactive exploration (with plotly)
results.plot_interactive_brain(factor_idx=0)
```

#### NIfTI Reconstruction and Export
```python
# Reconstruct specific factors as NIfTI images
global_factor_img = results.to_nifti(factor_idx=0)  # Global factor
subject_factor_img = results.to_nifti(factor_idx=0, subject_id='sub-01')  # Subject-specific

# Save in BIDS derivatives format
results.save_results('/path/to/derivatives/htfa/')
# Creates:
# - desc-global_factors.nii.gz
# - desc-{subject}_factors.nii.gz  
# - desc-{subject}_weights.tsv
# - desc-model_params.json
```

### Preprocessing Pipeline Integration

#### Automatic Preprocessing
```python
class HTFAPreprocessor:
    """Automatic preprocessing pipeline for BIDS fMRI data."""
    
    def __init__(self, 
                 smoothing_fwhm=6.0,
                 standardize=True,
                 detrend=True,
                 high_pass_filter=0.01,
                 motion_correction=True):
        self.smoothing_fwhm = smoothing_fwhm
        # ... other parameters
        
    def fit_transform(self, bids_path, **kwargs):
        """Load, preprocess, and prepare data for HTFA."""
        # 1. Load BIDS data with pybids
        # 2. Apply brain masking
        # 3. Smooth, filter, standardize
        # 4. Quality control checks
        # 5. Return preprocessed data + metadata
```

## Implementation Phases

### Phase 1: Core Implementation (Issues #60-#62)
**Timeline**: 3-5 weeks

**Deliverables**:
- [ ] Complete TFA implementation with optimization
- [ ] Complete HTFA hierarchical algorithm  
- [ ] HTFAResults class with visualization and NIfTI export
- [ ] Basic BIDS integration with pybids
- [ ] Comprehensive test suite with synthetic data
- [ ] Basic API documentation

**Key Components**:
- K-means initialization
- Non-linear least squares optimization
- Ridge regression weight estimation
- Convergence checking
- Multi-subject processing
- Factor matching algorithms
- NIfTI reconstruction via nilearn
- Basic brain plotting capabilities

### Phase 2: BIDS Integration and User Interface (New Issue)
**Timeline**: 3-4 weeks

**Deliverables**:
- [ ] Complete `htfa.fit_bids()` function with sensible defaults
- [ ] HTFAPreprocessor class for automatic data preprocessing
- [ ] Automatic factor count estimation (K selection)
- [ ] Quality control and outlier detection
- [ ] Advanced HTFAResults plotting methods
- [ ] BIDS derivatives output format

**Key Components**:
- BIDS dataset parsing and validation
- Automatic brain masking and coordinate extraction
- Preprocessing pipeline (smoothing, detrending, standardization)
- Information criteria for model selection
- Interactive plotting with plotly
- BIDS derivatives specification compliance

### Phase 3: Documentation and Tutorials (Issue #63)
**Timeline**: 2-3 weeks

**Deliverables**:
- [ ] Sphinx documentation site
- [ ] API reference documentation
- [ ] Installation and quickstart guides
- [ ] Jupyter notebook tutorials with real BIDS data
- [ ] Sample datasets and examples
- [ ] BIDS integration tutorial

**Key Components**:
- Sphinx setup with RTD theme
- Automated API documentation
- Interactive tutorials with BIDS datasets
- Real-world neuroimaging examples
- Performance benchmarks
- Gallery of brain network visualizations

### Phase 4: Performance Optimization (Issue #64)
**Timeline**: 4-6 weeks

**Deliverables**:
- [ ] Performance profiling and bottleneck identification
- [ ] Optimized matrix operations
- [ ] Hardware acceleration support (GPU/Metal)
- [ ] Modern ML framework integration (JAX/PyTorch)
- [ ] Scalability improvements

**Key Components**:
- Profiling and benchmarking framework
- JAX/NumPy array API compliance
- GPU acceleration via CuPy/PyTorch
- Numba JIT compilation
- Distributed computing support

### Phase 5: Advanced Research (Issue #65)
**Timeline**: 8-12 weeks

**Deliverables**:
- [ ] Literature review of modern factor analysis
- [ ] Advanced algorithm variants
- [ ] Neural network-based approaches
- [ ] Evaluation framework
- [ ] Research publications

**Key Components**:
- Sparse HTFA variants
- Variational inference approaches
- Attention mechanisms
- Graph neural networks
- Federated learning

## Technical Specifications

### Performance Requirements
- **Memory**: Efficient handling of datasets with >100k voxels
- **Speed**: 10x improvement over current baseline
- **Scalability**: Support for 100+ subjects
- **Hardware**: CPU optimization + optional GPU acceleration

### API Design Principles
- **Scikit-learn Compatibility**: Follow sklearn estimator interface
- **Consistent Naming**: Clear, descriptive method and parameter names
- **Flexible Input**: Support multiple data formats and structures
- **Comprehensive Output**: Rich fitted model objects with multiple views

### Code Quality Standards
- **Type Hints**: Full typing support for all APIs
- **Documentation**: Google-style docstrings with examples
- **Testing**: >90% test coverage with unit and integration tests
- **Linting**: Black formatting, isort imports, mypy type checking
- **Performance**: Regular benchmarking and performance regression testing

## Risk Assessment and Mitigation

### Technical Risks
1. **Algorithm Complexity**: HTFA optimization is mathematically complex
   - *Mitigation*: Start with working BrainIAK implementation as reference
   - *Mitigation*: Comprehensive unit testing of individual components

2. **Performance**: Optimization may be slow for large datasets  
   - *Mitigation*: Profile early and optimize bottlenecks incrementally
   - *Mitigation*: Implement progressive enhancement (basic → accelerated)

3. **Numerical Stability**: Matrix operations may be ill-conditioned
   - *Mitigation*: Use robust numerical algorithms from SciPy
   - *Mitigation*: Add regularization and stability checks

### Research Risks
1. **Algorithm Validity**: Implementation may not match theoretical performance
   - *Mitigation*: Validate against known ground truth datasets
   - *Mitigation*: Compare results with BrainIAK implementation

2. **Generalization**: Algorithm may not work on diverse neuroimaging data
   - *Mitigation*: Test on multiple real-world datasets
   - *Mitigation*: Collaborate with domain experts for validation

## Success Metrics

### Quantitative Metrics
- **Installation Time**: < 5 minutes on standard systems
- **Dependencies**: Core dependencies optimized for functionality
- **Performance**: 10x speedup over baseline
- **Test Coverage**: >90%
- **Documentation Coverage**: 100% API coverage
- **BIDS Compatibility**: Support for standard neuroimaging datasets
- **Visualization Quality**: Professional-grade brain network plots

### Qualitative Metrics
- **User Experience**: Single-line analysis of BIDS datasets
- **Intuitive Results**: Easy-to-understand HTFAResults objects
- **Research Impact**: Citations and academic usage
- **Code Quality**: High maintainability scores
- **Community**: Active contributor base
- **Neuroimaging Integration**: Seamless workflow with existing tools

## Future Directions

### Short-term (6 months)
- Complete core implementation and documentation
- Establish community adoption and feedback
- Optimize performance for common use cases
- Publish initial research findings

### Medium-term (1-2 years)
- Advanced algorithm variants and extensions
- Integration with major neuroimaging ecosystems
- GPU acceleration and distributed computing
- Commercial or clinical applications

### Long-term (2+ years)
- Next-generation algorithms using modern ML
- Real-time processing capabilities
- Multi-modal data integration
- Standardization in neuroimaging community

## Conclusion

This technical design provides a roadmap for creating a world-class HTFA toolbox that balances theoretical rigor with practical usability. The phased approach allows for iterative development, early feedback, and risk mitigation while maintaining focus on core objectives.

The success of this project will provide the neuroimaging community with an accessible, high-performance tool for hierarchical factor analysis, enabling new research and discoveries in brain network analysis.