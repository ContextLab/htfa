# Synthetic HTFA Validation Dataset

This is a synthetic BIDS dataset created for validating the HTFA algorithm.

## Dataset Properties
- 10 participants (sub-01 to sub-10)
- 1000 voxels per participant (10x10x10 grid)
- 100 timepoints
- 4 RBF spatial factors with known centers and widths
- TR = 2.0 seconds
- Voxel size = 10x10x10 mm

## Ground Truth
See `factor_params.json` for complete ground truth parameters including:
- Factor centers and widths
- Per-participant factor loadings
- Temporal weight patterns
- Spatial coordinates

## Usage
This dataset can be loaded with pybids or directly with the HTFA package.
