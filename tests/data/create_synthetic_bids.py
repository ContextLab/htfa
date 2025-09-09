"""Create synthetic BIDS dataset for HTFA validation.

This script generates a complete BIDS-compliant dataset with known ground truth
parameters for validating HTFA algorithm accuracy.
"""

import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def create_bids_structure(base_path: Path) -> None:
    """Create the BIDS directory structure."""
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Create participant directories
    for i in range(1, 11):
        participant_dir = base_path / f"sub-{i:02d}" / "func"
        participant_dir.mkdir(parents=True, exist_ok=True)


def create_dataset_description(base_path: Path) -> None:
    """Create dataset_description.json."""
    description = {
        "Name": "Synthetic HTFA Validation Dataset",
        "BIDSVersion": "1.6.0",
        "DatasetType": "raw",
        "Authors": ["HTFA Development Team"],
        "Description": "Synthetic dataset with known ground truth for HTFA validation",
        "License": "CC0",
        "ReferencesAndLinks": ["https://github.com/jeremymanning/htfa"]
    }
    
    with open(base_path / "dataset_description.json", "w") as f:
        json.dump(description, f, indent=2)


def create_participants_tsv(base_path: Path) -> None:
    """Create participants.tsv file."""
    participants = pd.DataFrame({
        "participant_id": [f"sub-{i:02d}" for i in range(1, 11)],
        "age": np.random.randint(20, 60, 10),
        "sex": np.random.choice(["M", "F"], 10),
        "handedness": ["R"] * 10
    })
    
    participants.to_csv(base_path / "participants.tsv", sep="\t", index=False)


def generate_rbf_factors(
    spatial_coords: np.ndarray,
    factor_centers: np.ndarray,
    factor_widths: np.ndarray
) -> np.ndarray:
    """Generate RBF spatial factors.
    
    Parameters
    ----------
    spatial_coords : array-like, shape (n_voxels, 3)
        3D coordinates for each voxel
    factor_centers : array-like, shape (n_factors, 3)
        Centers of RBF factors
    factor_widths : array-like, shape (n_factors,)
        Width parameters for each RBF
        
    Returns
    -------
    factors : array-like, shape (n_voxels, n_factors)
        Spatial factor loadings
    """
    n_voxels = spatial_coords.shape[0]
    n_factors = factor_centers.shape[0]
    factors = np.zeros((n_voxels, n_factors))
    
    for k in range(n_factors):
        # Compute distances from each voxel to factor center
        distances = np.sqrt(np.sum((spatial_coords - factor_centers[k])**2, axis=1))
        # Apply RBF (Gaussian) kernel
        factors[:, k] = np.exp(-(distances**2) / (2 * factor_widths[k]**2))
    
    # Normalize factors
    factors = factors / np.linalg.norm(factors, axis=0, keepdims=True)
    
    return factors


def generate_temporal_patterns(n_timepoints: int, n_factors: int) -> np.ndarray:
    """Generate temporal activation patterns.
    
    Parameters
    ----------
    n_timepoints : int
        Number of time points
    n_factors : int
        Number of factors
        
    Returns
    -------
    weights : array-like, shape (n_factors, n_timepoints)
        Temporal weights for each factor
    """
    weights = np.zeros((n_factors, n_timepoints))
    
    # Create distinct temporal patterns for each factor
    t = np.linspace(0, 4*np.pi, n_timepoints)
    
    # Factor 1: Sinusoidal pattern
    weights[0, :] = np.sin(t) + 0.5 * np.sin(3*t)
    
    # Factor 2: Square wave-like pattern
    weights[1, :] = np.sign(np.sin(0.5*t)) + 0.3 * np.sin(2*t)
    
    # Factor 3: Ramp pattern
    ramp = np.mod(t, 2*np.pi) / (2*np.pi) - 0.5
    weights[2, :] = ramp + 0.2 * np.sin(4*t)
    
    # Factor 4: Complex oscillation
    weights[3, :] = np.cos(t) * np.exp(-t/20) + np.sin(2*t) * 0.3
    
    # Add some smoothing
    from scipy.ndimage import gaussian_filter1d
    for k in range(n_factors):
        weights[k, :] = gaussian_filter1d(weights[k, :], sigma=1.5)
    
    return weights


def create_nifti_data(
    participant_id: str,
    base_path: Path,
    factor_centers: np.ndarray,
    factor_widths: np.ndarray,
    n_voxels: int = 1000,
    n_timepoints: int = 100,
    noise_level: float = 0.1,
    seed: int = None
) -> dict:
    """Create synthetic NIfTI file for one participant.
    
    Parameters
    ----------
    participant_id : str
        Participant ID (e.g., "sub-01")
    base_path : Path
        Base directory for BIDS dataset
    factor_centers : array-like, shape (n_factors, 3)
        Centers of RBF factors
    factor_widths : array-like, shape (n_factors,)
        Width parameters for each RBF
    n_voxels : int
        Number of voxels
    n_timepoints : int
        Number of time points
    noise_level : float
        Amount of noise to add (proportion of signal)
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    params : dict
        Dictionary containing generated parameters
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate spatial coordinates (10x10x10 grid)
    grid_size = int(np.cbrt(n_voxels))
    x = np.linspace(0, 100, grid_size)
    y = np.linspace(0, 100, grid_size)
    z = np.linspace(0, 100, grid_size)
    
    # Create 3D grid
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    spatial_coords = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])[:n_voxels]
    
    # Generate spatial factors
    factors = generate_rbf_factors(spatial_coords, factor_centers, factor_widths)
    
    # Generate temporal patterns
    weights = generate_temporal_patterns(n_timepoints, len(factor_centers))
    
    # Add participant-specific variations
    participant_weights = weights.copy()
    participant_weights += np.random.randn(*weights.shape) * 0.1
    
    # Reconstruct data: Y = F @ W
    data = factors @ participant_weights
    
    # Add noise
    noise = np.random.randn(*data.shape) * noise_level * np.std(data)
    data += noise
    
    # Reshape to 4D (10x10x10xT)
    data_4d = np.zeros((grid_size, grid_size, grid_size, n_timepoints))
    data_4d.flat[:data.size] = data.T.ravel()
    
    # Create NIfTI image with proper affine matrix
    affine = np.eye(4)
    affine[:3, :3] = np.diag([10, 10, 10])  # 10mm voxel size
    affine[:3, 3] = [-50, -50, -50]  # Center the image
    
    img = nib.Nifti1Image(data_4d.astype(np.float32), affine)
    
    # Add metadata
    img.header['descrip'] = b'Synthetic HTFA validation data'
    img.header['pixdim'][4] = 2.0  # TR = 2s
    
    # Save NIfTI file
    output_path = base_path / participant_id / "func"
    output_file = output_path / f"{participant_id}_task-rest_bold.nii.gz"
    nib.save(img, output_file)
    
    # Create associated JSON sidecar
    json_metadata = {
        "TaskName": "rest",
        "RepetitionTime": 2.0,
        "EchoTime": 0.03,
        "FlipAngle": 90,
        "SliceTiming": [0.0] * grid_size,
        "PhaseEncodingDirection": "j-",
        "EffectiveEchoSpacing": 0.00055,
        "BandwidthPerPixelPhaseEncode": 20.0,
        "TotalReadoutTime": 0.055
    }
    
    json_file = output_path / f"{participant_id}_task-rest_bold.json"
    with open(json_file, "w") as f:
        json.dump(json_metadata, f, indent=2)
    
    return {
        "participant_id": participant_id,
        "spatial_coords": spatial_coords.tolist(),
        "factors": factors.tolist(),
        "weights": participant_weights.tolist(),
        "noise_level": noise_level,
        "n_voxels": n_voxels,
        "n_timepoints": n_timepoints,
        "affine": affine.tolist()
    }


def save_ground_truth(base_path: Path, all_params: list, factor_centers: np.ndarray, factor_widths: np.ndarray) -> None:
    """Save ground truth parameters to JSON."""
    ground_truth = {
        "dataset_info": {
            "n_participants": len(all_params),
            "n_voxels": all_params[0]["n_voxels"],
            "n_timepoints": all_params[0]["n_timepoints"],
            "n_factors": len(factor_centers),
            "noise_level": all_params[0]["noise_level"]
        },
        "factor_centers": factor_centers.tolist(),
        "factor_widths": factor_widths.tolist(),
        "participants": {}
    }
    
    for params in all_params:
        ground_truth["participants"][params["participant_id"]] = {
            "factors": params["factors"],
            "weights": params["weights"],
            "spatial_coords": params["spatial_coords"],
            "affine": params["affine"]
        }
    
    with open(base_path / "factor_params.json", "w") as f:
        json.dump(ground_truth, f, indent=2)


def main():
    """Create the complete synthetic BIDS dataset."""
    # Set paths
    base_path = Path("/Users/jmanning/htfa/tests/data/synthetic")
    
    print("Creating synthetic BIDS dataset...")
    
    # Create directory structure
    create_bids_structure(base_path)
    print("✓ Created BIDS directory structure")
    
    # Create metadata files
    create_dataset_description(base_path)
    create_participants_tsv(base_path)
    print("✓ Created BIDS metadata files")
    
    # Define ground truth factor parameters
    factor_centers = np.array([
        [25, 30, 15],  # Factor 1: Left frontal
        [75, 30, 15],  # Factor 2: Right frontal
        [50, 60, 15],  # Factor 3: Central anterior
        [50, 30, 45]   # Factor 4: Central posterior
    ])
    
    factor_widths = np.array([5.0, 5.0, 5.0, 5.0])
    
    # Generate data for each participant
    all_params = []
    for i in range(1, 11):
        participant_id = f"sub-{i:02d}"
        print(f"Generating data for {participant_id}...")
        
        params = create_nifti_data(
            participant_id=participant_id,
            base_path=base_path,
            factor_centers=factor_centers,
            factor_widths=factor_widths,
            n_voxels=1000,
            n_timepoints=100,
            noise_level=0.1,
            seed=42 + i  # Different seed for each participant
        )
        all_params.append(params)
    
    print("✓ Generated NIfTI data for all participants")
    
    # Save ground truth parameters
    save_ground_truth(base_path, all_params, factor_centers, factor_widths)
    print("✓ Saved ground truth parameters")
    
    # Create README
    readme_content = """# Synthetic HTFA Validation Dataset

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
"""
    
    with open(base_path / "README.md", "w") as f:
        f.write(readme_content)
    
    print("\n✅ Synthetic BIDS dataset created successfully!")
    print(f"Location: {base_path}")


if __name__ == "__main__":
    main()