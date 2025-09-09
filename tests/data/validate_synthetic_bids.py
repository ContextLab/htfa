"""Validate the synthetic BIDS dataset."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
from bids import BIDSLayout


def validate_bids_structure():
    """Validate BIDS dataset structure and contents."""
    base_path = Path("/Users/jmanning/htfa/tests/data/synthetic")

    print("Validating synthetic BIDS dataset...")
    print("=" * 60)

    # Load dataset with pybids
    try:
        layout = BIDSLayout(base_path, validate=False)
        print("✓ Dataset loaded successfully with pybids")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return False

    # Check participants
    participants = layout.get_subjects()
    expected_participants = [f"{i:02d}" for i in range(1, 11)]

    if sorted(participants) == sorted(expected_participants):
        print(
            f"✓ Found all {len(participants)} participants: {', '.join(sorted(participants))}"
        )
    else:
        print(
            f"✗ Participant mismatch. Expected: {expected_participants}, Found: {participants}"
        )
        return False

    # Check functional files for each participant
    print("\nChecking functional data:")
    for sub in participants:
        func_files = layout.get(subject=sub, suffix="bold", extension="nii.gz")
        if len(func_files) == 1:
            func_file = func_files[0]

            # Load and check NIfTI dimensions
            img = nib.load(func_file.path)
            data = img.get_fdata()

            if data.shape == (10, 10, 10, 100):
                print(f"  ✓ sub-{sub}: Shape {data.shape} (correct)")
            else:
                print(f"  ✗ sub-{sub}: Shape {data.shape} (expected (10, 10, 10, 100))")
                return False

            # Check for JSON sidecar
            json_files = layout.get(subject=sub, suffix="bold", extension="json")
            if len(json_files) == 1:
                with open(json_files[0].path) as f:
                    metadata = json.load(f)
                    if metadata.get("RepetitionTime") == 2.0:
                        print(f"    ✓ JSON sidecar present with TR=2.0s")
                    else:
                        print(
                            f"    ✗ Incorrect TR in JSON: {metadata.get('RepetitionTime')}"
                        )
            else:
                print(f"    ✗ Missing JSON sidecar")
                return False
        else:
            print(
                f"  ✗ sub-{sub}: Found {len(func_files)} functional files (expected 1)"
            )
            return False

    # Check ground truth file
    print("\nChecking ground truth parameters:")
    ground_truth_path = base_path / "factor_params.json"
    if ground_truth_path.exists():
        with open(ground_truth_path) as f:
            ground_truth = json.load(f)

        info = ground_truth["dataset_info"]
        print(f"  ✓ Ground truth file exists")
        print(f"    - Participants: {info['n_participants']}")
        print(f"    - Voxels: {info['n_voxels']}")
        print(f"    - Timepoints: {info['n_timepoints']}")
        print(f"    - Factors: {info['n_factors']}")
        print(f"    - Noise level: {info['noise_level']}")

        # Verify factor centers
        centers = np.array(ground_truth["factor_centers"])
        expected_centers = np.array(
            [[25, 30, 15], [75, 30, 15], [50, 60, 15], [50, 30, 45]]
        )

        if np.allclose(centers, expected_centers):
            print("  ✓ Factor centers match expected values")
        else:
            print("  ✗ Factor centers don't match expected values")
            return False

        # Check that each participant has data
        for sub in participants:
            sub_key = f"sub-{sub}"
            if sub_key in ground_truth["participants"]:
                participant_data = ground_truth["participants"][sub_key]
                if all(
                    k in participant_data
                    for k in ["factors", "weights", "spatial_coords", "affine"]
                ):
                    factors = np.array(participant_data["factors"])
                    weights = np.array(participant_data["weights"])
                    print(
                        f"  ✓ sub-{sub}: Factors shape {factors.shape}, Weights shape {weights.shape}"
                    )
                else:
                    print(f"  ✗ sub-{sub}: Missing ground truth data")
                    return False
            else:
                print(f"  ✗ sub-{sub}: Not in ground truth")
                return False
    else:
        print("  ✗ Ground truth file not found")
        return False

    # Check dataset metadata files
    print("\nChecking BIDS metadata files:")

    # dataset_description.json
    desc_path = base_path / "dataset_description.json"
    if desc_path.exists():
        with open(desc_path) as f:
            desc = json.load(f)
            if desc.get("BIDSVersion") and desc.get("Name"):
                print(
                    f"  ✓ dataset_description.json present (BIDS v{desc['BIDSVersion']})"
                )
            else:
                print("  ✗ dataset_description.json incomplete")
                return False
    else:
        print("  ✗ dataset_description.json missing")
        return False

    # participants.tsv
    participants_path = base_path / "participants.tsv"
    if participants_path.exists():
        import pandas as pd

        participants_df = pd.read_csv(participants_path, sep="\t")
        if len(participants_df) == 10:
            print(f"  ✓ participants.tsv present ({len(participants_df)} participants)")
        else:
            print(
                f"  ✗ participants.tsv has wrong number of participants: {len(participants_df)}"
            )
            return False
    else:
        print("  ✗ participants.tsv missing")
        return False

    print("\n" + "=" * 60)
    print("✅ All validation checks passed!")
    return True


if __name__ == "__main__":
    success = validate_bids_structure()
    if not success:
        print("\n❌ Validation failed!")
        exit(1)
