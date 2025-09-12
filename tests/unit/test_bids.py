"""Tests for htfa.bids module without mocks."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from htfa.bids import (
    build_bids_query,
    extract_bids_metadata,
    fit_bids,
    load_bids_data,
    load_bids_for_htfa,
    parse_bids_dataset,
    validate_bids_structure,
)


class TestValidateBidsStructure:
    """Test validate_bids_structure function."""

    def test_validate_bids_valid_directory(self):
        """Test validation of valid BIDS directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            # Create minimal BIDS structure
            (path / "dataset_description.json").write_text(
                '{"Name": "Test", "BIDSVersion": "1.6.0"}'
            )
            (path / "participants.tsv").write_text(
                "participant_id\tsex\tage\nsub-01\tM\t25"
            )
            (path / "sub-01").mkdir()
            (path / "sub-01" / "func").mkdir(parents=True)

            result = validate_bids_structure(path)

            assert result["valid"] is True
            assert result["errors"] == []
            # We created participants.tsv, so no warning about it
            assert result["summary"]["n_subjects"] == 1

    def test_validate_bids_missing_description(self):
        """Test validation with missing dataset_description.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / "participants.tsv").touch()
            (path / "sub-01").mkdir()

            result = validate_bids_structure(path)

            assert result["valid"] is False
            assert any("dataset_description.json" in err for err in result["errors"])
            assert "Missing participants.tsv" in str(result["warnings"])

    def test_validate_bids_no_subjects(self):
        """Test validation with no subject directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / "dataset_description.json").write_text('{"Name": "Test"}')
            (path / "participants.tsv").touch()

            result = validate_bids_structure(path)

            # The validation should pass but show 0 subjects
            assert result["valid"] is True  # No subjects is not an error
            assert result["summary"]["n_subjects"] == 0


class TestBuildBidsQuery:
    """Test build_bids_query function."""

    def test_build_query_empty(self):
        """Test building empty query."""
        query = build_bids_query()
        assert query == {}

    def test_build_query_with_filters(self):
        """Test building query with filters."""
        query = build_bids_query(subject="01", task="rest", session="1", run=2)

        assert query["subject"] == "01"
        assert query["task"] == "rest"
        assert query["session"] == "1"
        assert query["run"] == 2

    def test_build_query_with_lists(self):
        """Test building query with list values."""
        query = build_bids_query(subject=["01", "02"], task=["rest", "task"])

        assert query["subject"] == ["01", "02"]
        assert query["task"] == ["rest", "task"]

    def test_build_query_filters_none(self):
        """Test that None values are filtered out."""
        query = build_bids_query(subject="01", task=None, session="1")

        assert "subject" in query
        assert "task" not in query
        assert "session" in query


class TestParseBidsDataset:
    """Test parse_bids_dataset function with real BIDS structures."""

    def create_minimal_bids_dataset(self, path):
        """Helper to create a minimal but valid BIDS dataset."""
        # Create dataset_description.json
        desc = {
            "Name": "TestDataset",
            "BIDSVersion": "1.6.0",
            "License": "CC0",
            "Authors": ["Test Author"],
        }
        (path / "dataset_description.json").write_text(json.dumps(desc, indent=2))

        # Create participants.tsv
        (path / "participants.tsv").write_text(
            "participant_id\tsex\tage\n" "sub-01\tM\t25\n" "sub-02\tF\t30\n"
        )

        # Create subject directories with func data
        for sub in ["sub-01", "sub-02"]:
            sub_dir = path / sub / "func"
            sub_dir.mkdir(parents=True)

            # Create a simple BOLD JSON sidecar
            bold_json = {
                "TaskName": "rest",
                "RepetitionTime": 2.0,
                "EchoTime": 0.03,
                "FlipAngle": 90,
                "SliceTiming": [0.0, 0.5, 1.0, 1.5],
            }
            json_file = sub_dir / f"{sub}_task-rest_bold.json"
            json_file.write_text(json.dumps(bold_json, indent=2))

            # Create a dummy NIfTI file (just touch it for now)
            nii_file = sub_dir / f"{sub}_task-rest_bold.nii.gz"
            nii_file.touch()

    def test_parse_bids_basic(self):
        """Test basic BIDS dataset parsing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            self.create_minimal_bids_dataset(path)

            # Parse the dataset
            layout = parse_bids_dataset(path)

            # Check basic properties
            assert layout is not None
            subjects = layout.get_subjects()
            assert len(subjects) == 2
            assert "01" in subjects
            assert "02" in subjects

    def test_parse_bids_with_filters(self):
        """Test BIDS parsing with filters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            self.create_minimal_bids_dataset(path)

            # Parse with derivatives disabled and validation enabled
            layout = parse_bids_dataset(path, derivatives=False, validate=False)

            assert layout is not None
            # Check that we can query the layout
            bold_files = layout.get(suffix="bold", extension="nii.gz")
            assert len(bold_files) == 2


class TestExtractBidsMetadata:
    """Test extract_bids_metadata function."""

    def test_extract_metadata_basic(self):
        """Test basic metadata extraction from real BIDS layout."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            # Create BIDS dataset
            desc = {"Name": "TestDataset", "BIDSVersion": "1.6.0"}
            (path / "dataset_description.json").write_text(json.dumps(desc))
            (path / "participants.tsv").write_text("participant_id\nsub-01\nsub-02")

            for sub in ["sub-01", "sub-02"]:
                (path / sub / "func").mkdir(parents=True)
                (path / sub / "func" / f"{sub}_task-rest_bold.nii.gz").touch()

            # Parse and extract metadata
            layout = parse_bids_dataset(path)
            metadata = extract_bids_metadata(layout)

            assert metadata["n_subjects"] == 2
            assert metadata["dataset_name"] == "TestDataset"
            assert "subjects" in metadata
            assert len(metadata["subjects"]) == 2
