"""Shared pytest configuration and fixtures for HTFA ecosystem testing."""

from typing import Any, Dict, Generator, List

import os
import sys
import tempfile
from pathlib import Path

import time

import numpy as np
import pytest
from _pytest.config import Config
from _pytest.nodes import Item

sys.path.insert(0, str(Path(__file__).parent.parent))


def pytest_configure(config: Config) -> None:
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line(
        "markers", "integration: Integration tests for component interactions"
    )
    config.addinivalue_line("markers", "e2e: End-to-end tests for complete workflows")
    config.addinivalue_line("markers", "benchmark: Performance benchmark tests")
    config.addinivalue_line("markers", "slow: Tests that take >5 seconds to run")
    config.addinivalue_line("markers", "epic_core: Tests for core HTFA functionality")
    config.addinivalue_line(
        "markers", "epic_scalability: Tests for scalability features"
    )
    config.addinivalue_line(
        "markers", "epic_deployment: Tests for deployment infrastructure"
    )
    config.addinivalue_line(
        "markers", "epic_integration: Tests for ecosystem integration"
    )
    config.addinivalue_line("markers", "gpu: Tests requiring GPU resources")
    config.addinivalue_line("markers", "hpc: Tests requiring HPC resources")


def pytest_collection_modifyitems(config: Config, items: List[Item]) -> None:
    """Modify test collection to add automatic markers."""
    for item in items:
        # Add markers based on test location
        test_path = Path(item.fspath)

        if "unit" in test_path.parts:
            item.add_marker(pytest.mark.unit)
        elif "integration" in test_path.parts:
            item.add_marker(pytest.mark.integration)
        elif "e2e" in test_path.parts:
            item.add_marker(pytest.mark.e2e)
        elif "benchmarks" in test_path.parts:
            item.add_marker(pytest.mark.benchmark)

        # Add epic markers based on test module
        module_name = item.module.__name__ if hasattr(item, "module") else ""
        if "core" in module_name:
            item.add_marker(pytest.mark.epic_core)
        elif "scalability" in module_name or "scale" in module_name:
            item.add_marker(pytest.mark.epic_scalability)
        elif "deployment" in module_name or "deploy" in module_name:
            item.add_marker(pytest.mark.epic_deployment)
        elif "integration" in module_name or "integrate" in module_name:
            item.add_marker(pytest.mark.epic_integration)


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Provide path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test isolation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="session")
def sample_neuroimaging_data() -> Dict[str, np.ndarray]:
    """Generate sample neuroimaging data for testing."""
    np.random.seed(42)
    n_subjects = 5
    n_timepoints = 100
    n_voxels = 1000

    return {
        f"subject_{i}": np.random.randn(n_timepoints, n_voxels)
        for i in range(n_subjects)
    }


@pytest.fixture(scope="session")
def sample_coordinates() -> np.ndarray:
    """Generate sample voxel coordinates."""
    np.random.seed(42)
    n_voxels = 1000
    return np.random.randn(n_voxels, 3) * 50


@pytest.fixture(scope="function")
def mock_bids_dataset(temp_dir: Path) -> Path:
    """Create a mock BIDS dataset structure."""
    bids_dir = temp_dir / "bids_dataset"
    bids_dir.mkdir()

    # Create dataset_description.json
    dataset_desc = bids_dir / "dataset_description.json"
    dataset_desc.write_text(
        """{
        "Name": "Test Dataset",
        "BIDSVersion": "1.8.0"
    }"""
    )

    # Create participants.tsv
    participants = bids_dir / "participants.tsv"
    participants.write_text("participant_id\tsex\tage\n")

    # Create subject directories
    for i in range(3):
        sub_dir = bids_dir / f"sub-{i:02d}" / "func"
        sub_dir.mkdir(parents=True)

        # Create dummy functional files
        bold_file = sub_dir / f"sub-{i:02d}_task-rest_bold.nii.gz"
        bold_file.touch()

        # Create JSON sidecar
        json_file = sub_dir / f"sub-{i:02d}_task-rest_bold.json"
        json_file.write_text(
            """{
            "TaskName": "rest",
            "RepetitionTime": 2.0
        }"""
        )

    return bids_dir


@pytest.fixture(scope="session")
def performance_thresholds() -> Dict[str, float]:
    """Define performance thresholds for benchmarks."""
    return {
        "tfa_fit_time": 5.0,  # seconds
        "htfa_fit_time": 30.0,  # seconds
        "memory_usage": 1024,  # MB
        "cpu_usage": 80,  # percent
    }


@pytest.fixture(scope="function")
def capture_metrics() -> Generator[Dict[str, Any], None, None]:
    """Capture performance metrics during test execution."""
    import time

    import psutil

    metrics = {
        "start_time": time.time(),
        "start_memory": psutil.Process().memory_info().rss / 1024 / 1024,
        "start_cpu": psutil.cpu_percent(interval=0.1),
    }

    yield metrics

    metrics.update(
        {
            "end_time": time.time(),
            "end_memory": psutil.Process().memory_info().rss / 1024 / 1024,
            "end_cpu": psutil.cpu_percent(interval=0.1),
            "duration": time.time() - metrics["start_time"],
            "memory_delta": (psutil.Process().memory_info().rss / 1024 / 1024)
            - metrics["start_memory"],
        }
    )


@pytest.fixture(scope="session")
def epic_components() -> Dict[str, List[str]]:
    """Map epic names to their component modules."""
    return {
        "core": ["htfa.core", "htfa.tfa", "htfa.htfa"],
        "scalability": ["htfa.optimization", "htfa.backends"],
        "deployment": ["htfa.cloud", "htfa.containers"],
        "integration": ["htfa.bids", "htfa.validation_infrastructure", "htfa.results"],
    }


@pytest.fixture(autouse=True)
def reset_random_state():
    """Reset random state before each test for reproducibility."""
    np.random.seed(42)
    import random

    random.seed(42)

    yield

    # No cleanup needed


@pytest.fixture(scope="session")
def ci_environment() -> Dict[str, str]:
    """Detect CI environment and provide relevant information."""
    env_info = {
        "is_ci": os.getenv("CI", "false").lower() == "true",
        "ci_provider": os.getenv("GITHUB_ACTIONS", ""),
        "branch": os.getenv("GITHUB_REF_NAME", "local"),
        "commit": os.getenv("GITHUB_SHA", "local"),
        "pr_number": os.getenv("GITHUB_PR_NUMBER", ""),
    }
    return env_info


@pytest.fixture(scope="function")
def mock_hpc_resources() -> Dict[str, Any]:
    """Mock HPC resource availability for testing."""
    return {
        "nodes": 4,
        "cpus_per_node": 32,
        "memory_per_node": 128,  # GB
        "gpus_per_node": 2,
        "queue": "test",
        "walltime": "01:00:00",
    }


class PerformanceMonitor:
    """Monitor and track performance during tests."""

    def __init__(self):
        self.metrics = []

    def record(self, name: str, value: float, unit: str = "ms"):
        """Record a performance metric."""
        self.metrics.append(
            {
                "name": name,
                "value": value,
                "unit": unit,
                "timestamp": time.time(),
            }
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of recorded metrics."""
        if not self.metrics:
            return {}

        summary = {}
        for metric in self.metrics:
            name = metric["name"]
            if name not in summary:
                summary[name] = []
            summary[name].append(metric["value"])

        return {
            name: {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
            }
            for name, values in summary.items()
        }


@pytest.fixture(scope="function")
def performance_monitor() -> PerformanceMonitor:
    """Provide performance monitoring for tests."""
    return PerformanceMonitor()


@pytest.fixture(scope="session")
def quality_thresholds() -> Dict[str, float]:
    """Define quality thresholds for code metrics."""
    return {
        "coverage": 90.0,  # percent
        "complexity": 10,  # cyclomatic complexity
        "maintainability": 20,  # maintainability index
        "duplication": 5.0,  # percent
    }
