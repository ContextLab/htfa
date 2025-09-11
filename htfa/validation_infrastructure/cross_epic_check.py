"""Cross-epic compatibility validation for HTFA ecosystem."""

from typing import Dict, List, Optional, Tuple

import importlib
import sys
from dataclasses import dataclass

import numpy as np


@dataclass
class ComponentStatus:
    """Status of a component in the ecosystem."""

    name: str
    epic: str
    available: bool
    version: Optional[str] = None
    dependencies: List[str] = None
    issues: List[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.issues is None:
            self.issues = []


class CrossEpicValidator:
    """Validate cross-epic compatibility and integration."""

    EPIC_COMPONENTS = {
        "core": [
            "htfa.core.tfa",
            "htfa.core.htfa",
            "htfa.core.base",
        ],
        "scalability": [
            "htfa.optimization",
            "htfa.backends.jax_backend",
            "htfa.backends.pytorch_backend",
        ],
        "deployment": [
            "htfa.cloud",
            "htfa.containers",
            "htfa.orchestration",
        ],
        "integration": [
            "htfa.bids",
            "htfa.validation_infrastructure",
            "htfa.results",
            "htfa.fit",
        ],
    }

    COMPATIBILITY_MATRIX = {
        ("core", "scalability"): [
            "htfa.optimization.OptimizedHTFA",
            "htfa.backends.Backend",
        ],
        ("core", "integration"): [
            "htfa.fit.fit_bids",
            "htfa.results.HTFAResults",
        ],
        ("scalability", "deployment"): [
            "htfa.cloud.CloudHTFA",
            "htfa.containers.DockerizedHTFA",
        ],
        ("integration", "deployment"): [
            "htfa.orchestration.Pipeline",
        ],
    }

    def __init__(self):
        self.component_status: Dict[str, ComponentStatus] = {}
        self.compatibility_issues: List[str] = []

    def check_component(self, module_path: str, epic: str) -> ComponentStatus:
        """Check if a component is available and functional."""
        try:
            module = importlib.import_module(module_path)

            # Get version if available
            version = getattr(module, "__version__", None)

            # Check for required attributes
            required_attrs = self._get_required_attributes(module_path)
            missing_attrs = [
                attr for attr in required_attrs if not hasattr(module, attr)
            ]

            issues = []
            if missing_attrs:
                issues.append(f"Missing attributes: {', '.join(missing_attrs)}")

            return ComponentStatus(
                name=module_path,
                epic=epic,
                available=True,
                version=version,
                issues=issues,
            )

        except ImportError as e:
            return ComponentStatus(
                name=module_path,
                epic=epic,
                available=False,
                issues=[f"Import error: {str(e)}"],
            )
        except Exception as e:
            return ComponentStatus(
                name=module_path,
                epic=epic,
                available=False,
                issues=[f"Unexpected error: {str(e)}"],
            )

    def _get_required_attributes(self, module_path: str) -> List[str]:
        """Get required attributes for a module."""
        requirements = {
            "htfa.core.tfa": ["TFA"],
            "htfa.core.htfa": ["HTFA"],
            "htfa.optimization": ["OptimizedHTFA", "AdamOptimizer"],
            "htfa.bids": ["load_bids_data"],
            "htfa.fit": ["fit_bids"],
            "htfa.results": ["HTFAResults"],
        }
        return requirements.get(module_path, [])

    def validate_all_components(self) -> Dict[str, List[ComponentStatus]]:
        """Validate all components across epics."""
        results = {}

        for epic, components in self.EPIC_COMPONENTS.items():
            epic_results = []
            for component in components:
                status = self.check_component(component, epic)
                self.component_status[component] = status
                epic_results.append(status)
            results[epic] = epic_results

        return results

    def check_cross_epic_compatibility(self) -> List[Tuple[str, str, List[str]]]:
        """Check compatibility between epic components."""
        compatibility_results = []

        for (epic1, epic2), required_interfaces in self.COMPATIBILITY_MATRIX.items():
            issues = []

            # Check if both epics have available components
            epic1_available = any(
                self.component_status.get(c, ComponentStatus(c, epic1, False)).available
                for c in self.EPIC_COMPONENTS.get(epic1, [])
            )
            epic2_available = any(
                self.component_status.get(c, ComponentStatus(c, epic2, False)).available
                for c in self.EPIC_COMPONENTS.get(epic2, [])
            )

            if not epic1_available:
                issues.append(f"Epic {epic1} has no available components")
            if not epic2_available:
                issues.append(f"Epic {epic2} has no available components")

            # Check required interfaces
            for interface in required_interfaces:
                try:
                    module_path, class_name = interface.rsplit(".", 1)
                    module = importlib.import_module(module_path)
                    if not hasattr(module, class_name):
                        issues.append(f"Missing interface: {interface}")
                except ImportError:
                    issues.append(f"Cannot import interface: {interface}")
                except Exception as e:
                    issues.append(f"Error checking interface {interface}: {str(e)}")

            compatibility_results.append((epic1, epic2, issues))

        return compatibility_results

    def run_integration_tests(self) -> Dict[str, bool]:
        """Run basic integration tests between components."""
        test_results = {}

        # Test 1: Core TFA functionality
        try:
            from htfa.core.tfa import TFA

            tfa = TFA(n_components=10)
            X = np.random.randn(100, 1000)
            coords = np.random.randn(1000, 3)
            tfa.fit(X, coords)
            test_results["core_tfa_basic"] = True
        except Exception as e:
            test_results["core_tfa_basic"] = False
            self.compatibility_issues.append(f"Core TFA test failed: {str(e)}")

        # Test 2: Core-Integration interface
        try:
            from htfa.results import HTFAResults

            HTFAResults(
                factors=np.random.randn(10, 1000),
                weights=np.random.randn(100, 10),
                coordinates=np.random.randn(1000, 3),
            )
            test_results["core_integration_interface"] = True
        except Exception as e:
            test_results["core_integration_interface"] = False
            self.compatibility_issues.append(f"Core-Integration test failed: {str(e)}")

        # Test 3: Optimization backend
        try:
            from htfa.optimization import AdamOptimizer

            optimizer = AdamOptimizer(learning_rate=0.001)
            params = np.random.randn(100)
            grad = np.random.randn(100)
            optimizer.update(params, grad)
            test_results["optimization_backend"] = True
        except Exception as e:
            test_results["optimization_backend"] = False
            self.compatibility_issues.append(f"Optimization test failed: {str(e)}")

        return test_results

    def generate_report(self) -> str:
        """Generate a comprehensive validation report."""
        report_lines = [
            "=" * 80,
            "HTFA Ecosystem Cross-Epic Validation Report",
            "=" * 80,
            "",
        ]

        # Component availability
        report_lines.append("## Component Availability")
        report_lines.append("-" * 40)

        epic_results = self.validate_all_components()
        for epic, components in epic_results.items():
            available = sum(1 for c in components if c.available)
            total = len(components)
            report_lines.append(f"\n### Epic: {epic.upper()}")
            report_lines.append(f"Available: {available}/{total}")

            for component in components:
                status = "✓" if component.available else "✗"
                report_lines.append(f"  {status} {component.name}")
                if component.issues:
                    for issue in component.issues:
                        report_lines.append(f"    - {issue}")

        # Cross-epic compatibility
        report_lines.append("\n## Cross-Epic Compatibility")
        report_lines.append("-" * 40)

        compatibility = self.check_cross_epic_compatibility()
        for epic1, epic2, issues in compatibility:
            status = "✓" if not issues else "✗"
            report_lines.append(f"\n{status} {epic1} <-> {epic2}")
            for issue in issues:
                report_lines.append(f"  - {issue}")

        # Integration tests
        report_lines.append("\n## Integration Tests")
        report_lines.append("-" * 40)

        test_results = self.run_integration_tests()
        for test_name, passed in test_results.items():
            status = "✓" if passed else "✗"
            report_lines.append(f"{status} {test_name}")

        # Summary
        report_lines.append("\n## Summary")
        report_lines.append("-" * 40)

        total_components = sum(len(c) for c in epic_results.values())
        available_components = sum(
            1
            for epic_components in epic_results.values()
            for c in epic_components
            if c.available
        )

        compatibility_pass = sum(1 for _, _, issues in compatibility if not issues)
        compatibility_total = len(compatibility)

        tests_pass = sum(1 for passed in test_results.values() if passed)
        tests_total = len(test_results)

        report_lines.append(
            f"Components: {available_components}/{total_components} available"
        )
        report_lines.append(
            f"Compatibility: {compatibility_pass}/{compatibility_total} passing"
        )
        report_lines.append(f"Integration Tests: {tests_pass}/{tests_total} passing")

        if self.compatibility_issues:
            report_lines.append("\n## Issues Found")
            report_lines.append("-" * 40)
            for issue in self.compatibility_issues:
                report_lines.append(f"- {issue}")

        report_lines.append("\n" + "=" * 80)

        return "\n".join(report_lines)


def main():
    """Run cross-epic validation as a standalone script."""
    validator = CrossEpicValidator()
    report = validator.generate_report()
    print(report)

    # Exit with error code if issues found
    if validator.compatibility_issues:
        sys.exit(1)

    # Check if all tests passed
    test_results = validator.run_integration_tests()
    if not all(test_results.values()):
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
