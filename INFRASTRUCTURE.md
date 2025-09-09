# HTFA Ecosystem Infrastructure

## Overview

This document describes the unified CI/CD, testing, and monitoring infrastructure for the HTFA ecosystem, supporting coordinated development across four component epics.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Unified CI/CD Pipeline                    │
├─────────────────┬─────────────────┬─────────────────────────┤
│  Quality Gates  │     Testing     │    Monitoring          │
├─────────────────┼─────────────────┼─────────────────────────┤
│ • Code Quality  │ • Unit Tests    │ • Health Checks        │
│ • Security      │ • Integration   │ • Performance          │
│ • Type Checking │ • E2E Tests     │ • Resource Usage       │
│ • Dependencies  │ • Benchmarks    │ • Documentation        │
└─────────────────┴─────────────────┴─────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Component Epics                           │
├──────────────┬──────────────┬──────────────┬───────────────┤
│     Core     │ Scalability  │  Deployment  │  Integration  │
└──────────────┴──────────────┴──────────────┴───────────────┘
```

## Workflows

### 1. Unified CI/CD Pipeline (`unified-ci.yml`)

**Purpose**: Orchestrates builds, tests, and validation across all epic components.

**Triggers**:
- Push to main, develop, or epic-* branches
- Pull requests
- Manual workflow dispatch with epic selection

**Key Features**:
- Matrix builds across Python versions (3.10, 3.11, 3.12)
- Epic-specific test execution
- Performance benchmarking
- Cross-epic integration validation
- Documentation generation

**Jobs**:
1. **Setup**: Prepares build matrix and cache keys
2. **Quality Gates**: Code formatting, linting, security scanning
3. **Testing**: Unit and integration tests with coverage
4. **Performance Benchmarks**: Automated performance regression detection
5. **Integration Validation**: Cross-epic compatibility checks
6. **Documentation**: Sphinx documentation build
7. **Deploy Preview**: PR preview environments
8. **Publish Metrics**: Unified reporting

### 2. Quality Gates (`quality-gates.yml`)

**Purpose**: Enforces code quality standards across the ecosystem.

**Checks**:
- **Coverage**: Minimum 90% code coverage
- **Complexity**: Maximum cyclomatic complexity of 10
- **Duplication**: Maximum 5% code duplication
- **Security**: Bandit security scanning
- **Type Safety**: MyPy type checking
- **Dependencies**: Vulnerability and outdated dependency checks

**Outputs**:
- Quality report as PR comment
- Detailed artifacts for each check
- Summary dashboard

### 3. Monitoring (`monitoring.yml`)

**Purpose**: Continuous ecosystem health monitoring and resource tracking.

**Schedule**: Every 6 hours + on-demand

**Components**:
- **Health Check**: Cross-epic validation and component status
- **Resource Monitoring**: GitHub Actions usage and cost estimation
- **Documentation Build**: Automated documentation deployment
- **Performance Tracking**: Historical benchmark data

## Testing Framework

### Test Categories

```python
@pytest.mark.unit          # Component-level tests
@pytest.mark.integration   # Cross-component tests
@pytest.mark.e2e          # End-to-end workflows
@pytest.mark.benchmark    # Performance tests
@pytest.mark.epic_core    # Core functionality
@pytest.mark.epic_scalability  # Scalability features
@pytest.mark.epic_deployment   # Deployment infrastructure
@pytest.mark.epic_integration  # Ecosystem integration
```

### Shared Fixtures

Located in `tests/conftest.py`:

- `test_data_dir`: Test data directory path
- `temp_dir`: Isolated temporary directory
- `sample_neuroimaging_data`: Generated fMRI data
- `mock_bids_dataset`: Mock BIDS dataset structure
- `performance_monitor`: Performance metric tracking
- `capture_metrics`: Resource usage monitoring

### Cross-Epic Validation

The `htfa.validation.cross_epic_check` module provides:

1. **Component Availability**: Verifies all epic components are importable
2. **Interface Compatibility**: Checks required interfaces between epics
3. **Integration Testing**: Runs basic cross-epic integration tests
4. **Report Generation**: Comprehensive validation reports

## Quality Standards

### Coverage Requirements

- Overall: ≥90%
- Unit tests: ≥95%
- Integration tests: ≥85%
- E2E tests: ≥80%

### Performance Benchmarks

| Metric | Threshold | Measurement |
|--------|-----------|-------------|
| TFA fit time | <5 seconds | Single subject, 1000 voxels |
| HTFA fit time | <30 seconds | 5 subjects, 1000 voxels |
| Memory usage | <1 GB | Peak RSS during fit |
| CPU efficiency | >80% | Core utilization |

### Code Quality Metrics

| Metric | Standard | Tool |
|--------|----------|------|
| Formatting | Black, 88 chars | `black` |
| Import sorting | Project style | `isort` |
| Linting | No errors | `ruff` |
| Type checking | No errors | `mypy` |
| Security | No high severity | `bandit` |
| Complexity | CC ≤ 10 | `radon` |

## Resource Management

### Compute Resources

- **CI/CD**: GitHub-hosted runners (ubuntu-latest)
- **HPC Integration**: Optional self-hosted runners for large-scale tests
- **GPU Testing**: Optional GPU runners for backend validation

### Storage

- **Artifacts**: 90-day retention for test results and reports
- **Cache**: Poetry dependencies, up to 10 GB
- **Documentation**: GitHub Pages deployment

### Monitoring

- **Health Checks**: Every 6 hours
- **Performance**: Continuous benchmark tracking
- **Alerts**: Automatic issue creation on failures
- **Dashboards**: HTML reports in artifacts

## Usage

### Running Tests Locally

```bash
# Install dependencies
make install

# Run all tests
make test

# Run specific epic tests
poetry run pytest -m epic_core

# Run with coverage
poetry run pytest --cov=htfa --cov-report=html

# Run benchmarks
poetry run pytest tests/benchmarks/ --benchmark-only
```

### Triggering CI/CD

```bash
# Manual workflow dispatch
gh workflow run unified-ci.yml -f epic=core

# Create PR to trigger quality gates
git checkout -b feature/my-feature
git push origin feature/my-feature
gh pr create
```

### Monitoring

```bash
# View latest health check
gh run list --workflow=monitoring.yml --limit=1
gh run view <run-id>

# Download artifacts
gh run download <run-id>
```

## Deployment

### Preview Environments

Pull requests automatically deploy to:
```
https://pr-<number>.htfa-preview.dev
```

### Production Deployment

Merges to main trigger:
1. Full test suite execution
2. Documentation build and deployment
3. Performance benchmark updates
4. Health dashboard refresh

## Troubleshooting

### Common Issues

1. **Coverage Below Threshold**
   - Run `make test` locally to identify gaps
   - Add tests for uncovered code paths
   - Check for unreachable code

2. **Type Errors**
   - Run `poetry run mypy htfa/`
   - Add type annotations where missing
   - Update type stubs if needed

3. **Security Vulnerabilities**
   - Run `poetry run safety check`
   - Update vulnerable dependencies
   - Review bandit findings

4. **Cross-Epic Failures**
   - Run `poetry run python -m htfa.validation.cross_epic_check`
   - Check import paths and module structure
   - Verify interface compatibility

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## Support

For infrastructure issues:
- Create an issue with label `infrastructure`
- Check workflow run logs
- Review monitoring dashboard artifacts