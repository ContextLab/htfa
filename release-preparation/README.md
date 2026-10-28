# Publication is blocked pending review

Jeremy must carefully review and explicitly approve publication. Nothing in this
folder is an active workflow. Do not move the example into `.github/workflows`
or configure/dispatch publication until that approval is given.

The active candidate workflow builds/tests wheels and executes notebooks only.
The example publication workflow consumes the exact Ubuntu/Python 3.12 artifacts
from a successful candidate run, checks the approved full commit SHA and source
repository/workflow, and does not rebuild packages after their validation.

After approval:

1. Configure GitHub environments `testpypi` and `pypi` with required review by
   Jeremy and no self-approval. Configure matching Trusted Publishers on both
   indexes for ContextLab/htfa, workflow `publish.yml`, respective environment.
2. Activate the reviewed workflow. Obtain separate approval for the TestPyPI
   step, dispatch against the approved SHA/run, install from TestPyPI in a fresh
   environment and run the package/tutorial smoke checks.
3. Only after explicit PyPI approval, publish the same validated artifacts. Record
   their SHA-256s and confirm the index serves those files. Tag the approved source,
   finalize the GitHub release notes, and verify README links and RTD stable.

Configure index identity carefully: project name is `htfa`, version `0.1.0`.
GitHub environment protection and index trusted-publisher setup are NOT complete
merely because this example exists. No TestPyPI or PyPI upload has been performed.

References: [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/using-a-publisher/),
[PyPA publication guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/).
