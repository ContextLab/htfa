"""Run non-network tests and package doctests away from the source checkout.

The calling environment must already have installed the candidate wheel.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    source = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="htfa-installed-") as directory:
        target = Path(directory)
        for name in ("tests", "docs"):
            shutil.copytree(
                source / name,
                target / name,
                ignore=shutil.ignore_patterns("_build", "auto_examples", "generated"),
            )
        shutil.copy2(source / "pyproject.toml", target / "pyproject.toml")
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        probe = subprocess.check_output(
            [
                sys.executable,
                "-c",
                "from pathlib import Path; import htfa; "
                "print(Path(htfa.__file__).resolve().parent)",
            ],
            cwd=target,
            env=env,
            text=True,
        ).strip()
        installed = Path(probe)
        if source == installed.parent or "site-packages" not in installed.parts:
            raise RuntimeError(f"Expected installed wheel, got {installed}")
        print("Testing installed package:", installed, flush=True)
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "tests", str(installed)],
            cwd=target,
            env=env,
            check=True,
        )


if __name__ == "__main__":
    main()
