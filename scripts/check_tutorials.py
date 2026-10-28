"""Execute tutorials using an installed package, retaining executed notebooks.

Usage: python scripts/check_tutorials.py OUTPUT [PREFIX ...]
No prefixes selects every tutorial. Run with the wheel's environment and extras.
"""

import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    prefixes = tuple(sys.argv[2:])
    source = Path(__file__).resolve().parents[1] / "docs" / "tutorials"
    os.environ.pop("PYTHONPATH", None)
    for path in sorted(source.glob("*.ipynb")):
        if prefixes and not path.name.startswith(prefixes):
            continue
        notebook = nbformat.read(path, as_version=4)
        notebook.cells.insert(
            0,
            nbformat.v4.new_code_cell(
                "import sys\nfrom pathlib import Path\nimport htfa\n"
                "print(sys.executable, htfa.__file__)\n"
                'assert "site-packages" in Path(htfa.__file__).resolve().parts'
            ),
        )
        print("Executing", path.name, flush=True)
        client = NotebookClient(
            notebook,
            timeout=7200,
            kernel_name="python3",
            resources={"metadata": {"path": str(output)}},
        )
        try:
            client.execute()
        finally:
            nbformat.write(notebook, output / path.name)
        print("Passed", path.name, flush=True)


if __name__ == "__main__":
    main()
