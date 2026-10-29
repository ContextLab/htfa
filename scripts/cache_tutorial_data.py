"""Prepare one shared tutorial download cache, or verify a transferred cache.

Usage: python scripts/cache_tutorial_data.py ROOT [--verify]
Set HTFA_DATA_DIR/HTFA_DATA to ROOT/htfa-data and NILEARN_DATA to ROOT/nilearn-data
when executing notebooks. Preparation downloads inputs only: no fits or rendering.
Transport failures receive bounded retries; validation failures are never skipped.
"""

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import requests


def retry_download(label, operation, attempts=5):
    """Retry transient HTTP/connection failures, including OSF's intermittent 403."""
    for attempt in range(attempts):
        try:
            return operation()
        except (
            urllib.error.URLError,
            requests.RequestException,
            ConnectionError,
        ) as exc:
            status = getattr(exc, "code", None)
            if isinstance(exc, requests.HTTPError) and exc.response is not None:
                status = exc.response.status_code
            transient = status is None or status in (403, 408, 429) or status >= 500
            if not transient or attempt + 1 == attempts:
                raise
            delay = min(5 * 2**attempt, 40)
            print(
                f"{label}: transport failure; retrying in {delay}s: {exc}", flush=True
            )
            time.sleep(delay)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text())
    for relative, expected in manifest.items():
        path = root / relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Tutorial cache integrity failure: {relative}")
    print(f"Verified {len(manifest)} tutorial input files", flush=True)


def prepare(root):
    data = root / "htfa-data"
    nilearn = root / "nilearn-data"
    os.environ.update(
        HTFA_DATA_DIR=str(data), HTFA_DATA=str(data), NILEARN_DATA=str(nilearn)
    )
    import htfa
    from htfa.brainnet import fetch_mesh

    # Tutorial 06 uses five adults; 10 uses twelve; 11 uses six; 15 uses three.
    # Tutorial 13 uses ten adults plus the first ten children of a 13-person
    # mixed-age fetch. These two calls cover each notebook's selection exactly.
    for count, adults in [(12, True), (13, False)]:
        retry_download(
            f"development_fmri ({count}, adults_only={adults})",
            lambda count=count, adults=adults: htfa.datasets.development_fmri_files(
                n_subjects=count, adults_only=adults, data_dir=nilearn
            ),
        )
    for name in htfa.datasets.TUTORIAL_FITS:
        retry_download(name, lambda name=name: htfa.datasets.fetch_tutorial_fit(name))
    for mesh in ("ch2", "icbm152", "icbm152_smoothed"):
        retry_download(mesh, lambda mesh=mesh: fetch_mesh(mesh))
    mitchell = data / "mitchell2008"
    mitchell.mkdir(parents=True, exist_ok=True)
    base = "https://www.cs.cmu.edu/afs/cs/project/theo-73/www/science2008"
    for participant in range(1, 10):
        path = mitchell / f"data-science-P{participant}.mat"

        def download(path=path):
            if path.exists():
                return
            partial = path.with_suffix(".part")
            urllib.request.urlretrieve(f"{base}/{path.name}", partial)
            partial.replace(path)

        retry_download(path.name, download)
        # Parse the MAT directory so an HTTP-success error page cannot be cached.
        from scipy.io import whosmat

        variables = {name for name, _, _ in whosmat(path)}
        if not {"data", "meta", "info"} <= variables:
            raise ValueError(f"Missing Mitchell dataset variables: {path}")
    manifest = {
        path.relative_to(root).as_posix(): digest(path)
        for folder in (data, nilearn)
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path.suffix != ".part"
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify(root)


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if "--verify" in sys.argv[2:]:
        verify(root)
    else:
        prepare(root)
