"""Where htfa caches downloaded files, and their checksums."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

DATA_DIR_ENV = "HTFA_DATA_DIR"


def data_dir() -> Path:
    """Root folder for files htfa downloads.

    ``$HTFA_DATA_DIR`` if that variable is set, otherwise ``%LOCALAPPDATA%\\htfa``
    on Windows and ``${XDG_CACHE_HOME:-~/.cache}/htfa`` elsewhere.
    """
    root = os.environ.get(DATA_DIR_ENV)
    if root:
        return Path(root).expanduser()
    if sys.platform == "win32" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "htfa"
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base).expanduser() / "htfa"


def sha256(path: Path) -> str:
    """Hex SHA-256 digest of the file at ``path``."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
