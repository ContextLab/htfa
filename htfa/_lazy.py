"""Import optional dependencies, installing them on demand.

The pip specs installed are read back from the installed package metadata, so
the extras declared in ``pyproject.toml`` are the single source of truth.
"""

import importlib
import importlib.metadata
import importlib.util
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType

from packaging.requirements import Requirement

ENV_VAR = "HTFA_AUTO_INSTALL"
TORCH_INDEX = "https://download.pytorch.org/whl/"
# Highest-first; verified 2026-10-23 to carry torch 2.14 wheels.
CUDA_WHEEL_TAGS = {(13, 2): "cu132", (13, 0): "cu130", (12, 6): "cu126"}

_override: bool | None = None


def set_autoinstall(enabled: bool) -> None:
    """Turn on-demand installation of optional dependencies on or off."""
    global _override
    _override = bool(enabled)


def autoinstall_enabled() -> bool:
    """Whether missing optional dependencies may be pip-installed."""
    if _override is not None:
        return _override
    return os.environ.get(ENV_VAR, "1").strip().lower() not in ("0", "false", "no")


def _find_spec(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def extra_requirements(extra: str) -> list[str]:
    """Requirement specs that htfa's metadata declares for ``extra``."""
    found = []
    for line in importlib.metadata.requires("htfa") or []:
        req = Requirement(line)
        if req.marker is None:
            continue
        if req.marker.evaluate({"extra": extra}) and not req.marker.evaluate(
            {"extra": ""}
        ):
            req.marker = None
            found.append(str(req))
    if not found:
        raise ValueError(f"htfa declares no requirements for extra {extra!r}")
    return found


def _manual(extra: str) -> str:
    return f'pip install "htfa[{extra}]"'


def _pip_install(args: list[str]) -> None:
    # -I and a neutral cwd keep a planted pip.py in the caller's cwd off sys.path
    subprocess.run(
        [sys.executable, "-I", "-m", "pip", "install", "-q", *args],
        check=True,
        cwd=tempfile.gettempdir(),
    )
    importlib.invalidate_caches()


def lazy_import(module: str, extra: str, purpose: str | None = None) -> ModuleType:
    """Import ``module``, installing the htfa ``extra`` that provides it if missing."""
    try:
        return importlib.import_module(module)
    except ImportError as first:
        need = f" (needed for {purpose})" if purpose else ""
        manual = _manual(extra)
        if not autoinstall_enabled():
            raise ImportError(
                f"{module} is not installed{need}. Install it with `{manual}` "
                "(automatic installation is off)."
            ) from first
        try:
            reqs = extra_requirements(extra)
            print(f"htfa: installing {', '.join(reqs)}{need}...", file=sys.stderr)
            _pip_install(reqs)
            return importlib.import_module(module)
        except (subprocess.CalledProcessError, ImportError, ValueError) as second:
            raise ImportError(
                f"{module} is not installed{need}, and installing it automatically "
                f"failed ({type(second).__name__}). Install it with `{manual}`."
            ) from second


def torch_index_url(nvidia_smi_output: str | None, system: str) -> str | None:
    """PyTorch wheel index suited to the NVIDIA driver, or None for PyPI default."""
    if system == "Darwin" or not nvidia_smi_output:
        return None
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", nvidia_smi_output)
    if match is None:
        return None
    driver = (int(match.group(1)), int(match.group(2)))
    for version in sorted(CUDA_WHEEL_TAGS, reverse=True):
        if version <= driver:
            return TORCH_INDEX + CUDA_WHEEL_TAGS[version]
    return TORCH_INDEX + "cpu"


def _find_nvidia_smi() -> str | None:
    """Locate nvidia-smi without searching the current directory."""
    cwd = os.path.normcase(os.path.abspath(os.getcwd()))
    safe = [
        d
        for d in os.environ.get("PATH", "").split(os.pathsep)
        if d not in ("", ".") and os.path.normcase(os.path.abspath(d)) != cwd
    ]
    found = shutil.which("nvidia-smi", path=os.pathsep.join(safe))
    if found:
        return found
    candidates = ["/usr/bin/nvidia-smi"]
    root = os.environ.get("SystemRoot")
    if root:
        candidates.append(os.path.join(root, "System32", "nvidia-smi.exe"))
    files = os.environ.get("ProgramFiles")
    if files:
        candidates.append(
            os.path.join(files, "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe")
        )
    return next((c for c in candidates if os.path.isfile(c)), None)


def ensure_torch() -> ModuleType:
    """Return the torch module, installing a suitable build first if needed."""
    if _find_spec("torch"):
        return importlib.import_module("torch")
    manual = _manual("gpu")
    if not autoinstall_enabled():
        raise ImportError(
            f"torch is not installed. Install it with `{manual}` "
            "(automatic installation is off)."
        )
    output = None
    exe = _find_nvidia_smi()
    if exe:
        try:
            run = subprocess.run(
                [exe], capture_output=True, text=True, check=False, timeout=10
            )
            output = run.stdout if run.returncode == 0 else None
        except (subprocess.TimeoutExpired, OSError):
            output = None
    url = torch_index_url(output, platform.system())
    if url is not None and url.endswith("/cpu"):
        print(
            "htfa: CUDA driver is older than the oldest supported wheels (12.6); "
            "installing CPU-only torch.",
            file=sys.stderr,
        )
    print(
        "htfa: installing torch (GPU backend; this can be a large download)...",
        file=sys.stderr,
    )
    args = ["torch>=2.2"] + (["--index-url", url] if url else [])
    try:
        _pip_install(args)
        return importlib.import_module("torch")
    except (subprocess.CalledProcessError, ImportError) as err:
        raise ImportError(
            f"torch could not be installed automatically ({type(err).__name__}). "
            f"Install it with `{manual}`."
        ) from err
