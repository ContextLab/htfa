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
from collections.abc import Callable, Sequence
from inspect import getmodulename
from types import ModuleType

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

ENV_VAR = "HTFA_AUTO_INSTALL"
TORCH_INDEX = "https://download.pytorch.org/whl/"
# Highest-first; verified 2026-10-23 to carry torch 2.14 wheels.
CUDA_WHEEL_TAGS = {(13, 2): "cu132", (13, 0): "cu130", (12, 6): "cu126"}

_override: bool | None = None


def set_autoinstall(enabled: bool) -> None:
    """Turn on-demand installation of optional dependencies on or off.

    By default a missing optional dependency (for example matplotlib for plots
    or pillow for 2D images) is installed with pip the first time it is needed.
    The ``HTFA_AUTO_INSTALL`` environment variable (``0``, ``false`` or ``no``
    turns it off) sets the default; this function overrides the variable for the
    current process.

    Parameters
    ----------
    enabled : bool
        ``True`` allows installation; ``False`` makes a missing dependency raise
        ``ImportError`` with the ``pip install`` command to run.

    Examples
    --------
    >>> import htfa
    >>> htfa.set_autoinstall(False)
    >>> from htfa._lazy import autoinstall_enabled
    >>> autoinstall_enabled()
    False
    >>> htfa.set_autoinstall(True)
    """
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


_DISTRIBUTIONS = {"PIL": "pillow", "sklearn": "scikit-learn"}


def declared_requirement(module: str) -> str:
    """Requirement spec htfa's metadata declares for the library ``module``.

    The version constraints of every extra that lists the library are combined.
    """
    top = module.split(".")[0]
    name = canonicalize_name(_DISTRIBUTIONS.get(top, top))
    found = None
    spec = SpecifierSet()
    for line in importlib.metadata.requires("htfa") or []:
        req = Requirement(line)
        if canonicalize_name(req.name) == name:
            found = req.name
            spec &= req.specifier
    if found is None:
        raise ImportError(
            f"{module} is not installed, and htfa does not declare it in any extra, "
            "so it is not installed automatically; install it yourself, or pass "
            "the htfa extra that provides it"
        )
    return f"{found}{spec}"


def _succeeds(cmd: list[str], cwd: str) -> bool:
    try:
        run = subprocess.run(cmd, capture_output=True, check=False, cwd=cwd)
    except OSError:
        return False
    return run.returncode == 0


def _installer(cwd: str) -> list[str]:
    """Command prefix that installs packages into this interpreter's environment.

    pip if it works; else uv (a bare ``uv venv`` has no pip); else pip
    bootstrapped with ensurepip. ``cwd`` is the private directory they run in.
    pip runs without ``-q``: its output is captured, and only a failed run's is
    shown, which then includes the constraint that blocked the resolution.
    """
    # -I keeps a planted pip.py off sys.path; pip still reads the user's and
    # the global pip.conf (a corporate index keeps working), never the cwd.
    pip = [sys.executable, "-I", "-m", "pip"]
    if _succeeds([*pip, "--version"], cwd):
        return [*pip, "install"]
    uv = _find_executable("uv")
    if uv is not None:
        print(f"htfa: pip is not available; installing with {uv}", file=sys.stderr)
        # uv reads uv.toml / pyproject.toml [tool.uv] from the cwd and every
        # parent, which could redirect the index; --no-config ignores them. It
        # also ignores the user's own uv.toml (UV_* environment variables, such
        # as UV_INDEX_URL, still apply).
        return [uv, "--no-config", "pip", "install", "-q", "--python", sys.executable]
    print(
        "htfa: pip is not available; bootstrapping it with ensurepip", file=sys.stderr
    )
    if _succeeds([sys.executable, "-I", "-m", "ensurepip"], cwd) and _succeeds(
        [*pip, "--version"], cwd
    ):
        return [*pip, "install"]
    raise ImportError(
        f"no installer works for {sys.executable}: pip is missing, uv is not on "
        "PATH and ensurepip failed. Install pip (`apt-get install python3-pip` "
        "on Debian/Ubuntu, or `python -m ensurepip`) or uv, then rerun"
    )


class _RestartRequired(ImportError):
    """The install needs another version of a package this process has loaded."""


def imported_distributions() -> dict[str, str]:
    """Installed version of each distribution that provides a loaded module.

    Keys are canonical distribution names; a top-level name in ``sys.modules``
    is mapped to its distributions with ``packages_distributions`` and, as
    Python 3.11+ does, the top-level entries of each distribution's file list.
    A version that is not PEP 440 cannot be written as a pin and is left out.
    """
    providers = {
        module: set(dists)
        for module, dists in importlib.metadata.packages_distributions().items()
    }
    # Python 3.10's packages_distributions reads only top_level.txt, which
    # wheels not built by setuptools (numpy, packaging) do not have.
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        for path in dist.files or ():
            top = path.parts[0] if len(path.parts) > 1 else getmodulename(path.name)
            if name and top and "." not in top:  # skips *.dist-info and ../bin
                providers.setdefault(top, set()).add(name)
    names = {
        canonicalize_name(dist)
        for module in list(sys.modules)
        if "." not in module
        for dist in providers.get(module, ())
    }
    pins: dict[str, str] = {}
    for name in sorted(names):
        try:
            version = importlib.metadata.version(name)
            Version(version)
        except (importlib.metadata.PackageNotFoundError, InvalidVersion):
            continue
        pins[name] = version
    return pins


def _restart(changed: list[str], manual: str) -> _RestartRequired:
    return _RestartRequired(
        f"the automatic install would replace {', '.join(changed)}, already "
        "imported in this Python process, and a loaded package cannot be replaced "
        "safely. Restart the process (in Jupyter or Colab: restart the kernel or "
        f"runtime), then run `{manual}` before importing anything else."
    )


def _pip_install(args: list[str], manual: str) -> None:
    """Install ``args`` without changing any distribution that is loaded.

    Every distribution with an imported module is pinned to its installed
    version through a constraints file, so the installer either leaves them
    alone or fails before changing anything; a failure that names a pin raises
    ``_RestartRequired``.
    """
    pins = imported_distributions()
    # A fresh private directory as cwd: nothing in the shared temp dir is read.
    work = tempfile.mkdtemp(prefix="htfa-install-")
    try:
        constraints = os.path.join(work, "constraints.txt")
        with open(constraints, "w", encoding="utf-8") as f:
            f.writelines(f"{name}=={version}\n" for name, version in pins.items())
        cmd = [*_installer(work), "-c", constraints, *args]
        run = subprocess.run(
            cmd, cwd=work, capture_output=True, text=True, errors="replace"
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)
    importlib.invalidate_caches()
    if run.returncode != 0:
        output = run.stdout + run.stderr
        print(output, end="", file=sys.stderr)
        # pip and uv both quote the constraint that blocks the resolution.
        blocked = [
            f"{name} {version}"
            for name, version in pins.items()
            if re.search(
                rf"(?<![\w.-]){re.escape(name)}=={re.escape(version)}(?![\w.+-])",
                output,
                re.IGNORECASE,
            )
        ]
        if blocked:
            raise _restart(blocked, manual)
        raise subprocess.CalledProcessError(run.returncode, cmd, run.stdout, run.stderr)


def _import(
    module: str, requirements: Callable[[], list[str]], manual: str, need: str
) -> ModuleType:
    try:
        return importlib.import_module(module)
    except ImportError as first:
        if not autoinstall_enabled():
            raise ImportError(
                f"{module} is not installed{need}. Install it with `{manual}` "
                "(automatic installation is off)."
            ) from first
        try:
            reqs = requirements()
            print(f"htfa: installing {', '.join(reqs)}{need}...", file=sys.stderr)
            _pip_install(reqs, manual)
            return importlib.import_module(module)
        except _RestartRequired:
            raise
        except (subprocess.CalledProcessError, ImportError, ValueError) as second:
            raise ImportError(
                f"{module} is not installed{need}, and installing it automatically "
                f"failed ({type(second).__name__}: {second}). "
                f"Install it with `{manual}`."
            ) from second


def lazy_import(module: str, extra: str, purpose: str | None = None) -> ModuleType:
    """Import ``module``, installing the htfa ``extra`` that provides it if missing."""
    need = f" (needed for {purpose})" if purpose else ""
    return _import(module, lambda: extra_requirements(extra), _manual(extra), need)


def require(module: str, extra: str | None = None) -> ModuleType:
    """Import a third-party library, installing it on first use.

    Installation follows the same rules as htfa's own optional dependencies (see
    :func:`set_autoinstall`): it uses the version constraints declared in htfa's
    package metadata, and is off when ``HTFA_AUTO_INSTALL=0``. Only libraries
    that htfa declares are installed.

    Parameters
    ----------
    module : str
        Name to import, for example ``"hypertools"`` or ``"matplotlib.pyplot"``.
    extra : str, optional
        The htfa extra to install if ``module`` is missing, for example
        ``"tutorials"`` (hypertools, sentence-transformers). By default only the
        library itself is installed, with the constraints that htfa's extras
        declare for it. torch is always installed as for ``backend='torch'``
        (a build matching the CUDA driver).

    Returns
    -------
    module
        The imported module.

    Raises
    ------
    ValueError
        If ``module`` is not an absolute module name.
    ImportError
        If the module is missing and cannot be installed: installation is off,
        htfa does not declare it, or the installer failed. The message names the
        command to run.

    Examples
    --------
    >>> import htfa
    >>> np = htfa.require("numpy")
    >>> np.__name__
    'numpy'
    """
    if not module or module.startswith("."):
        raise ValueError(f"require needs an absolute module name, got {module!r}")
    if module.split(".")[0] == "torch":
        ensure_torch()  # picks the wheel index that matches the CUDA driver
        return importlib.import_module(module)
    if extra is not None:
        return lazy_import(module, extra)
    try:
        return importlib.import_module(module)
    except ImportError:
        pass
    spec = declared_requirement(module)
    return _import(module, lambda: [spec], f'pip install "{spec}"', "")


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


def _under(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:  # different drives
        return False


def _find_executable(name: str, candidates: Sequence[str] = ()) -> str | None:
    """Locate ``name`` on PATH, or else at ``candidates``, never in the cwd."""
    cwd = os.path.realpath(os.getcwd())
    exts = [""]
    if sys.platform == "win32":
        exts = os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD").split(";")
    for d in os.environ.get("PATH", "").split(os.pathsep):
        if not d or not os.path.isabs(d) or _under(os.path.realpath(d), cwd):
            continue
        for ext in exts:
            path = os.path.join(d, name + ext)
            if os.path.isfile(path) and os.access(path, os.X_OK):
                real = os.path.realpath(path)
                if os.path.isabs(real) and not _under(real, cwd):
                    return path
    return next((c for c in candidates if os.path.isfile(c)), None)


def _find_nvidia_smi() -> str | None:
    """Locate nvidia-smi without ever consulting the current directory."""
    candidates = ["/usr/bin/nvidia-smi"]
    root = os.environ.get("SystemRoot")
    if root:
        candidates.append(os.path.join(root, "System32", "nvidia-smi.exe"))
    files = os.environ.get("ProgramFiles")
    if files:
        candidates.append(
            os.path.join(files, "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe")
        )
    return _find_executable("nvidia-smi", candidates)


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
        _pip_install(args, manual)
        return importlib.import_module("torch")
    except _RestartRequired:
        raise
    except (subprocess.CalledProcessError, ImportError) as err:
        raise ImportError(
            f"torch could not be installed automatically ({type(err).__name__}). "
            f"Install it with `{manual}`."
        ) from err
