"""Tests for on-demand dependency installation (htfa._lazy)."""

import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from htfa import _lazy

REPO = Path(__file__).resolve().parent.parent


def smi(cuda: str) -> str:
    return (
        "| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      "
        f"CUDA Version: {cuda}     |"
    )


@pytest.fixture(autouse=True)
def _reset_autoinstall(monkeypatch):
    monkeypatch.delenv("HTFA_AUTO_INSTALL", raising=False)
    monkeypatch.setattr(_lazy, "_override", None, raising=False)


@pytest.mark.parametrize("value", ["0", "false", "no", "FALSE", "No"])
def test_autoinstall_env_var_disables(monkeypatch, value):
    monkeypatch.setenv("HTFA_AUTO_INSTALL", value)
    assert _lazy.autoinstall_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", ""])
def test_autoinstall_env_var_enabled(monkeypatch, value):
    monkeypatch.setenv("HTFA_AUTO_INSTALL", value)
    assert _lazy.autoinstall_enabled() is True


def test_autoinstall_default_on():
    assert _lazy.autoinstall_enabled() is True


def test_set_autoinstall_overrides_env(monkeypatch):
    monkeypatch.setenv("HTFA_AUTO_INSTALL", "0")
    _lazy.set_autoinstall(True)
    assert _lazy.autoinstall_enabled() is True
    _lazy.set_autoinstall(False)
    assert _lazy.autoinstall_enabled() is False


def test_set_autoinstall_public():
    import htfa

    assert htfa.set_autoinstall is _lazy.set_autoinstall


def test_lazy_import_present_module():
    import json

    assert _lazy.lazy_import("json", extra="images") is json


def test_lazy_import_disabled_message():
    _lazy.set_autoinstall(False)
    with pytest.raises(ImportError) as exc:
        _lazy.lazy_import("htfa_nonexistent_pkg", extra="images", purpose="tests")
    assert 'pip install "htfa[images]"' in str(exc.value)


def test_extra_requirements_reads_metadata():
    reqs = _lazy.extra_requirements("images")
    assert any(r.startswith("pillow") for r in reqs)
    assert any(r.startswith("tifffile") for r in reqs)
    assert not any("pytest" in r for r in reqs)
    assert _lazy.extra_requirements("gpu") == ["torch>=2.2"]


def test_torch_index_url_parsing():
    base = "https://download.pytorch.org/whl/"
    assert _lazy.torch_index_url(smi("13.2"), "Linux") == base + "cu132"
    assert _lazy.torch_index_url(smi("13.0"), "Linux") == base + "cu130"
    assert _lazy.torch_index_url(smi("12.8"), "Linux") == base + "cu126"
    assert _lazy.torch_index_url(smi("11.8"), "Linux") == base + "cpu"
    assert _lazy.torch_index_url(smi("13.2"), "Darwin") is None
    assert _lazy.torch_index_url(None, "Linux") is None
    assert _lazy.torch_index_url("no gpu here", "Linux") is None


def test_torch_index_url_is_silent(capsys):
    _lazy.torch_index_url(smi("11.8"), "Linux")
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def _plant(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    fake = directory / "nvidia-smi"
    fake.write_text("#!/bin/sh\necho planted\n")
    fake.chmod(0o755)
    return fake


@pytest.mark.skipif(sys.platform == "win32", reason="uses a POSIX shell script")
def test_find_nvidia_smi_ignores_cwd(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    fake = _plant(cwd)
    _plant(cwd / "subdir")
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("PATH", os.pathsep.join(["", ".", "subdir", str(cwd)]))
    found = _lazy._find_nvidia_smi()
    assert found is None or Path(found).resolve() != fake.resolve()
    assert found is None or "cwd" not in Path(found).parts


@pytest.mark.skipif(sys.platform == "win32", reason="uses a POSIX shell script")
def test_find_nvidia_smi_finds_absolute_path_entry(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    good = _plant(tmp_path / "bin")
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("PATH", os.pathsep.join(["", ".", str(good.parent)]))
    found = _lazy._find_nvidia_smi()
    assert found is not None
    assert Path(found).resolve() == good.resolve()
    assert os.path.isabs(found)


def test_ensure_torch_disabled_message():
    if _lazy._find_spec("torch"):
        pytest.skip("torch is installed, so the missing-torch path cannot be hit")
    _lazy.set_autoinstall(False)
    with pytest.raises(ImportError) as exc:
        _lazy.ensure_torch()
    assert 'pip install "htfa[gpu]"' in str(exc.value)


@pytest.mark.network
def test_lazy_import_installs_into_fresh_venv(tmp_path):
    env = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(env)], check=True)
    py = env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    uv = shutil.which("uv")
    if uv:
        install = [uv, "pip", "install", "--python", str(py), "-q", "-e", str(REPO)]
    else:
        install = [str(py), "-m", "pip", "install", "-q", "-e", str(REPO)]
    subprocess.run(install, check=True, timeout=600)
    # the images extra must be absent before the call
    pre = subprocess.run(
        [str(py), "-c", "import tifffile"], capture_output=True, text=True
    )
    assert pre.returncode != 0
    code = (
        "from htfa._lazy import lazy_import; "
        "m = lazy_import('tifffile', extra='images'); print(m.__version__)"
    )
    out = subprocess.run(
        [str(py), "-c", code], capture_output=True, text=True, timeout=600
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1][0].isdigit(), out.stdout


def test_require_public():
    import htfa

    assert htfa.require is _lazy.require


def test_require_present_module():
    import json

    assert _lazy.require("json") is json
    assert _lazy.require("json", extra="tutorials") is json


def test_require_disabled_message_names_extra():
    _lazy.set_autoinstall(False)
    with pytest.raises(ImportError) as exc:
        _lazy.require("htfa_nonexistent_pkg", extra="tutorials")
    print(exc.value)
    assert 'pip install "htfa[tutorials]"' in str(exc.value)
    assert "automatic installation is off" in str(exc.value)


def test_require_disabled_message_names_declared_spec():
    _lazy.set_autoinstall(False)
    if _lazy._find_spec("hypertools"):
        pytest.skip("hypertools is installed, so the missing path cannot be hit")
    with pytest.raises(ImportError) as exc:
        _lazy.require("hypertools")
    print(exc.value)
    assert 'pip install "hypertools>=' in str(exc.value)


def test_require_undeclared_module_is_not_installed(capsys):
    # autoinstall is on, but htfa never pip-installs a name it does not declare
    with pytest.raises(ImportError) as exc:
        _lazy.require("htfa_nonexistent_pkg")
    print(exc.value)
    assert "does not declare" in str(exc.value)
    assert "installing" not in capsys.readouterr().err


def test_declared_requirement_reads_metadata():
    pil = _lazy.declared_requirement("PIL.Image")
    print(pil)
    # pillow is declared >=10 (images, docs) and >=10.1 (brain3d, dev)
    assert pil.startswith("pillow") and ">=10.1" in pil
    assert _lazy.declared_requirement("matplotlib.pyplot") == "matplotlib>=3.7"
    assert _lazy.declared_requirement("sentence_transformers").startswith(
        "sentence-transformers>="
    )
    assert _lazy.declared_requirement("imageio_ffmpeg").startswith("imageio-ffmpeg>=")
    with pytest.raises(ImportError):
        _lazy.declared_requirement("htfa_nonexistent_pkg")


def test_extra_requirements_tutorials():
    reqs = _lazy.extra_requirements("tutorials")
    print(reqs)
    names = sorted(r.split(">")[0].split("=")[0] for r in reqs)
    assert names == ["hypertools", "sentence-transformers"]


@pytest.mark.network
def test_installer_uses_pip_when_available(tmp_path):
    # A venv seeded with pip (the dev environment from `make install` has none).
    py = _uv_venv(tmp_path, seed=True)
    work = tmp_path / "work"
    work.mkdir()
    code = f"from htfa import _lazy; print(_lazy._installer({str(work)!r}))"
    out = subprocess.run(
        [str(py), "-c", code], capture_output=True, text=True, timeout=300
    )
    print(out.stdout, out.stderr)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == str([str(py), "-I", "-m", "pip", "install"])


def _uv_venv(tmp_path: Path, seed: bool = False) -> Path:
    """A uv venv with htfa installed; with pip if ``seed``, else without."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not on PATH")
    env = tmp_path / "venv"
    subprocess.run(
        [
            uv,
            "venv",
            "-q",
            *(["--seed"] if seed else []),
            "-p",
            sys.executable,
            str(env),
        ],
        check=True,
    )
    py = env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    subprocess.run(
        [uv, "pip", "install", "--python", str(py), "-q", "-e", str(REPO)],
        check=True,
        timeout=900,
    )
    pip = subprocess.run(
        [str(py), "-m", "pip", "--version"], capture_output=True, text=True
    )
    assert (pip.returncode == 0) == seed, f"pip in the venv: {pip.returncode == 0}"
    return py


def _require_in(
    py: Path,
    path: str,
    module: str,
    extra: str | None,
    cwd: Path | None = None,
    tmpdir: Path | None = None,
) -> str:
    code = (
        "import sys; from htfa import require; "
        f"m = require({module!r}, extra={extra!r}); "
        "print(getattr(m, '__version__', 'ok'))"
    )
    env = dict(os.environ, PATH=path)
    env.pop("HTFA_AUTO_INSTALL", None)
    if tmpdir is not None:
        env["TMPDIR"] = str(tmpdir)
    out = subprocess.run(
        [str(py), "-c", code],
        capture_output=True,
        text=True,
        timeout=1800,
        env=env,
        cwd=cwd,
    )
    print(out.stdout, out.stderr)
    assert out.returncode == 0, out.stderr
    return out.stderr


@pytest.mark.network
def test_require_installs_with_uv_when_pip_is_missing(tmp_path):
    py = _uv_venv(tmp_path)
    uv_dir = os.path.dirname(shutil.which("uv"))
    err = _require_in(
        py, os.pathsep.join([uv_dir, "/usr/bin", "/bin"]), "tifffile", "images"
    )
    assert "installing" in err and "uv" in err
    # uv installed into the venv without adding pip
    nopip = subprocess.run([str(py), "-m", "pip", "--version"], capture_output=True)
    assert nopip.returncode != 0


@pytest.mark.network
def test_uv_install_ignores_planted_config(tmp_path):
    # A uv.toml in the (shared) temp dir and the caller's cwd points every index
    # at a dead URL; uv reads such files from its cwd and all parents.
    py = _uv_venv(tmp_path)
    planted = tmp_path / "shared_tmp"
    planted.mkdir()
    (planted / "uv.toml").write_text(
        '[[index]]\nurl = "http://127.0.0.1:9/simple"\ndefault = true\n'
    )
    (planted / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0"\n'
        '[tool.uv]\nindex-url = "http://127.0.0.1:9/simple"\n'
    )
    # the planted config is live: uv honours it when run there without --no-config
    uv = shutil.which("uv")
    probe = subprocess.run(
        [uv, "pip", "install", "--dry-run", "--python", str(py), "tifffile"],
        cwd=planted,
        capture_output=True,
        text=True,
        timeout=300,
    )
    print(probe.stderr)
    assert probe.returncode != 0 and "127.0.0.1" in probe.stderr
    uv_dir = os.path.dirname(uv)
    err = _require_in(
        py,
        os.pathsep.join([uv_dir, "/usr/bin", "/bin"]),
        "tifffile",
        "images",
        cwd=planted,
        tmpdir=planted,
    )
    assert "installing" in err and "127.0.0.1" not in err
    # the private working directory was removed
    assert sorted(p.name for p in planted.iterdir()) == ["pyproject.toml", "uv.toml"]


@pytest.mark.network
def test_require_bootstraps_pip_with_ensurepip(tmp_path):
    py = _uv_venv(tmp_path)
    path = os.pathsep.join(["/usr/bin", "/bin"])
    if shutil.which("uv", path=path):
        pytest.skip("uv is installed system-wide, so the ensurepip path cannot be hit")
    err = _require_in(py, path, "tifffile", "images")
    assert "ensurepip" in err
    pip = subprocess.run([str(py), "-m", "pip", "--version"], capture_output=True)
    assert pip.returncode == 0


@pytest.mark.network
@pytest.mark.slow
def test_require_installs_tutorials_extra(tmp_path):
    py = _uv_venv(tmp_path)
    uv_dir = os.path.dirname(shutil.which("uv"))
    path = os.pathsep.join([uv_dir, "/usr/bin", "/bin"])
    _require_in(py, path, "hypertools", "tutorials")
    # the whole extra was installed, so this imports without installing
    err = _require_in(py, path, "sentence_transformers", None)
    assert "installing" not in err


@pytest.mark.parametrize("name", [".x", "", "..htfa"])
def test_require_rejects_relative_names(name):
    with pytest.raises(ValueError, match="absolute module name"):
        _lazy.require(name)


def test_require_torch_goes_through_ensure_torch():
    if _lazy._find_spec("torch"):
        import torch

        assert _lazy.require("torch") is torch
        assert _lazy.require("torch.nn") is torch.nn
        return
    _lazy.set_autoinstall(False)
    with pytest.raises(ImportError) as exc:
        _lazy.require("torch", extra="tutorials")
    print(exc.value)
    # ensure_torch's message, not a bare `pip install "torch>=2.2"`
    assert 'pip install "htfa[gpu]"' in str(exc.value)


def test_imported_distributions_pins_only_loaded_packages():
    # A clean interpreter: numpy is imported, pytest is installed but not imported.
    code = (
        "import json, numpy; from htfa import _lazy; "
        "print(json.dumps(_lazy.imported_distributions()))"
    )
    out = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True
    )
    print(out.stdout, out.stderr)
    assert out.returncode == 0, out.stderr
    pins = json.loads(out.stdout)
    assert pins["numpy"] == importlib.metadata.version("numpy")
    assert pins["packaging"] == importlib.metadata.version("packaging")
    assert "pytest" not in pins
    assert "json" not in pins  # the standard library has no distribution


def _venv_with(tmp_path: Path, installer: str, *packages: str) -> tuple[Path, str]:
    """A venv with htfa and ``packages``; pip or uv does the on-demand install."""
    py = _uv_venv(tmp_path, seed=installer == "pip")
    uv = shutil.which("uv")
    subprocess.run(
        [uv, "pip", "install", "--python", str(py), "-q", *packages],
        check=True,
        timeout=900,
    )
    path = os.environ["PATH"]
    if installer == "uv":
        path = os.pathsep.join([os.path.dirname(uv), "/usr/bin", "/bin"])
    return py, path


def _run_in(py: Path, path: str, code: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PATH=path)
    env.pop("HTFA_AUTO_INSTALL", None)
    out = subprocess.run(
        [str(py), "-c", code], capture_output=True, text=True, timeout=1800, env=env
    )
    print(out.stdout, out.stderr)
    return out


@pytest.mark.network
@pytest.mark.slow
@pytest.mark.parametrize("installer", ["pip", "uv"])
def test_install_keeps_versions_of_imported_packages(tmp_path, installer):
    # The Colab failure: packaging (imported by htfa itself) is older than the
    # newest nilearn needs. The install must pick a nilearn that works with the
    # loaded packaging rather than upgrade packaging under the running process.
    py, path = _venv_with(tmp_path, installer, "packaging==24.2")
    code = (
        "import json, importlib.metadata as md, packaging; "
        "from htfa._lazy import lazy_import; "
        "lazy_import('nilearn.datasets', 'nilearn'); "
        "print(json.dumps({'disk': md.version('packaging'), "
        "'memory': packaging.__version__, 'nilearn': md.version('nilearn')}))"
    )
    out = _run_in(py, path, code)
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got["disk"] == got["memory"] == "24.2", got
    assert "installing" in out.stderr


@pytest.mark.network
@pytest.mark.parametrize("installer", ["pip", "uv"])
def test_install_conflicting_with_imported_package_asks_for_restart(
    tmp_path, installer
):
    # The images extra needs tifffile>=2023.1; an older tifffile is imported.
    py, path = _venv_with(tmp_path, installer, "tifffile==2022.10.10")
    code = (
        "import sys, tifffile; from htfa import require\n"
        "try:\n"
        "    require('PIL', extra='images')\n"
        "except ImportError as err:\n"
        "    print(err); sys.exit(3)\n"
    )
    out = _run_in(py, path, code)
    assert out.returncode == 3
    message = out.stdout.strip().splitlines()[-1]
    # the installer's own explanation is shown too
    assert "tifffile==2022.10.10" in out.stderr
    assert "tifffile 2022.10.10" in message
    assert "restart" in message
    assert 'pip install "htfa[images]"' in message
    # nothing was installed or changed: no half-upgraded environment
    check = _run_in(
        py,
        path,
        "import importlib.metadata as md, importlib.util as u; "
        "print(md.version('tifffile'), u.find_spec('PIL') is None)",
    )
    assert check.stdout.split() == ["2022.10.10", "True"]
