"""Tests for on-demand dependency installation (htfa._lazy)."""

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


def test_installer_uses_pip_when_available():
    # this test environment has pip
    assert _lazy._installer() == [sys.executable, "-I", "-m", "pip", "install", "-q"]


def _uv_venv(tmp_path: Path) -> Path:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not on PATH")
    env = tmp_path / "venv"
    subprocess.run([uv, "venv", "-q", "-p", sys.executable, str(env)], check=True)
    py = env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    subprocess.run(
        [uv, "pip", "install", "--python", str(py), "-q", "-e", str(REPO)],
        check=True,
        timeout=900,
    )
    nopip = subprocess.run(
        [str(py), "-m", "pip", "--version"], capture_output=True, text=True
    )
    assert nopip.returncode != 0, "a bare uv venv is expected to have no pip"
    return py


def _require_in(py: Path, path: str, module: str, extra: str | None) -> str:
    code = (
        "import sys; from htfa import require; "
        f"m = require({module!r}, extra={extra!r}); "
        "print(getattr(m, '__version__', 'ok'))"
    )
    env = dict(os.environ, PATH=path)
    env.pop("HTFA_AUTO_INSTALL", None)
    out = subprocess.run(
        [str(py), "-c", code], capture_output=True, text=True, timeout=1800, env=env
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
