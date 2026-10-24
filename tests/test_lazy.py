"""Tests for on-demand dependency installation (htfa._lazy)."""

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
