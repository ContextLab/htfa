"""Tests for rendering without a display (headless Linux, Colab)."""

import ctypes.util
import os
import subprocess
import sys
import time

import numpy as np
import pytest

from htfa import brainnet

LINUX = sys.platform.startswith("linux")
NATIVE = {"darwin": "vtkCocoaRenderWindow", "win32": "vtkWin32OpenGLRenderWindow"}


@pytest.fixture
def no_display(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("VTK_DEFAULT_OPENGL_WINDOW", raising=False)
    monkeypatch.setattr(brainnet, "_auto_window", None)
    monkeypatch.setattr(brainnet, "_no_window", False)


def _working_window() -> str | None:
    """A window class the probe should accept here, if any."""
    if not LINUX:
        return NATIVE[sys.platform]
    if os.environ.get("DISPLAY"):
        return None
    return next(
        (w for w in brainnet.HEADLESS_WINDOWS if brainnet._gl_window_works(w)), None
    )


def test_gl_window_works_detects_native_window():
    if LINUX:
        pytest.skip("the native Linux window needs an X server")
    assert brainnet._gl_window_works(NATIVE[sys.platform]) is True


def test_gl_window_works_rejects_unknown_window():
    pytest.importorskip("vtkmodules", reason="needs VTK (the brain3d extra)")
    good = _working_window()
    if good is None:
        pytest.skip("no window renders here, so a rejection would mean nothing")
    # the probe accepts a working window ...
    assert brainnet._gl_window_works(good) is True
    # ... and rejects an unknown one: VTK falls back to another class
    assert brainnet._gl_window_works("vtkNoSuchRenderWindow") is False


@pytest.mark.network
def test_gl_window_works_sees_runtime_sys_path(tmp_path):
    # VTK reachable only through an entry added to sys.path at runtime (like a
    # --user or PYTHONPATH install): the probe must find it as the parent does.
    vtkmodules = pytest.importorskip("vtkmodules")
    good = _working_window()
    if good is None:
        pytest.skip("no window renders here")
    from test_lazy import _uv_venv

    py = _uv_venv(tmp_path)
    site = os.path.dirname(os.path.dirname(vtkmodules.__file__))
    code = (
        "import importlib.util as u, sys\n"
        "assert u.find_spec('vtkmodules') is None, 'venv must lack VTK'\n"
        f"sys.path.append({site!r})\n"
        "from htfa import brainnet\n"
        f"print(brainnet._gl_window_works({good!r}))\n"
    )
    run = subprocess.run([str(py), "-c", code], capture_output=True, text=True)
    print(run.stdout, run.stderr)
    assert run.returncode == 0, run.stderr
    assert run.stdout.split()[-1] == "True"


@pytest.mark.skipif(not LINUX, reason="OSMesa probing is Linux-only")
@pytest.mark.skipif(bool(ctypes.util.find_library("OSMesa")), reason="OSMesa present")
def test_gl_window_works_survives_a_crashing_backend():
    # Without libOSMesa, VTK's OSMesa window segfaults; the probe must contain it.
    assert brainnet._gl_window_works("vtkOSOpenGLRenderWindow") is False


def test_headless_window_leaves_displays_alone(monkeypatch):
    monkeypatch.delenv("VTK_DEFAULT_OPENGL_WINDOW", raising=False)
    if LINUX:
        monkeypatch.setenv("DISPLAY", ":0")
    assert brainnet._headless_window() is None
    assert "VTK_DEFAULT_OPENGL_WINDOW" not in os.environ


@pytest.mark.skipif(not LINUX, reason="headless selection is Linux-only")
def test_headless_window_respects_user_choice(no_display, monkeypatch):
    monkeypatch.setenv("VTK_DEFAULT_OPENGL_WINDOW", "vtkOSOpenGLRenderWindow")
    assert brainnet._headless_window() == "vtkOSOpenGLRenderWindow"


@pytest.mark.skipif(not LINUX, reason="headless selection is Linux-only")
def test_headless_window_picks_first_working_backend(no_display):
    works = [w for w in brainnet.HEADLESS_WINDOWS if brainnet._gl_window_works(w)]
    print("working offscreen windows:", works)
    if not works:
        with pytest.raises(RuntimeError, match="apt-get install"):
            brainnet._headless_window()
        return
    assert brainnet._headless_window() == works[0]
    assert os.environ["VTK_DEFAULT_OPENGL_WINDOW"] == works[0]


@pytest.mark.skipif(not LINUX, reason="headless selection is Linux-only")
def test_headless_window_failure_is_cached(no_display):
    if _working_window() is not None:
        pytest.skip("an offscreen window works here")
    with pytest.raises(RuntimeError, match="apt-get install"):
        brainnet._headless_window()
    t0 = time.perf_counter()
    with pytest.raises(RuntimeError, match="apt-get install"):
        brainnet._headless_window()
    assert time.perf_counter() - t0 < 0.5, "the failed probe was re-run"


@pytest.mark.skipif(not LINUX, reason="headless selection is Linux-only")
def test_headless_window_dropped_when_display_appears(no_display, monkeypatch):
    window = brainnet._headless_window() if _working_window() else None
    if window is None:
        pytest.skip("no offscreen window works here")
    assert os.environ["VTK_DEFAULT_OPENGL_WINDOW"] == window
    monkeypatch.setenv("DISPLAY", ":99")  # e.g. Xvfb started later
    assert brainnet._headless_window() is None
    assert "VTK_DEFAULT_OPENGL_WINDOW" not in os.environ
    # a window the user set themselves is never dropped
    monkeypatch.setenv("VTK_DEFAULT_OPENGL_WINDOW", "vtkOSOpenGLRenderWindow")
    assert brainnet._headless_window() is None
    assert os.environ["VTK_DEFAULT_OPENGL_WINDOW"] == "vtkOSOpenGLRenderWindow"


RENDER = """
import sys
import numpy as np
import htfa
c = np.array([[-40, -60, 0], [-40, 40, 0], [-40, 0, 50], [-40, 0, -30]], float)
E = np.array([[0, 2, -2, 0], [2, 0, 0, 2], [-2, 0, 0, 0], [0, 2, 0, 0]])
img = htfa.plot_network(c, E, mesh=None, snap=False, size=(400, 250), out=sys.argv[1])
a = np.asarray(img).astype(float)
red = ((a[..., 0] - a[..., 1] > 60) & (a[..., 0] - a[..., 2] > 60)).sum()
blue = ((a[..., 2] - a[..., 0] > 60) & (a[..., 2] - a[..., 1] > 60)).sum()
gray = ((abs(a[..., 0] - 128) < 40) & (abs(a[..., 0] - a[..., 2]) < 8)).sum()
print(a.shape, red, blue, gray)
assert red > 20 and blue > 20 and gray > 100, "blank or incomplete render"
"""


def test_plot_network_renders_without_display(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "DISPLAY"}
    env.pop("VTK_DEFAULT_OPENGL_WINDOW", None)
    out = tmp_path / "net.png"
    run = subprocess.run(
        [sys.executable, "-c", RENDER, str(out)],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    print(run.stdout, run.stderr)
    if (
        LINUX
        and not os.environ.get("DISPLAY")
        and not any(brainnet._gl_window_works(w) for w in brainnet.HEADLESS_WINDOWS)
    ):
        # No offscreen library here: the call must fail cleanly (no segfault)
        # with the error that names the packages to install.
        assert run.returncode == 1, run.stderr
        assert "RuntimeError" in run.stderr and "apt-get install" in run.stderr
        return
    assert run.returncode == 0, run.stderr
    assert out.stat().st_size > 1000
    assert np.asarray(brainnet._open(out)).std() > 5
