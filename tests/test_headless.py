"""Tests for rendering without a display (headless Linux, Colab)."""

import ctypes.util
import os
import subprocess
import sys

import numpy as np
import pytest

from htfa import brainnet

LINUX = sys.platform.startswith("linux")
NATIVE = {"darwin": "vtkCocoaRenderWindow", "win32": "vtkWin32OpenGLRenderWindow"}


@pytest.fixture
def no_display(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("VTK_DEFAULT_OPENGL_WINDOW", raising=False)


def test_gl_window_works_detects_native_window():
    if LINUX:
        pytest.skip("the native Linux window needs an X server")
    assert brainnet._gl_window_works(NATIVE[sys.platform]) is True


def test_gl_window_works_rejects_unknown_window():
    # VTK falls back to another window class, which does not count
    assert brainnet._gl_window_works("vtkNoSuchRenderWindow") is False


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
    assert run.returncode == 0, run.stderr
    assert out.stat().st_size > 1000
    assert np.asarray(brainnet._open(out)).std() > 5
