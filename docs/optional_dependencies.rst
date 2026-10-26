Optional dependencies
=====================

Extras
------

These extras are declared in ``pyproject.toml``:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Extra
     - Contents
   * - ``plot``
     - matplotlib, nilearn (``HTFAResults.plot_*``)
   * - ``images``
     - pillow, tifffile (2D image inputs)
   * - ``nilearn``
     - nilearn (``mask='epi'``, ``mask='background'``, and fitted ``NiftiMasker`` masks)
   * - ``brain3d``
     - pyvista, imageio, imageio-ffmpeg, pillow (``htfa.plot_network``, ``htfa.animate_network``; the BrainNet Viewer meshes are downloaded from NITRC on first use)
   * - ``tutorials``
     - hypertools, sentence-transformers (only to re-execute the word-decoding tutorial)
   * - ``gpu``
     - torch (``backend='torch'``, ``'cuda'`` or ``'mps'``)
   * - ``tutorials``
     - hypertools, sentence-transformers (libraries some tutorials use besides htfa)
   * - ``dev``
     - pytest, pytest-cov, ruff, mypy, build, twine, plus the optional libraries above
   * - ``docs``
     - sphinx, furo, numpydoc, sphinx-gallery, nbsphinx, and others

Install an extra with, for example, ``pip install "htfa[plot]"``.

Installation on demand
----------------------

The first time a feature needs a missing optional library, htfa installs the
extra that provides it with pip, using the version constraints declared for the
extra, and prints a line to standard error naming what it installs. For example,
the first call to ``HTFAResults.plot_nodes`` installs matplotlib, and the first
2D image read installs pillow.

When installation is switched off, or fails, the call raises ``ImportError``
with the ``pip install "htfa[extra]"`` command to run.

htfa installs with ``python -m pip`` when pip works. In an environment without
pip (for example a bare ``uv venv``), it uses ``uv pip install --python
<interpreter>`` when ``uv`` is on ``PATH``, and otherwise bootstraps pip with
``python -m ensurepip``. If none of these works, the error names the command to
run. Installers are never looked up in the current directory, and they run in a
fresh private directory that is removed afterwards. pip keeps reading your user
and global configuration (for example a corporate index in ``pip.conf``). uv runs
with ``--no-config``, because it would otherwise read ``uv.toml`` and
``pyproject.toml`` from its working directory and every parent (a file planted in
a shared temporary directory could redirect the package index). This also skips
your own ``uv.toml``; set the index with environment variables such as
``UV_INDEX_URL`` instead, which uv still reads.

Other libraries
---------------

:func:`htfa.require` gives your own code (and the tutorials) the same behavior
for the libraries htfa declares: it imports a module, installing it first if it
is missing.

.. code-block:: python

   import htfa

   hyp = htfa.require("hypertools", extra="tutorials")  # installs htfa[tutorials]
   plt = htfa.require("matplotlib.pyplot")  # installs matplotlib>=3.7 alone

Without ``extra``, only the library itself is installed, with the version
constraints htfa's extras declare for it. A library htfa does not declare is
never installed: :func:`htfa.require` raises ``ImportError`` instead.

Turning it off
--------------

Set the environment variable ``HTFA_AUTO_INSTALL`` to ``0``, ``false`` or ``no``
before starting Python, or call :func:`htfa.set_autoinstall` in the running
process. The function overrides the environment variable.

.. code-block:: python

   import htfa

   htfa.set_autoinstall(False)

torch
-----

``backend='torch'``, ``'cuda'`` or ``'mps'`` imports torch, installing
``torch>=2.2`` when it is missing. On hosts other than macOS where ``nvidia-smi``
is found and reports a CUDA driver, the installer picks the PyTorch wheel index
for the newest supported CUDA version the driver can run (13.2, 13.0 or 12.6),
or the CPU-only index when the driver is older than 12.6. Without an NVIDIA
driver, and on macOS, it uses the default PyPI wheel.
``backend='auto'`` never installs torch: it uses torch only if it is already
installed (see :doc:`performance`).

Rendering without a display
---------------------------

The network plots (:func:`htfa.plot_network`, :func:`htfa.animate_network`)
render with VTK off screen. On macOS and Windows, and on Linux with an X
display, this needs nothing more. On Linux without a display (``DISPLAY``
unset, as on Google Colab, in containers and on most servers), htfa selects one
of VTK's display-free OpenGL windows before the first render, by setting
``VTK_DEFAULT_OPENGL_WINDOW``:

1. ``vtkEGLRenderWindow``, which needs ``libEGL`` and a driver for it: Mesa's
   software renderer (llvmpipe) or a GPU driver. Colab includes these
   (``libegl1``, ``libegl-mesa0``).
2. ``vtkOSOpenGLRenderWindow``, which needs Mesa's ``libOSMesa``.

Each is first tried in a short subprocess (VTK crashes the process when a
backend's library is missing), and the first that works is used. A value of
``VTK_DEFAULT_OPENGL_WINDOW`` that you set yourself is kept. If neither works,
the call raises ``RuntimeError`` naming the packages to install; on
Debian/Ubuntu:

.. code-block:: bash

   apt-get install libegl1 libegl-mesa0

These are system libraries, so htfa does not install them itself. pip wheels
of VTK do not include them.
