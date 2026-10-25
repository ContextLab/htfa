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
   * - ``gpu``
     - torch (``backend='torch'``, ``'cuda'`` or ``'mps'``)
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
