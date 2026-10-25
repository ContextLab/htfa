Installation
============

htfa requires Python 3.10 or newer. It is not on PyPI yet; install it from GitHub:

.. code-block:: bash

   pip install git+https://github.com/ContextLab/htfa.git

For development, using `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: bash

   git clone https://github.com/ContextLab/htfa.git
   cd htfa
   uv venv
   uv pip install -e ".[dev]"

Core dependencies
-----------------

numpy, scipy, scikit-learn, nibabel, joblib, threadpoolctl and packaging. On
Apple Silicon Macs (``sys_platform == 'darwin'`` and ``platform_machine ==
'arm64'``), torch is also a core dependency, so that the Apple GPU (MPS) backend
is available without an extra install step.

The lower bounds in ``pyproject.toml`` (numpy 1.24, scipy 1.10, scikit-learn
1.3, nibabel 5, joblib 1.3, threadpoolctl 3, torch 2.2) are the versions the
test suite is run against in CI (``uv pip install --resolution lowest-direct``).
scikit-learn 1.3 is the first release whose k-means++ seeding uses
``sample_weight``, which the ``'ncorr'`` and ``'power'`` starts rely on; older
releases ignore the weights during seeding. torch's own metadata requires a
recent ``typing_extensions``; pip and uv upgrade it when they install torch, so
an error from ``typing_extensions`` while importing torch (such as
``ImportError: cannot import name 'deprecated'``) means the environment was
assembled without that requirement (for example with ``--no-deps``);
``pip install -U typing_extensions`` fixes it.

Optional dependencies are listed on :doc:`optional_dependencies`. A missing
optional dependency is installed with pip the first time it is needed.

Building the documentation
--------------------------

.. code-block:: bash

   uv pip install -e ".[docs,plot,images,nilearn]"
   sphinx-build -W -b html docs docs/_build/html

Building the tutorials needs `pandoc <https://pandoc.org>`_ on the path.
