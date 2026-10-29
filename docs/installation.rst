Installation
============

htfa requires Python 3.10 or newer. Install the
`PyPI release <https://pypi.org/project/htfa/>`_:

.. code-block:: bash

   pip install htfa

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

The Linux minimum-dependency CI job uses
``uv pip install --resolution lowest-direct`` for its selected direct
dependencies, including numpy, scipy, scikit-learn, nibabel, joblib and
threadpoolctl. This does not validate every optional dependency floor or
the Apple Silicon torch requirement against its minimum version.
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
