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
'arm64'``), torch is also a core dependency.

Optional dependencies are listed on :doc:`optional_dependencies`. A missing
optional dependency is installed with pip the first time it is needed.

Building the documentation
--------------------------

.. code-block:: bash

   uv pip install -e ".[docs,plot,images,nilearn]"
   sphinx-build -W -b html docs docs/_build/html

Building the tutorials needs `pandoc <https://pandoc.org>`_ on the path.
