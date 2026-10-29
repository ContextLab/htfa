Tutorials
=========

.. include:: _cover.inc

Each tutorial opens in Google Colab from the "Open in Colab" badge at its top.
Its first code cell installs the PyPI release of htfa when htfa is not installed. The
other packages a tutorial uses (for example matplotlib, nilearn, PyVista and
hypertools) are installed the first time they are needed (see
:doc:`optional_dependencies`), and the datasets, the embedding model and the
brain meshes are downloaded on first use.

.. toctree::
   :glob:
   :maxdepth: 1

   tutorials/*
