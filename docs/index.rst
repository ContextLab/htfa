HTFA Documentation
==================

Hierarchical Topographic Factor Analysis (HTFA) - A lightweight Python implementation for neuroimaging data analysis.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/modules
   infrastructure
   contributing

Overview
--------

HTFA is a dimensionality reduction technique specifically designed for neuroimaging data. It decomposes fMRI time series data into a set of spatial factors and temporal weights, enabling the identification of functional brain networks.

Key Features
------------

* **Hierarchical Structure**: Multi-subject analysis with shared global factors
* **Spatial Constraints**: Incorporates voxel coordinates for topographic organization
* **Scalable**: Optimized backends for large-scale neuroimaging data
* **BIDS Compatible**: Native support for BIDS-formatted datasets
* **Extensible**: Modular architecture for custom optimizers and backends

Installation
------------

.. code-block:: bash

   pip install htfa

Quick Start
-----------

.. code-block:: python

   from htfa import fit_bids
   
   # Analyze a BIDS dataset
   results = fit_bids(
       bids_dir="/path/to/bids/dataset",
       output_dir="/path/to/output",
       n_components=20,
       task="rest"
   )
   
   # Visualize results
   results.plot_factors(n_factors=5)
   
   # Export to NIfTI
   results.to_nifti("factors.nii.gz")

Infrastructure
--------------

The HTFA ecosystem includes comprehensive CI/CD infrastructure:

* **Unified Pipeline**: Coordinated builds across component epics
* **Quality Gates**: Automated code quality enforcement
* **Performance Monitoring**: Continuous benchmark tracking
* **Documentation**: Automated builds and deployment

See :doc:`infrastructure` for details.

Contributing
------------

We welcome contributions! Please see :doc:`contributing` for guidelines.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`