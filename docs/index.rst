htfa
====

.. include:: _cover.inc

htfa is a Python implementation of topographic factor analysis (TFA) and
hierarchical topographic factor analysis (HTFA), written from scratch after the
algorithm in `BrainIAK <https://brainiak.org>`_. It does not import BrainIAK.

TFA models a data matrix ``X`` (locations by samples) as ``X = F W + noise``.
Each of the ``K`` factors is a radial basis function with a center and a width in
coordinate space, and ``W`` holds the factors' weights over samples. HTFA fits TFA
to several subjects at once, tying each subject's factors to a global template.

.. code-block:: python

   import htfa

   res = htfa.fit("path/to/folder", K=50, random_state=0)
   res.global_centers   # (K, 3) template centers, in world (mm) coordinates
   res.weights[0]       # (K, T) factor time courses of the first subject
   res.connectivity(0)  # (K, K) factor connectivity of the first subject

.. toctree::
   :maxdepth: 1
   :caption: User guide

   installation
   inputs
   optional_dependencies
   performance
   tutorials
   auto_examples/index
   validation

.. toctree::
   :maxdepth: 1
   :caption: Reference

   api
   changelog
