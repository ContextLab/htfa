Tutorials
=========

.. include:: _cover.inc

The notebooks are worked examples of the general API, not required experimental
protocols. Synthetic examples provide known factors for checking recovery; 2D
images illustrate the spatial model without a brain dataset. The real-data
examples use two different experiments:

- :doc:`tutorials/09_word_decoding_mitchell2008` uses fMRI responses to nouns
  and accompanying drawings to illustrate mappings between spatial factor
  weights and word representations.
- :doc:`tutorials/10_dynamic_isfc_animation`,
  :doc:`tutorials/11_plotting_and_appearance`,
  :doc:`tutorials/13_group_hierarchy`, and
  :doc:`tutorials/15_real_fmri_kernel_comparison` use participants watching the
  same short film. They illustrate aligned-time connectivity, visualization,
  grouped templates, and comparisons of spatial kernels, respectively.

Each notebook introduces its data, the question being illustrated, and the
limits of the resulting comparison.

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
