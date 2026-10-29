.. Cover image (scripts/make_cover.py); furo shows one variant per theme.

.. image:: _static/cover_light.png
   :class: only-light
   :alt: HTFA factor network: 50 template factors drawn as spheres inside a
         translucent brain, with the strongest inter-subject correlations as
         edges
   :width: 100%

.. image:: _static/cover_dark.png
   :class: only-dark
   :alt: HTFA factor network: 50 template factors drawn as spheres inside a
         translucent brain, with the strongest inter-subject correlations as
         edges
   :width: 100%

Illustrative movie-viewing analysis: an HTFA template (K = 50) fitted to 12
adults watching Pixar's *Partly Cloudy*, from Nilearn's ``development_fmri``
dataset, viewed from the left, above and right. Inter-subject functional
correlation (ISFC) compares participants' factor time courses at matching
moments of the film. Edges show the strongest 5% of factor pairs
(\|ISFC\| ≥ 0.137; red positive, blue negative). Node size represents factor
width; node colour identifies communities in the positive-ISFC graph. See
:doc:`tutorials/10_dynamic_isfc_animation` for the analysis workflow.
