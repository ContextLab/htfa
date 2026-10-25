"""Sphinx configuration for the htfa documentation."""

import logging
from pathlib import Path

import htfa

project = "htfa"
author = "Contextual Dynamics Lab"
copyright = "2026, Contextual Dynamics Lab"
version = htfa.__version__
release = htfa.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "numpydoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_gallery.gen_gallery",
    "nbsphinx",
]

exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "superpowers",
    "API.md",
    "auto_examples/*.ipynb",
]
default_role = "py:obj"
autosummary_generate = True
autodoc_typehints = "none"
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "nibabel": ("https://nipy.org/nibabel/", None),
    "nilearn": ("https://nilearn.github.io/stable/", None),
}

# Notebooks are executed locally and their outputs are saved in the .ipynb files.
nbsphinx_execute = "never"

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"plot_",
    "download_all_examples": False,
    "remove_config_comments": True,
    "doc_module": ("htfa",),
    "backreferences_dir": "generated/backreferences",
    "reference_url": {"htfa": None},
}

# sphinx_gallery_conf is not picklable into the environment cache.
suppress_warnings = ["config.cache"]


class _EmptyTutorialsGlob(logging.Filter):
    """Drop the empty-glob warning for tutorials.rst while no notebook exists."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "glob pattern 'tutorials/*'" not in record.getMessage()


def setup(app):
    if any((Path(__file__).parent / "tutorials").glob("*.ipynb")):
        return
    for handler in logging.getLogger("sphinx").handlers:
        handler.filters.insert(0, _EmptyTutorialsGlob())


html_theme = "furo"
html_title = "htfa"
html_static_path = ["_static"]
html_favicon = "_static/favicon.ico"
html_css_files = ["custom.css"]

_GITHUB = "https://github.com/ContextLab/htfa"
html_theme_options = {
    "source_repository": _GITHUB,
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {
        "color-brand-primary": "#007030",
        "color-brand-content": "#007030",
        "color-link--hover": "#005524",
        "font-stack": "'Nunito Sans', -apple-system, BlinkMacSystemFont, "
        "'Segoe UI', Helvetica, Arial, sans-serif",
    },
    "dark_css_variables": {
        "color-brand-primary": "#4CAF50",
        "color-brand-content": "#4CAF50",
        "color-link--hover": "#81C784",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": _GITHUB,
            "html": '<svg stroke="currentColor" fill="currentColor" '
            'stroke-width="0" viewBox="0 0 16 16"><path fill-rule="evenodd" '
            'd="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 '
            "0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13"
            "-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66"
            ".07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15"
            "-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 "
            "1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 "
            "1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 "
            '1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z">'
            "</path></svg>",
            "class": "",
        },
    ],
}
