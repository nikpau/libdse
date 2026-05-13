# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

# Make the `libdse` package importable without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------
project = "Denoising AutoEncoder"
copyright = "2026, Niklas Paulig"
author = "Niklas Paulig"
release = "0.1.0"

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",  # Pull docstrings from source
    "sphinx.ext.napoleon",  # Google / NumPy style docstring support
    "sphinx.ext.intersphinx",  # Cross-links to Python / PyTorch / NumPy docs
    "sphinx.ext.viewcode",  # [source] link on every object
    "sphinx.ext.autosummary",  # Auto-generate summary tables
]

# autodoc: show members in source order; include private (_) helpers if needed
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "special-members": "__call__, __repr__, __iter__, __len__",
}

# napoleon: accept both Google and NumPy docstring styles
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# intersphinx: resolve links to stdlib, PyTorch, NumPy, and librosa docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

# autosummary: generate stub .rst files automatically
autosummary_generate = True

# ---------------------------------------------------------------------------
# Templates & exclusions
# ---------------------------------------------------------------------------
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# ---------------------------------------------------------------------------
# HTML output — furo theme
# ---------------------------------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}

# Short title used in the browser tab and navigation bar
html_title = "Denoising AutoEncoder"
