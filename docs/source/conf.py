# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use pathlib to make it absolute, like shown here.
#
from pathlib import Path
import sys

sys.path.insert(0, Path("../..").absolute())
import swiftgalaxy


# -- Project information -----------------------------------------------------

project = "SWIFTGalaxy"
copyright = "2022, Kyle Oman"
author = "Kyle Oman"

# The full version, including alpha/beta/rc tags
release = swiftgalaxy.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "recommonmark",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_design",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

source_suffix = [".rst", ".md"]
master_doc = "index"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"


def setup(app):
    app.add_css_file("custom.css")


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

intersphinx_mapping = dict(
    swiftsimio=("https://swiftsimio.readthedocs.io/en/latest/", None),
    velociraptor=("https://velociraptor-python.readthedocs.io/en/latest/", None),
    scipy=("https://docs.scipy.org/doc/scipy/", None),
    unyt=("https://unyt.readthedocs.io/en/stable/", None),
    numpy=("https://numpy.org/doc/stable/", None),
    caesar=("https://caesar.readthedocs.io/en/latest/", None),
)
