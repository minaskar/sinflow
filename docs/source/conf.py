# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('./../../'))

import sinflow


# -- Project information -----------------------------------------------------

project = 'sinflow'
copyright = '2024, Minas Karamanis'
author = 'Minas Karamanis'

# The full version, including alpha/beta/rc tags
release = sinflow.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "myst_nb",
]

master_doc = "index"

myst_enable_extensions = ["dollarmath", "colon_fence"]

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = "sphinx_book_theme"
#html_theme = "sphinx_rtd_theme"
#html_theme = 'pydata_sphinx_theme'
#html_theme = 'sphinx_material'
html_theme = "furo"
html_title = "sinflow"
#html_logo = "./../../logo.png"
#logo_only = True

html_theme_options = {
    #"logo_only" : True,
    #'collapse_navigation': True, 
    #'navigation_depth': 4,
    #"announcement": (
    #    "⚠️ The new release 1.1.0 includes major performance and quality-of-life updates. Please check the new syntax and features! ⚠️"
    #),
    'sidebar_hide_name': False,
}

nb_execution_mode = "off"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path = ['_static']