# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from pathlib import Path

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../gaussianfeels'))
sys.path.insert(0, os.path.abspath('../camera'))
sys.path.insert(0, os.path.abspath('../tactile'))
sys.path.insert(0, os.path.abspath('../fusion'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GaussianFeels'
copyright = '2024, Krishi Attri, Seoul National University'
author = 'Krishi Attri'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'myst_parser',
    'sphinxcontrib.mermaid'
]

# Enable MyST parser for Markdown files
source_suffix = {
    '.rst': None,
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Theme options
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# -- Extension configuration -------------------------------------------------

# autodoc configuration
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# autosummary configuration
autosummary_generate = True

# napoleon configuration
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# MathJax configuration
mathjax3_config = {
    'tex': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'displayMath': [['$$', '$$'], ['\\[', '\\]']],
        'macros': {
            'R': '\\mathbb{R}',
            'C': '\\mathbb{C}',
            'N': '\\mathbb{N}',
            'Z': '\\mathbb{Z}',
            'Q': '\\mathbb{Q}',
            'argmin': '\\operatorname{argmin}',
            'argmax': '\\operatorname{argmax}',
        }
    }
}

# Todo configuration
todo_include_todos = True

# Mermaid configuration
mermaid_version = "latest"
mermaid_init_js = "mermaid.initialize({startOnLoad:true});"

# -- Custom configuration for academic documentation ------------------------

# Add custom CSS for mathematical notation
html_css_files = [
    'custom.css',
]

# Add LaTeX preamble for PDF output
latex_elements = {
    'preamble': r'''
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{bm}
\usepackage{mathtools}
\DeclareMathOperator*{\argmin}{arg\,min}
\DeclareMathOperator*{\argmax}{arg\,max}
\newcommand{\R}{\mathbb{R}}
\newcommand{\C}{\mathbb{C}}
\newcommand{\N}{\mathbb{N}}
\newcommand{\Z}{\mathbb{Z}}
\newcommand{\Q}{\mathbb{Q}}
''',
}

# Document title
latex_documents = [
    ('index', 'gaussianfeels.tex', 'GaussianFeels Documentation',
     'GaussianFeels Team', 'manual'),
]

# -- Academic documentation configuration -----------------------------------

# Add academic formatting options
html_context = {
    'display_github': True,
    'github_user': 'username',  # Update with actual username
    'github_repo': 'gaussianfeels',
    'github_version': 'main',
    'conf_py_path': '/docs/',
}

# Add custom roles for academic writing
def setup(app):
    """Setup custom Sphinx extensions for academic documentation."""
    app.add_css_file('custom.css')
    
    # Add custom roles
    from docutils.nodes import Element, TextElement
    from docutils.parsers.rst import roles
    from sphinx.util.docutils import SphinxRole
    
    class EquationRole(SphinxRole):
        """Custom role for inline equations."""
        def run(self):
            text = self.text
            return [TextElement(rawsource=f'${text}$', text=f'${text}$')], []
    
    app.add_role('eq', EquationRole())
    app.add_role('math', EquationRole())
    
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }