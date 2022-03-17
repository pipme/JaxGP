# -*- coding: utf-8 -*-

import jaxgp

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_nb",
]

myst_enable_extensions = ["dollarmath", "colon_fence"]
master_doc = "index"
source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
}
templates_path = ["_templates"]

# General information about the project.
project = "jaxgp"
copyright = jaxgp.__copyright__
version = jaxgp.__version__
release = jaxgp.__version__

exclude_patterns = ["_build"]
html_theme = "sphinx_book_theme"
html_title = "jaxgp"
# html_logo = "_static/zap.svg"
# html_favicon = "_static/favicon.png"
html_static_path = ["_static"]
html_show_sourcelink = False
html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/pipme/JaxGP",
    "repository_branch": "main",
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "notebook_interface": "jupyterlab",
        "colab_url": "https://colab.research.google.com/",
    },
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
}
html_baseurl = "https://jaxgp.readthedocs.io/en/latest/"
jupyter_execute_notebooks = "off"
# execution_excludepatterns = ["benchmarks.ipynb"]
execution_timeout = -1

autodoc_type_aliases = {
    "Array": "jaxgp.helpers.Array",
    "Dataset": "jaxgp.helpers.Dataset",
}
