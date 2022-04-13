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
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath('../mavis'))
sys.path.insert(0, os.path.abspath('../../modules/'))


# -- Project information -----------------------------------------------------

project = 'Mavis'
copyright = '2020, Tobias Schiele'
author = 'Tobias Schiele'

# The full version, including alpha/beta/rc tags
release = '1.0'

# -- Internationalization ------------------------------------------------
# specifying the natural language populates some key tags
language = "en"


# -- General configuration ---------------------------------------------------

# Allows using numbered figures in latex and html output
numfig = True

# Root document containing initial toc
master_doc = "index"

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "recommonmark",  # Allow processing md files
    'sphinx.ext.viewcode',  # Link sources to documentation
    'autoapi.extension',  # generate documentation from docstrings
    'sphinx_markdown_tables',  # allow markdown style tables (only works for html output sadly)
    'sphinx.ext.napoleon',
    #'sphinxcontrib.autohttp.web', # generate web endpoints from source code
    #'sphinxcontrib.bibtex',  # Allows using :cite:`bibtexentry` in .rst files
]


# Map file-extensions to sphinx parser
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

# -- AUTOAPI ---------------------------------------------------

autoapi_type = 'python'
autoapi_dirs = [
    "../mavis",
    #"../../modules/"
]
autoapi_ignore = []#"*/lib/*", "*/playground/*", "*/model/*", "*/models/*", "*/nima/*", "*/qtclient/*", "*webclient/config*"]
autoapi_template_dir = '_templates'
autoapi_add_toctree_entry = False
autoapi_member_order = "groupwise"
autoapi_keep_files = True

# autodoc_inherit_docstrings = False


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    "autoapi/index.rst",
    "requirements.txt",
    "README.md"
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
html_logo = "../mavis/assets/images/MAVIS_logo.png"
html_favicon = '../mavis/assets/images/M_icon.png'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_sidebars = {'index': ["search-field.html", 'sidebar-nav-bs.html']}


html_theme_options = {
    "github_url": "https://github.com/thetoby9944/mavis_core",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/mavis_core",
            "icon": "fas fa-box",
        },
    ],
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "show_nav_level": 2,
    # "search_bar_position": "navbar",  # TODO: Deprecated - remove in future version
    # "navbar_align": "left",  # [left, content, right] For testing that the navbar items align properly
    # "navbar_start": ["navbar-logo", "navbar-version"],
    # "navbar_center": ["navbar-nav", "navbar-version"],  # Just for testing
    "navbar_end": ["navbar-icon-links"],
}

html_context = {
    "github_user": "thetoby9944",
    "github_repo": "mavis_core",
    "github_version": "master",
    "doc_path": "docs",
}

# -- Options for Latex output -------------------------------------------------

#latex_docclass = {
#   'howto': 'ausarbeitung',
#   'manual': 'ausarbeitung',
#}
with open(Path("latex/preamble.tex")) as f:
    preamble = f.read()

with open(Path("latex/maketitle.tex")) as f:
    make_title = f.read()


asset_src = Path("../mavis/assets")
asset_dst_1 = Path("mavis/assets")
asset_dst_2 = Path("assets")


# Python 3.7 fix for copytree not able to ignore existing dirs
def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)


copytree(asset_src, asset_dst_1)
copytree(asset_src, asset_dst_2)

latex_documents = [
    (master_doc, 'master.tex', u' MAVIS',
     u'Tobias Schiele', 'report'), #"report"
]

latex_logo = 'assets/images/MAVIS_logo.png'
latex_engine = 'lualatex'
latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    'papersize': 'a4paper',
    'releasename':" ",

    # Sonny, Lenny, Glenn, Conny, Rejne, Bjarne and Bjornstrup
    #'fncychap': '\\usepackage[Glenn]{fncychap}',
    #'fncychap': '\\usepackage{fncychap}',
    'fontpkg': '\\usepackage{amsmath,amsfonts,amssymb,amsthm}',

    # Latex figure (float) alignment
    #
    'figure_align':'htbp',

    # The font size ('10pt', '11pt' or '12pt').
    #
    'pointsize': '11pt',

    'inputenc': '',
    'utf8extra': '',


    # Additional stuff for the LaTeX preamble.
    #
    'preamble': preamble,

    # Latex title
    # + Toc, tof, lot
    #
    'maketitle': make_title,

    # Sphinx Setup
    #
    'sphinxsetup': \
        'hmargin={0.75in,0.75in}, vmargin={1in,1in}, \
        verbatimwithframe=false, \
        inlineliteralwraps=true, \
        parsedliteralwraps=true, \
        TitleColor={rgb}{0.505,0.04,0.115}, \
        HeaderFamily=\\rmfamily\\bfseries, \
        InnerLinkColor={rgb}{0.1,0,0.5}, \
        OuterLinkColor={rgb}{0,0,0.2}, \
        VerbatimColor={rgb}{0.93,0.93,0.93}, \
        verbatimborder=0.1pt',

    # Table of contens
    # - defined in title
    #
    'tableofcontents':' ',

}
