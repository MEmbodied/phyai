import os
import sys
import textwrap


sys.path.insert(0, os.path.abspath("../phyai/src"))
sys.path.insert(0, os.path.abspath("../phyai-kernel"))
sys.path.insert(0, os.path.abspath("../phyai-ext/src"))
sys.path.insert(0, os.path.abspath("../phyai-model-optimizer/src"))
sys.path.insert(0, os.path.abspath("../phyai-utils-tools/src"))
sys.path.insert(0, os.path.abspath("../"))
autodoc_mock_imports = ["torch", "flashinfer", "triton", "numpy"]
project = "PhyAI"
version = "0.1.0"
release = "0.1.0"
author = "PhyAI Contributors"
copyright = "2025-2026, %s" % author

enable_doxygen = os.environ.get("PHYAI_ENABLE_DOXYGEN", "false").lower() == "true"

extensions = [
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "autoapi.extension",
    "myst_parser",
]

# API Doc Info
autoapi_type = "python"
autoapi_dirs = [
    "../phyai/src",
    "../phyai-kernel",
    "../phyai-ext/src",
    "../phyai-model-optimizer/src",
    "../phyai-utils-tools/src",
]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
]
autoapi_keep_files = False  # Useful for debugging the generated rst files
autoapi_generate_api_docs = True
autodoc_typehints = "description"
autoapi_ignore = []

if enable_doxygen:
    extensions.extend(["breathe", "exhale"])

this_file_dir = os.path.abspath(os.path.dirname(__file__))
if enable_doxygen:
    doxygen_xml_dir = os.path.join(this_file_dir, "xml")
    breathe_projects = {"phyai": doxygen_xml_dir}
    breathe_default_project = "phyai"

    repo_root = os.path.dirname(this_file_dir)

    # Setup the exhale extension
    exhale_args = {
        "containmentFolder": f"{os.path.join(this_file_dir, 'CppAPI')}",
        "rootFileName": "library_root.rst",
        "rootFileTitle": "Library API",
        "doxygenStripFromPath": repo_root,
        "exhaleExecutesDoxygen": True,
        "exhaleUseDoxyfile": True,
        "verboseBuild": True,
        "contentsDirectives": False,
        "pageLevelConfigMeta": ":github_url: https://github.com/MEmbodied/phyai",
        "contentsTitle": "Page Contents",
        "kindsWithContentsDirectives": ["class", "file", "namespace", "struct"],
        "afterTitleDescription": textwrap.dedent(
            """
            Welcome to the developer reference for the PhyAI C++ API.
        """
        ),
    }

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
language = "en"
exclude_patterns = ["build"]
pygments_style = "sphinx"
todo_include_todos = False

# == html settings
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "PhyAI"
html_show_sourcelink = False

html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_center": [],
    "navbar_end": ["navbar-nav", "theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "header_links_before_dropdown": 6,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/MEmbodied/phyai",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],
    "show_nav_level": 1,
    "show_toc_level": 2,
    "use_edit_page_button": True,
    "footer_start": ["copyright"],
    "footer_end": [],
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "pygments_light_style": "default",
    "pygments_dark_style": "github-dark",
}

html_context = {
    "github_user": "MEmbodied",
    "github_repo": "phyai",
    "github_version": "main",
    "doc_path": "docs",
    "default_mode": "auto",
}

# Hide left sidebar on the landing page so the hero stands alone.
html_sidebars = {
    "index": [],
}

# == latex
latex_engine = "xelatex"
