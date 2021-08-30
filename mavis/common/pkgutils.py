import glob
from pathlib import Path

import streamlit as st

from os.path import join, dirname, basename, isfile
from runpy import run_module

from pipelines import Files, References, Images, ML, Analysis, Visualization, Classical, Custom


packages = [
    Files,
    Images,
    References,
    Classical,
    ML,
    Analysis,
    Visualization,
    Custom
]


def get_modules(interactive=False, search=""):
    with open(Path('data/license_keys.txt')) as f:
        licensed = f.read()

    for pack in packages:
        if pack.__name__.split('.')[-1] not in licensed:
            print(pack.__name__.split('.')[-1])
            continue

        if interactive:
            expander = st.sidebar.beta_expander(pack.__name__.split('.')[-1], expanded=bool(search))

        modules = glob.glob(join(dirname(pack.__file__), "*.py"))
        modules = [basename(f)[:-3] for f in modules if isfile(f)
                   and not f.endswith('__init__.py')]

        for module_name in sorted(modules):
            if search and search.lower() not in module_name.lower():
                continue

            full_module_name = f"{pack.__name__}.{module_name}"

            if interactive and expander.checkbox(module_name):
                st.markdown(f"# {module_name.split('.')[-1]}")
                execute(full_module_name)

            yield full_module_name


def execute(module_name):
    run_module(module_name)
