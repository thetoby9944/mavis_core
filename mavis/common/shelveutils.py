import glob
import os
import time
import traceback
import streamlit as st
import pandas as pd

from pathlib import Path

import shelve
from dill import Pickler, Unpickler


shelve.Pickler = Pickler
shelve.Unpickler = Unpickler


from mavis.stutils.sessionstate import get


def _project_names():
    names = set([Path(p).stem for p in glob.glob(str(Path("data") / get(username="default").username / "*"))])
    names.remove("CONFIG")
    return list(names)


def get_all_projects():
    all_projects = _project_names()

    if "Default" not in all_projects:
        with shelve.open(config_path()) as c:
            create_new_project(c, "Default")
        time.sleep(0.2)
        all_projects = _project_names()

    return all_projects


def project_path(name):
    return str(Path("data") / get(username="default").username / name)


def current_data_dir():
    path = Path(current_project()) / "data"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def current_model_dir():
    path = Path(current_project()) / "data" / "models"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def workflow_path():
    return str(Path("data") / "WORKFLOW")


def config_path():
    path = Path("data") / get(username="default").username
    path.mkdir(exist_ok=True, parents=True)
    return str(path / "CONFIG")


def current_project():
    with shelve.open(config_path()) as c:
        if "Last" not in c:
            get_all_projects()
            c["Last"] = "Default"
        return project_path(c["Last"])


def save_globals(name, config):
    with shelve.open(current_project()) as d:
        base_key = Path(name).stem+"_config"
        config_dict = {key:config[key] for key in config.keys() if key.startswith("cfg_")}
        try:
            d[base_key] = config_dict
        except TypeError:
            print('ERROR shelving: {0}'.format(base_key))


def load_globals(name):
    with shelve.open(current_project()) as d:
        base_key = Path(name).stem+"_config"
        if base_key not in d:
            return {}
        return d[base_key]


def create_new_project(c, name):
    if name in _project_names():
        st.info(f"Project {name} already exists")
        return

    with shelve.open(project_path(name)) as d:
        d["df"] = pd.DataFrame({})
    c["Last"] = name
    st.info("Created project " + name)
    st.info("Selected project" + name)


def delete_project(c, name):
    time.sleep(0.2)
    try:
        os.remove(f'{project_path(name)}.dat')
        os.remove(f'{project_path(name)}.bak')
        os.remove(f'{project_path(name)}.dir')
    except:
        st.warning("Could not delete DB files")
        st.code(traceback.format_exc())

    time.sleep(0.2)

    if "Default" in _project_names():
        c["Last"] = "Default"
        st.info("Selected Project Default")
    else:
        st.warning("The Default project will be freshly created as fallback.")
        create_new_project(c, "Default")


@st.cache(
    max_entries=1,
    suppress_st_warning=True,
    allow_output_mutation=True,
    # hash_funcs={pd.DataFrame: lambda _: None},
)
def load_df(path):
    if path is None:
        return None
    with st.spinner(text='Updating Data Table Cache'):
        with shelve.open(path) as db:
            if "df" not in db:
                st.error("NO DATA TABLE IN THIS PROJECT. CREATING EMPTY")
                db["df"] = pd.DataFrame({})
                
            df = db["df"]
    st.success("Success")
    return df


def load_presets(default=None):
    with shelve.open(config_path()) as db:
        if "presets" not in db:
            db["presets"] = {default.name: default}

        preset_dict = db["presets"]
    return preset_dict


def last_preset(default=None):
    presets = load_presets(default=default)
    return presets[list(presets.keys())[-1]]


def save_preset(preset):
    preset_dict = load_presets()
    preset_dict[preset.name] = preset

    with shelve.open(config_path()) as db:
        db["presets"] = preset_dict
    st.info("Saved Preset")


def load_model_paths():
    with shelve.open(config_path()) as db:
        if "models" not in db:
            db["models"] = []
        preset_dict = db["models"]
    return preset_dict


def save_model_path(model_path):
    model_paths = load_model_paths() + [model_path]
    with shelve.open(config_path()) as db:
        db["models"] = model_paths

