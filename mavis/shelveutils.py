import glob
import json
import os
import pprint
import shelve
import time
import tkinter as tk
import traceback
from abc import ABC
from pathlib import Path
from tkinter import filedialog

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from dill import Pickler, Unpickler

# HACK This only works when we've installed streamlit with pipenv, so the
# permissions during install are the same as the running process

STREAMLIT_STATIC_PATH = Path(st.__path__[0]) / 'static'
DOWNLOADS_PATH = (STREAMLIT_STATIC_PATH / "downloads")
# We create a downloads directory within the streamlit static asset directory
# and we write output files to it
DOWNLOADS_PATH.mkdir(exist_ok=True)

# Use dill for shelve to be able to pickle more complex objects
shelve.Pickler = Pickler
shelve.Unpickler = Unpickler

# Use session state hack for login
from stutils.sessionstate import get


def credential_path():
    return str(DataPathDAO().get() / "login")


def projects_path():
    path = DataPathDAO().get() / get(username="default")
    path.mkdir(parents=True, exist_ok=True)
    return path


def project_path(name):
    path = DataPathDAO().get() / get(username="default") / name
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def current_data_dir():
    path = Path(ProjectDAO().get()) / "data"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def current_model_dir():
    path = Path(current_data_dir()) / "models"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def workflow_path():
    return str(DataPathDAO().get() / "WORKFLOW")


def config_path():
    username = "default" # get(username="default")
    path = Path("data") / "config" / username
    path.mkdir(exist_ok=True, parents=True)
    return str(path / "CONFIG")


def save_df(df, base_path, new_dir, suffix, dir_name=None):
    p = Path(base_path)
    path = str(maybe_new_dir(p, dir_name, new_dir) / (p.stem + suffix))
    df.to_csv(path, sep=";", decimal=",")
    return path


def maybe_new_dir(path: Path, dir_name=None, new_dir=False) -> Path:
    folder = path if path.is_dir() else path.parent
    if new_dir:
        folder = folder.parent / dir_name
        folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_json(to_save, base_path, new_dir, suffix, dir_name=None):
    p = Path(base_path)
    path = str(maybe_new_dir(p, dir_name, new_dir) / (p.stem + suffix))
    with open(path, "w") as f:
        json.dump(to_save, f)
    return path


def save_pil(img, base_path, new_dir, suffix, dir_name=None, stem=None, prefix=None):
    if not isinstance(img, Image.Image):
        return img

    p = Path(base_path)
    prefix = "" if not prefix else (prefix + "_")
    stem = p.stem if not stem else stem
    path = str(maybe_new_dir(p, dir_name, new_dir) / (prefix + stem + suffix))
    img.save(path)
    return path


def identifier(path: str, path_indices, include_stem=True) -> str:
    if not type(path) == str:
        return None

    p = Path(path)
    parts = "_".join(np.array(p.parts)[path_indices])
    parts = (parts and parts + "_") + (p.stem if include_stem else "")
    return parts


def save_globals(name, config):
    with shelve.open(ProjectDAO().get()) as d:
        base_key = Path(name).stem + "_config"
        config_dict = {key: config[key] for key in config.keys() if key.startswith("cfg_")}
        try:
            d[base_key] = config_dict
        except TypeError:
            print('ERROR shelving: {0}'.format(base_key))


def load_globals(name):
    with shelve.open(ProjectDAO().get()) as d:
        base_key = Path(name).stem + "_config"
        if base_key not in d:
            return {}
        return d[base_key]


class BaseDAO:
    ACTIVE_PIPELINE = None


class LoginDAO:
    def __init__(self):
        with shelve.open(credential_path()) as login:
            if "passwords" not in login:
                login["passwords"] = {"default": "wg"}

    def check_session(self, username, password):
        with shelve.open(credential_path()) as login:
            # username, password = get(username="default", password="")
            check_user = username in login["passwords"]
            check_user = check_user and password == login["passwords"][username]
            return check_user, username, password


class DFDAO(BaseDAO):
    @staticmethod
    @st.cache(
        max_entries=1,
        suppress_st_warning=True,
        allow_output_mutation=True,
        # hash_funcs={pd.DataFrame: lambda _: None},
    )
    def get(path):
        if path is None:
            # This is used to refresh the cache
            return None
        with st.spinner(text='Updating Data Table Cache'):
            with shelve.open(path) as db:
                if "df" not in db:
                    st.error("NO DATA TABLE IN THIS PROJECT. CREATING EMPTY")
                    db["df"] = pd.DataFrame({})

                df = db["df"]
        st.success("Success. Press **`R`** to reload.")
        return df

    def check_len(self, df1, df2):
        if len(df1.dropna(how="all")) <= len(df2.dropna(how="all")):
            return True
        else:
            st.warning(
                "Will not update table. Updating would cause data loss because the new table contains less data.")
            return False

    def set(self, new_df, project, allow_loss=False):
        if allow_loss or self.check_len(self.get(project), new_df):
            with shelve.open(ProjectDAO().get()) as db:
                db["df"] = new_df.drop(["Index"], errors="ignore")
            self.get(None)
        return self.get(project)


class ProjectDAO(BaseDAO):
    def _project_names(self):
        path = projects_path()
        names = set([
            Path(p).stem
            for p in glob.glob(str(path / "*.bak")) + glob.glob(str(path / "*.db"))
            if Path(p).stem not in "CONFIG"
        ])
        return list(names)

    def get(self):
        with shelve.open(config_path()) as c:
            if "Last" not in c or c["Last"] is None:
                c["Last"] = "Default"

            last = c["Last"]
            all_projects = self._project_names()
            if last not in all_projects:
                c["Last"] = "Default"
                self.get_all()

        return project_path(last)

    def get_all(self):
        all_projects = self._project_names()
        if "Default" not in all_projects:
            self.add("Default")
            time.sleep(0.2)
            all_projects = self._project_names()
        return all_projects

    def add(self, name, overwrite=False):
        with shelve.open(config_path()) as c:
            if name in self._project_names():
                st.info(f"Project {name} already exists")
                if not overwrite:
                    return
            c["Last"] = name

        with shelve.open(project_path(name)) as d:
            d["df"] = pd.DataFrame({})
        st.info("Created project " + name)
        st.info("Selected project" + name)
        time.sleep(0.1)

    def delete(self, name):
        try:
            os.remove(f'{project_path(name)}.dat')
            os.remove(f'{project_path(name)}.bak')
            os.remove(f'{project_path(name)}.dir')
        except:
            st.warning("Could not delete DB files")
            st.code(traceback.format_exc())

        time.sleep(0.2)
        with shelve.open(config_path()) as c:
            if "Default" in self._project_names():
                c["Last"] = "Default"
                st.info("Selected Project Default")
            else:
                st.warning("The Default project will be freshly created as fallback.")
                self.add("Default")

    def set(self, selection):
        with shelve.open(config_path()) as c:
            c["Last"] = selection
        time.sleep(0.1)
        DFDAO().get(None)
        DFDAO().get(self.get())


class UserDAO(BaseDAO):
    def create(self, username, password):
        with shelve.open(credential_path()) as login:
            if not username or not password:
                st.info("Cannot Create empty User")
                return
            if username in login["passwords"]:
                st.info("User already exists")
                return
            passwords = login["passwords"]
            passwords[username] = password
            login["passwords"] = passwords
            st.info("Created User " + username)
            os.mkdir(str(DataPathDAO().get() / username))

    def delete(self, username, password):
        with shelve.open(credential_path()) as login:
            passwords = login["passwords"]
            if username in passwords and passwords[username] == password:
                del passwords[username]
                login["passwords"] = passwords
                st.info("Deleted User " + username)
            else:
                st.info("Wrong credentials for " + username)


class SimpleListDAO(BaseDAO):
    @property
    def key(self):
        raise NotImplementedError

    @property
    def default(self):
        return []

    def get_all(self):
        with shelve.open(config_path()) as db:
            if self.key not in db:
                db[self.key] = self.default
            all_entries = db[self.key]
        return all_entries

    def add(self, entry):
        all_entries = self.get_all()
        if entry not in all_entries:
            with shelve.open(config_path()) as db:
                db[self.key] = all_entries + [entry]

    def reset(self):
        with shelve.open(config_path()) as db:
            db[self.key] = self.default


class PresetListDAO(SimpleListDAO):
    @property
    def key(self):
        return "all_presets"

    @property
    def default(self):
        return ["Default"]


class ModelDAO(SimpleListDAO):
    @property
    def key(self):
        return f"{BaseDAO.ACTIVE_PIPELINE}_models"


class SimpleKeyDAO(BaseDAO):
    @property
    def key(self):
        raise NotImplementedError

    @property
    def default(self):
        raise NotImplementedError

    def get(self):
        with shelve.open(config_path()) as db:
            if self.key not in db:
                db[self.key] = self.default
            value = db[self.key]
        return value

    def set(self, value):
        with shelve.open(config_path()) as db:
            db[self.key] = value

    def reset(self):
        with shelve.open(config_path()) as db:
            db[self.key] = self.default


class LocalFolderBrowserMixin:
    def browse(self) -> None or Path:
        clicked = st.button('Select Folder')
        if clicked:
            root = tk.Tk()
            root.wm_attributes('-topmost', 1)
            root.withdraw()
            dialog = filedialog.askdirectory(
                master=root,
                initialdir="bullshit"
            )
            dirname = st.text_input(
                'Selected folder:',
                dialog
            )

            if dirname:
                return Path(dirname)


class SimplePathDAO(SimpleKeyDAO, ABC, LocalFolderBrowserMixin):
    def get(self):
        with shelve.open(config_path()) as db:
            if self.key not in db:
                db[self.key] = self.default
            module_path = db[self.key]
        return Path(module_path)


class ModulePathDAO(SimplePathDAO):
    @property
    def key(self):
        return "module_path"

    @property
    def default(self):
        return "pipelines"


class LogPathDAO(SimplePathDAO):
    @property
    def key(self):
        return "log_path"

    @property
    def default(self):
        return "logs"


class DataPathDAO(SimplePathDAO):
    @property
    def key(self):
        return "data_path"

    @property
    def default(self):
        return "data"


class ActivePresetDAO(SimpleKeyDAO):
    @property
    def key(self):
        return f"{BaseDAO.ACTIVE_PIPELINE}_active_preset"

    @property
    def default(self):
        return "Default"


class WorkflowDAO:
    def __init__(self):
        self.current = "Default"
        self.workflow = []
        self.pipeline = 0
        self.update()

    def update(self):
        with shelve.open(workflow_path()) as d:
            # No current workflow
            if "current" not in d:
                d["current"] = "Default"

            self.current = d["current"]

            # No workflow db
            if "workflows" not in d:
                d["workflows"] = {"Default": []}
                self.workflow = []

            # Default workflow has been Deleted
            if "Default" not in d["workflows"]:
                workflows = d["workflows"]
                workflows.update({"Default": []})
                d["workflows"] = workflows

            # Workflow DB ready to use
            if self.current not in d["workflows"]:
                d["current"] = "Default"
                self.current = "Default"

            # Set current workflow
            self.workflow = d["workflows"][self.current]

            # No active pipeline
            if "pipeline" not in d:
                d["pipeline"] = 0
                self.pipeline = 0
            else:
                self.pipeline = d["pipeline"]

    def set(self, name, pipelines):
        with shelve.open(workflow_path()) as d:
            workflows = d["workflows"]
            workflows[name] = pipelines
            d["workflows"] = workflows
        self.activate(name)

    def activate(self, name):
        with shelve.open(workflow_path()) as d:
            d["current"] = name
            d["pipeline"] = 0
        self.update()
        st.info("Activated Workflow: " + self.current)

    def get_all(self):
        with shelve.open(workflow_path()) as d:
            return list(d["workflows"].keys())

    def get(self):
        with shelve.open(workflow_path()) as d:
            return d["current"]

    def iterate(self):
        with shelve.open(workflow_path()) as d:
            if st.button("Next Pipeline"):
                self.pipeline = (self.pipeline + 1) % len(self.workflow)
                d["pipeline"] = self.pipeline

        if self.workflow:
            return self.workflow[(self.pipeline - 1)]
        else:
            st.error("Workflow has no configured pipelines")

    def delete(self):
        with shelve.open(workflow_path()) as d:
            if self.current in d["workflows"]:
                wf = d["workflows"]
                del wf[self.current]
                d["workflows"] = wf
                st.info("Deleted Workflow " + self.current)
        self.activate("Default")


class ConfigDAO:
    def __init__(self, default=None):
        self.default = default
        self.active = ActivePresetDAO().get()
        with shelve.open(config_path()) as db:
            if self.active not in db:
                db[self.active] = {}

    def __setitem__(self, key, value):
        with shelve.open(config_path()) as db:
            config = db[self.active]
            config[key] = value
            db[self.active] = config

    def __getitem__(self, key):
        with shelve.open(config_path()) as db:
            config = db[self.active]
        if key not in config:
            self[key] = self.default
            return self.default
        return config[key]

    def print_preset(self):
        with shelve.open(config_path()) as db:
            config = db[self.active]
        formatted = pprint.pformat({
            k: v.get_config() if k == "OPTIMIZER" and v is not None else v
            for k, v in config.items() if "__" not in k and k.isupper()
        })
        return formatted

    def reset(self):
        with shelve.open(config_path()) as db:
            db[self.active] = {}

