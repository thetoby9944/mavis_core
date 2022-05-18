import json
import shelve
from abc import ABC
from pathlib import Path
from typing import Any, Union, List, Literal

import streamlit as st

from keras.optimizer_v2.optimizer_v2 import OptimizerV2

from pydantic import BaseModel, ValidationError, Field

ACTIVE_PRESET = "Default"
CONFIG_DB_PATH = "config.db"


class ListView:
    def __init__(self, options: List[str], default: List[str], label):
        self.label = label
        self.options = options

        new_defaults = st.session_state.get(f"{label}_defaults_ListView", "") != default
        if new_defaults:
            self.default = st.session_state[f"{label}_defaults_ListView"] = default
        else:
            self.default = st.session_state.get(f"{label}_current_ListView", None)
            if self.default is None:
                self.default = default

    def _grid_layout(self, n_rows, column_spec):
        grid_layout = [
            [
                column.empty()
                for column in st.columns(column_spec)
            ]
            for _ in range(n_rows)
        ]
        return grid_layout

    def _option_grid(self, default_options):
        new_list = default_options.copy()
        for i, option in enumerate(default_options):
            option_col, move_down_col, move_up_col, del_col = st.columns((3,1,1,1))
            with option_col:
                st.code(f"{i+1}. {option}")
            with move_down_col:
                if i < (len(default_options) - 1) and st.form_submit_button(f"▼ {i+1}. "):#, key=f"{option}{i}down"):
                    new_list[i + 1], new_list[i] = default_options[i], default_options[i + 1]
            with move_up_col:
                if i > 0 and st.form_submit_button(f"▲ {i+1}. "):#, key=f"{option}{i}up"):
                    new_list[i], new_list[i - 1] = default_options[i - 1], default_options[i]
            with del_col:
                if st.form_submit_button(f"🞭 {i+1}. "):#, key=f"{option}{i}delete"):
                    del new_list[i]
        return new_list

    def selection(self):
        new_options = st.multiselect(
            self.label,
            self.options,
        )

        if st.form_submit_button(f"Update {self.label}"):
            self.default.extend(new_options)

        new_list = self._option_grid(
            default_options=self.default
        )
        st.session_state[f"{self.label}_current_ListView"] = new_list

        if new_list != self.default:
            st.experimental_rerun()

        return new_list


class ConfigDAO:
    def __init__(self, default=None):
        self.default = default
        self.active = ACTIVE_PRESET
        with shelve.open(CONFIG_DB_PATH) as db:
            if self.active not in db:
                db[self.active] = {}

    def __setitem__(self, key, value):
        with shelve.open(CONFIG_DB_PATH) as db:
            config = db[self.active]
            config[key] = value
            db[self.active] = config

    def __getitem__(self, key):
        with shelve.open(CONFIG_DB_PATH) as db:
            config = db[self.active]
        if key not in config:
            self[key] = self.default
            return self.default
        return config[key]

    def reset(self):
        with shelve.open(CONFIG_DB_PATH) as db:
            db[self.active] = {}



class BaseProperty(BaseModel, ABC):
    class Config:
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True
        json_encoders = {
            OptimizerV2: lambda o: json.dumps(o.get_config()),
            Path: lambda p: str(p)
        }

    _name = ""

    @property
    def st(self):
        """
        Import streamlit locally to avoid errors
        when accessing properties without a streamlit context
        """
        import streamlit as st
        return st.empty()

    def parameter_block(self):
        """Shows parameters for the property"""
        raise NotImplementedError

    def get(self):
        """Construct the underlying object from the properties, if any"""
        pass


class BaseConfig(BaseProperty):
    """
    This model will try to initialize itself from the database upon every instantiation,
    based on the config_key property

    It further supplies an update function, that allows to write back any changes into the database, under its key.

    When to model fails to initialize, it will be restored from default parameters or will take any passed
    arguments to initialize itself.
    """

    @property
    def config_key(self):
        return self.__class__.__name__

    def __init__(self, **data: Any):
        """
        Use a key DAO to load a dict that can initialize the model
        :param data: keyword data to initialize the model, takes precedence over
        """
        try:
            super().__init__(**ConfigDAO({})[self.config_key], **data)
        # In case model gets initialized with outside data or the database data is inconsistent
        except (ValidationError, TypeError) as e:
            print(f"FAILED to initialize Model from {self.config_key}. Restoring from {data or 'default'}")
            super().__init__(**data)
            self.update()
            # raise e

    def update(self):
        """
        Use a key DAO to persist the dict that represents the model
        :return: None
        """
        ConfigDAO()[self.config_key] = self.dict()

    def get(self):
        pass


class PropertyContainer(BaseProperty):
    ACTIVE: str = ""

    def parameter_block(self):
        options = list(self.all.keys())

        self.ACTIVE = self.st.selectbox(
            self._name, options,
            options.index(self.ACTIVE)
            if self.ACTIVE in options
            else 0
        )
        self.st.form_submit_button(f"Update {self._name}")

        self.all[self.ACTIVE].parameter_block()

    def get(self):
        return self.all[self.ACTIVE].get()

    def dict(self, *args, **kwargs):
        exclude = {
            attr
            for attr, value in self.__dict__.items()
            if isinstance(value, BaseProperty) and self.ACTIVE is not value._name
        }
        return super().dict(exclude=exclude)

    @property
    def all(self):
        return {
            value._name: value
            for attr, value in self.__dict__.items()
            if isinstance(value, BaseProperty)
        }


class PropertiesContainerProperty(BaseProperty):
    name: Literal[""] = ""
    i: int = 0

    def __init__(self, i, *args, **kwargs):
        super().__init__(**kwargs)
        self.i = i

    def parameter_block(self):
        st.write(f"### Step {self.i + 1}: {self.name}")

    def reindex(self, i, inplace=True):
        assert inplace, "Only inplace supported"
        self.i = i
        return self


class PropertiesContainer(BaseConfig, ABC):
    ACTIVE: List[Union[
        PropertiesContainerProperty
    ]] = Field([], discriminator="name")
    n_properties: int = 0
    name: str = ""

    @property
    def _parent_property_cls(self):
        raise NotImplementedError

    def parameter_block(self):
        options = list(self.all().keys())
        selected = ListView(
            options,
            default=[item.name for item in self.ACTIVE],
            label="Options"
        ).selection()

        self.n_properties = len(selected)

        existing_keys = [
            element.name
            for element in self.ACTIVE
        ]

        active_properties_list = []
        for i, key in enumerate(selected):

            if key in existing_keys:
                index = existing_keys.index(key)
                existing_keys.pop(index)
                active_property = self.ACTIVE.pop(index)
                active_property.reindex(i, inplace=True)
            else:
                active_property = self.all()[key](i)

            active_properties_list.append(
                active_property
            )

        self.ACTIVE = active_properties_list
        for active_property in self.ACTIVE:
            active_property.parameter_block()

    def all(self):
        return {
            subclass(0).name: subclass
            for subclass in self._parent_property_cls.__subclasses__()
        }

