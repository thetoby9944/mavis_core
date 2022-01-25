import io
import json
import zipfile
from abc import ABC
from datetime import datetime
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any

import h5py
import streamlit as st
import tensorflow as tf
from keras.optimizer_v2.optimizer_v2 import OptimizerV2

from pydantic import BaseModel, ValidationError

from db import ConfigDAO, ModelDAO, ActivePresetDAO, PresetListDAO


class ExtendedEnum(Enum):
    """
    Allows to access the enums values as a list
    """

    @classmethod
    def values(cls):
        return list(map(lambda c: c.value, cls))


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


class BaseConfig(BaseProperty, ABC):
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


class PresetHandler(ABC):

    @staticmethod
    def export_preset(file_name: str):
        st.write("###### Export Preset")

        from v2.config import  MLConfig

        zip_model = st.checkbox("Zip model")
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for subclass in BaseConfig.__subclasses__():
                zip_file.writestr(f"{subclass.__name__}.json", subclass().json())
            model_path = Path(MLConfig().MODEL.MODEL_PATH)
            if zip_model and model_path.is_file():
                zip_file.write(model_path, model_path.name)

        st.download_button(
            "Export Preset",
            zip_buffer.getvalue(),
            file_name
        )


    @staticmethod
    def import_preset():
        st.write("###### Import Preset")
        preset_config_file = st.file_uploader("Import from file", type="zip")
        if not preset_config_file:
            return

        if st.button("Update preset"):
            with zipfile.ZipFile(preset_config_file) as the_zip:
                PresetHandler._handle_preset_zip(the_zip)

    @staticmethod
    def _handle_preset_zip(
            zip_file: zipfile.ZipFile
    ):
        model_path = None
        for zip_info in zip_file.infolist():
            with zip_file.open(zip_info) as archive_fp:
                if zip_info.filename.endswith(".h5"):
                    st.info("Reading .h5 file")
                    with h5py.File(archive_fp, 'r') as h5_file:
                        st.info("Loading model")
                        model = tf.keras.models.load_model(h5_file, compile=False)
                        model_name = zip_info.filename

                    st.info("Saving model")
                    model_path = ModelDAO().save(model, model_name)

        for zip_info in zip_file.infolist():
            with zip_file.open(zip_info) as archive_fp:
                for subclass in BaseConfig.__subclasses__():
                    if zip_info.filename == f"{subclass.__name__}.json":
                        st.info(f"Parsing {subclass.__name__}")
                        config: BaseConfig = subclass.parse_obj(json.load(archive_fp))

                        if model_path is not None and hasattr(config, "MODEL"):
                            config.MODEL.MODEL_PATH = model_path

                        config.update()

    @staticmethod
    def access(name):
        def wrapper(fn):
            @wraps(fn)
            def access(self, *args, **kwargs):
                res = None
                # try:
                with st.expander(name):
                    fn(self, *args, **kwargs)
                    if st.checkbox("Show summary as JSON"):
                        st.write(self.__class__.__name__)
                        st.write(json.loads(self.json()))

                    st.write("---")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        PresetHandler._update(self, name)
                    with col2:
                        PresetHandler.import_preset()
                    with col3:
                        PresetHandler.export_preset(
                        file_name=f"{ActivePresetDAO().get()}.zip"
                    )

                # except Exception as e:
                #    st.error(f"Preset raised a message in **{name}**.")
                #    st.code(traceback.format_exc())
                return res

            return access

        return wrapper

    @staticmethod
    def select():
        current = ActivePresetDAO().get()
        all_preset_names = PresetListDAO().get_all()
        selection = st.selectbox(
            "Select Preset to configure",
            all_preset_names,
            all_preset_names.index(current)
            if current in all_preset_names else 0,
            key="sel_pres",
            help="**Presets are persisted on a per project basis.**  \n"
                 " - The name of the preset is the identifying property per project. "
                 "I.e. each project has its own 'Default' preset. \n"
                 " - Changing the default preset for one project will *not* cause changes for another project. \n "
                 " - A newly created preset will only be available in the project it has been created in. \n "
                 " - To pass presets between project use the import / export functionality."
        )
        if current != selection:
            PresetHandler._set(selection)

    @staticmethod
    def _set(new_config):
        ActivePresetDAO().set(new_config)
        PresetListDAO().add(new_config)
        # st.experimental_rerun()
        st.success(f"Preset is: {new_config}. Press **`R`** for refresh.")
        st.button("Refresh")

    @staticmethod
    def _update(self: BaseConfig, key_name):
        current = ActivePresetDAO().get()
        btn = st.empty()

        if st.checkbox("New", key=f"custom_model{key_name}"):
            new_name = f"{datetime.now():%y%m%d_%H-%M}"
            if st.checkbox("Use custom name"):
                new_name = st.text_input("Preset name", key=f"custom_model_name_{key_name}")
                if not new_name:
                    st.warning("Please specify a name")

            if btn.button(f"🞥 Save as new preset: {new_name}", key="sav_new" + key_name):
                PresetHandler._set(new_name)
                self.update()

        elif btn.button(f"⭯ Update preset: {current}", key="upd_nam" + key_name):
            PresetHandler._set(current)
            self.update()