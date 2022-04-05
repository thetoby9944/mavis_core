from abc import ABC
from contextlib import nullcontext
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from keras import Model
from tensorflow.keras.models import load_model

from mavis.db import ActivePresetDAO, LogPathDAO, LogDAO
from mavis.config import MLConfig
from mavis.ml.dataset.base import TFDatasetWrapper
from mavis.ml.train import TrainingHandler
from mavis.ui.processors.base import BaseProcessor


class TfModelProcessor(BaseProcessor, ABC):
    config: MLConfig = None  # Abstract Field
    dataset: TFDatasetWrapper = None  # Abstract Field

    def __init__(self, *args, **kwargs):
        self.dataset.config = self.config
        super().__init__(*args, **kwargs)
        self.continue_training = False
        self.inference_after_training = False
        # self.presets = []
        self.multiprocessing = True
        self.save_weights_only = False
        self.dry_run = False

    def model_from_path(self, model_path=None):
        if model_path is None:
            model_path = self.config.MODEL.MODEL_PATH

        return load_model(model_path, compile=False)

    def prog_perc(self, n, i):
        return min(1., i / (n // self.config.TRAIN.BATCH_SIZE - 1)) if n > self.config.TRAIN.BATCH_SIZE else 1.

    def inference(self, img_paths):
        model = self.model_from_path()

        data_generator = self.dataset
        data_generator.create(img_paths, None)
        n = len(img_paths)

        bar = st.progress(0)
        for batch_i, batch in enumerate(data_generator.ds):
            bar.progress(self.prog_perc(n, batch_i))
            print("predicting on batch", batch_i)
            predictions = model.predict_on_batch(x=batch)
            print("returning batch", batch_i)
            for pred_i, pred in enumerate(predictions):
                # Convert to numpy array
                if isinstance(pred, np.ndarray):
                    pass
                if isinstance(pred, tf.Tensor):
                    pred = pred.numpy()

                # Crop potential padding
                image_i = batch_i * self.config.TRAIN.BATCH_SIZE + pred_i
                image_path = img_paths[image_i]
                original_image: Image.Image = Image.open(image_path)
                original_w, original_h = original_image.width, original_image.height

                yield pred[0:original_h, 0:original_w, ...]

    def preview(self, n):
        img_path = self.input_args(dropna_jointly=False)[0][n]
        col1, col2 = st.columns(2)
        col1.image(Image.open(img_path))
        if st.button("Preview"):
            preds = self.inference([img_path])
            if preds is not None:
                pred = next(preds)
                with col2:
                    self.dataset.display_pred(pred)

    def train_keras(self, model, ds, val_ds):
        return TrainingHandler(
            config=self.config,
            model=model,
            train_data=ds,
            validation_data=val_ds,
            multiprocessing=self.multiprocessing,
            save_weights_only=self.save_weights_only
        ).train_model()

    def training(self):
        st.info(f"Training on preset {ActivePresetDAO().get()}")
        # Create a MirroredStrategy.
        try:
            strategy = tf.distribute.MirroredStrategy()
            scope = strategy.scope()
            st.write("Loaded Mirrored Strategy")
            st.write(scope)
            st.write(f'Number of replicas in sync: `{strategy.num_replicas_in_sync}`')
            st.code(tf.config.get_visible_devices())
        except:
            st.warning("Failed to load MirroredStrategy")
            scope = nullcontext()

        try:
            st.write(
                "Memory Info for GPUs:  \n"
                "- `'current'`: The current memory used by the device, in bytes.  \n"
                "- `'peak'`: The peak memory used by the device across the run of the program, in bytes."
            )
            for device in tf.config.list_logical_devices("GPU"):
                st.write(device.name)
                st.code(tf.config.experimental.get_memory_info(device.name))
        except Exception as e:
            st.warning("Cannot display current memory info. Install tf>=2.5 for memory logging")
            with st.expander("Details"):
                st.exception(e)
        # Open a strategy scope in case its not a GAN
        with st.spinner("Compiling Model"):
            with scope:
                # Everything that creates variables should be under the strategy scope.
                # In general this is only model construction & `compile()`.

                base_model = self.config.TRAIN.CONTINUE_TRAINING
                if Path(base_model).is_file():
                    model = self.model_from_path(base_model)
                    st.info("Successfully loaded base model from:")
                    st.code(str(base_model))
                else:
                    model = self.models()[0]

                # Name the model (not relevant for mask rcnn)
                model._name = ActivePresetDAO().get() + model._name
                with st.expander("Settings"):
                    st.write(self.config.dict())
                # First Compile the model, so that all variables for dataset creation have been initialized
                model.compile(**self.config.compile_args())
        # end mirrored strategy scope

        data_generator = self.dataset
        data_generator.create(*self.input_args())
        data_generator.peek()

        if self.dry_run:
            return

        model, loss = self.train_keras(
            model,
            data_generator.ds,
            data_generator.val_ds
        )

        st.success(
            f"Loss improved to {loss}. "
            f"Keeping {model.name}"
        )

        if self.save_weights_only:
            model.save_weights(self.config.MODEL.MODEL_PATH)
        else:
            model.save(self.config.MODEL.MODEL_PATH)

    def training_store(self):
        self.training()
        if self.inference_after_training:
            st.info("Running Inference on Training Data once")
            self.df = self.inference_store()
        return self.df

    def inference_store(self):
        preds = self.inference(self.input_args(dropna_jointly=False)[0])
        self.df[self.column_out] = np.nan
        return self.store_preds(preds, self.df)

    def inference_block(self):
        if st.button("Inference"):
            LogDAO(self.input_columns, self.column_out).add("Inference")
            df = self.inference_store()
            self.save_new_df(df)

        model_path = Path(self.config.MODEL.MODEL_PATH)
        if model_path.is_file() and st.form_submit_button(f"Get download link"):
            model: Model = load_model(model_path, compile=False)
            new_path = model_path.parent / (model_path.stem + "no_optimizer.h5")
            model.save(new_path, include_optimizer=False)
            with open(new_path, "rb") as model_file:
                st.download_button(
                    label="Download Model",
                    data=model_file,
                    file_name=model_path.name,
                )

    def tensorboard_info(self):
        import socket
        st.write("Mavis connected and started tensorboard for you with:")
        st.code(f"tensorboard --logdir {LogPathDAO().get().resolve()} --bind_all serve")

        def get_ip():
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                # doesn't even have to be reachable
                s.connect(('10.255.255.255', 1))
                IP = s.getsockname()[0]
            except Exception:
                IP = '127.0.0.1'
            finally:
                s.close()
            return IP

        st.markdown(
            f"View at:  \n "
            f"[http://{get_ip()}:6006](http://{get_ip()}:6006)  \n"
            f"[http://127.0.0.1:6006](http://127.0.0.1:6006)"
        )

    def training_block(self):
        #preset_names = st.multiselect(
        #    "Run Training on these Presets",
        #    list(PresetListDAO().get_all()),
        #    self.presets
        #)

        self.continue_training = self.config.TRAIN.CONTINUE_TRAINING
        # self.inference_after_training = st.checkbox("Run inference after Training", False)

        self.dry_run = st.button(
            "Check Settings",
            help = "Run a dry-run. It will check the preset and *peek the dataset*. "
            "This allows you to see your data augmentation and whether the model compiles."
        )

        if st.button("Start Training") or self.dry_run:
            # self.presets = preset_names
            # if len(self.presets) == 0:
            #    st.warning("No presets selected!")
            #    return
            self.tensorboard_info()
            preset = self.config.dict()
            if not self.dry_run:
                LogDAO(self.input_columns, self.column_out, preset).add("Training")
            else:
                LogDAO(self.input_columns, self.column_out, preset).add("Dry Run")
            df = self.training_store()
            self.save_new_df(df)

    def core(self):
        st.write("--- \n ### Inference")
        # self.inference_parameter()
        self.preview_block(expanded=False)
        self.inference_block()
        st.write("--- \n ### Training")
        self.training_block()

    def store_preds(self, preds, df) -> pd.DataFrame:
        raise NotImplementedError

    def models(self) -> List[Model]:
        return [self.model()]

    def model(self) -> Model:  # list of (model:tf.keras.models.Model, name:str)
        raise NotImplementedError
