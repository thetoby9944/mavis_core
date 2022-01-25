from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model

from db import ActivePresetDAO, PresetListDAO, LogPathDAO, LogDAO
from v2.config import MLConfig
from v2.ml.train import TrainingHandler
from v2.ui.processors.base import BaseProcessor


class TfModelProcessor(BaseProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.continue_training = False
        self.inference_after_training = False
        self.dataset = None
        self.presets = []
        self.multiprocessing = True
        self.save_weights_only = False
        self.dry_run = False

        self.config: Optional[MLConfig] = None

    def model_from_path(self, model_path=None):
        if model_path is None:
            model_path = self.config.MODEL.MODEL_PATH

        return load_model(model_path, compile=False)

    def inference(self, img_paths):
        model = self.model_from_path()

        ds = self.dataset.create(img_paths, None)
        n = len(img_paths)

        def prog_perc(x):
            return min(1., x / (n // self.config.TRAIN.BATCH_SIZE - 1)) if n > self.config.TRAIN.BATCH_SIZE else 1.

        bar = st.progress(0)
        for i, batch in enumerate(ds):
            bar.progress(prog_perc(i))
            print("predicting on batch", i)
            preds = model.predict_on_batch(x=batch)
            print("returning batch", i)
            for pred in preds:
                if isinstance(pred, np.ndarray):
                    yield pred
                if isinstance(pred, tf.Tensor):
                    yield pred.numpy()

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
        final_model, final_loss = None, np.inf
        for preset in self.presets:
            ActivePresetDAO().set(preset)

            st.info(f"Loaded preset {ActivePresetDAO().get()}")
            # Create a MirroredStrategy.
            strategy = tf.distribute.MirroredStrategy()
            scope = strategy.scope()
            st.write("Loaded Mirrored Strategy")
            st.write(scope)
            st.write(f'Number of replicas in sync: `{strategy.num_replicas_in_sync}`')
            st.code(tf.config.get_visible_devices())
            try:
                for device in tf.config.list_logical_devices("GPU"):
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
                    if base_model:
                        model = self.model_from_path(base_model)
                    else:
                        model = self.models()[0]

                    # Name the model (not relevant for mask rcnn)
                    model._name = ActivePresetDAO().get() + model._name
                    with st.expander("Settings"):
                        st.write(self.config.dict())
                    # First Compile the model, so that all variables for dataset creation have been initialized
                    model.compile(**self.config.compile_args())

            ### end strategy
            self.dataset.create(*self.input_args())
            self.dataset.peek()

            if self.dry_run:
                continue

            model, loss = self.train_keras(model, self.dataset.ds, self.dataset.val_ds)

            # Calculate if the model is better or if its the first iteration
            if final_loss == np.inf or loss < final_loss:
                final_model = model
                st.success(
                    f"Overall loss improved from {final_loss} to {loss}. "
                    f"Keeping {model.name}"
                )
                final_loss = loss

            # Every iteration save the best model only
            if self.save_weights_only:
                final_model.save_weights(self.config.MODEL.MODEL_PATH)
            else:
                final_model.save(self.config.MODEL.MODEL_PATH)

        return final_model

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

    def tensorboard_info(self):
        import socket
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
            f"View tensorboard:  \n "
            f"[http://{get_ip()}:6006](http://{get_ip()}:6006)  \n"
            f"[http://127.0.0.1:6006](http://127.0.0.1:6006)"
        )

    def training_block(self):
        preset_names = st.multiselect(
            "Run Training on these Presets",
            list(PresetListDAO().get_all()),
            self.presets
        )

        self.continue_training = self.config.TRAIN.CONTINUE_TRAINING
        # self.inference_after_training = st.checkbox("Run inference after Training", False)
        st.write(
            "Try a dry-run with checking the preset "
            "to see the data augmentation and whether the model compiles."
        )
        self.dry_run = st.button("Check preset")

        if st.button("Start Training") or self.dry_run:
            self.presets = preset_names
            if len(self.presets) is 0:
                st.warning("No presets selected!")
                return
            self.tensorboard_info()
            if not self.dry_run:
                preset = self.config.dict()
                LogDAO(self.input_columns, self.column_out, preset).add("Training")
                LogDAO(self.input_columns, self.column_out, preset).add("Dry Run")
            df = self.training_store()
            self.save_new_df(df)

    def run(self):
        st.write("--- \n ### Inference")
        # self.inference_parameter()
        self.preview_block(expanded=False)
        self.inference_block()
        st.write("--- \n ### Training")
        self.training_block()

    def store_preds(self, preds, df) -> pd.DataFrame:
        raise NotImplementedError

    def models(self) -> list:  # list of (model:tf.keras.models.Model, name:str)
        raise NotImplementedError

    def compile_args(self) -> dict:
        return {}
