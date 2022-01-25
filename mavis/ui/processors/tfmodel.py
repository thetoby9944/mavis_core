from contextlib import nullcontext

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model

import config
from db import ConfigDAO, ActivePresetDAO, PresetListDAO, LogPathDAO, LogDAO
from ml.train import train_model
from ui.processors.base import BaseProcessor


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

    def model_from_path(self, model_path=None):
        if model_path is None:
            model_path = ConfigDAO()["MODEL_PATH"]

        return load_model(model_path, compile=False)

    def inference(self, img_paths):
        model = self.model_from_path()

        ds = self.dataset.create(img_paths, None)
        n = len(img_paths)

        def prog_perc(x):
            return min(1., x / (n // ConfigDAO()["BATCH_SIZE"] - 1)) if n > ConfigDAO()["BATCH_SIZE"] else 1.

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
        return train_model(
            ds, model, val_ds,
            multiprocessing=self.multiprocessing,
            save_weights_only=self.save_weights_only
        )

    def training(self):
        final_model, final_loss = None, np.inf
        for preset in self.presets:
            ActivePresetDAO().set(preset)

            st.info(f"Loaded preset {ActivePresetDAO().get()}")
            # Create a MirroredStrategy.
            strategy = tf.distribute.MirroredStrategy()
            scope = (
                strategy.scope()
                if ConfigDAO()["ARCHITECTURE"] not in 'Reconstruction'
                else nullcontext()
            )
            st.write(scope)
            print(f'Number of replicas in sync: {strategy.num_replicas_in_sync}')

            # Open a strategy scope in case its not a GAN
            with scope:
                # Everything that creates variables should be under the strategy scope.
                # In general this is only model construction & `compile()`.

                base_model = ConfigDAO()["CONTINUE_TRAINING"]
                if base_model:
                    model = self.model_from_path(base_model)
                else:
                    model = self.models()[0]

                # Name the model (not relevant for mask rcnn)
                model._name = ActivePresetDAO().get() + model._name
                st.code(ConfigDAO().print_preset())

                # First Compile the model, so that all variables for dataset creation have been initialized
                st.write("compiling model")
                with st.spinner("Compiling Model"):
                    model.compile(**self.compile_args())

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
                final_model.save_weights(ConfigDAO()["MODEL_PATH"])
            else:
                final_model.save(ConfigDAO()["MODEL_PATH"])

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

    def training_block(self):
        preset_names = st.multiselect(
            "Run Training on these Presets",
            list(PresetListDAO().get_all()),
            self.presets
        )

        self.continue_training = ConfigDAO()["CONTINUE_TRAINING"]
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

            if not self.dry_run:
                LogDAO(self.input_columns, self.column_out).add("Training")
                st.markdown(f"Started Training. Run tensorboard to see progress.")
                st.code(f"tensorboard --logdir {LogPathDAO().get().resolve()} --bind_all serve")
                LogDAO(self.input_columns, self.column_out).add("Dry Run")
            df = self.training_store()
            self.save_new_df(df)

    def run(self):
        config.Preset().common_model_parameter()
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
