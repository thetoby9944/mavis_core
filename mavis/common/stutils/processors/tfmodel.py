import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from PIL import Image
from tensorflow.python.keras.models import load_model

from mavis import config
from mavis.shelveutils import load_presets
from mavis.stutils.processors.base import BaseProcessor
from mavis.tfutils.train import train_model


class TfModelProcessor(BaseProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.continue_training = False
        self.inference_after_training = False
        self.dataset = None
        self.presets = []
        self.multiprocessing = True
        self.save_weights_only = False

    def model_from_path(self):
        return load_model(config.c.MODEL_PATH, compile=False)

    def inference(self, img_paths):
        model = self.model_from_path()

        ds = self.dataset.create(img_paths, None)
        n = len(img_paths)

        def prog_perc(x):
            return min(1., x / (n // config.c.BATCH_SIZE - 1)) if n > 1 else 1

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
        st.image(Image.open(img_path))
        if st.button("Preview"):
            preds = self.inference([img_path])
            if preds is not None:
                pred = next(preds)
                self.dataset.display_pred(pred)

    def train_keras(self, model, ds, val_ds):
        return train_model(ds, model, val_ds,
                           multiprocessing=self.multiprocessing,
                           save_weights_only=self.save_weights_only)

    def training(self):
        final_model, final_loss = None, np.inf
        for preset in self.presets:
            config.c = preset

            st.info(f"Training with preset {config.c.name}")

            if preset.MODEL_PATH is not None and self.continue_training:
                model = self.model_from_path()
            else:
                model = self.models()[0]

            # Name the model (not relevant for mask rcnn)
            model._name = preset.name + model._name
            st.code(config.c.print_preset())

            # First Compile the model, so that all variables for dataset creation have been initialized
            with st.spinner("Compiling Model"):
                model.compile(**self.compile_args())

            ds, val_ds = self.dataset.create(*self.input_args())
            self.dataset.peek()

            model, loss = self.train_keras(model, ds, val_ds)

            # Calculate if the model is better or if its the first iteration
            if final_loss == np.inf or loss < final_loss:
                final_model = model
                st.success(f"Overall loss improved from {final_loss} to {loss}. "
                           f"Keeping {model.name}")
                final_loss = loss

            # Every iteration save the best model only
            if self.save_weights_only:
                final_model.save_weights(config.c.MODEL_PATH)
            else:
                final_model.save(config.c.MODEL_PATH)

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
            df = self.inference_store()
            self.save_new_df(df)

    def training_block(self):
        preset_names = st.multiselect(
            "Run Training on these Presets",
            list(load_presets().keys()),
            [preset.name for preset in self.presets]
        )

        # self.continue_training = st.checkbox("Continue Training", False)
        self.inference_after_training = st.checkbox("Run inference after Training", False)

        if st.button("Start Training"):
            presets = load_presets()
            self.presets = [presets[k] for k in preset_names]
            st.markdown(f"Started Training. "
                        f"See [tensorboard](http://141.18.61.112:6006) for progress.")
            df = self.training_store()
            self.save_new_df(df)

    def run(self):
        config.c.common_model_parameter()
        st.write("--- \n ### Inference")
        # self.inference_parameter()
        self.preview_block(expanded=False)
        self.inference_block()
        st.write("--- \n ### Training")
        self.training_parameter()
        self.training_block()

    def store_preds(self, preds, df) -> pd.DataFrame:
        raise NotImplementedError

    def models(self) -> list:  # list of (model:tf.keras.models.Model, name:str)
        raise NotImplementedError

    def compile_args(self) -> dict:
        return {}

    def training_parameter(self):
        raise NotImplementedError

    def inference_parameter(self):
        raise NotImplementedError
