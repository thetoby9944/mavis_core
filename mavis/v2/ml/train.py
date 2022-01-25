import io
import json
import os
import re
from datetime import datetime
from pathlib import Path

import streamlit as st
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow_addons.optimizers import CyclicalLearningRate

from db import ProjectDAO, LogPathDAO, ActivePresetDAO
from v2.config import MLConfig
from v2.ml.callbacks import CheckPoint, ReduceCyclicalLROnPlateau


class TrainingHandler:
    def __init__(
            self,
            config: MLConfig,
            model: tf.keras.models.Model,
            train_data: tf.data.Dataset,
            validation_data: tf.data.Dataset,
            multiprocessing=True,
            save_weights_only=False
    ):
        self.train_data = train_data
        self.model = model
        self.validation_data = validation_data
        self.multiprocessing = multiprocessing
        self.save_weights_only = save_weights_only
        self.log_dir = self.create_log_dir()
        self.config = config

    def train_model(self) -> (tf.keras.Model, float):
        model = self.model
        validation_data = self.validation_data
        train_data = self.train_data
        config = self.config

        try:
            self.summary()
            self.remember_training_in_tensorboard()
        except:
            pass

        tensorboard = TensorBoard(
            log_dir=self.log_dir,
            write_images=True,
            histogram_freq=1,
            profile_batch=0
        )

        metric = "val_loss" if validation_data else "loss"
        checkpoint_path = config.MODEL.MODEL_PATH
        model_checkpoint = CheckPoint(
            model_path=checkpoint_path,
            save_weights_only=self.save_weights_only,
            metric=metric
        )

        early_stopping = EarlyStopping(
            monitor=metric,
            min_delta=0,
            patience=config.TRAIN.EARLY_STOPPING,
            verbose=0,
            mode='auto',
            restore_best_weights=False
        )

        cbs = [
            tensorboard,
            model_checkpoint,
            early_stopping,
        ]

        if config.TRAIN.RED_ON_PLATEAU_PATIENCE:
            if isinstance(model.optimizer.lr, CyclicalLearningRate):
                lr_cb = ReduceCyclicalLROnPlateau
            else:
                lr_cb = ReduceLROnPlateau

            reduce_lr_on_plateau = lr_cb(
                monitor="loss",
                patience=config.TRAIN.RED_ON_PLATEAU_PATIENCE,
                min_delta=0.01,
                factor=0.5,
                verbose=1,
            )
            cbs.append(reduce_lr_on_plateau)

        workers = (
            0
            if os.name == 'nt' else
            max(os.cpu_count() - 1, 1)
        )

        model.fit(
            train_data,
            epochs=config.TRAIN.EPOCHS,
            steps_per_epoch=config.TRAIN.STEPS_PER_EPOCH,
            callbacks=cbs,
            use_multiprocessing=self.multiprocessing,
            workers=workers,
            validation_data=validation_data,
            validation_steps=config.TRAIN.VAL_SPLIT,
            max_queue_size=workers * 2
        )

        model.load_weights(checkpoint_path)
        st.info("Evaluating")
        ev = model.evaluate(
            validation_data if validation_data else train_data,
            steps=config.TRAIN.VAL_SPLIT
        )

        if "accuracy" in model.metrics_names:
            loss = 1 - ev[model.metrics_names.index("accuracy")]
        elif "iou_score" in model.metrics_names:
            loss = 1 - ev[model.metrics_names.index("iou_score")]
        elif "rpn_bbox_loss" in model.metrics_names:
            loss = ev[model.metrics_names.index("rpn_bbox_loss")]
        elif isinstance(ev, list) and len(ev) > 0:
            loss = ev[0]
        elif isinstance(ev, float):
            loss = ev
        else:
            st.warning(
                "No evaluation possible, "
                "please specify a evaluation metric during compile"
            )
            loss = 0

        return model, loss

    def summary(self):
        stream = io.StringIO()
        self.model.summary(
            print_fn=lambda x: stream.write(x + '\n'),
            line_length=120
        )
        summary_string = stream.getvalue()
        stream.close()

        with st.expander("Model Info"):
            st.code(summary_string)
            st.info(f"Train Data Shapes {self.train_data}")
            st.info(f"Validation Data Shapes {self.validation_data}")

        return summary_string

    def create_log_dir(self):
        log_dir = (
            Path(LogPathDAO().get()) /
            "tensorboard" /
            Path(ProjectDAO().get()).stem /
            f'{datetime.now().strftime("%Y%m%d-%H%M%S")}-{self.model.name}'
        )
        st.info(f"Logging directory is:")
        st.code(str(log_dir))
        log_dir.mkdir(exist_ok=True, parents=True)
        return str(log_dir)

    def remember_training_in_tensorboard(self):
        log_dir = self.log_dir
        config = self.config

        with tf.summary.create_file_writer(log_dir).as_default():
            description = "\n".join([
                f"```{line}```  "
                for line in re.split("\\n|<\\?br>", json.dumps(config.json()))
            ])

            val_batches = self.validation_data.take(config.TRAIN.VAL_SPLIT)
            for i, (image_batch, label_batch) in enumerate(val_batches):
                tf.summary.image(
                    f"Validation Image Data",
                    image_batch,
                    max_outputs=config.TRAIN.BATCH_SIZE,
                    step=0,
                    description=f"##Active Preset\n###{ActivePresetDAO().get()}\n\n{description}"
                )

                if hasattr(config.TRAIN, "BINARY") and config.TRAIN.BINARY:
                    tf.expand_dims(label_batch, -1)

                tf.summary.image(
                    f"Validation Label Data",
                    label_batch,
                    max_outputs=config.TRAIN.BATCH_SIZE,
                    step=0
                )
