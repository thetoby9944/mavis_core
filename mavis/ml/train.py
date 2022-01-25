import io
import os
import re
from datetime import datetime
from pathlib import Path

import streamlit as st
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from db import ProjectDAO, LogPathDAO, ConfigDAO, ActivePresetDAO
from ml.callbacks import CheckPoint


def get_model_summary(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'), line_length=120)
    summary_string = stream.getvalue()
    stream.close()
    return summary_string


def train_model(
        train_data: tf.data.Dataset,
        model: tf.keras.models.Model,
        validation_data: tf.data.Dataset,
        multiprocessing=True,
        save_weights_only=False
) -> (tf.keras.Model, float):
    # local import to avoid circular

    log_dir = str(
        Path(LogPathDAO().get()) /
        "tensorboard" /
        ProjectDAO().get() /
        f'{datetime.now().strftime("%Y%m%d-%H%M%S")}-{model.name}'
    )
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    with tf.summary.create_file_writer(log_dir).as_default():
        description = "\n".join([f"```{line}```  " for line in re.split("\\n|<\\?br>", ConfigDAO().print_preset())])
        try:
            # tf.summary.text("Configuration", data="", description=description)
            for i, (image_batch, label_batch) in enumerate(validation_data.take(ConfigDAO()["VAL_SPLIT"])):
                tf.summary.image(f"Validation Image Data", image_batch, max_outputs=ConfigDAO()["BATCH_SIZE"], step=0,
                                 description=f"##Active Preset\n###{ActivePresetDAO().get()}\n\n{description}")
                if ConfigDAO()["BINARY"]:
                    tf.expand_dims(label_batch, -1)
                tf.summary.image(f"Validation Label Data", label_batch, max_outputs=ConfigDAO()["BATCH_SIZE"], step=0)
        except:
            pass

    metric = "val_loss" if validation_data else "loss"

    checkpoint_path = ConfigDAO()["MODEL_PATH"]
    model_checkpoint = CheckPoint(
        model_path=checkpoint_path,
        save_weights_only=save_weights_only,
        metric=metric
    )

    early_stopping = EarlyStopping(
        monitor=metric,
        min_delta=0,
        patience=ConfigDAO()["EARLY_STOPPING"],
        verbose=0,
        mode='auto',
        restore_best_weights=False
    )

    reduce_lr_on_plateau = ReduceLROnPlateau(
        monitor="loss",
        patience=ConfigDAO()["RED_ON_PLATEAU_PATIENCE"],
        min_delta=0.01,
        factor=0.5,
        verbose=1,
    )

    cbs = [tensorboard, model_checkpoint, early_stopping, reduce_lr_on_plateau]
    workers = 0 if os.name == 'nt' else max(os.cpu_count() - 1, 1)

    with st.expander("Model Info"):
        st.code(get_model_summary(model))
        st.info(f"Train Data Shapes {train_data}")
        st.info(f"Validation Data Shapes {validation_data}")

    model.fit(
        train_data,
        epochs=ConfigDAO()["EPOCHS"],
        steps_per_epoch=ConfigDAO()["STEPS_PER_EPOCH"],
        callbacks=cbs,
        use_multiprocessing=multiprocessing,
        workers=workers,
        validation_data=validation_data,
        validation_steps=ConfigDAO()["VAL_SPLIT"],
        max_queue_size=workers * 2
    )

    model.load_weights(checkpoint_path)

    st.info("Evaluating")
    ev = model.evaluate(
        validation_data if validation_data else train_data,
        steps=ConfigDAO()["VAL_SPLIT"]
    )
    if "accuracy" in model.metrics_names:
        loss = 1 - ev[model.metrics_names.index("accuracy")]
    elif "iou_score" in model.metrics_names:
        loss = 1 - ev[model.metrics_names.index("iou_score")]
    elif "rpn_bbox_loss" in model.metrics_names:
        loss = ev[model.metrics_names.index("rpn_bbox_loss")]
    elif len(ev) > 0:
        loss = ev[0]
    else:
        st.warning(
            "No evaluation possible, "
            "please specify a evaluation metric during compile"
        )
        loss = 0

    return model, loss
