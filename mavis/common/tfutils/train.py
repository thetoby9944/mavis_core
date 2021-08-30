import os
import re
from datetime import datetime
from pathlib import Path

import tensorflow as tf
import streamlit as st
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from mavis.shelveutils import current_project
from mavis.tfutils.callbacks import CheckPoint


def train_model(
        train_data: tf.data.Dataset,
        model: tf.keras.models.Model,
        validation_data: tf.data.Dataset,
        multiprocessing=True,
        save_weights_only=False
) -> (tf.keras.Model, float):
    # local import to avoid circular
    from mavis import config

    log_dir = str(
        Path(f'logs')
        / current_project()
        / f'{datetime.now().strftime("%Y%m%d-%H%M%S")}-{model.name}'
    )
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    with tf.summary.create_file_writer(log_dir).as_default():
        description = "\n".join([f"```{line}```  " for line in re.split("\\n|<\\?br>", config.c.print_preset())])
        try:
            # tf.summary.text("Configuration", data="", description=description)
            for i, (image_batch, label_batch) in enumerate(validation_data.take(config.c.VAL_SPLIT)):
                tf.summary.image(f"Validation Image Data", image_batch, max_outputs=config.c.BATCH_SIZE, step=0,
                                 description=f"##Active Preset\n###{config.c.name}\n\n{description}")
                if config.c.BINARY:
                    tf.expand_dims(label_batch, -1)
                tf.summary.image(f"Validation Label Data", label_batch, max_outputs=config.c.BATCH_SIZE, step=0)
        except:
            pass
    checkpoint_path = config.c.MODEL_PATH
    model_checkpoint = CheckPoint(
        model_path=checkpoint_path,
        save_weights_only=save_weights_only
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=config.c.PATIENCE,
        verbose=0,
        mode='auto',
        restore_best_weights=False
    )

    reduce_lr_on_plateau = ReduceLROnPlateau(
        patience=config.c.PATIENCE / 2
    )

    cbs = [tensorboard, model_checkpoint, early_stopping, reduce_lr_on_plateau]
    workers = 0 if os.name == 'nt' else max(os.cpu_count() - 1, 1)

    st.info(f"Fitting Model ... ")
    model.fit(
        train_data,
        epochs=config.c.EPOCHS,
        steps_per_epoch=config.c.STEPS_PER_EPOCH,
        callbacks=cbs,
        use_multiprocessing=multiprocessing,
        workers=workers,
        validation_data=validation_data,
        validation_steps=config.c.VAL_SPLIT,
        max_queue_size=workers * 2
    )

    model.load_weights(checkpoint_path)

    st.info("Evaluating")
    ev = model.evaluate(
        validation_data if validation_data else train_data,
        steps=config.c.VAL_SPLIT
    )

    return model, ev[0]
