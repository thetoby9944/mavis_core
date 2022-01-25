import tensorflow as tf
from tensorflow.keras.models import Sequential

from db import ConfigDAO


def ResNet_Transfer(len_classes, dropout_rate=0.2):
    base_model = tf.keras.applications.ResNet50V2(input_shape=(ConfigDAO()["SIZE"], ConfigDAO()["SIZE"], 3),
                                                  include_top=False,
                                                  weights='imagenet')
    base_model.trainable = True

    model = Sequential([
        tf.keras.layers.Input(shape=(ConfigDAO()["SIZE"], ConfigDAO()["SIZE"], 3)),
        # tf.keras.layers.experimental.preprocessing.Rescaling(2. / 1., offset=-1),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len_classes, activation="softmax")

        # Fully connected layer 1
        # tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1),
        # tf.keras.layers.Dropout(dropout_rate),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Activation('relu'),

        # Fully connected layer 2
        # tf.keras.layers.Conv2D(filters=len_classes, kernel_size=1, strides=1),
        # tf.keras.layers.Dropout(dropout_rate),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.GlobalMaxPooling2D(),
        # tf.keras.layers.Activation('softmax')
    ])

    # model = tf.keras.Model(inputs=x_in, outputs=predictions)
    return model
