import tensorflow as tf
from tensorflow.keras import layers


def simple_conv(len_classes):
    x_in = layers.Input(shape=(None, None, 3))
    x = layers.Conv2D(24, (3,3), activation="relu")(x_in)
    x = layers.BatchNormalization(momentum=0.01)(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(48, (3,3), activation="relu")(x)
    x = layers.BatchNormalization(momentum=0.01)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(96, (3, 3), activation="relu")(x)
    x = layers.BatchNormalization(momentum=0.01)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(len_classes, (3, 3), activation="relu")(x)
    x = layers.BatchNormalization(momentum=0.01)(x)
    x = layers.GlobalMaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Softmax()(x)
    return tf.keras.models.Model([x_in], [x])