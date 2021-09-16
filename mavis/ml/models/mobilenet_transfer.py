import tensorflow as tf

from db import ConfigDAO


def MobileNet_Transfer(len_classes, dropout_rate=0.2):
    x_in = tf.keras.layers.Input(shape=(None, None, 3))

    x = tf.keras.layers.experimental.preprocessing.Rescaling(2. / 1., offset=-1)(x_in)
    x = tf.keras.applications.MobileNetV2(input_shape=(ConfigDAO()["SIZE"], ConfigDAO()["SIZE"], 3),
                                          include_top=False,
                                          weights='imagenet')(x, training=False)
    x.trainable = False
    # x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Fully connected layer 1
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # Fully connected layer 2
    x = tf.keras.layers.Conv2D(filters=len_classes, kernel_size=1, strides=1)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalMaxPooling2D()(x)
    predictions = tf.keras.layers.Activation('softmax')(x)

    model = tf.keras.Model(inputs=x_in, outputs=predictions)

    return model
