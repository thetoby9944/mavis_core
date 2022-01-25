import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization, Activation, GlobalMaxPooling2D


def fcn(n_classes=5, dropout_rate=0.2):
    x_in = Input(shape=(None, None, 3))

    x = Conv2D(filters=32, kernel_size=3, strides=1)(x_in)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # x = MaxPooling2D()(x)

    x = Conv2D(filters=64, kernel_size=3, strides=1)(x)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # x = MaxPooling2D()(x)

    x = Conv2D(filters=128, kernel_size=3, strides=2)(x)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # x = MaxPooling2D()(x)

    x = Conv2D(filters=256, kernel_size=3, strides=2)(x)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # x = MaxPooling2D()(x)

    x = Conv2D(filters=512, kernel_size=3, strides=2)(x)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Fully connected layer 1
    x = Conv2D(filters=64, kernel_size=1, strides=1)(x)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Fully connected layer 2
    x = Conv2D(filters=n_classes, kernel_size=1, strides=1)(x)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)
    x = GlobalMaxPooling2D()(x)
    y = tf.keras.layers.Activation('softmax')(x)

    model = tf.keras.Model(inputs=x_in, outputs=y)
    return model
