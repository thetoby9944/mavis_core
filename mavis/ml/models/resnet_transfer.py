from keras import Input
from keras.applications.resnet_v2 import ResNet50V2
from keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential


def ResNet_Transfer(len_classes, dropout_rate=0.2):
    base_model = ResNet50V2(
        input_shape=(None, None, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = True

    model = Sequential([
        Input(shape=(None, None, 3)),
        # tf.keras.layers.experimental.preprocessing.Rescaling(2. / 1., offset=-1),
        base_model,
        GlobalAveragePooling2D(),
        Dense(len_classes, activation="softmax")

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

    return model
