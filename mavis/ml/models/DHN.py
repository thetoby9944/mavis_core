import keras.layers
from keras import Model, Input
from keras.layers import Conv2D, Conv2DTranspose, Layer

CHANNELS = 3

def ConvBlock(n_filters, x):
    x = Conv2D(n_filters, kernel_size=3, padding="same", activation="relu")(x)
    x = Conv2D(n_filters, kernel_size=3, padding="same")(x)
    return x


def DoubleConvBlock(n_filters, x):
    x = keras.layers.Add()([ConvBlock(n_filters, x), x])
    x = keras.layers.Add()([ConvBlock(n_filters, x), x])
    return x


def Encoder(x):
    # Conv1
    x = Conv2D(32, kernel_size=3, padding="same")(x)
    x = DoubleConvBlock(32, x)
    # Conv2
    x = Conv2D(64, kernel_size=3, strides=2, padding="same")(x)
    x = DoubleConvBlock(64, x)
    # Conv3
    x = Conv2D(128, kernel_size=3, strides=2, padding="same")(x)
    x = DoubleConvBlock(128, x)
    return x


def Decoder(x):
    # Deconv3
    x = DoubleConvBlock(128, x)
    x = Conv2DTranspose(64, kernel_size=4, strides=2, padding="same")(x)
    # Deconv2
    x = DoubleConvBlock(64, x)
    x = Conv2DTranspose(32, kernel_size=4, strides=2, padding="same")(x)
    # Deconv1
    x = DoubleConvBlock(32, x)
    x = Conv2D(CHANNELS, kernel_size=3, padding="same")(x)
    return x


def DHN(depth=4, *args, **kwargs):
    encoders = [Encoder for _ in range(depth)]
    decoders = [Decoder for _ in range(depth)]

    inputs = Input(shape=(None, None, 3))

    features = [encoders[-1](inputs)]
    residuals = [decoders[-1](features[0])]

    for i in range(1, depth):
        features.append(encoders[depth - i - 1](keras.layers.Add()([inputs, residuals[i-1]])))
        residuals.append(decoders[depth - i - 1](keras.layers.Add()([features[i], features[i-1]])))

    model = Model(
        inputs=inputs,
        outputs=residuals[-1]
    )
    model._name = "DHN"
    return model






