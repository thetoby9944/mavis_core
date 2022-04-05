import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow_probability.python.distributions import Chi2
from tqdm import tqdm

N_Z = 1000


class GAIA(tf.keras.Model):
    """a basic gaia class for tensorflow

    Extends:
        tf.keras.Model
    """

    def __init__(self, **kwargs):
        super(GAIA, self).__init__()
        self.__dict__.update(kwargs)
        self._name = "GAIA"

        self.enc = tf.keras.Sequential(self.enc)
        self.dec = tf.keras.Sequential(self.dec)

        inputs, outputs = self.unet_function()
        self.disc = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    def summary(self, **kwargs):
        print("=== ENCODER ===")
        print(self.enc.summary())
        print("=== DECODER ===")
        print(self.dec.summary())
        print("=== DISCRIMINATOR ===")
        print(self.disc.summary())

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        return self.dec(z)

    def discriminate(self, x):
        return self.disc(x)

    def regularization(self, x1, x2):
        return tf.reduce_mean(tf.square(x1 - x2))

    @tf.function
    def network_pass(self, x):
        z = self.encode(x)
        print("=== z", z)

        xg = self.decode(z)
        print("=== xg ", xg)

        zi = self._interpolate_z(z)
        print("=== zi ", zi)

        xi = self.decode(zi)
        print("=== xi ", xi)

        d_xi = self.discriminate(xi)
        print("=== dxi ", d_xi)

        d_x = self.discriminate(x)
        print("=== dx ", d_x)

        d_xg = self.discriminate(xg)
        print("=== dxg ", d_xg)

        return z, xg, zi, xi, d_xi, d_x, d_xg

    @tf.function
    def compute_loss(self, x):
        # run through network
        z, xg, zi, xi, d_xi, d_x, d_xg = self.network_pass(x)

        # compute losses
        xg_loss = self.regularization(x, xg)
        d_xg_loss = self.regularization(x, d_xg)
        d_xi_loss = self.regularization(xi, d_xi)
        d_x_loss = self.regularization(x, d_x)

        return d_xg_loss, d_xi_loss, d_x_loss, xg_loss

    @tf.function
    def compute_gradients(self, x):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            d_xg_loss, d_xi_loss, d_x_loss, xg_loss = self.compute_loss(x)

            gen_loss = d_xg_loss + d_xi_loss
            disc_loss = d_xg_loss + d_x_loss - tf.clip_by_value(d_xi_loss, 0, d_x_loss)

        gen_gradients = gen_tape.gradient(
            gen_loss, self.enc.trainable_variables + self.dec.trainable_variables
        )
        disc_gradients = disc_tape.gradient(disc_loss, self.disc.trainable_variables)
        return gen_gradients, disc_gradients

    @tf.function
    def apply_gradients(self, gen_gradients, disc_gradients):
        self.gen_optimizer.apply_gradients(
            zip(
                gen_gradients,
                self.enc.trainable_variables + self.dec.trainable_variables,
            )
        )
        self.disc_optimizer.apply_gradients(
            zip(disc_gradients, self.disc.trainable_variables)
        )

    @tf.function
    def train(self, x):
        gen_gradients, disc_gradients = self.compute_gradients(x)
        self.apply_gradients(gen_gradients, disc_gradients)

    def _interpolate_z(self, z):
        """ takes the dot product of some random tensor of batch_size,
         and the z representation of the batch as the interpolation
        """
        if self.chsq.df != z.shape[0]:
            self.chsq = Chi2(df=1 / z.shape[0])
        ip = tf.convert_to_tensor(self.chsq.sample((z.shape[0], z.shape[0])), dtype=tf.float32)
        ip = ip / tf.reduce_sum(ip, axis=0)
        zi = tf.transpose(tf.tensordot(tf.transpose(z), ip, axes=1))
        return zi

    def compile(self, **kwargs):
        pass

    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_batch_size=None,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False):
        # a pandas dataframe to save the loss information to
        losses = pd.DataFrame(columns=['d_xg_loss', 'd_xi_loss', 'd_x_loss', 'xg_loss'])
        for epoch in range(epochs):
            # train
            for batch, (train_x, train_y) in tqdm(zip(range(steps_per_epoch), x),
                                                  total=steps_per_epoch):
                self.train(train_x)
            # test on holdout
            if validation_data is not None:
                loss = []
                for test_x, test_y in validation_data:
                    loss.append(self.compute_loss(test_x))

                losses.loc[len(losses)] = np.mean(loss, axis=0)
                print(f"Epoch: {epoch} Losses {np.mean(loss, axis=0)}")

    def evaluate(self,
                 x=None,
                 y=None,
                 batch_size=None,
                 verbose=1,
                 sample_weight=None,
                 steps=None,
                 callbacks=None,
                 max_queue_size=10,
                 workers=1,
                 use_multiprocessing=False,
                 return_dict=False):
        loss = []
        print("Started Evaluation")
        for batch, (test_x, test_y) in tqdm(zip(range(steps), x),
                                            total=steps):
            loss.append(self.compute_loss(test_x))
        return np.mean(loss, axis=0)

    def predict(self,
                x,
                batch_size=None,
                verbose=0,
                steps=None,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False):
        print("Started Prediction")
        res = []
        for sample in x:
            z, xg, zi, xi, d_xi, d_x, d_xg = self.network_pass(sample)
            res += [xg]
        return np.stack(res, axis=0)

    def save(self, filepath, **kwargs):
        self.enc.save(filepath + "_enc", **kwargs)
        self.dec.save(filepath + "_dec", **kwargs)
        self.disc.save(filepath + "_disc", **kwargs)


def sigmoid(x, shift=0.0, mult=20):
    """ squashes a value with a sigmoid
    """
    return tf.constant(1.0) / (
            tf.constant(1.0) + tf.exp(-tf.constant(1.0) * ((x + tf.constant(shift)) * mult))
    )


def unet_convblock_down(
        _input,
        channels=16,
        kernel=(3, 3),
        activation="relu",
        pool_size=(2, 2),
        kernel_initializer="he_normal",
):
    """ An upsampling convolutional block for a UNET
    """
    conv = tf.keras.layers.Conv2D(
        channels,
        kernel,
        activation=activation,
        padding="same",
        kernel_initializer=kernel_initializer,
    )(_input)
    conv = tf.keras.layers.Conv2D(
        channels,
        kernel,
        activation=activation,
        padding="same",
        kernel_initializer=kernel_initializer,
    )(conv)
    pool = tf.keras.layers.MaxPooling2D(pool_size=pool_size)(conv)
    return conv, pool


def unet_convblock_up(
        last_conv,
        cross_conv,
        channels=16,
        kernel=(3, 3),
        activation="relu",
        pool_size=(2, 2),
        kernel_initializer="he_normal",
):
    """ A downsampling convolutional block for a UNET
    """
    up_conv = tf.keras.layers.Conv2DTranspose(
        filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation="sigmoid"
    )(last_conv)

    merge = tf.keras.layers.concatenate([up_conv, cross_conv], axis=3)
    conv = tf.keras.layers.Conv2D(
        channels,
        kernel,
        activation=activation,
        padding="same",
        kernel_initializer=kernel_initializer,
    )(merge)
    conv = tf.keras.layers.Conv2D(
        channels,
        kernel,
        activation=activation,
        padding="same",
        kernel_initializer=kernel_initializer,
    )(conv)
    return conv


def unet_mnist():
    """ the architecture for a UNET specific to MNIST
    """
    inputs = tf.keras.layers.Input(shape=(128, 128, 3))
    up_1, pool_1 = unet_convblock_down(inputs, channels=32)
    up_2, pool_2 = unet_convblock_down(pool_1, channels=64)
    conv_middle = tf.keras.layers.Conv2D(
        128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(pool_2)
    conv_middle = tf.keras.layers.Conv2D(
        128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(conv_middle)
    down_2 = unet_convblock_up(conv_middle, up_2, channels=64)
    down_1 = unet_convblock_up(down_2, up_1, channels=32)
    outputs = tf.keras.layers.Conv2D(3, (1, 1), activation="sigmoid")(down_1)
    return inputs, outputs


def gaia_encoder():
    return [
        tf.keras.layers.InputLayer(input_shape=[128, 128, 3]),
        tf.keras.layers.Conv2D(
            filters=63, kernel_size=3, strides=(2, 2), activation="relu", padding="SAME"
        ),
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, strides=(2, 2), activation="relu", padding="SAME"
        ),
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, strides=(2, 2), activation="relu", padding="SAME"
        ),
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, strides=(2, 2), activation="relu", padding="SAME"
        ),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=N_Z * 2),
    ]


def gaia_decoder():
    return [
        # tf.keras.layers.InputLayer(input_shape=[N_Z*2]),
        tf.keras.layers.Dense(units=8 * 8 * 64, activation="relu", input_shape=[N_Z * 2]),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Reshape(target_shape=(8, 8, 64)),
        tf.keras.layers.Conv2DTranspose(
            filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
        ),
        tf.keras.layers.Conv2DTranspose(
            filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
        ),
        tf.keras.layers.Conv2DTranspose(
            filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
        ),
        tf.keras.layers.Conv2DTranspose(
            filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
        ),
        tf.keras.layers.Conv2DTranspose(
            filters=3, kernel_size=3, strides=(1, 1), padding="SAME", activation="sigmoid"
        ),

    ]
