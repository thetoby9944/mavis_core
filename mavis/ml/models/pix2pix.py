import time
from pathlib import Path

import streamlit as st
import tensorflow as tf
from keras.engine import data_adapter
from tensorflow.keras.models import load_model

from mavis.pilutils import pil


class Pix2Pix(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(Pix2Pix, self).__init__(*args, **kwargs)
        self.OUTPUT_CHANNELS = 3
        self.LAMBDA = 100

        self.generator = self.Generator()
        self.discriminator = self.Discriminator()

    def summary(self, **kwargs):
        print("=== GENERATOR ===")
        print(self.generator.summary())
        print("=== DISCRIMINATOR ===")
        print(self.discriminator.summary())

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)

        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss

    def generator_loss(self, disc_generated_output, gen_output, target):
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

        # mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

        total_gen_loss = gan_loss + (self.LAMBDA * l1_loss)

        return total_gen_loss, gan_loss, l1_loss

    def upsample(self, filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))

        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())

        return result

    def downsample(self, filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                   kernel_initializer=initializer, use_bias=False))

        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())

        return result

    def Discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
        tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

        x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

        down1 = self.downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
        down2 = self.downsample(128, 4)(down1)  # (bs, 64, 64, 128)
        down3 = self.downsample(256, 4)(down2)  # (bs, 32, 32, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                      kernel_initializer=initializer,
                                      use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

        last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                      kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

        return tf.keras.Model(inputs=[inp, tar], outputs=last)

    def Generator(self):
        inputs = tf.keras.layers.Input(shape=[256, 256, 3])

        down_stack = [
            self.downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
            self.downsample(128, 4),  # (bs, 64, 64, 128)
            self.downsample(256, 4),  # (bs, 32, 32, 256)
            self.downsample(512, 4),  # (bs, 16, 16, 512)
            self.downsample(512, 4),  # (bs, 8, 8, 512)
            self.downsample(512, 4),  # (bs, 4, 4, 512)
            self.downsample(512, 4),  # (bs, 2, 2, 512)
            self.downsample(512, 4),  # (bs, 1, 1, 512)
        ]

        up_stack = [
            self.upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
            self.upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
            self.upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
            self.upsample(512, 4),  # (bs, 16, 16, 1024)
            self.upsample(256, 4),  # (bs, 32, 32, 512)
            self.upsample(128, 4),  # (bs, 64, 64, 256)
            self.upsample(64, 4),  # (bs, 128, 128, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(self.OUTPUT_CHANNELS, 4,
                                               strides=2,
                                               padding='same',
                                               kernel_initializer=initializer,
                                               activation='tanh')  # (bs, 256, 256, 3)

        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)

    def train_step(self, data):
        """
        The logic for one training step.

        This method can be overridden to support custom training logic.
        This method is called by `Model.make_train_function`.

        This method should contain the mathemetical logic for one step of training.
        This typically includes the forward pass, loss calculation, backpropagation,
        and metric updates.

        Configuration details for *how* this logic is run (e.g. `tf.function` and
        `tf.distribute.Strategy` settings), should be left to
        `Model.make_train_function`, which can also be overridden.

        Arguments:
          data: A nested structure of `Tensor`s.

        Returns:
          A `dict` containing values that will be passed to
          `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
          values of the `Model`'s metrics are returned. Example:
          `{'loss': 0.2, 'accuracy': 0.7}`.

        """
        data = data_adapter.expand_1d(data)
        input_image, target, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(disc_generated_output, gen_output, target)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                     self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                     self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                         self.discriminator.trainable_variables))

        return {
            "loss": gen_total_loss,
            "accuracy": 1 - disc_loss,
            "gen_total_loss": gen_total_loss,
            "gen_l2_loss": gen_l1_loss,
            "gen_gan_los": gen_gan_loss,
            "disc_loss": disc_loss,
        }

    def test_step(self, data):
        """The logic for one evaluation step.

        This method can be overridden to support custom evaluation logic.
        This method is called by `Model.make_test_function`.

        This function should contain the mathemetical logic for one step of
        evaluation.
        This typically includes the forward pass, loss calculation, and metrics
        updates.

        Configuration details for *how* this logic is run (e.g. `tf.function` and
        `tf.distribute.Strategy` settings), should be left to
        `Model.make_test_function`, which can also be overridden.

        Arguments:
          data: A nested structure of `Tensor`s.

        Returns:
          A `dict` containing values that will be passed to
          `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
          values of the `Model`'s metrics are returned.
        """
        data = data_adapter.expand_1d(data)
        input_image, target, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        gen_output = self.generator(input_image, training=False)

        disc_real_output = self.discriminator([input_image, target], training=False)
        disc_generated_output = self.discriminator([input_image, gen_output], training=False)

        gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(disc_generated_output, gen_output, target)
        disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        return {
            "loss": gen_total_loss,
            "accuracy": 1 - disc_loss,
            "gen_total_loss": gen_total_loss,
            "gen_l2_loss": gen_l1_loss,
            "gen_gan_los": gen_gan_loss,
            "disc_loss": disc_loss,
        }

    def predict_step(self, data):
        """The logic for one inference step.

        This method can be overridden to support custom inference logic.
        This method is called by `Model.make_predict_function`.

        This method should contain the mathemetical logic for one step of inference.
        This typically includes the forward pass.

        Configuration details for *how* this logic is run (e.g. `tf.function` and
        `tf.distribute.Strategy` settings), should be left to
        `Model.make_predict_function`, which can also be overridden.

        Arguments:
          data: A nested structure of `Tensor`s.

        Returns:
          The result of one inference step, typically the output of calling the
          `Model` on data.
        """
        data = data_adapter.expand_1d(data)
        x, _, _ = data_adapter.unpack_x_y_sample_weight(data)
        return self.generator(x, training=False)

    def compile(
            self,
            optimizer='rmsprop',
            loss=None,
            metrics=None,
            loss_weights=None,
            weighted_metrics=None,
            run_eagerly=None,
            steps_per_execution=None,
            **kwargs):
        from keras.engine import base_layer
        from keras.engine import compile_utils

        base_layer.keras_api_gauge.get_cell('compile').set(True)
        with self.distribute_strategy.scope():
            if 'experimental_steps_per_execution' in kwargs:
                if not steps_per_execution:
                    steps_per_execution = kwargs.pop('experimental_steps_per_execution')

            # When compiling from an already-serialized model, we do not want to
            # reapply some processing steps (e.g. metric renaming for multi-output
            # models, which have prefixes added for each corresponding output name).
            from_serialized = kwargs.pop('from_serialized', False)

            self._validate_compile(optimizer, metrics, **kwargs)
            self._run_eagerly = run_eagerly

            self.optimizer = self._get_optimizer(optimizer)
            self.compiled_loss = compile_utils.LossesContainer(
                losses="MSE",
                loss_weights=loss_weights,
                output_names=self.output_names
            )
            self.compiled_metrics = compile_utils.MetricsContainer(
                metrics, weighted_metrics, output_names=self.output_names,
                from_serialized=from_serialized)

            self.loss_object = tf.keras.losses.BinaryCrossentropy(
                from_logits=True,
                reduction=tf.keras.losses.Reduction.SUM
            )
            self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
            self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

            self._configure_steps_per_execution(steps_per_execution or 1)

            # Initializes attrs that are reset each time `compile` is called.
            self._reset_compile_cache()
            self._is_compiled = True

            self.loss = loss or {}  # Backwards compat.

    def _handle_file_path(self, filepath):
        filepath = Path(filepath)
        filepath = filepath.parent / filepath.stem
        filepath = str(filepath)
        return filepath + "_disc.h5", filepath + "_gen.h5"

    def load_weights(
            self,
            filepath,
            by_name=False,
            skip_mismatch=False,
            options=None
    ):
        """
        Warning doesnt load weights, restores a model with load_model
        Parameters
        ----------
        filepath
        by_name
        skip_mismatch
        options

        Returns
        -------

        """
        disc_path, gen_path = self._handle_file_path(filepath)
        self.discriminator = load_model(disc_path, compile=False)
        self.generator = load_model(gen_path, compile=False)

    def save_weights(
            self,
            filepath,
            **kwargs
    ):
        disc_path, gen_path = self._handle_file_path(filepath)
        self.discriminator.save(disc_path, include_optimizer=False, **kwargs)
        self.generator.save(gen_path, include_optimizer=False, **kwargs)

    def save(
            self,
            filepath,
            **kwargs
    ):
        disc_path, gen_path = self._handle_file_path(filepath)
        self.discriminator.save(disc_path, include_optimizer=False, **kwargs)
        self.generator.save(gen_path, include_optimizer=False, **kwargs)

    def predict_old(
            self,
            x,
            batch_size=None,
            verbose=0,
            steps=None,
            callbacks=None,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False
    ):
        print("Started Prediction")
        return self.generator(x).numpy()

    def evaluate_old(
            self,
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
            return_dict=False
    ):
        # loss = []
        # print("Started Evaluation")
        # for batch, (test_x, test_y) in tqdm(zip(range(steps), x),
        #                           total=steps):
        #    loss.append(self.compute_loss(test_x))
        # return np.mean(loss, axis=0)
        return [0]

    def fit_old(
            self,
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
            use_multiprocessing=False
    ):
        st.text("Image:")
        image_pl = st.empty()
        st.text("Target:")
        label_pl = st.empty()
        st.text("Generated:")
        gen_pl = st.empty()
        for epoch in range(epochs):

            start = time.time()

            # Train
            for n, (x_batch, y_batch) in zip(range(steps_per_epoch), x):
                print('.', end='')
                if (n + 1) % 100 == 0:
                    print()
                self.train_step(x_batch, y_batch, epoch)
            print()

            # saving (checkpoint) the model every 20 epochs
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                               time.time() - start))

        from tqdm.notebook import tqdm, trange

        # Populate with typical keras callbacks
        _callbacks = callbacks

        callbacks = tf.keras.callbacks.CallbackList(
            _callbacks, add_history=True, model=self)

        logs = {}
        callbacks.on_train_begin(logs=logs)

        # Presentation
        epochs = trange(
            epochs,
            desc="Epoch",
            unit="Epoch",
            postfix="loss = {loss:.4f}, accuracy = {accuracy:.4f}")
        epochs.set_postfix(loss=0, accuracy=0)

        # Get a stable test set so epoch results are comparable
        test_batches = validation_data

        for epoch in epochs:
            callbacks.on_epoch_begin(epoch, logs=logs)

            # Presentation
            enumerated_batches = tqdm(
                zip(range(steps_per_epoch), x),
                desc="Batch",
                unit="batch",
                postfix="loss = {loss:.4f}, accuracy = {accuracy:.4f}",
                position=1,
                leave=False)

            for batch, (x_batch, y_batch) in enumerated_batches:
                callbacks.on_batch_begin(batch, logs=logs)
                callbacks.on_train_batch_begin(batch, logs=logs)

                self.train_step(x_batch, y_batch, epoch)

                callbacks.on_train_batch_end(batch, logs=logs)
                callbacks.on_batch_end(batch, logs=logs)

                # Presentation
                enumerated_batches.set_postfix(
                    loss=float(logs["loss"]),
                    accuracy=float(logs["accuracy"]))

            for (batch, (x_val_batch, y_val_batch)) in enumerate(test_batches):
                callbacks.on_batch_begin(batch, logs=logs)
                callbacks.on_test_batch_begin(batch, logs=logs)

                logs = self.evaluate(x=x_val_batch, y=y_val_batch, return_dict=True)

                callbacks.on_test_batch_end(batch, logs=logs)
                callbacks.on_batch_end(batch, logs=logs)

            # Presentation
            epochs.set_postfix(
                loss=float(logs["loss"]),
                accuracy=float(logs["accuracy"]))

            callbacks.on_epoch_end(epoch, logs=logs)

            # NOTE: This is a decent place to check on your early stopping
            # callback.
            # Example: use training_model.stop_training to check for early stopping

            image_pl.image(pil(x_batch.numpy()[0]))
            gen_pl.image(pil(self.generator(x_batch).numpy()[0]))
            label_pl.image(pil(y_batch.numpy()[0]))

        callbacks.on_train_end(logs=logs)

        # Fetch the history object we normally get from keras.fit
        history_object = None
        for cb in callbacks:
            if isinstance(cb, tf.keras.callbacks.History):
                history_object = cb
        assert history_object is not None
