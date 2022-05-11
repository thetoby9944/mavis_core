import keras.callbacks
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from mavis.ml.dataset.base import TFDatasetWrapper


class CheckPoint(tf.keras.callbacks.Callback):
    def __init__(self, model_path, save_weights_only=False, metric="val_loss"):
        super(CheckPoint, self).__init__()
        self.model_path = model_path
        self.metric = metric
        self.best = np.Inf
        self.save_weights_only = save_weights_only

    def save(self):
        if self.save_weights_only:
            self.model.save_weights(self.model_path, save_format="h5")
        else:
            self.model: tf.keras.Model = self.model
            self.model.save(self.model_path)

    def on_train_begin(self, logs=None):
        print("Begin Training ...")
        self.best = np.Inf
        self.save()

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.metric)
        if current is not None and np.less(current, self.best):
            self.best = current
            print(f"Saving new model with loss {self.best} to {self.model_path}")
            self.save()
            print(f"Saved model {self.model_path}")


class ReduceCyclicalLROnPlateau(Callback):
    """Reduce learning rate when a metric has stopped improving.

    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This callback monitors a
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Example:

    ```python
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.001)
    model.fit(X_train, Y_train, callbacks=[reduce_lr])
    ```

    Args:
        monitor: quantity to be monitored.
        factor: factor by which the learning rate will be reduced.
          `new_lr = lr * factor`.
        patience: number of epochs with no improvement after which learning rate
          will be reduced.
        verbose: int. 0: quiet, 1: update messages.
        mode: one of `{'auto', 'min', 'max'}`. In `'min'` mode,
          the learning rate will be reduced when the
          quantity monitored has stopped decreasing; in `'max'` mode it will be
          reduced when the quantity monitored has stopped increasing; in `'auto'`
          mode, the direction is automatically inferred from the name of the
          monitored quantity.
        min_delta: threshold for measuring the new optimum, to only focus on
          significant changes.
        cooldown: number of epochs to wait before resuming normal operation after
          lr has been reduced.
        min_lr: lower bound on the learning rate.
    """

    def __init__(self,
                 monitor='val_loss',
                 factor=0.1,
                 patience=10,
                 verbose=0,
                 mode='auto',
                 min_delta=1e-4,
                 cooldown=0,
                 min_lr=0,
                 **kwargs):
        super(ReduceCyclicalLROnPlateau, self).__init__()


        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau ' 'does not support a factor >= 1.0.')
        if 'epsilon' in kwargs:
            min_delta = kwargs.pop('epsilon')

        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['auto', 'min', 'max']:
            self.mode = 'auto'
        if (self.mode == 'min' or
                (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        self._reset()


    def get_lr(self):
        return self.model.optimizer.lr.maximal_learning_rate

    def set_lr(self, lr):
        self.model.optimizer.lr.maximal_learning_rate = lr

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = self.get_lr()
        current = logs.get(self.monitor)
        if current is None:
            pass

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = self.get_lr()
                    if old_lr > np.float32(self.min_lr):
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        new_lr = np.float32(new_lr)
                        self.set_lr(new_lr)
                        if self.verbose > 0:
                            print('\nEpoch %05d: ReduceLROnPlateau reducing maximal learning '
                                  'rate to %s.' % (epoch + 1, new_lr))
                        self.cooldown_counter = self.cooldown
                        self.wait = 0

    def in_cooldown(self):
        return self.cooldown_counter > 0


class PredictionCallback(keras.callbacks.Callback):
    def __init__(self, inference_fn):
        super().__init__()
        self.inference_fn = inference_fn

    def on_epoch_end(self, epoch, logs={}):
        self.inference_fn()
