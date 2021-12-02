import numpy as np
import tensorflow as tf


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
            self.model.save(self.model_path, save_format="h5")

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
