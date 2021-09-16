import numpy as np
import tensorflow as tf


class CheckPoint(tf.keras.callbacks.Callback):
    """
    Stop training when the loss is at its min, i.e. the loss stops decreasing.
    """

    def __init__(self, model_path, save_weights_only=False, metric="val_loss"):
        super(CheckPoint, self).__init__()
        self.model_path = model_path
        self.metric = metric
        self.best = np.Inf
        self.save_weights_only = save_weights_only
        # self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        # self.best_weights = None

    def save(self):
        if self.save_weights_only:
            self.model.save_weights(self.model_path, save_format="h5")
        else:
            self.model.save(self.model_path, save_format="h5")

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        # self.wait = 0
        # The epoch the training stops at.
        # self.stopped_epoch = 0
        # Initialize the best as infinity.
        print("Begin Training ...")
        self.best = np.Inf
        self.save()

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.metric)
        if current is not None and np.less(current, self.best):
            self.best = current
            # self.wait = 0
            # Record the best weights if current results is better (less).
            print(f"Saving new model with loss {self.best} to {self.model_path}")
            self.save()
            print(f"Saved model {self.model_path}")

        # else:
        #    self.wait += 1
        #    if self.wait >= self.patience:
        #        self.stopped_epoch = epoch
        #        self.model.stop_training = True
        #        print("Restoring model weights from the end of the best epoch.")
        #        self.model.set_weights(self.best_weights)

    # def on_train_end(self, logs=None):
    #    if self.stopped_epoch > 0:
    #        print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))
