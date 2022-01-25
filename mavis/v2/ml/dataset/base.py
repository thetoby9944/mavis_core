from typing import Optional

import numpy as np
import tensorflow as tf
import albumentations as A
from keras.utils.data_utils import Sequence

from v2.config import MLConfig

auto = tf.data.experimental.AUTOTUNE


class TFDatasetWrapper:
    @staticmethod
    def decode_img(file_path):
        img = tf.io.read_file(file_path)
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.convert_image_dtype(img, tf.float32, saturate=True)
        return img

    @staticmethod
    def process_image_path(file_path):
        # load the raw data from the file as a string
        img = TFDatasetWrapper.decode_img(file_path)
        # img = resize(img)
        return img

    @staticmethod
    def prepare_batch(ds: tf.data.Dataset, config: MLConfig) -> tf.data.Dataset:
        ds = ds.batch(config.TRAIN.BATCH_SIZE)
        # `prefetch` lets the dataset fetch batches in the background while the model is training.
        ds = ds.prefetch(buffer_size=auto)
        return ds

    @staticmethod
    def paths_to_image_ds(paths):
        return paths.map(TFDatasetWrapper.process_image_path, num_parallel_calls=auto)

    def __init__(self, config: MLConfig, *args, **kwargs):
        # List of tensorflow functions to preprocess labels
        self.label_preprocessing = []
        # List of tensorflow functions to preprocess images
        self.image_preprocessing = []
        # Infinite tf.data.DataSet with random augmentation
        self.ds = None
        # Finite tf.data.Dataset with random augmentation
        self.val_ds = None
        # Whether labels are augmented (e.g. masks)
        self.augment_label = True
        # ImgAug sequential pipeline

        self.config: MLConfig = config
        self.iaa: A.Compose = self.config.DATASET.AUGMENTATION.get()

    def _apply_img_aug(self, img_tensor, lbl_tensor):
        lbl_shape = tf.shape(lbl_tensor)
        if self.augment_label:
            lbl_tensor = tf.image.convert_image_dtype(lbl_tensor, tf.uint8)
            lbl_shape = tf.shape(lbl_tensor)
        lbl_dtype = lbl_tensor.dtype

        img_tensor = tf.image.convert_image_dtype(img_tensor, tf.uint8)
        img_shape = tf.shape(img_tensor)
        img_dtype = img_tensor.dtype

        img_tensor, lbl_tensor = tf.numpy_function(
            self.img_aug,
            [img_tensor, lbl_tensor],
            [img_dtype, lbl_dtype]
        )

        # img_tensor = tf.reshape(img_tensor, shape=tf.shape())
        # lbl_tensor = tf.reshape(lbl_tensor, shape=lbl_shape)

        if self.augment_label:
            lbl_tensor = tf.image.convert_image_dtype(lbl_tensor, tf.float32)
        img_tensor = tf.image.convert_image_dtype(img_tensor, tf.float32)

        return img_tensor, lbl_tensor

    def img_aug(self, image: np.ndarray, label: np.ndarray) -> (np.ndarray, np.ndarray):
        raise NotImplementedError

    def augment(self):
        self.ds = self.ds.map(self._apply_img_aug, num_parallel_calls=auto)

        for fn in self.label_preprocessing:
            self.ds = self.ds.map(lambda x, y: (x, fn(y)), num_parallel_calls=auto)

        for fn in self.image_preprocessing:
            self.ds = self.ds.map(lambda x, y: (fn(x), y), num_parallel_calls=auto)

    def split_and_batch(self):
        n_val = self.config.TRAIN.VAL_SPLIT * self.config.TRAIN.BATCH_SIZE
        if n_val != 0:
            self.val_ds = self.ds.take(n_val)
            self.val_ds = self.prepare_batch(self.val_ds, self.config)
            self.val_ds = self.val_ds.repeat()
            self.ds = self.ds.skip(n_val)

        if self.config.DATASET.RESHUFFLE_EACH_ITERATION:
            self.ds = self.ds.shuffle(
                buffer_size=int(self.config.DATASET.BUFFER_SIZE),
                reshuffle_each_iteration=True
            )

        self.ds = self.ds.repeat()
        self.ds = self.prepare_batch(self.ds, self.config)

    def create(self, img_paths, labels):
        # Convert to str if paths are windows paths
        img_paths = [str(img_path) for img_path in img_paths]
        if labels is None:
            return self.create_inference(img_paths)
        else:
            return self.create_train(img_paths, labels)

    def create_inference(self, img_paths) -> (tf.data.Dataset or Sequence):
        """
        Prepare Inference Dataset
        """
        image_paths = tf.data.Dataset.from_tensor_slices(img_paths)
        ds = self.paths_to_image_ds(image_paths)

        for fn in self.image_preprocessing:
            ds = ds.map(fn, num_parallel_calls=auto)

        print("Preparing Batches")
        self.ds = self.prepare_batch(ds, self.config)
        return self.ds

    def create_train(self, img_paths, labels) -> None:
        raise NotImplementedError

    def peek(self) -> None:
        raise NotImplementedError

    def display_pred(self, pred: np.array) -> None:
        raise NotImplementedError
