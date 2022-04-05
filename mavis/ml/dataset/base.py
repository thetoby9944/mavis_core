from abc import ABC

import albumentations as A
import numpy as np
import segmentation_models as sm
import tensorflow as tf
from keras.utils.data_utils import Sequence

from mavis.config import MLConfig
from mavis.pilutils import pil

auto = tf.data.experimental.AUTOTUNE


class TFDatasetWrapper(ABC):
    config: MLConfig = None  # Inject from processor

    @staticmethod
    def py_unet_preprocessing(img: tf.Tensor) -> tf.Tensor:
        if np.min(img) < 0:
            img -= np.min(img)
        return sm.get_preprocessing("resnet50")(img)

    @staticmethod
    def resnet_preprocess_img(img: tf.Tensor) -> tf.Tensor:
        img = tf.py_function(TFDatasetWrapper.py_unet_preprocessing, [img], tf.float32)
        img.set_shape([None for _ in range(3)])
        return img

    @staticmethod
    def pad_image_to_divisor(image: tf.Tensor, divisor=64) -> tf.Tensor:
        """
        Pads width and height with zeros to make them multiples of `divisor`.
        E.g. The multiple of 64 is needed to ensure smooth scaling of feature
        maps up and down a-6 leveled Encoder (2**6=64).

        Returns:
            image: the resized image
            window: (y1, x1, y2, x2). Padding might
                be added to the returned image. If so, this window is the
                coordinates of the image part of the full image (excluding
                the padding). The x2, y2 pixels are not included.
            scale: The scale factor used to resize the image
            padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
        """
        # Keep track of image dtype and return results in the same dtype

        h, w = tf.shape(image)[0], tf.shape(image)[1]
        # Both sides must be divisible by 64
        # Height
        if h % divisor > 0:
            bottom_pad = divisor - (h % divisor)
        else:
            bottom_pad = 0
        # Width
        if w % divisor > 0:
            right_pad = divisor - (w % divisor)
        else:
            right_pad = 0
        padding = [(0, bottom_pad), (0, right_pad), (0, 0)]
        # window = (0, 0, h, w)
        image = tf.pad(image, padding, mode='constant', constant_values=0)
        return image

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
    def prepare_batch(ds: tf.data.Dataset, batch_size=1) -> tf.data.Dataset:
        ds = ds.batch(batch_size)
        # `prefetch` lets the dataset fetch batches in the background while the model is training.
        ds = ds.prefetch(buffer_size=auto)
        return ds

    @staticmethod
    def paths_to_image_ds(paths):
        return paths.map(TFDatasetWrapper.process_image_path, num_parallel_calls=auto)

    def __init__(self, *args, **kwargs):
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

    @property
    def iaa(self) -> A.Compose:
        """
        Transform accepts named arguments in __call__ i.e.
        image = img
        mask = mask
        label = label

        Returns
        -------
        an albumentations composed transform
        """
        return self.config.DATASET.AUGMENTATION.get()

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
        batch_size = self.config.TRAIN.BATCH_SIZE

        n_val = self.config.TRAIN.VAL_SPLIT
        if n_val != 0:
            self.val_ds = self.ds.take(n_val)
            self.val_ds = self.val_ds.repeat()
            self.val_ds = self.prepare_batch(self.val_ds, batch_size)
            self.ds = self.ds.skip(n_val)

        self.ds = self.ds.repeat()
        self.ds = self.prepare_batch(self.ds, batch_size)

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
        print("Padding images")
        ds = ds.map(self.pad_image_to_divisor, num_parallel_calls=auto)

        print("Preprocessing")
        for fn in self.image_preprocessing:
            ds = ds.map(fn, num_parallel_calls=auto)

        print("Preparing Batches")
        self.ds = self.prepare_batch(ds, batch_size=1)

        return self.ds

    def peek_augmentation(self):
        """
        # TODO use peek during data augmentation pipeline
        Parameters
        ----------
        ds

        Returns
        -------

        """
        import streamlit as st
        for image_batch in self.ds.take(2):
            batch = image_batch.numpy()
            st.info(f"Batch Shape: {batch.shape}")
            for i, img_np in zip(np.arange(20), batch):
                st.image(
                    pil(img_np),
                    use_column_width=True,
                    output_format="png"
                )
                st.info(
                    f"Image: "
                    f"Min {np.min(img_np)} "
                    f"Max {np.max(img_np)} "
                    f"Shape {img_np.shape}"
                )

    def create_train(self, img_paths, labels) -> None:
        raise NotImplementedError

    def peek(self) -> None:
        raise NotImplementedError

    def display_pred(self, pred: np.array) -> None:
        raise NotImplementedError
