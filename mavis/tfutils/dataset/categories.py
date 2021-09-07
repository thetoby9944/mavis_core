import numpy as np
import streamlit as st
import tensorflow as tf

from pilutils import pil
from shelveutils import ConfigDAO
from tfutils.dataset.base import TFDatasetWrapper
from tfutils.dataset.preprocessing import resnet_preprocess_img, one_hot, auto, paths_to_image_ds


class ImageToCategoryDataset(TFDatasetWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_preprocessing = [resnet_preprocess_img]
        self.augment_label = False

    def display_pred(self, pred):
        st.write(ConfigDAO()["CLASS_NAMES"][np.argmax(pred)])

    def img_aug(self, image, label):
        image = self.iaa(image=image)
        return image, label

    def create_train(self, img_paths, labels):
        """
                Prepare Inference or Training Dataset, works with classes or segmentation maps
                If you prepare a training dataset either pass image_label_path_list or class_label_list + CLASS_NAMES
                """

        def set_shape(x: tf.Tensor) -> tf.Tensor:
            x.set_shape([None])
            return x

        image_paths = tf.data.Dataset.from_tensor_slices(img_paths)
        image_paths = image_paths.shuffle(buffer_size=ConfigDAO()["BUFFER_SIZE"], seed=0)
        ds = paths_to_image_ds(image_paths)

        label_paths = tf.data.Dataset.from_tensor_slices(labels).shuffle(
            buffer_size=ConfigDAO()["BUFFER_SIZE"], seed=0
        )
        label_ds = label_paths.map(
            lambda x: tf.py_function(one_hot, [x], [tf.int8]),
            num_parallel_calls=auto
        )

        label_ds = label_ds.map(lambda x: set_shape(x))

        self.ds = tf.data.Dataset.zip((ds, label_ds))
        # ds = shuffle_and_cache(ds)
        self.augment()
        self.split_and_batch()

        return self.ds, self.val_ds

    def peek(self):
        for image_batch, label_batch in self.ds.take(1):
            st.image(pil(image_batch.numpy()[0]))
            st.write(label_batch.numpy()[0])
