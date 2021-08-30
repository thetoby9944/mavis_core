import numpy as np
import segmentation_models as sm
import tensorflow as tf

import config

auto = tf.data.experimental.AUTOTUNE


# @tf.function
def normalize_minus_one(img: tf.Tensor) -> tf.Tensor:
    return (img - 0.5) * 2


# @tf.function
def py_unet_preprocessing(img: tf.Tensor) -> tf.Tensor:
    if np.min(img) < 0:
        img -= np.min(img)
    return sm.get_preprocessing(config.c.BACKBONE)(img)


# @tf.function
def resnet_preprocess_img(img: tf.Tensor) -> tf.Tensor:
    img = tf.py_function(py_unet_preprocessing, [img], tf.float32)
    img.set_shape([None for _ in range(3)])
    return img


def one_hot(label):
    one_hot_label = np.asarray(label.numpy().decode() == config.c.CLASS_NAMES)
    return one_hot_label.reshape(len(config.c.CLASS_NAMES)).astype(np.int8),


# @tf.function
def resize(img: tf.Tensor) -> tf.Tensor:
    return tf.image.resize_with_pad(img, config.c.SIZE, config.c.SIZE,
                                    antialias=False,
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


# @tf.function
def mask_by_color(img: tf.Tensor, col: tf.Tensor) -> tf.Tensor:
    img = tf.cast(img == col, dtype=tf.uint8)
    img = tf.reduce_sum(img, axis=-1) == 3
    return tf.cast(img, dtype=tf.float32)


# @tf.function
def masking(img: tf.Tensor) -> tf.Tensor:
    img = tf.image.convert_image_dtype(img, tf.uint8, saturate=True)
    img = tf.stack([mask_by_color(img, col) for col in config.c.CLASS_COLORS], axis=-1)
    if config.c.BINARY:
        img = img[..., 1]
    return img


def decode_img(file_path):
    img = tf.io.read_file(file_path)
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32, saturate=True)
    return img


def process_image_path(file_path):
    # load the raw data from the file as a string
    img = decode_img(file_path)
    img = resize(img)
    return img


def prepare_batch(ds: tf.data.Dataset) -> tf.data.Dataset:
    ds = ds.batch(config.c.BATCH_SIZE)
    # `prefetch` lets the dataset fetch batches in the background while the model is training.
    ds = ds.prefetch(buffer_size=auto)
    return ds


def paths_to_image_ds(paths):
    return paths.map(process_image_path, num_parallel_calls=auto)