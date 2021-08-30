import numpy as np
import segmentation_models as sm
import tensorflow as tf

from mavis import config


# @tf.function
def normalize_minus_one(img: tf.Tensor) -> tf.Tensor:
    return (img - 0.5) * 2


# @tf.function
def invert_image(img: tf.Tensor) -> tf.Tensor:
    return 255 - img


# @tf.function
def py_unet_preprocessing(img: tf.Tensor) -> tf.Tensor:
    return sm.get_preprocessing(config.c.BACKBONE)(img)


# @tf.function
def resnet_preprocess_img(img: tf.Tensor) -> tf.Tensor:
    img = tf.py_function(py_unet_preprocessing, [img], tf.float32)
    img.set_shape([None for _ in range(3)])
    return img


# @tf.function
def to_grayscale(img: tf.Tensor) -> tf.Tensor:
    return tf.image.rgb_to_grayscale(img)


# @tf.function
def clip_img_minus_one_to_one(img: tf.Tensor) -> tf.Tensor:
    return tf.clip_by_value(img, -1, 1)


def one_hot(label):
    one_hot_label = np.asarray(label.numpy().decode() == config.c.CLASS_NAMES)
    return one_hot_label.reshape(len(config.c.CLASS_NAMES)).astype(np.int8),


# @tf.function
def resize(img: tf.Tensor) -> tf.Tensor:
    return tf.image.resize(img, [config.c.SIZE, config.c.SIZE],
                           antialias=False,
                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


# @tf.function
def convert_binary(img: tf.Tensor) -> tf.Tensor:
    img = tf.where(img > 0., 1., 0.)
    return img


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
