import numpy as np
import tensorflow as tf

import tensorflow_addons as tfa

from mavis import config


@tf.function
def rand_rotate(img: tf.Tensor, angle: tf.Tensor) -> tf.Tensor:
    if config.c.AUG_ROTATE == "mirror":
        img_ext = tf.image.flip_left_right(img)
        img = tf.concat([img, img_ext, img], axis=1)
        img_ext = tf.image.flip_up_down(img)
        img = tf.concat([img, img_ext, img], axis=0)

        img = tfa.image.rotate(img, angle)
        img = tf.image.central_crop(img, 1 / 3)

    else:
        img = tfa.image.rotate(img, angle)

    return img


@tf.function
def rotate_pair(img: tf.Tensor, lbl: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    angle = tf.random.uniform([], minval=-np.pi / 4, maxval=np.pi / 4)
    return rand_rotate(img, angle), rand_rotate(lbl, angle)


@tf.function
def rotate_single(img: tf.Tensor, lbl: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    angle = tf.random.uniform([], minval=-np.pi / 4, maxval=np.pi / 4)
    return rand_rotate(img, angle), lbl


@tf.function
def rand_upscale(img: tf.Tensor, rand_size: tf.Tensor) -> tf.Tensor:
    return tf.image.resize(img, [rand_size, rand_size],
                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


@tf.function
def zoom_pair(x: tf.Tensor, y: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    rand_size = tf.random.uniform([],
                                  minval=config.c.SIZE,
                                  maxval=int(config.c.SIZE * (1 + config.c.AUG_ZOOM_PERCENT)),
                                  dtype=tf.int32)
    x = rand_upscale(x, rand_size)
    y = rand_upscale(y, rand_size)
    xy = tf.image.random_crop(tf.stack([x, y], axis=0),
                              size=[2, config.c.SIZE, config.c.SIZE, 3])
    return xy[0], xy[1]


@tf.function
def zoom_single(x: tf.Tensor, y: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    rand_size = tf.random.uniform([],
                                  minval=config.c.SIZE,
                                  maxval=int(config.c.SIZE * 1.1),
                                  dtype=tf.int32)
    return tf.image.random_crop(rand_upscale(x, rand_size),
                                size=[config.c.SIZE, config.c.SIZE, 3]), y


@tf.function
def flip_pair(img: tf.Tensor, lbl: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    img, lbl = tf.cond(tf.random.uniform([], minval=0, maxval=1) > tf.constant(0.5),
                       lambda: (tf.image.flip_left_right(img), tf.image.flip_left_right(lbl)),
                       lambda: (img, lbl))
    return img, lbl


@tf.function
def flip_single(img: tf.Tensor, lbl: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    img = tf.cond(tf.random.uniform([], minval=0, maxval=1) > tf.constant(0.5),
                  lambda: tf.image.flip_left_right(img),
                  lambda: img)
    return img, lbl


@tf.function
def random_color(x: tf.Tensor, y: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    x = tf.image.random_saturation(x,
                                   config.c.AUG_SATURATION_MIN,
                                   config.c.AUG_SATURATION_MAX)
    x = tf.image.random_brightness(x,
                                   config.c.AUG_BRIGHTNESS)
    x = tf.image.random_contrast(x,
                                 config.c.AUG_CONTRAST_MIN,
                                 config.c.AUG_CONTRAST_MAX)
    return x, y


@tf.function
def noise(x: tf.Tensor, y: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    rand_noise = tf.random.normal(shape=tf.shape(x),
                                  mean=0.0,
                                  stddev=config.c.AUG_NOISE / 255,
                                  dtype=tf.float32)
    x = tf.clip_by_value(x + rand_noise, 0.0, 1.0)
    return x, y


@tf.function
def unsharp(x: tf.Tensor, y: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    # sigma = tf.random.normal(shape=[],
    #                         mean=0.0,
    #                         stddev= config.c.AUG_GAUSS_SIGMA,
    #                         dtype=tf.float32)
    # sigma = tf.cond(sigma < tf.constant(0.),
    #                lambda: sigma,
    #                lambda: tf.constant(0.))
    sigma = np.random.normal(scale=config.c.AUG_GAUSS_SIGMA)
    if sigma < 0.01:
        return x, y

    # with tf.device("/gpu:0"):
    x = tfa.image.gaussian_filter2d(x, filter_shape=[config.c.AUG_GAUSS_FILTER_RADIUS * 2 + 1] * 2, sigma=sigma)
    x = tf.clip_by_value(x, 0.0, 1.0)
    return x, y


@tf.function
def jpg_quality(x: tf.Tensor, y: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    return tf.image.random_jpeg_quality(
        x, config.c.AUG_MIN_JPG_QUALITY, 100
    ), y
