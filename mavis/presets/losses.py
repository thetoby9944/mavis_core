from abc import ABC
from typing import Dict, List

import numpy as np
import segmentation_models as sm
import tensorflow as tf
import tensorflow_addons as tfa
from segmentation_models.losses import CategoricalFocalLoss, DiceLoss

from mavis.presets.base import BaseProperty, PropertyContainer


class LossBase(BaseProperty, ABC):
    _class_names: List = []

    def parameter_block(self):
        pass


class CFLConfig(LossBase):
    _name = "Categorical Focal Loss"

    def get(self):
        return sm.losses.CategoricalFocalLoss(4, 4)


class FDLConfig(LossBase):
    _name = "Focal Dice Loss"

    def get(self):
        return (100 * CategoricalFocalLoss()) + DiceLoss()


class FJLConfig(LossBase):
    _name = "Focal Jacquard Loss"

    def get(self):
        return sm.losses.categorical_focal_jaccard_loss


class JACConfig(LossBase):
    _name = "Jacquard Loss"

    def get(self):
        return sm.losses.jaccard_loss


class WJFConfig(LossBase):
    _name = "Weighted Jacquard Focal Loss"
    CLASS_WEIGHT: Dict[int, float] = {
        i: 1.
        for i in range(2)
    }

    def parameter_block(self):
        self.CLASS_WEIGHT = {i: self.st.number_input(
            f"Treat every occurrence of a {name} value as "
            f"`X` instances during training loss calculation",
            1., 1000.,
            self.CLASS_WEIGHT[i] if i in self.CLASS_WEIGHT else 1.,
            key=f"class_input_{i}"
        ) for i, name in enumerate(self._class_names)}

    def get(self):
        if self.CLASS_WEIGHT is not None:
            class_weights = np.array(list(self.CLASS_WEIGHT.values()))
            class_weights_normalized = class_weights / np.sum(class_weights)
        else:
            class_weights_normalized = np.ones(len(self._class_names))

        return sm.losses.JaccardLoss(
            class_weights=tf.convert_to_tensor(class_weights_normalized, dtype=tf.float32),
            per_image=False
        ) + sm.losses.categorical_focal_loss

class DistTransformLossConfig(LossBase):
    _name = "Dist Transoform Loss"

    def get(self):
        def local_convolutional_distance_transform(inputs, k):
            dt = tfa.image.gaussian_filter2d(
                inputs,
                filter_shape=(k, k),
                sigma=1
            )
            return dt

        def normalize_per_image_per_channel(inputs):
            args = dict(axis=[1, 2], keepdims=True)

            d = tf.cast(inputs, float)

            min_val = tf.reduce_min(d, **args)
            max_val = tf.reduce_max(d, **args)

            d = tf.where((max_val - min_val) > 0, (d - min_val) / (max_val - min_val), d)

            max_val = tf.reduce_max(d, **args)
            d = tf.where(max_val > 1., d / max_val, d)
            return d


        def get_scaled_distance_map(
                inputs: tfa.types.TensorLike
        ):
            # https://www.is.uni-due.de/fileadmin/literatur/publikation/pham20gcpr_website.pdf

            tf.cast(inputs, float)
            k = 3

            diag = tf.shape(inputs)[1] * np.sqrt(2.)

            s = tf.math.ceil(diag / (k // 2))

            d = inputs * 0
            for i in tf.range(int(s)):
                d_i = local_convolutional_distance_transform(inputs, k)
                d_i = tf.cast(d_i, float)

                scaling = (i * k // 2)
                scaling = tf.cast(scaling, float)

                d = tf.where(d_i > 0, d + scaling + d_i, d)

                inputs = tf.where(d_i > 0, 1., inputs)

            d = normalize_per_image_per_channel(d)
            return d


        def cleaned_dist_transform(
                gt: tfa.types.TensorLike,
        ):
            """
            Calculate the dist transofrm for a tensor with applying soft thresholding
            """
            # https://openreview.net/pdf?id=B1eIcvS45V

            n_channels = tf.shape(gt)[-1]
            probability_threshold = (n_channels - 1) / n_channels

            gt_relu = tf.keras.activations.relu(gt - tf.cast(probability_threshold, float))
            gt_dist = get_scaled_distance_map(tf.identity(gt))

            return gt_dist

        def categorical_focal_loss(gt, pr, gamma=2.0, alpha=0.25, class_indexes=None, **kwargs):
            r"""Implementation of Focal Loss from the paper in multiclass classification

            Formula:
                loss = - gt * alpha * ((1 - pr)^gamma) * log(pr)

            Args:
                gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
                pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
                alpha: the same as weighting factor in balanced cross entropy, default 0.25
                gamma: focusing parameter for modulating factor (1-p), default 2.0
                class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.

            """

            backend = tf.keras.backend

            # clip to prevent NaN's and Inf's
            pr = backend.clip(pr, backend.epsilon(), 1.0 - backend.epsilon())

            # Calculate eucldiean dist transforms
            gt_dist = cleaned_dist_transform(gt)

            # Calculate focal loss
            loss = - gt * (alpha * backend.pow((1 - pr), gamma) * backend.log(pr))

            loss = loss * (gt_dist + 1)

            return 100 * backend.mean(loss)

        return categorical_focal_loss


class LossConfig(PropertyContainer):
    _name = "Loss"

    CFL_PARAMETER: CFLConfig = CFLConfig()
    FDL_PARAMETER: FDLConfig = FDLConfig()
    FJL_PARAMETER: FJLConfig = FJLConfig()
    JAC_PARAMETER: JACConfig = JACConfig()
    WJF_PARAMETER: WJFConfig = WJFConfig()
    DST_PARAMETER: DistTransformLossConfig = DistTransformLossConfig()

    ACTIVE: str = "Weighted Jacquard Focal Loss"
