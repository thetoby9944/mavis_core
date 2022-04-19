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
        def euclidean_dist_transform_loss(
                gt: tfa.types.TensorLike,
                pred: tfa.types.TensorLike
        ):
            def get_scaled_distance_map(
                    images: tfa.types.TensorLike
            ):
                dist = tfa.image.euclidean_dist_transform(
                    images=tf.image.convert_image_dtype(images, tf.uint8),
                )
                return dist / (tf.reduce_max(dist, axis=0) + tf.keras.backend.epsilon())

            """
            Calculate the difference between gt and pred euclidean distance transforms 
            """

            pred_dist = get_scaled_distance_map(pred)
            gt_dist = get_scaled_distance_map(gt)

            return (
                    100 * sm.losses.CategoricalFocalLoss()
                    + sm.losses.DiceLoss()
            )(gt_dist, pred_dist)

        return euclidean_dist_transform_loss


class LossConfig(PropertyContainer):
    _name = "Loss"

    CFL_PARAMETER: CFLConfig = CFLConfig()
    FDL_PARAMETER: FDLConfig = FDLConfig()
    FJL_PARAMETER: FJLConfig = FJLConfig()
    JAC_PARAMETER: JACConfig = JACConfig()
    WJF_PARAMETER: WJFConfig = WJFConfig()
    DST_PARAMETER: DistTransformLossConfig = DistTransformLossConfig()

    ACTIVE: str = "Weighted Jacquard Focal Loss"
