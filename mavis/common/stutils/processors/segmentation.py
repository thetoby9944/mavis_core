import numpy as np
import segmentation_models as sm
import tensorflow as tf

from mavis import config
from mavis.pilutils import save_pil
from mavis.stutils.processors.tfmodel import TfModelProcessor
from mavis.tfutils.dataset import ImageToImageDataset


class SegmentationModelProcessor(TfModelProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(
            input_labels=[
                "Images: Select Column with Input Images Paths for Segmentation",
                "Labels: Select Segmentation Maps (Color Encoded Images)"
            ],
            output_label="Segmented",
            class_info_required=True,
            *args,
            **kwargs
        )

        self.all_backbones = [
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
            'seresnet18', 'seresnet34', 'seresnet50', 'seresnet101', 'seresnet152',
            'resnext50', 'resnext101',
            # 'inceptionv3', 'inceptionresnetv2',
            # 'mobilenet', 'mobilenetv2',
            # 'efficientnetb0', 'efficientnetb1', 'efficientnetb2',
            # 'efficientnetb3', 'efficientnetb4', 'efficientnetb5',
            # 'efficientnetb6', 'efficientnetb7'
        ]

        self.default_backbone = 'resnet34'

        self.all_architectures = {
            "U-Net": sm.Unet,
            "LinkNet": sm.Linknet,
            # "PSPNet": sm.PSPNet,
            # "FPN": sm.FPN
        }

        self.default_architecture = "U-Net"

        if hasattr(config.c, "CLASS_WEIGHT"):
            class_weights = np.array(list(config.c.CLASS_WEIGHT.values()))
            class_weights_normalized = class_weights / np.sum(class_weights)
        else:
            class_weights_normalized = np.ones(len(config.c.CLASS_NAMES))

        self.all_losses = {
            "Focal Loss": sm.losses.categorical_focal_loss,
            "Focal Dice Loss": sm.losses.categorical_focal_dice_loss,
            "Focal Jaccard Loss": sm.losses.categorical_focal_jaccard_loss,
            "Jaccard Loss": sm.losses.jaccard_loss,
            "Weighted Jaccard Focal Loss": sm.losses.JaccardLoss(
                class_weights=tf.convert_to_tensor(class_weights_normalized, dtype=tf.float32),
                per_image=False
            ) + sm.losses.categorical_focal_loss
        }

        self.default_loss = "Weighted Jaccard Focal Loss"

        self.dataset = ImageToImageDataset()

    def models(self):
        yield self.model()

    def model(self):
        self.architecture = config.c.ARCHITECTURE
        model = self.all_architectures[config.c.ARCHITECTURE](
            config.c.BACKBONE,
            classes=1 if config.c.BINARY else len(config.c.CLASS_NAMES),
            encoder_weights='imagenet',
            encoder_freeze=config.c.ENCODER_FREEZE,
            activation='sigmoid' if config.c.BINARY else 'softmax'
        )
        model._name = f"{config.c.ARCHITECTURE}-{config.c.BACKBONE}"
        yield model

    def compile_args(self):
        return {
            "optimizer": config.c.OPTIMIZER,
            "loss": sm.losses.binary_focal_jaccard_loss if config.c.BINARY else self.all_losses[config.c.LOSS],
            "metrics": [sm.metrics.iou_score],
        }

    def training_parameter(self):
        config.c.semantic_segmentation_training_args(self)

    def inference_parameter(self):
        config.c.semantic_segmentation_inference_args(self)

    def store_preds(self, preds, df):
        for i, (path, pred) in enumerate(zip(self.input_args(dropna_jointly=False)[0], preds)):
            img = self.dataset.pred_to_pil(pred)
            df[self.column_out].loc[i] = save_pil(img, path, self.new_dir, self.suffix, self.column_out)
        return df