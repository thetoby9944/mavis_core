from pathlib import Path

import tensorflow as tf
import wget
from pixellib.custom_train import instance_custom_training
from pixellib.mask_rcnn import MaskRCNN
from tensorflow.keras import regularizers

from cvutils import LabelMeJsonWrapper
from db import ConfigDAO
from ml.dataset.pixellib import PixelLibDataset


class PixelLibWrapper:
    def __init__(self):
        self.train_instance: instance_custom_training = instance_custom_training()
        self.dataset: PixelLibDataset = None
        self._name = "MaskRCNN"

    def init(self, dataset: PixelLibDataset):
        self.dataset: PixelLibDataset = dataset

    def _compile_keras(self):
        model = self.train_instance.model.keras_model
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        # Optimizer object
        optimizer = ConfigDAO()["OPTIMIZER"]

        # Add Losses
        loss_names = [
            "rpn_class_loss", "rpn_bbox_loss",
            "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"
        ]

        for name in loss_names:
            layer = model.get_layer(name)
            if layer.output in model.losses:
                continue
            loss = tf.reduce_mean(
                input_tensor=layer.output,
                keepdims=True
            ) * self.train_instance.config.LOSS_WEIGHTS.get

            model.add_loss(loss)

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [
            regularizers.l2(self.train_instance.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(input=w), tf.float32)
            for w in model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name
        ]
        model.add_loss(tf.add_n(reg_losses))

        print("compiling keras")
        # Compile
        model.compile(
            optimizer=optimizer,
            loss=[None] * len(model.outputs)
        )

        # Add metrics for losses
        for name in loss_names:
            if name in model.metrics_names:
                continue

            layer = model.get_layer(name)
            model.metrics_names.append(name)
            loss = tf.reduce_mean(
                input_tensor=layer.output,
                keepdims=True
            ) * self.train_instance.config.LOSS_WEIGHTS.get

            model.add_metric(
                loss,
                name=name,
                aggregation='mean'
            )

        return model

    def compile(self, **kwargs):
        self.train_instance.modelConfig(
            network_backbone=ConfigDAO()["BACKBONE"],
            num_classes=len(self.dataset.class_names_from_dataset),
            batch_size=ConfigDAO()["BATCH_SIZE"],
            image_resize_mode="square",
            image_max_dim=ConfigDAO()["SIZE"],
            image_min_dim=ConfigDAO()["SIZE"]
        )

        # Fixed by resnet
        # self.train_instance.BACKBONE_STRIDES = [4, 8, 16, 32, 64]

        # Length of square anchor side in pixels
        self.train_instance.RPN_ANCHOR_SCALES = (1, 2, 4, 8, 16, 32, 64, 128)

        # Non-max suppression threshold to filter RPN proposals.
        # You can increase this during training to generate more propsals.
        self.train_instance.config.RPN_NMS_THRESHOLD = 0.8

        # Minimum scaling ratio. Checked after MIN_IMAGE_DIM and can force further
        # up scaling. For example, if set to 2 then images are scaled up to double
        # the width and height, or more, even if MIN_IMAGE_DIM doesn't require it.
        # However, in 'square' mode, it can be overruled by IMAGE_MAX_DIM.
        self.train_instance.config.IMAGE_MIN_SCALE = 2

        # Number of ROIs per image to feed to classifier/mask heads
        # The Mask RCNN paper uses 512 but often the RPN doesn't generate
        # enough positive proposals to fill this and keep a positive:negative
        # ratio of 1:3. You can increase the number of proposals by adjusting
        # the RPN NMS threshold.
        self.train_instance.config.TRAIN_ROIS_PER_IMAGE = 512

        # Maximum number of ground truth instances to use in one image
        self.train_instance.config.MAX_GT_INSTANCES = 200

        # Max number of final detections
        self.train_instance.config.DETECTION_MAX_INSTANCES = 200
        if (Path("data") / "mask_rcnn_coco.h5").is_file() is not True:
            wget.download(
                "https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.2/mask_rcnn_coco.h5",
                out=str(Path("data"))
            )
        self.train_instance.load_pretrained_model(str(Path("data") / "mask_rcnn_coco.h5"))

        m: MaskRCNN = self.train_instance.model

        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
        layers = "heads" if ConfigDAO()["TRAIN_HEADS_ONLY"] else "all"
        layers = layer_regex[layers]

        m.set_trainable(layers)
        m.keras_model = self._compile_keras()

    def pred_to_mask(self, image, r):
        boxes, masks, class_ids = r['rois'], r['masks'], r['class_ids']

        n_instances = boxes.shape[0]
        if not n_instances:
            print('NO INSTANCES TO DISPLAY')
        else:
            assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

        for i in range(n_instances):
            image[masks[..., i]] = ConfigDAO()["CLASS_COLORS"][list(ConfigDAO()["CLASS_NAMES"]).index(
                (["Background"] + self.dataset.class_names_from_dataset)[class_ids[i]]
            )]

        return image

    def pred_to_labelme(self,
                        img_path,
                        pixel_lib_result,
                        exclude_classes,
                        save_with_contents=False,
                        save_full_path=False):

        boxes, masks, class_ids = pixel_lib_result['rois'], pixel_lib_result['masks'], pixel_lib_result['class_ids']
        label_me = LabelMeJsonWrapper(img_path, save_with_contents, save_full_path)
        print(f"found {len(boxes)} in {img_path}")

        for box, mask, i in zip(boxes, masks.transpose(2, 0, 1), class_ids):
            if ConfigDAO()["CLASS_NAMES"][i] in exclude_classes:
                continue
            label_me.add(mask, i)

        return label_me.labels
