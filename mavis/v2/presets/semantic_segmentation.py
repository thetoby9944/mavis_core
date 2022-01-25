from abc import ABC
from typing import Optional, List, Any

import segmentation_models as sm

from v2.config import MLConfig
from v2.presets.base import BaseProperty, PropertyContainer, BaseConfig, PresetHandler
from v2.presets.losses import LossConfig
from v2.presets.optimizer import OptimizerConfig


class ArchitectureBase(BaseProperty, ABC):
    def parameter_block(self):
        pass


class UNetConfig(ArchitectureBase):
    _name = "U-Net"

    def get(self):
        return sm.Unet


class LinkNetConfig(ArchitectureBase):
    _name = "LinkNet"

    def get(self):
        return sm.Linknet


class ArchitectureConfig(PropertyContainer):
    _name = "Architecture"

    U_NET_PARAMETER: UNetConfig = UNetConfig()
    LINK_NET_PARAMETER: LinkNetConfig = LinkNetConfig()

    ACTIVE = "U-Net"



class BackboneConfig(BaseProperty):
    ACTIVE: str = "resnet50"

    def parameter_block(self):
        self.ACTIVE = self.st.selectbox(
            "Model Backbone", self.all,
            self.all.index(self.ACTIVE)
        )

    def get(self):
        return self.all[self.all.index(self.ACTIVE)]

    @property
    def all(self) -> List[str]:
        return [
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
            'seresnet18', 'seresnet34', 'seresnet50', 'seresnet101', 'seresnet152',
            'resnext50', 'resnext101',
            # 'inceptionv3', 'inceptionresnetv2',
            # 'mobilenet', 'mobilenetv2',
            # 'efficientnetb0', 'efficientnetb1', 'efficientnetb2',
            # 'efficientnetb3', 'efficientnetb4', 'efficientnetb5',
            # 'efficientnetb6', 'efficientnetb7'
        ]


class SegmentationModelsConfig(MLConfig, BaseConfig):

    LOSS: LossConfig = LossConfig()
    OPTIMIZER: OptimizerConfig = OptimizerConfig()

    ARCHITECTURE: ArchitectureConfig = ArchitectureConfig()
    BACKBONE: BackboneConfig = BackboneConfig()

    INSPECT_CHANNEL: Optional[str] = None
    ENCODER_FREEZE: bool = False
    BINARY: bool = False
    WEIGHT_DECAY: float = 0.0001

    def compile_args(self):
        return {
            "optimizer": self.OPTIMIZER.get(),
            "loss": (
                sm.losses.binary_focal_jaccard_loss
                if self.BINARY
                else self.LOSS.get()
            ),
            "metrics":
                [
                    # Total IoU Score
                    sm.metrics.iou_score
                ] + [
                    # Per class IoU Sores
                    sm.metrics.IOUScore(
                        name=f"{str(name).replace(' ', '_')}_IoU",
                        class_indexes=i
                    )
                    for i, name in enumerate(self.CLASSES.CLASS_NAMES)
                ]
        }

    @PresetHandler.access("Semantic Segmentation Settings")
    def parameter_block(self):
        super().parameter_block()
        self.st.markdown("##### Model Optimization")
        self.OPTIMIZER.parameter_block()

        self.WEIGHT_DECAY = self.st.number_input(
            "L2 Kernel Regularization",
            0., 1.,
            self.WEIGHT_DECAY,
            format="%0.6f"
        )

        self.st.write("##### Model Specification")
        self.ARCHITECTURE.parameter_block()
        self.BACKBONE.parameter_block()
        self.LOSS.parameter_block()

        self.ENCODER_FREEZE = self.st.checkbox(
            "Freeze pretrained model",
            self.ENCODER_FREEZE
        )

        self.st.write("##### Output Options")
        if len(self.CLASSES.CLASS_NAMES) == 2:
            self.BINARY = self.st.checkbox(
                "Check to use Sigmoid Output. Default is Softmax",
                self.BINARY
            )

        if not self.BINARY:
            output_opts = ["None"] + list(self.CLASSES.CLASS_NAMES)

            self.INSPECT_CHANNEL = self.st.selectbox(
                "Get probability masks for a specific class. "
                "Select `None` to get color encoded output for all classes. "
                "Color encoding is then based on maximum probability for each class per pixel",
                output_opts,
                output_opts.index(self.INSPECT_CHANNEL)
                if self.INSPECT_CHANNEL in output_opts else 0
            )


class ReconstructionModelConfig(SegmentationModelsConfig):
    RECONSTRUCTION_CLASS = "None"
