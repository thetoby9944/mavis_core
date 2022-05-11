import random
from abc import ABC
from typing import List, Tuple, Union, Literal

import albumentations as A
import cv2
import numpy as np
from pydantic import Field

from mavis.presets.base import ExtendedEnum, PropertiesContainer, PropertiesContainerProperty


class BorderMode(str, ExtendedEnum):
    """
    Border mode values enum for rotation border treatment in the `imgaug` library
    """
    REFLECT = "reflect"
    CONSTANT = "constant"
    EDGE = "edge"
    SYMMETRIC = "symmetric"
    WRAP = "wrap"

    @staticmethod
    def to_cv2(border_mode: str):
        return dict(
            reflect=cv2.BORDER_REFLECT,
            constant=cv2.BORDER_CONSTANT,
            edge=cv2.BORDER_REFLECT_101,
            symmetric=cv2.BORDER_REPLICATE,
            wrap=cv2.BORDER_WRAP
        )[border_mode]


class AugmentationBaseConfig(PropertiesContainerProperty):
    p: float = 0.5
    apply_on_label: bool = True

    def key(self, key):
        return f"{key}{self.name}{self.i}"

    def get(self) -> A.BasicTransform:
        raise NotImplementedError

    def parameter_block(self):
        super().parameter_block()
        self.p = float(self.st.slider(
            f"{self.name} Probability", 0., 1.,
            float(self.p),
            key=self.key("prob")
        ))
        self.apply_on_label = bool(self.st.checkbox(
            f"Apply on Label",
            self.apply_on_label,
            key=self.key("apply"),
            help="For categorical and mask data, this field has no effect"
        ))


class FlipUDConfig(AugmentationBaseConfig):
    """
    kwargs for the iaa.FlipUP
    """
    name: Literal["Flip Up Down"] = "Flip Up Down"

    def get(self):
        return A.VerticalFlip(
            p=self.p
        )


class FlipLRConfig(AugmentationBaseConfig):
    """
    kwargs for the iaa.FlipLR
    """
    name: Literal["Flip LR"] = "Flip LR"

    def get(self):
        return A.HorizontalFlip(
            p=self.p
        )


class ZoomOutConfig(AugmentationBaseConfig):
    """
    kwargs for iaa.Crop functions
    """
    name: Literal["Zoom Out"] = "Zoom Out"
    sample_independently: bool = False
    percent: float = 0.2
    pad_mode: BorderMode = BorderMode.REFLECT

    def get(self):
        return A.CropAndPad(
            p=self.p,
            sample_independently=self.sample_independently,
            percent=(0, self.percent),
            pad_mode=BorderMode.to_cv2(self.pad_mode)
        )

    def parameter_block(self):
        self.pad_mode = self.st.selectbox(
            "Void Area Treatment for Zoom Out. Extend border values:",
            BorderMode.values(),
            BorderMode.values().index(self.pad_mode)
            if self.pad_mode in BorderMode.values()
            else 0,
            key=self.key("")
        )


class ZoomInConfig(AugmentationBaseConfig):
    name: Literal["Zoom In"] = "Zoom In"

    sample_independently: bool = False
    percent: float = 0.2

    def get(self):
        return A.CropAndPad(
            p=self.p,
            sample_independently=self.sample_independently,
            percent=(-self.percent, 0),
        )


class RandomCropConfig(AugmentationBaseConfig):
    name: Literal["Random Crop"] = "Random Crop"

    width: int = 32
    height: int = 32

    def parameter_block(self):
        self.st.write(f"### Step {self.i + 1}: {self.name}")

        self.width = int(self.st.number_input(
            "Random Crop Width", 1, 10 * 100,
            value=int(self.width),
            key=self.key("width")
        ))
        self.height = int(self.st.number_input(
            "Random Crop Height", 1, 10 * 100,
            value=int(self.height),
            key=self.key("height")
        ))

    def get(self):
        return A.RandomCrop(
            p=1.,
            # p=1. Must be 1 because otherwise
            # dataset has inconsistently sized patches
            width=self.width,
            height=self.height,
            always_apply=True
        )


class CenterCropConfig(AugmentationBaseConfig):
    name: Literal["Center Crop"] = "Center Crop"

    width: int = 32
    height: int = 32

    def parameter_block(self):
        self.st.write(f"### Step {self.i + 1}: {self.name}")

        self.width = int(self.st.number_input(
            "Center Crop Width", 1, 10 * 100,
            value=int(self.width),
            key=self.key("width")
        ))
        self.height = int(self.st.number_input(
            "Center Crop Height", 1, 10 * 100,
            value=int(self.height),
            key=self.key("height")
        ))

    def get(self):
        return A.CenterCrop(
            p=1.,
            # p=1. Must be 1 because otherwise
            # dataset has inconsistently sized patches
            width=self.width,
            height=self.height,
            always_apply=True
        )


class CustomCropNonEmptyMaskIfExists(A.CropNonEmptyMaskIfExists):
    _target_str_singular = "mask"
    _target_str_plural = "masks"

    def sample_more_likely_in_the_middle(self, range_length):
        epsilon = 0.000001
        return int(np.clip(random.betavariate(5, 5), 0, 1 - epsilon) * range_length)

    def update_params(self, params, **kwargs):
        super().update_params(params, **kwargs)
        if self._target_str_singular in kwargs:
            mask = self._preprocess_mask(kwargs[self._target_str_singular])
        elif self._target_str_plural in kwargs and len(kwargs[self._target_str_plural]):
            masks = kwargs[self._target_str_plural]
            mask = self._preprocess_mask(masks[0])
            for m in masks[1:]:
                mask |= self._preprocess_mask(m)
        else:
            raise RuntimeError("Can not find mask for CropNonEmptyMaskIfExists")

        mask_height, mask_width = mask.shape[:2]

        if mask.any():
            mask = mask.sum(axis=-1) if mask.ndim == 3 else mask
            non_zero_yx = np.argwhere(mask)
            y, x = random.choice(non_zero_yx)
            x_min = x - self.sample_more_likely_in_the_middle(self.width)
            y_min = y - self.sample_more_likely_in_the_middle(self.height)
            x_min = np.clip(x_min, 0, mask_width - self.width)
            y_min = np.clip(y_min, 0, mask_height - self.height)
        else:
            x_min = random.randint(0, mask_width - self.width)
            y_min = random.randint(0, mask_height - self.height)

        x_max = x_min + self.width
        y_max = y_min + self.height

        params.update({"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max})
        return params


class MaskCropConfig(AugmentationBaseConfig):
    width: int = 32
    height: int = 32

    name: Literal["Random Crop patch where Mask is not empty if exists"] = (
        "Random Crop patch where Mask is not empty if exists"
    )

    class MaskCropNonEmptyMaskIfExists(CustomCropNonEmptyMaskIfExists):
        _target_str_singular = "mask"
        _target_str_plural = "masks"

    def parameter_block(self):
        super().parameter_block()
        self.width = int(self.st.number_input(
            "Crop Width", 1, 10 * 100,
            value=int(self.width),
            key=self.key("width")
        ))
        self.height = int(self.st.number_input(
            "Crop Height", 1, 10 * 100,
            value=int(self.height),
            key=self.key("height")
        ))

    def get(self):
        return self.MaskCropNonEmptyMaskIfExists(
            p=self.p,
            width=self.width,
            height=self.height,
            ignore_channels=[0]
        )


class HardNegativeCropConfig(AugmentationBaseConfig):
    width: int = 32
    height: int = 32

    class HardNegatiaveCropNonEmptyMaskIfExists(CustomCropNonEmptyMaskIfExists):
        _target_str_singular = "wrong_predictions"
        _target_str_plural = "wrong_predictions"

    name: Literal["Random Crop patch where Wrong Prediction is not empty if exists"] = (
        "Random Crop patch where Wrong Prediction is not empty if exists"
    )

    def parameter_block(self):
        super().parameter_block()
        self.width = int(self.st.number_input(
            "Crop Width", 1, 10 * 100,
            value=int(self.width),
            key=self.key("width")
        ))
        self.height = int(self.st.number_input(
            "Crop Height", 1, 10 * 100,
            value=int(self.height),
            key=self.key("height")
        ))

    def get(self):
        return self.HardNegatiaveCropNonEmptyMaskIfExists(
            p=self.p,
            width=self.width,
            height=self.height,
            ignore_channels=[0]
        )



class BlurConfig(AugmentationBaseConfig):
    """
    kwargs for iaa.GaussianBlur
    """
    name: Literal["Gaussian Blur"] = "Gaussian Blur"
    kernel_radius: Tuple[int, int] = (3, 7)

    def get(self):
        return A.GaussianBlur(
            p=self.p,
            blur_limit=self.kernel_radius
        )

    def parameter_block(self):
        super().parameter_block()
        self.kernel_radius = (int(self.st.number_input(
            "Min kernel diameter (px)", 1, 100,
            int(self.kernel_radius[0]),
            key=self.key("min")
        )), int(self.st.number_input(
            "Max kernel diameter (px)", 1, 100,
            value=int(self.kernel_radius[1]),
            key=self.key("max")
        )))
        assert (
                self.kernel_radius[0] % 2 == 1 and
                self.kernel_radius[1] % 2 == 1
        ), "Diameters muss be odd"


class ContrastConfig(AugmentationBaseConfig):
    """
    kwargs for iaa.Contrast
    """
    name: Literal["Contrast"] = "Contrast"
    alpha: Tuple[float, float] = (-0.2, 0.2)

    def get(self):
        return A.RandomBrightnessContrast(
            p=self.p,
            contrast_limit=self.alpha,
            brightness_limit=0
        )

    def parameter_block(self):
        super().parameter_block()
        self.alpha = (self.st.number_input(
            "Min contrast multiplier", -1., 0.,
            float(self.alpha[0]),
            key=self.key("min")
        ), self.st.number_input(
            "Max contrast  multiplier", 0., 1.,
            value=float(self.alpha[1]),
            key=self.key("max")
        ))
        assert self.alpha[0] < self.alpha[1], "Min must be smaller than max"


class SaturationConfig(AugmentationBaseConfig):
    """
    kwargs for iaa.Contrast
    """
    name: Literal["Saturation"] = "Saturation"
    mul_saturation: Tuple[int, int] = (1, 6)

    def get(self):
        return A.HueSaturationValue(
            p=self.p,
            hue_shift_limit=0,
            val_shift_limit=0,
            sat_shift_limit=self.mul_saturation
        )

    def parameter_block(self):
        super().parameter_block()
        self.mul_saturation = (int(self.st.number_input(
            "Min saturation shift limit [0-255]", 0., 1.,
            float(self.mul_saturation[0]),
            key=self.key("min")
        )), int(self.st.number_input(
            "Max saturation shift limit [1-255]", 1, 255,
            value=int(self.mul_saturation[1]),
            key=self.key("max")
        )))
        assert self.mul_saturation[0] < self.mul_saturation[1], "Min must be smaller than max"


class BrightnessConfig(AugmentationBaseConfig):
    name: Literal["Brightness"] = "Brightness"
    mul: Tuple[float, float] = (-0.05, 0.05)

    def get(self):
        return A.RandomBrightnessContrast(
            p=self.p,
            brightness_limit=self.mul,
            contrast_limit=0
        )

    def parameter_block(self):
        super().parameter_block()
        self.mul = (self.st.number_input(
            "Min brightness multiplier", -1., 0.,
            float(self.mul[0]),
            key=self.key("min")
        ), self.st.number_input(
            "Max brightness multiplier", 0., 1.,
            value=float(self.mul[1]),
            key=self.key("max")
        ))
        assert self.mul[0] < self.mul[1], "Min must be smaller than max"


class CompressionConfig(AugmentationBaseConfig):
    """
    kwargs for iaa.JPGCompression
    """
    name: Literal["JPG Compression"] = "JPG Compression"
    compression: List[int] = [0, 95]

    def get(self):
        return A.JpegCompression(
            p=self.p,
            quality_lower=self.compression[0],
            quality_upper=self.compression[1]
        )

    def parameter_block(self):
        super().parameter_block()
        self.compression = [self.st.number_input(
            "Lowest JPG quality (percent of original)", 0., 100.,
            float(self.compression[0]),
            key=self.key("min")
        ), self.st.number_input(
            "Highest JPG quality (percent of original)", 0., 100.,
            float(self.compression[1]),
            key=self.key("max")
        )]
        assert self.compression[0] < self.compression[1], "Min must be smaller than max"


class AdditiveGaussianNoiseConfig(AugmentationBaseConfig):
    name: Literal["Additive Gaussian Noise"] = "Additive Gaussian Noise"
    scale: Tuple[float, float] = [0., 15.]

    def get(self):
        return A.GaussNoise(
            p=self.p,
            var_limit=self.scale
        )

    def parameter_block(self):
        super().parameter_block()
        self.scale = (self.st.number_input(
            "Minimum additive noise in pixel values", 0., 255.,
            float(self.scale[0]),
            key=self.key("min")
        ), self.st.number_input(
            "Maximum additive noise in pixel values", 0., 255.,
            float(self.scale[1]),
            key=self.key("max")
        ))
        assert self.scale[0] < self.scale[1], "Min must be smaller than max"


class ChangeColorTemperatureConfig(AugmentationBaseConfig):
    name: Literal["Color Temperature"] = "Color Temperature"

    def get(self):
        return A.RandomGamma(
            p=self.p
        )


class ColorJitterConfig(AugmentationBaseConfig):
    name: Literal["Color Temperature"] = "Random HSV Color Jitter"

    def get(self):
        return A.ColorJitter(
            p=self.p
        )


class AffineTransformConfig(AugmentationBaseConfig):
    name: Literal["Rotation"] = "Rotation"

    mode: BorderMode = BorderMode.REFLECT
    rotate: Tuple[float, float] = (-360., 360.)

    def get(self):
        return A.Affine(
            p=self.p,
            mode=BorderMode.to_cv2(self.mode),
            rotate=self.rotate
            # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            # shear=(-8, 8)
        )

    def parameter_block(self):
        super().parameter_block()
        self.rotate = (self.st.number_input(
            "Min rotation angles", -360., 0.,
            float(self.rotate[0]),
            key=self.key("min")
        ), self.st.number_input(
            "Max rotation angles", 0., 360.,
            float(self.rotate[1]),
            key=self.key("max")
        ))
        self.mode = self.st.selectbox(
            "Void Area Treatment for rotations. Extend border values:",
            BorderMode.values(),
            BorderMode.values().index(self.mode)
            if self.mode in BorderMode.values()
            else 0,
            key=self.key("mode")
        )


class InvertBlendConfig(AugmentationBaseConfig):
    name: Literal["Blended Cloudy Masks"] = "Blended Cloudy Masks"

    def get(self):
        return A.RandomFog()


class ElasticTransformConfig(AugmentationBaseConfig):
    name: Literal["Elastic Transform"] = "Elastic Transform"

    def get(self):
        return A.ElasticTransform()


class CutoutConfig(AugmentationBaseConfig):
    name: Literal["Cut-Out (Coarse Dropout)"] = "Cut-Out (Coarse Dropout)"

    max_holes: int = 8
    min_holes: int = 0
    min_width: int = 1
    max_width: int = 8
    min_height: int = 1
    max_height: int = 8
    mask_fill = False

    def parameter_block(self):
        super().parameter_block()
        self.max_holes = int(self.st.number_input(
            "Max holes", 0, 1000,
            int(self.max_holes),
            key=self.key("max_holes")
        ))
        self.max_height = int(self.st.number_input(
            "Max height", 0, 1000,
            int(self.max_height),
            key=self.key("max_height")
        ))
        self.max_width = int(self.st.number_input(
            "Max width", 0, 1000,
            int(self.max_width),
            key=self.key("max_width")
        ))

        self.min_holes = int(self.st.number_input(
            "Min holes", 0, 1000,
            int(self.min_holes),
            key=self.key("min_holes")
        ))
        self.min_height = int(self.st.number_input(
            "Min height", 0, 1000,
            int(self.min_height),
            key=self.key("min_height")
        ))
        self.min_width = int(self.st.number_input(
            "Min width", 0, 1000,
            int(self.min_width),
            key=self.key("min_width")
        ))
        self.mask_fill = self.st.checkbox(
            "Dropout Masks",
            self.mask_fill,
            key=self.key("mask")
        )

    def get(self) -> A.BasicTransform:
        return A.CoarseDropout(
            p=self.p,
            max_holes=self.max_holes,
            max_width=self.max_width,
            max_height=self.max_height,
            min_holes=self.min_holes or None,
            min_width=self.min_width or None,
            min_height=self.min_height or None,
            mask_fill_value=(0 if self.mask_fill else None)
        )


class AugmentationConfig(PropertiesContainer):
    ACTIVE: List[Union[
        RandomCropConfig,
        MaskCropConfig,
        FlipUDConfig,
        FlipLRConfig,
        ZoomInConfig,
        ZoomOutConfig,
        BlurConfig,
        AdditiveGaussianNoiseConfig,
        ContrastConfig,
        SaturationConfig,
        BrightnessConfig,
        CompressionConfig,
        ChangeColorTemperatureConfig,
        AffineTransformConfig,
        InvertBlendConfig,
        ElasticTransformConfig,
        CutoutConfig,
        ColorJitterConfig,
        CenterCropConfig,
        HardNegativeCropConfig
    ]] = Field([], discriminator="name")

    @property
    def _parent_property_cls(self):
        return AugmentationBaseConfig

    GLOBAL_PROBABILITY: float = 1.
    RANDOM_ORDER: bool = False

    def parameter_block(self):
        super().parameter_block()

        if self.ACTIVE:
            self.st.write("### General Data Augmentation Settings ")
            self.GLOBAL_PROBABILITY = self.st.slider(
                "Global probability factor for each augmentation to be applied", 0., 1.,
                self.GLOBAL_PROBABILITY,
                help="Set to 1. if random crop is applied, otherwise crops can have different sizes"
            )
            self.RANDOM_ORDER = self.st.checkbox(
                "Random order of Data Augmentations",
                self.RANDOM_ORDER
            )

            if False and self.st.checkbox("Preview"):
                preview_image_file = self.st.file_uploader("Sample image", "png")
                if preview_image_file is not None:
                    # preview_image = PIL.Image(preview_image_file.read())
                    self.get()
            self.st.write("---")

    def get(self) -> A.Compose:
        return A.Compose(
            transforms=[
                A.Compose(
                    p=self.GLOBAL_PROBABILITY,
                    transforms=[a.get()],
                    additional_targets=(
                        {
                            "wrong_predictions": "mask",
                            "label": "image"
                        }
                        if a.apply_on_label
                        else None
                    )
                )
                for a in self.ACTIVE
            ]
        )
