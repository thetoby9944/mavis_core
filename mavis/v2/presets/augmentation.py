from typing import List, Tuple
import random
import numpy as np

import albumentations as A
import cv2

from v2.presets.base import ExtendedEnum, BaseProperty


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


class AugmentationTypes(str, ExtendedEnum):
    RAND_CROP = "Random Crop"
    MASK_CROP = "Random Crop patch where Mask is not empty if exists"
    FLIP_UD = "Flip Up Down"
    FLIP_LR = "Flip LR"
    ZOOM_IN = "Zoom In"
    ZOOM_OUT = "Zoom Out"
    BLUR = "Gaussian Blur"
    ADDITIVE_NOISE = "Additive Gaussian Noise"
    CONTRAST = "Contrast"
    SATURATION = "Saturation"
    BRIGHTNESS = "Brightness"
    COMPRESSION = "JPG Compression"
    COLOR_TEMPERATURE = "Color Temperature"
    ROTATION = "Rotation"
    BLEND = "Blended Cloudy Masks"
    ELASTIC_TRANSFORM = "Elastic Transform"
    CUTOUT = "Cut-Out (Coarse Dropout)"


class AugmentationBaseConfig(BaseProperty):
    _name: AugmentationTypes = None
    p: float = 0.5

    def get(self) -> A.BasicTransform:
        raise NotImplementedError

    def parameter_block(self):
        self.p = float(self.st.slider(
            f"{self.name} Probability", 0., 1.,
            float(self.p)
        ))

    @property
    def name(self):
        return self._name


class FlipUDConfig(AugmentationBaseConfig):
    """
    kwargs for the iaa.FlipUP
    """
    _name = AugmentationTypes.FLIP_UD

    def get(self):
        return A.VerticalFlip(
            p=self.p
        )


class FlipLRConfig(AugmentationBaseConfig):
    """
    kwargs for the iaa.FlipLR
    """
    _name = AugmentationTypes.FLIP_LR

    def get(self):
        return A.HorizontalFlip(
            p=self.p
        )


class ZoomOutConfig(AugmentationBaseConfig):
    """
    kwargs for iaa.Crop functions
    """
    _name = AugmentationTypes.ZOOM_OUT
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
            else 0
        )


class ZoomInConfig(AugmentationBaseConfig):
    _name = AugmentationTypes.ZOOM_IN

    sample_independently: bool = False
    percent: float = 0.2

    def get(self):
        return A.CropAndPad(
            p=self.p,
            sample_independently=self.sample_independently,
            percent=(-self.percent, 0),
        )


class RandomCropConfig(AugmentationBaseConfig):
    _name = AugmentationTypes.RAND_CROP

    width: int = 32
    height: int = 32

    def parameter_block(self):
        self.width = int(self.st.number_input(
            "Random Crop Width", 1, 10 * 100,
            value=int(self.width)
        ))
        self.height = int(self.st.number_input(
            "Random Crop Height", 1, 10 * 100,
            value=int(self.height)
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


class MaskCropConfig(AugmentationBaseConfig):
    class CustomCropNonEmptyMaskIfExists(A.CropNonEmptyMaskIfExists):

        def sample_more_likely_in_the_middle(self, range_length):
            epsilon = 0.000001
            return int(np.clip(random.betavariate(5, 5), 0, 1-epsilon) * range_length)

        def update_params(self, params, **kwargs):
            super().update_params(params, **kwargs)
            if "mask" in kwargs:
                mask = self._preprocess_mask(kwargs["mask"])
            elif "masks" in kwargs and len(kwargs["masks"]):
                masks = kwargs["masks"]
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

    _name = AugmentationTypes.MASK_CROP

    width: int = 32
    height: int = 32

    def parameter_block(self):
        super().parameter_block()
        self.width = int(self.st.number_input(
            "Crop Width", 1, 10 * 100,
            value=int(self.width)
        ))
        self.height = int(self.st.number_input(
            "Crop Height", 1, 10 * 100,
            value=int(self.height)
        ))

    def get(self):
        return self.CustomCropNonEmptyMaskIfExists(
            p=self.p,
            width=self.width,
            height=self.height,
            ignore_channels=[0]
        )


class BlurConfig(AugmentationBaseConfig):
    """
    kwargs for iaa.GaussianBlur
    """
    _name = AugmentationTypes.BLUR
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
            int(self.kernel_radius[0])
        )), int(self.st.number_input(
            "Max kernel diameter (px)", 1, 100,
            value=int(self.kernel_radius[1])
        )))
        assert (
                self.kernel_radius[0] % 2 == 1 and
                self.kernel_radius[1] % 2 == 1
        ), "Diameters muss be odd"


class ContrastConfig(AugmentationBaseConfig):
    """
    kwargs for iaa.Contrast
    """
    _name = AugmentationTypes.CONTRAST
    alpha: Tuple[float, float] = (-0.2, 0.2)

    def get(self):
        return A.RandomContrast(
            p=self.p,
            limit=self.alpha
        )

    def parameter_block(self):
        super().parameter_block()
        self.alpha = (self.st.number_input(
            "Min contrast multiplier", -1., 0.,
            float(self.alpha[0])
        ), self.st.number_input(
            "Max contrast  multiplier", 0., 1.,
            value=float(self.alpha[1])
        ))
        assert self.alpha[0] < self.alpha[1], "Min must be smaller than max"


class SaturationConfig(AugmentationBaseConfig):
    """
    kwargs for iaa.Contrast
    """
    _name = AugmentationTypes.SATURATION
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
            float(self.mul_saturation[0])
        )), int(self.st.number_input(
            "Max saturation shift limit [1-255]", 1, 255,
            value=int(self.mul_saturation[1])
        )))
        assert self.mul_saturation[0] < self.mul_saturation[1], "Min must be smaller than max"


class BrightnessConfig(AugmentationBaseConfig):
    _name = AugmentationTypes.BRIGHTNESS
    mul: Tuple[float, float] = (-0.05, 0.05)

    def get(self):
        return A.RandomBrightness(
            p=self.p,
            limit=self.mul
        )

    def parameter_block(self):
        super().parameter_block()
        self.mul = (self.st.number_input(
            "Min brightness multiplier", -1., 0.,
            float(self.mul[0])
        ), self.st.number_input(
            "Max brightness multiplier", 0., 1.,
            value=float(self.mul[1])
        ))
        assert self.mul[0] < self.mul[1], "Min must be smaller than max"


class CompressionConfig(AugmentationBaseConfig):
    """
    kwargs for iaa.JPGCompression
    """
    _name = AugmentationTypes.COMPRESSION
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
            float(self.compression[0])
        ), self.st.number_input(
            "Highest JPG quality (percent of original)", 0., 100.,
            float(self.compression[1])
        )]
        assert self.compression[0] < self.compression[1], "Min must be smaller than max"


class AdditiveGaussianNoiseConfig(AugmentationBaseConfig):
    _name = AugmentationTypes.ADDITIVE_NOISE
    scale: Tuple[float, float] = [0., 15.]

    def get(self):
        return A.IAAAdditiveGaussianNoise(
            p=self.p,
            scale=self.scale
        )

    def parameter_block(self):
        super().parameter_block()
        self.scale = (self.st.number_input(
            "Minimum additive noise in pixel values", 0., 255.,
            float(self.scale[0])
        ), self.st.number_input(
            "Maximum additive noise in pixel values", 0., 255.,
            float(self.scale[1])
        ))
        assert self.scale[0] < self.scale[1], "Min must be smaller than max"


class ChangeColorTemperatureConfig(AugmentationBaseConfig):
    _name = AugmentationTypes.COLOR_TEMPERATURE

    def get(self):
        return A.RandomGamma(
            p=self.p
        )


class AffineTransformConfig(AugmentationBaseConfig):
    _name = AugmentationTypes.ROTATION

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
            float(self.rotate[0])
        ), self.st.number_input(
            "Max rotation angles", 0., 360.,
            float(self.rotate[1])
        ))
        self.mode = self.st.selectbox(
            "Void Area Treatment for rotations. Extend border values:",
            BorderMode.values(),
            BorderMode.values().index(self.mode)
            if self.mode in BorderMode.values()
            else 0
        )


class InvertBlendConfig(AugmentationBaseConfig):
    _name = AugmentationTypes.BLEND

    def get(self):
        return A.RandomFog()


class ElasticTransformConfig(AugmentationBaseConfig):
    _name = AugmentationTypes.ELASTIC_TRANSFORM

    def get(self):
        return A.ElasticTransform()


class CutoutConfig(AugmentationBaseConfig):
    _name = AugmentationTypes.CUTOUT

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
            int(self.max_holes)
        ))
        self.max_height = int(self.st.number_input(
            "Max height", 0, 1000,
            int(self.max_height)
        ))
        self.max_width = int(self.st.number_input(
            "Max width", 0, 1000,
            int(self.max_width)
        ))

        self.min_holes = int(self.st.number_input(
            "Min holes", 0, 1000,
            int(self.min_holes)
        ))
        self.min_height = int(self.st.number_input(
            "Min height", 0, 1000,
            int(self.min_height)
        ))
        self.min_width = int(self.st.number_input(
            "Min width", 0, 1000,
            int(self.min_width)
        ))
        self.mask_fill = self.st.checkbox("Dropout Masks", self.mask_fill)

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


class AugmentationConfig(BaseProperty):
    RANDOM_CROP: RandomCropConfig = RandomCropConfig()
    MASK_CROP: MaskCropConfig = MaskCropConfig()
    FLIP_UD: FlipUDConfig = FlipUDConfig()
    FLIP_LR: FlipLRConfig = FlipLRConfig()
    ZOOM_IN: ZoomInConfig = ZoomInConfig()
    ZOOM_OUT: ZoomOutConfig = ZoomOutConfig()
    BLUR: BlurConfig = BlurConfig()
    ADDITIVE_NOISE: AdditiveGaussianNoiseConfig = AdditiveGaussianNoiseConfig()
    CONTRAST: ContrastConfig = ContrastConfig()
    SATURATION: SaturationConfig = SaturationConfig()
    BRIGHTNESS: BrightnessConfig = BrightnessConfig()
    COMPRESSION: CompressionConfig = CompressionConfig()
    COLOR_TEMPERATURE: ChangeColorTemperatureConfig = ChangeColorTemperatureConfig()
    ROTATION: AffineTransformConfig = AffineTransformConfig()
    BLEND: InvertBlendConfig = InvertBlendConfig()
    ELASTIC_TRANSFORM: ElasticTransformConfig = ElasticTransformConfig()
    CUTOUT: CutoutConfig = CutoutConfig()

    ACTIVE_AUGMENTATIONS: List[AugmentationTypes] = []
    GLOBAL_PROBABILITY: float = 0.5
    RANDOM_ORDER: bool = False

    def parameter_block(self):
        self.ACTIVE_AUGMENTATIONS = self.st.multiselect(
            "Data Augmentations",
            AugmentationTypes.values(),
            [a for a in self.ACTIVE_AUGMENTATIONS]
        )

        if self.ACTIVE_AUGMENTATIONS:
            self.GLOBAL_PROBABILITY = self.st.slider(
                "Probability factor for each augmentation to be applied", 0., 1.,
                self.GLOBAL_PROBABILITY
            )

            self.RANDOM_ORDER = self.st.checkbox(
                "Random order of Data Augmentations",
                self.RANDOM_ORDER
            )

            for key in self.ACTIVE_AUGMENTATIONS:
                self.st.markdown(f"###### {key}")
                self.all_augmentations()[key].parameter_block()

            if self.st.checkbox("Preview"):
                preview_image_file = self.st.file_uploader("Sample image", "png")
                if preview_image_file is not None:
                    # preview_image = PIL.Image(preview_image_file.read())
                    self.get()

    def all_augmentations(self):
        return {
            AugmentationTypes.RAND_CROP: self.RANDOM_CROP,
            AugmentationTypes.MASK_CROP: self.MASK_CROP,
            AugmentationTypes.FLIP_UD: self.FLIP_UD,
            AugmentationTypes.FLIP_LR: self.FLIP_LR,
            AugmentationTypes.ZOOM_IN: self.ZOOM_IN,
            AugmentationTypes.ZOOM_OUT: self.ZOOM_OUT,
            AugmentationTypes.BLUR: self.BLUR,
            AugmentationTypes.ADDITIVE_NOISE: self.ADDITIVE_NOISE,
            AugmentationTypes.CONTRAST: self.CONTRAST,
            AugmentationTypes.SATURATION: self.SATURATION,
            AugmentationTypes.BRIGHTNESS: self.BRIGHTNESS,
            AugmentationTypes.COMPRESSION: self.COMPRESSION,
            AugmentationTypes.COLOR_TEMPERATURE: self.COLOR_TEMPERATURE,
            AugmentationTypes.ROTATION: self.ROTATION,
            AugmentationTypes.BLEND: self.BLEND,
            AugmentationTypes.ELASTIC_TRANSFORM: self.ELASTIC_TRANSFORM,
            AugmentationTypes.CUTOUT: self.CUTOUT
        }

    def dict(self, *args, **kwargs):
        exclude = {
            attr
            for attr, value in self.__dict__.items()
            if isinstance(value, AugmentationBaseConfig) and value._name not in self.ACTIVE_AUGMENTATIONS
        }
        return super().dict(exclude=exclude)

    def get(self) -> A.Compose:
        return A.Compose(
            transforms=[
                A.Compose(
                    p=self.GLOBAL_PROBABILITY,
                    transforms=[self.all_augmentations()[a].get()],
                    additional_targets={"label": "image"}
                )
                for a in self.ACTIVE_AUGMENTATIONS
                if a in self.all_augmentations()
            ],
            additional_targets={"label": "image"}
        )
