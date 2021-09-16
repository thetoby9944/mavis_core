import imgaug.augmenters as iaa
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.python.keras.utils.data_utils import Sequence

from ml.dataset.preprocessing import auto, prepare_batch, paths_to_image_ds
from config import BasePreset
from db import ConfigDAO


class DatasetSettings(BasePreset):
    def __init__(self):
        self.AUGMENTATIONS = ConfigDAO([])["AUGMENTATIONS"]
        self.AUG_ROTATE = ConfigDAO("mirror")["AUG_ROTATE"]
        self.AUG_MIN_JPG_QUALITY = ConfigDAO(65)["AUG_MIN_JPG_QUALITY"]
        self.AUG_NOISE = ConfigDAO(10)["AUG_NOISE"]
        self.AUG_CONTRAST_MIN = ConfigDAO(0.7)["AUG_CONTRAST_MIN"]
        self.AUG_CONTRAST_MAX = ConfigDAO(1.3)["AUG_CONTRAST_MAX"]
        self.AUG_SATURATION_MIN = ConfigDAO(0.6)["AUG_SATURATION_MIN"]
        self.AUG_SATURATION_MAX = ConfigDAO(1.6)["AUG_SATURATION_MAX"]
        self.AUG_BRIGHTNESS = ConfigDAO(0.05)["AUG_BRIGHTNESS"]
        self.AUG_ZOOM_PERCENT = ConfigDAO(0.1)["AUG_ZOOM_PERCENT"]
        self.AUG_GAUSS_SIGMA = ConfigDAO(1.)["AUG_GAUSS_SIGMA"]
        self.AUG_GAUSS_FILTER_RADIUS = ConfigDAO(1)["AUG_GAUSS_FILTER_RADIUS"]
        self.BUFFER_SIZE = ConfigDAO(1000)["BUFFER_SIZE"]
        self.AUG_PROBABILITY = ConfigDAO(0.5)["AUG_PROBABILITY"]
        self.AUG_RANDOM_ORDER = ConfigDAO(False)["AUG_RANDOM_ORDER"]

    def update(self):
        ConfigDAO()["AUGMENTATIONS"] = self.AUGMENTATIONS
        ConfigDAO()["AUG_ROTATE"] = self.AUG_ROTATE
        ConfigDAO()["AUG_MIN_JPG_QUALITY"] = self.AUG_MIN_JPG_QUALITY
        ConfigDAO()["AUG_NOISE"] = self.AUG_NOISE
        ConfigDAO()["AUG_CONTRAST_MIN"] = self.AUG_CONTRAST_MIN
        ConfigDAO()["AUG_CONTRAST_MAX"] = self.AUG_CONTRAST_MAX
        ConfigDAO()["AUG_SATURATION_MIN"] = self.AUG_SATURATION_MIN
        ConfigDAO()["AUG_SATURATION_MAX"] = self.AUG_SATURATION_MAX
        ConfigDAO()["AUG_BRIGHTNESS"] = self.AUG_BRIGHTNESS
        ConfigDAO()["AUG_ZOOM_PERCENT"] = self.AUG_ZOOM_PERCENT
        ConfigDAO()["AUG_GAUSS_SIGMA"] = self.AUG_GAUSS_SIGMA
        ConfigDAO()["AUG_GAUSS_FILTER_RADIUS"] = self.AUG_GAUSS_FILTER_RADIUS
        ConfigDAO()["BUFFER_SIZE"] = self.BUFFER_SIZE
        ConfigDAO()["AUG_PROBABILITY"] = self.AUG_PROBABILITY
        ConfigDAO()["AUG_RANDOM_ORDER"] = self.AUG_RANDOM_ORDER

    def _augmentation_parameter_block(self, model_processor):
        st.markdown("### Data Augmentation")
        all_aug = list(model_processor.all_augmentations.keys())
        if st.button("Add all"):
            self.AUGMENTATIONS = all_aug
        self.AUGMENTATIONS = st.multiselect(
            "Data Augmentations",
            all_aug,
            [a for a in self.AUGMENTATIONS if a in all_aug]
        )
        rotate_opts = ["constant", "edge", "symmetric", "reflect", "wrap"]
        self.AUG_ROTATE = st.selectbox(
            "Void Area Treatment. Extend border values:",
            rotate_opts,
            rotate_opts.index(self.AUG_ROTATE)
            if self.AUG_ROTATE in rotate_opts
            else 0
        )
        self.AUG_SATURATION_MIN = st.slider(
            "Random Saturation - Minimum multiplier", 0., 1.,
            self.AUG_SATURATION_MIN
        )
        self.AUG_SATURATION_MAX = st.slider(
            "Random Saturation - Maximum multiplier", 1., 2.,
            self.AUG_SATURATION_MAX
        )
        self.AUG_BRIGHTNESS = st.slider(
            "Random Brightness - Maximum deviation in percent", 0., 1.,
            self.AUG_BRIGHTNESS
        )
        self.AUG_CONTRAST_MIN = st.slider(
            "Random Contrast - Minimum multiplier", 0., 1.,
            self.AUG_CONTRAST_MIN
        )
        self.AUG_CONTRAST_MAX = st.slider(
            "Random Contrast - Maximum multiplier", 1., 2.,
            self.AUG_CONTRAST_MAX
        )
        self.AUG_MIN_JPG_QUALITY = st.slider(
            "JPG Quality - Minimum Percentage", 0, 100,
            self.AUG_MIN_JPG_QUALITY
        )
        self.AUG_NOISE = st.slider(
            "Random Noise - Std. Deviation in pixel values", 0, 255,
            self.AUG_NOISE
        )
        self.AUG_ZOOM_PERCENT = st.slider(
            "Zoom Percentage - Maximum Zoom / Crop multiplier", 0., 1.,
            self.AUG_ZOOM_PERCENT
        )
        self.AUG_GAUSS_FILTER_RADIUS = st.slider(
            "Gauss Filter Radius", 0, 3,
            self.AUG_GAUSS_FILTER_RADIUS
        )
        self.AUG_GAUSS_SIGMA = st.slider(
            "Gauss Std Dev. of Sigma Value", 0., 3.,
            self.AUG_GAUSS_SIGMA
        )
        self.AUG_RANDOM_ORDER = st.checkbox(
            "Random order of Data Augmentations",
            self.AUG_RANDOM_ORDER
        )
        self.AUG_PROBABILITY = st.slider(
            "Probability factor for each augmentation to be applied", 0., 1.,
            self.AUG_PROBABILITY
        )

    def _advanced_training_duration_parameter_block(self):
        self.BUFFER_SIZE = st.number_input(
            "Shuffle Buffer Size. 1 - no shuffling", 1, 100 * 1000,
            int(self.BUFFER_SIZE)
        )

    @BasePreset.access("Dataset Settings")
    def dataset_options(self, processor):
        self._advanced_training_duration_parameter_block()
        self._augmentation_parameter_block(processor)


class TFDatasetWrapper:
    def __init__(self, *args, **kwargs):
        # List of tensorflow functions to preprocess labels
        self.label_preprocessing = []
        # List of tensorflow functions to preprocess images
        self.image_preprocessing = []
        # Infinite tf.data.DataSet with random augmentation
        self.ds = None
        # Finite tf.data.Dataset with random augmentation
        self.val_ds = None
        # Whether labels are augmented (e.g. masks)
        self.augment_label = True
        # ImgAug sequential pipeline
        self.iaa = None
        self.border_mode = ConfigDAO()["AUG_ROTATE"] if ConfigDAO()["AUG_ROTATE"] in {
            "constant", "edge", "symmetric", "reflect", "wrap"
        } else "constant"

        DatasetSettings().dataset_options(self)

    @property
    def all_augmentations(self): return {
        "Flip Left Right": iaa.Fliplr(1),
        "Flip Up Down": iaa.Flipud(1),
        # random crops
        "Crop (Zoom In)": iaa.CropAndPad(
            sample_independently=False,
            percent=(-ConfigDAO()["AUG_ZOOM_PERCENT"], 0),
        ),
        "Zoom Out": iaa.CropAndPad(
            sample_independently=False,
            percent=(0, ConfigDAO()["AUG_ZOOM_PERCENT"]),
            pad_mode=self.border_mode

        ),
        "Gaussian Blur": iaa.GaussianBlur(
            sigma=(0.0, ConfigDAO()["AUG_GAUSS_SIGMA"])
        ),
        # Strengthen or weaken the contrast in each image.
        "Contrast": iaa.LinearContrast(
            alpha=(ConfigDAO()["AUG_CONTRAST_MIN"], ConfigDAO()["AUG_CONTRAST_MAX"])
        ),
        "JPG Compression": iaa.JpegCompression(
            compression=(0, int(100 - ConfigDAO()["AUG_MIN_JPG_QUALITY"]))
        ),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        "Additive Gaussian Noise": iaa.AdditiveGaussianNoise(
            loc=0,
            scale=(0.0, ConfigDAO()["AUG_NOISE"]),
            per_channel=0.1
        ),
        "Color Temperature": iaa.ChangeColorTemperature(),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        "Brightness": iaa.Multiply(
            mul=(1 - ConfigDAO()["AUG_BRIGHTNESS"], 1 + ConfigDAO()["AUG_BRIGHTNESS"]),
            per_channel=0.1),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        "Affine Transform": iaa.Affine(
            # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-180, 180),
            # shear=(-8, 8)
            mode=self.border_mode
        ),
        "Invert Blend": iaa.BlendAlphaMask(
            iaa.InvertMaskGen(0.5, iaa.VerticalLinearGradientMaskGen()),
            iaa.Clouds()
        ),
        "Elastic Transform": iaa.ElasticTransformation(
            alpha=(0, 10),
            sigma=(2, 4)
        )

    }

    def _init_iaa_augmentor(self):
        self.iaa = iaa.Sequential([
            iaa.Sometimes(
                ConfigDAO()["AUG_PROBABILITY"],
                self.all_augmentations[a]
            )
            for a in ConfigDAO()["AUGMENTATIONS"]
            if a in self.all_augmentations
        ], random_order=ConfigDAO()["AUG_RANDOM_ORDER"])

    def _apply_img_aug(self, img_tensor, lbl_tensor):

        lbl_shape = tf.shape(lbl_tensor)
        if self.augment_label:
            lbl_tensor = tf.image.convert_image_dtype(lbl_tensor, tf.uint8)
            lbl_shape = tf.shape(lbl_tensor)
        lbl_dtype = lbl_tensor.dtype

        img_tensor = tf.image.convert_image_dtype(img_tensor, tf.uint8)
        img_shape = tf.shape(img_tensor)
        img_dtype = img_tensor.dtype

        img_tensor, lbl_tensor = tf.numpy_function(
            self.img_aug,
            [img_tensor, lbl_tensor],
            [img_dtype, lbl_dtype]
        )

        img_tensor = tf.reshape(img_tensor, shape=img_shape)
        lbl_tensor = tf.reshape(lbl_tensor, shape=lbl_shape)

        if self.augment_label:
            lbl_tensor = tf.image.convert_image_dtype(lbl_tensor, tf.float32)
        img_tensor = tf.image.convert_image_dtype(img_tensor, tf.float32)

        return img_tensor, lbl_tensor

    def img_aug(self, image, label):
        raise NotImplementedError

    def augment(self):
        self._init_iaa_augmentor()

        self.ds = self.ds.map(self._apply_img_aug, num_parallel_calls=auto)

        for fn in self.label_preprocessing:
            self.ds = self.ds.map(lambda x, y: (x, fn(y)), num_parallel_calls=auto)

        for fn in self.image_preprocessing:
            self.ds = self.ds.map(lambda x, y: (fn(x), y), num_parallel_calls=auto)

    def split_and_batch(self):
        n_val = ConfigDAO()["VAL_SPLIT"] * ConfigDAO()["BATCH_SIZE"]
        if n_val != 0:
            self.val_ds = self.ds.take(n_val)
            self.val_ds = prepare_batch(self.val_ds)
            self.val_ds = self.val_ds.repeat()
            self.ds = self.ds.skip(n_val)

        self.ds = self.ds.shuffle(
            buffer_size=ConfigDAO()["BUFFER_SIZE"],
            reshuffle_each_iteration=True
        )
        self.ds = self.ds.repeat()
        self.ds = prepare_batch(self.ds)

    def create(self, img_paths, labels):
        # Convert to str if is windows path
        img_paths = [str(img_path) for img_path in img_paths]
        if labels is None:
            return self.create_inference(img_paths)
        else:
            return self.create_train(img_paths, labels)

    def create_inference(self, img_paths) -> (tf.data.Dataset or Sequence):
        """
        Prepare Inference or Training Dataset, works with classes or segmentation maps
        If you prepare a training dataset either pass image_label_path_list or class_label_list + CLASS_NAMES
        """
        image_paths = tf.data.Dataset.from_tensor_slices(img_paths)
        ds = paths_to_image_ds(image_paths)

        for fn in self.image_preprocessing:
            ds = ds.map(fn, num_parallel_calls=auto)

        print("Preparing Batches")
        self.ds = prepare_batch(ds)
        return self.ds

    def create_train(self, img_paths, labels) -> None:
        raise NotImplementedError

    def peek(self) -> None:
        raise NotImplementedError

    def display_pred(self, pred: np.array) -> None:
        raise NotImplementedError
