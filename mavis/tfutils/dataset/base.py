import imgaug.augmenters as iaa
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import Sequence

from shelveutils import ConfigDAO
from tfutils.dataset.preprocessing import auto, prepare_batch, paths_to_image_ds


class TFDatasetWrapper:
    def __init__(self, *args, **kwargs):
        self.label_preprocessing = []
        self.image_preprocessing = []
        self.all_augmentations = {
            "Flip Left Right": iaa.Fliplr(1),
            "Flip Up Down": iaa.Flipud(1),
            # random crops
            "Crop (Zoom In)": iaa.CropAndPad(
                sample_independently=False,
                percent=(-0.1, 0),
            ),
            "Zoom Out": iaa.CropAndPad(
                sample_independently=False,
                percent=(0, 0.5),
            ),
            "Gauss Filter": iaa.GaussianBlur(
                sigma=(0.0, ConfigDAO()["AUG_GAUSS_SIGMA"])
            ),
            # Strengthen or weaken the contrast in each image.
            "Contrast": iaa.LinearContrast(
                alpha=(ConfigDAO()["AUG_CONTRAST_MIN"], ConfigDAO()["AUG_CONTRAST_MAX"])
            ),
            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            "Noise": iaa.AdditiveGaussianNoise(
                loc=0,
                scale=(0.0, ConfigDAO()["AUG_NOISE"]),
                per_channel=0.1
            ),
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
                rotate=(-45, 45),
                # shear=(-8, 8)
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
        self.ds = None
        self.val_ds = None

    def _init_iaa_augmentor(self):
        self._iaa_augmentor = iaa.Sequential([
            iaa.Sometimes(
                0.5,
                self.all_augmentations[a]
            )
            for a in ConfigDAO()["AUGMENTATIONS"]
            if a in self.all_augmentations
        ], random_order=False)
        self._iaa_augmentor = self._iaa_augmentor.to_deterministic()

    def _iaa_aug(self, img, lbl):
        self._init_iaa_augmentor()

        def apply(tensor, iaa_augmentor):
            if iaa_augmentor is None:
                return tensor

            tensor = tf.image.convert_image_dtype(tensor, tf.uint8)
            tensor = tf.expand_dims(tensor, 0)
            img_shape = tf.shape(tensor)
            img_dtype = tensor.dtype
            tensor = tf.numpy_function(iaa_augmentor,
                                       [tensor],
                                       img_dtype)
            tensor = tf.reshape(tensor, shape=img_shape)
            tensor = tensor[0, ...]
            return tf.image.convert_image_dtype(tensor, tf.float32)

        img = apply(img, self.img_augmentor)
        lbl = apply(lbl, self.lbl_augmentor)

        return img, lbl

    def get_iaa_augmentor(self):
        return self._iaa_augmentor

    @property
    def img_augmentor(self):
        def apply(image):
            return self.get_iaa_augmentor().augment_images(image)

        return apply

    @property
    def lbl_augmentor(self):
        return None

    def augment(self):
        self.ds = self.ds.map(self._iaa_aug, num_parallel_calls=auto)

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
            self.ds = self.ds.repeat()

        else:
            self.ds = self.ds.repeat()

        print("Preparing Batches")
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
