import json
import traceback
from pathlib import Path

import imgaug
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from pixellib.custom_train import display_box_instances, instance_custom_training
from pixellib.mask_rcnn import DataGenerator, MaskRCNN
from pixellib.utils import extract_bboxes
from tensorflow.python.keras.utils.data_utils import Sequence

from mavis.pilutils import pil
from mavis.tfutils.data_augmentation import flip_single, zoom_single, rotate_single, random_color, flip_pair, \
    zoom_pair, rotate_pair, jpg_quality, noise, unsharp
from mavis.tfutils.preprocessing import one_hot, resize, masking, resnet_preprocess_img
from mavis import config

auto = tf.data.experimental.AUTOTUNE


class TFDatasetWrapper:
    def __init__(self, *args, **kwargs):
        self.label_preprocessing = []
        self.image_preprocessing = []
        self.all_augmentations = {}
        self.ds = None
        self.val_ds = None

    def augment(self):
        augmentations = [self.all_augmentations[a] for a in config.c.AUGMENTATIONS]

        for fn in augmentations:
            self.ds = self.ds.map(fn, num_parallel_calls=auto)

        for fn in self.label_preprocessing:
            self.ds = self.ds.map(lambda x, y: (x, fn(y)), num_parallel_calls=auto)

        for fn in self.image_preprocessing:
            self.ds = self.ds.map(lambda x, y: (fn(x), y), num_parallel_calls=auto)

    def split_and_batch(self):
        if config.c.VAL_SPLIT != 0:
            self.val_ds = self.ds.take(config.c.VAL_SPLIT * config.c.BATCH_SIZE)
            self.ds = self.ds.skip(config.c.VAL_SPLIT * config.c.BATCH_SIZE)
            self.ds = self.ds.repeat()

        else:
            self.ds = self.ds.repeat()

        print("Preparing Batches")
        self.ds = prepare_batch(self.ds)

    def create(self, img_paths, labels):
        if labels is None:
            return self.create_inference(img_paths)
        else:
            return self.create_train(img_paths, labels)

    def create_train(self, img_paths, labels) -> (tf.data.Dataset or Sequence,
                                                  tf.data.Dataset or Sequence):
        raise NotImplementedError

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

    def peek(self) -> None:
        raise NotImplementedError

    def display_pred(self, pred: np.array) -> None:
        raise NotImplementedError


class PixelLibDataset(TFDatasetWrapper):
    def __init__(self, dataset_location, train_instance, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_location = dataset_location
        self.train_instance = train_instance

        self.all_augmentations = {
            "Flip": imgaug.augmenters.Fliplr(0.5),
            "Zoom": imgaug.augmenters.Flipud(0.5),
            "Gauss Filter": imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
        }

        try:
            with open(Path(self.dataset_location) / "train.json") as f:
                json_config = json.load(f)
                self.class_names_from_dataset = [cat["name"] for cat in json_config["categories"]]
                st.info("Loaded COCO dataset. Predicting on classes " + str(self.class_names_from_dataset))
        except:
            st.error("No COCO dataset:")
            st.code(traceback.format_exc())

        try:
            c1, c2 = st.beta_columns([1, 3])
            c2.code(self.dataset_location)
            if c1.button(f"(Re)create COCO dataset from LabelMe files at:"):
                instance_custom_training().load_dataset(self.dataset_location)
        except:
            st.warning("Could not load dataset")
            st.code(traceback.format_exc())

    def create_train(self, img_paths, labels):
        m: MaskRCNN = self.train_instance.model

        augmentation = imgaug.augmenters.Sometimes(0.5, [
            self.all_augmentations[a]
            for a in config.c.AUGMENTATIONS
            if a in self.all_augmentations
        ])

        self.train_instance.load_dataset(self.dataset_location)
        # Data generators
        train_generator = DataGenerator(
            self.train_instance.dataset_train,
            m.config,
            shuffle=True,
            augmentation=augmentation
        )
        sample_in, sample_out = train_generator.__getitem__(0)

        val_generator = DataGenerator(
            self.train_instance.dataset_test,
            m.config,
            shuffle=True
        )

        print(tuple([np.array(net_in).shape for net_in in sample_in]))

        self.ds = train_generator
        self.val_ds = val_generator
        return self.ds, self.val_ds

    def create_inference(self, img_paths) -> (tf.data.Dataset or Sequence):
        raise NotImplementedError

    def peek(self):
        ds = self.train_instance.dataset_train
        image_id = np.random.choice(ds.image_ids)
        image = ds.load_image(image_id)
        mask, class_ids = ds.load_mask(image_id)
        bbox = extract_bboxes(mask)
        # Display image and instances
        out = display_box_instances(image, bbox, mask, class_ids, ds.class_names)
        st.image(pil(out))

    def display_pred(self, pred: np.array) -> None:
        if isinstance(pred, Image.Image):
            st.image(pred)
        else:
            st.write(pred)


class ImageToImageDataset(TFDatasetWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_augmentations = {
            "Flip": flip_pair,
            "Zoom": zoom_pair,
            "Rotate": rotate_pair,
            "Color": random_color,
            "Quality": jpg_quality,
            "Noise": noise,
            "Gauss Filter": unsharp
        }
        self.image_preprocessing = [resnet_preprocess_img]
        self.label_preprocessing = [masking]

    def create_train(self, img_paths, labels):
        """
        Prepare Inference or Training Dataset, works with classes or segmentation maps
        If you prepare a training dataset either pass image_label_path_list or class_label_list + CLASS_NAMES
        """
        print("Creating Dataset while size is", config.c.SIZE)
        image_paths = tf.data.Dataset.from_tensor_slices(img_paths)
        image_paths = image_paths.shuffle(buffer_size=config.c.BUFFER_SIZE, seed=0)
        ds = paths_to_image_ds(image_paths)

        label_paths = tf.data.Dataset.from_tensor_slices(labels).shuffle(
            buffer_size=config.c.BUFFER_SIZE, seed=0
        )
        label_ds = paths_to_image_ds(label_paths)

        self.ds = tf.data.Dataset.zip((ds, label_ds))
        # ds = shuffle_and_cache(ds)
        self.augment()
        self.split_and_batch()

        self.ds = prepare_batch(ds)

        return self.ds, self.val_ds

    def pred_to_pil(self, pred):
        if config.c.BINARY:
            return pil(pred)
        pred = np.argmax(pred, axis=-1)
        img = np.zeros((*pred.shape, 3), dtype=np.uint8)
        for i, col in enumerate(config.c.CLASS_COLORS):
            img[pred == i] = tuple(col)
        return pil(img)

    def display_pred(self, pred):
        st.image(self.pred_to_pil(pred))

    def peek(self):
        # Look at prepared image
        for image_batch, label_batch in self.ds.take(1):
            img_np = image_batch.numpy()[0]
            st.image(pil(img_np))
            st.info(f"Image: Min {np.min(img_np)} Max {np.max(img_np)} Shape {img_np.shape}")
            lbl_np = label_batch.numpy()[0]
            st.image(self.pred_to_pil(lbl_np))
            st.info(f"Label: Min {np.min(lbl_np)} Max {np.max(lbl_np)} Shape {lbl_np.shape}")


class ImageToCategoryDataset(TFDatasetWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_augmentations = {
            "Flip": flip_single,
            "Zoom": zoom_single,
            "Rotate": rotate_single,
            "Color": random_color,
        }
        self.image_preprocessing = [resnet_preprocess_img]

    def display_pred(self, pred):
        st.write(config.c.CLASS_NAMES[np.argmax(pred)])

    def create_train(self, img_paths, labels):
        """
                Prepare Inference or Training Dataset, works with classes or segmentation maps
                If you prepare a training dataset either pass image_label_path_list or class_label_list + CLASS_NAMES
                """
        def set_shape(x: tf.Tensor) -> tf.Tensor:
            x.set_shape([None])
            return x
        image_paths = tf.data.Dataset.from_tensor_slices(img_paths)
        image_paths = image_paths.shuffle(buffer_size=config.c.BUFFER_SIZE, seed=0)
        ds = paths_to_image_ds(image_paths)

        label_paths = tf.data.Dataset.from_tensor_slices(labels).shuffle(
            buffer_size=config.c.BUFFER_SIZE, seed=0
        )
        label_ds = label_paths.map(
            lambda x: tf.py_function(one_hot, [x], [tf.int8]),
            num_parallel_calls=auto
        )

        label_ds = label_ds.map(lambda x: set_shape(x))

        self.ds = tf.data.Dataset.zip((ds, label_ds))
        # ds = shuffle_and_cache(ds)
        self.augment()
        self.split_and_batch()

        return self.ds, self.val_ds

    def peek(self):
        for image_batch, label_batch in self.ds.take(1):
            st.image(pil(image_batch.numpy()[0]))
            st.write(label_batch.numpy()[0])


def decode_img(file_path):
    img = tf.io.read_file(file_path)
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32, saturate=True)
    return img


def process_image_path(file_path):
    # load the raw data from the file as a string
    img = decode_img(file_path)
    img = resize(img)
    return img


def shuffle_and_cache(ds: tf.data.Dataset) -> tf.data.Dataset:
    print("Shuffeling with buffer size ", config.c.BUFFER_SIZE)
    ds = ds
    # ds = ds.cache()
    # for i, batch in enumerate(ds.as_numpy_iterator()):
    #    if not i % 1000:
    #        print("Building cache ...")
    return ds


def prepare_batch(ds: tf.data.Dataset) -> tf.data.Dataset:
    ds = ds.batch(config.c.BATCH_SIZE)
    # `prefetch` lets the dataset fetch batches in the background while the model is training.
    ds = ds.prefetch(buffer_size=auto)
    return ds


def paths_to_image_ds(paths):
    return paths.map(process_image_path, num_parallel_calls=auto)
