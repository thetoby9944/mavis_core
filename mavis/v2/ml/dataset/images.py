import numpy as np
import segmentation_models as sm
import streamlit as st
import tensorflow as tf
from tensorflow.python.data import AUTOTUNE
from tqdm import tqdm

from pilutils import pil
from v2.ml.dataset.base import TFDatasetWrapper
from v2.presets.semantic_segmentation import SegmentationModelsConfig


class ImageToImageDataset(TFDatasetWrapper):
    @staticmethod
    def py_unet_preprocessing(img: tf.Tensor) -> tf.Tensor:
        if np.min(img) < 0:
            img -= np.min(img)
        return sm.get_preprocessing(SegmentationModelsConfig().BACKBONE.ACTIVE)(img)

    @staticmethod
    def resnet_preprocess_img(img: tf.Tensor) -> tf.Tensor:
        img = tf.py_function(
            ImageToImageDataset.py_unet_preprocessing,
            [img],
            tf.float32
        )
        img.set_shape([None for _ in range(3)])
        return img

    @staticmethod
    def mask_by_color(img: tf.Tensor, col: tf.Tensor) -> tf.Tensor:
        img = tf.cast(img == col, dtype=tf.uint8)
        img = tf.reduce_sum(img, axis=-1) == 3
        return tf.cast(img, dtype=tf.float32)

    @staticmethod
    def masking(img: tf.Tensor, stack_axis=-1, keep_dim_binary=False) -> tf.Tensor:
        img = tf.image.convert_image_dtype(img, tf.uint8, saturate=True)
        img = tf.stack([
            ImageToImageDataset.mask_by_color(img, col)
            for col in SegmentationModelsConfig().CLASSES.CLASS_COLORS
        ], axis=stack_axis)

        if not keep_dim_binary and SegmentationModelsConfig().BINARY:
            img = img[..., 1]
        return img

    def __init__(self, config: SegmentationModelsConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config
        self.image_preprocessing = [self.resnet_preprocess_img]
        self.label_preprocessing = [self.masking]

    def img_aug(self, image, label):
        """
        Image
        Parameters
        ----------
        image
        label: Color Coded mask

        Returns
        -------

        """
        mask = self.masking(
            tf.convert_to_tensor(label),
        ).numpy().astype(int)

        transformed = self.iaa(
            image=image,
            label=label,
            mask=mask
        )
        mask = transformed["mask"].astype(float)
        result = np.asarray(self.pred_to_pil(mask)).astype(np.uint8)
        return transformed["image"], result

    def create_train(self, img_paths, labels):
        """
        Prepare Training Dataset (self.ds and self.val_ds)
        """
        n_paths = len(img_paths)

        image_paths = tf.data.Dataset.from_tensor_slices(img_paths)
        label_paths = tf.data.Dataset.from_tensor_slices(labels)

        paths = tf.data.Dataset.zip((
            image_paths,
            label_paths)
        ).shuffle(
            buffer_size=self.config.DATASET.BUFFER_SIZE if self.config.DATASET.BUFFER_SIZE else n_paths,
            reshuffle_each_iteration=False,
            seed=0,
        )

        self.ds = paths.map(
            lambda x, y: (
                self.process_image_path(x),
                self.process_image_path(y)
            ),
            num_parallel_calls=AUTOTUNE
        )
        self.ds = self.ds.take(n_paths).cache()

        st.write("Caching Dataset")
        bar = st.progress(0.)
        for i, build_cache in tqdm(
                enumerate(self.ds.as_numpy_iterator()),
                total=n_paths,
                desc="Building Cache"
        ):
            bar.progress(i / n_paths)

        self.augment()
        self.split_and_batch()

        return self.ds, self.val_ds

    def pred_to_pil(self, pred):
        if self.config.BINARY:
            return pil(pred)
        if self.config.INSPECT_CHANNEL in self.config.CLASSES.CLASS_NAMES:
            return pil(pred[:, :, list(self.config.CLASSES.CLASS_NAMES).index(self.config.INSPECT_CHANNEL)])
        pred = np.argmax(pred, axis=-1)
        # print(np.max(pred), np.min(pred))
        img = np.zeros((*pred.shape, 3), dtype=np.uint8)
        for i, col in enumerate(self.config.CLASSES.CLASS_COLORS):
            img[pred == i] = tuple(col)
        return pil(img, verbose=False)

    def display_pred(self, pred):
        st.image(self.pred_to_pil(pred))

    def peek_dataset(self, ds, n_batches=3):
        if not n_batches:
            return
        # Look at prepared images
        st.info(f"Showing {n_batches} Batches)")
        for col, (image_batch, label_batch) in zip(st.columns(n_batches), ds.take(n_batches)):
            with col:
                batch = image_batch.numpy()
                label_batch = label_batch.numpy()
                st.info(f"Batch Shape: {batch.shape}")
                for i, img_np, lbl_np in zip(np.arange(20), batch, label_batch):
                    st.image(
                        pil(img_np),
                        use_column_width=True,
                        output_format="png"
                    )
                    st.info(
                        f"Image: "
                        f"Min {np.min(img_np)} "
                        f"Max {np.max(img_np)} "
                        f"Shape {img_np.shape}"
                    )
                    st.image(
                        self.pred_to_pil(lbl_np),
                        use_column_width=True,
                        output_format="png"
                    )
                    st.info(
                        f"Label: "
                        f"Min {np.min(lbl_np)} "
                        f"Max {np.max(lbl_np)} "
                        f"Median {np.median(lbl_np)} "
                        f"Shape {lbl_np.shape}"
                    )

    def peek(self):
        with st.expander("Train Dataset Peek"):
            self.peek_dataset(self.ds)
        with st.expander("Val Dataset Peek"):
            self.peek_dataset(self.val_ds, self.config.TRAIN.VAL_SPLIT)


class ImageToReconstructionDataset(ImageToImageDataset):
    @staticmethod
    def normalize_minus_one(img: tf.Tensor) -> tf.Tensor:
        return (img - 0.5) * 2

    def __init__(self, config: SegmentationModelsConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self.image_preprocessing = [self.resnet_preprocess_img, self.normalize_minus_one]
        self.label_preprocessing = [self.resnet_preprocess_img, self.normalize_minus_one]

    def img_aug(self, image, label):
        kwargs = {"image": image, "label": label}
        transformed = self.iaa(**kwargs)
        return transformed["image"], transformed["label"]

    def pred_to_pil(self, pred):
        return pil(pred)
