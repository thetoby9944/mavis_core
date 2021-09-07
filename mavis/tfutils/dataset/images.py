import numpy as np
import streamlit as st
import tensorflow as tf
from imgaug import SegmentationMapsOnImage

from pilutils import pil
from shelveutils import ConfigDAO
from tfutils.dataset.base import TFDatasetWrapper
from tfutils.dataset.preprocessing import resnet_preprocess_img, masking, process_image_path, normalize_minus_one


class ImageToImageDataset(TFDatasetWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_preprocessing = [resnet_preprocess_img]
        self.label_preprocessing = [masking]

    def img_aug(self, image, label):
        label = SegmentationMapsOnImage(label, shape=image.shape)
        image, label = self.iaa(image=image, segmentation_maps=label)
        return image, label.get_arr()

    def create_train(self, img_paths, labels):
        """
        Prepare Training Dataset (self.ds and self.val_ds)
        """
        print("Creating Dataset while size is", ConfigDAO()["SIZE"])
        image_paths = tf.data.Dataset.from_tensor_slices(img_paths)
        label_paths = tf.data.Dataset.from_tensor_slices(labels)

        paths = tf.data.Dataset.zip((
            image_paths,
            label_paths)
        ).shuffle(buffer_size=ConfigDAO()["BUFFER_SIZE"])

        self.ds = paths.map(lambda x, y: (
            process_image_path(x),
            process_image_path(y)
        ))

        self.augment()
        self.split_and_batch()

        return self.ds, self.val_ds

    def pred_to_pil(self, pred):
        if ConfigDAO()["BINARY"]:
            return pil(pred)
        if ConfigDAO()["INSPECT_CHANNEL"] in ConfigDAO()["CLASS_NAMES"]:
            return pil(pred[:, :, list(ConfigDAO()["CLASS_NAMES"]).index(ConfigDAO()["INSPECT_CHANNEL"])])
        pred = np.argmax(pred, axis=-1)
        img = np.zeros((*pred.shape, 3), dtype=np.uint8)
        for i, col in enumerate(ConfigDAO()["CLASS_COLORS"]):
            img[pred == i] = tuple(col)
        return pil(img)

    def display_pred(self, pred):
        st.image(self.pred_to_pil(pred))

    def peek_dataset(self, ds, n_batches=5):
        if not n_batches:
            return
        # Look at prepared images
        st.info(f"Showing {n_batches} Batches)")
        for col, (image_batch, label_batch) in zip(st.columns(n_batches), ds.take(n_batches)):
            with col:
                batch = image_batch.numpy()
                label_batch = label_batch.numpy()
                st.info(f"Batch Shape: {batch.shape}")
                for img_np, lbl_np in zip(batch, label_batch):
                    st.image(pil(img_np), use_column_width=True)
                    st.info(f"Image: Min {np.min(img_np)} Max {np.max(img_np)} Shape {img_np.shape}")
                    st.image(self.pred_to_pil(lbl_np), use_column_width=True)
                    st.info(f"Label: Min {np.min(lbl_np)} Max {np.max(lbl_np)} Shape {lbl_np.shape}")

    def peek(self):
        with st.expander("Train Dataset Peek"):
            self.peek_dataset(self.ds)
        with st.expander("Val Dataset Peek"):
            self.peek_dataset(self.val_ds, ConfigDAO()["VAL_SPLIT"])


class ImageToReconstructionDataset(ImageToImageDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.image_preprocessing = [resnet_preprocess_img, normalize_minus_one]
        self.label_preprocessing = [resnet_preprocess_img, normalize_minus_one]

    @property
    def lbl_augmentor(self):
        def apply(image):
            return self._iaa.augment_images(image)
        return apply

    def pred_to_pil(self, pred):
        return pil(pred)

