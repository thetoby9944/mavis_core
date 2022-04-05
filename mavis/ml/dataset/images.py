from abc import ABC

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.python.data import AUTOTUNE
from tqdm import tqdm

from mavis.pilutils import pil
from mavis.ml.dataset.base import TFDatasetWrapper


class ImageToImageDataset(TFDatasetWrapper, ABC):
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

        if self.config.DATASET.RESHUFFLE_EACH_ITERATION:
            keep_unshuffled = paths.take(self.config.TRAIN.VAL_SPLIT)
            shuffle_each_iteration = paths.skip(self.config.TRAIN.VAL_SPLIT).shuffle(
                buffer_size=self.config.DATASET.BUFFER_SIZE if self.config.DATASET.BUFFER_SIZE else n_paths,
                reshuffle_each_iteration=True,
                seed=0,
            )
            paths = keep_unshuffled.concatenate(shuffle_each_iteration)

        self.ds = paths.map(
            lambda x, y: (
                self.process_image_path(x),
                self.process_image_path(y)
            ),
            num_parallel_calls=AUTOTUNE
        )

        if not self.config.DATASET.RESHUFFLE_EACH_ITERATION:
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
        raise NotImplementedError

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

