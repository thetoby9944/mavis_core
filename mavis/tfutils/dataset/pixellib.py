import json
import traceback
from pathlib import Path

import numpy as np
import seaborn as sns
import streamlit as st
import tensorflow as tf
from PIL import Image
from pixellib.custom_train import instance_custom_training, display_box_instances
from pixellib.mask_rcnn import MaskRCNN, DataGenerator, load_image_gt
from tensorflow.python.keras.utils.data_utils import Sequence

from pilutils import pil
from tfutils.dataset.base import TFDatasetWrapper


def hex_2_rgb(hex_str:str)-> (int, int, int):
    return (
        int(hex_str.lstrip("#")[0: 2], 16),
        int(hex_str.lstrip("#")[2: 4], 16),
        int(hex_str.lstrip("#")[4: 6], 16)
    )

class PixelLibDataset(TFDatasetWrapper):
    def __init__(self, dataset_location, train_instance, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Dataset at:", dataset_location)
        self.dataset_location = dataset_location
        self.train_instance = train_instance

        try:
            with open(Path(self.dataset_location) / "train.json") as f:
                json_config = json.load(f)
            self._parse_coco_dataset(json_config)
        except:
            st.error("No dataset available. "
                     "Try to create or recreate. "
                     "Dataset creation expects train / test LabelMe Folders")
            st.code(traceback.format_exc())

        try:
            c1, c2 = st.columns([1, 3])
            c2.code(self.dataset_location)
            if c1.button(f"(Re)create dataset from LabelMe files at:"):
                instance_custom_training().load_dataset(self.dataset_location)
        except:
            st.warning("Could not load dataset. "
                       "Probably the LabelMe Folder is containing files or folders, "
                       "that do not belong in there")
            st.code(traceback.format_exc())

    def _parse_coco_dataset(self, json_config):
        self.class_names_from_dataset = [cat["name"] for cat in json_config["categories"]]
        st.info("Predicting on classes " + str(self.class_names_from_dataset))
        n_classes = len(self.class_names_from_dataset)

        if st.checkbox("Show classes in as .json preset"):
            class_colors = list([
                hex_2_rgb(hex_str) for hex_str in
                sns.color_palette("husl", n_colors=20).as_hex()[:n_classes]
            ])
            st.write({
                "Class Indices": list(range(n_classes + 1)),
                "Class Names": ["background"] + list(self.class_names_from_dataset),
                "Class Colors": [(0., 0., 0.)] + class_colors
            })

    def create_train(self, img_paths, labels):
        m: MaskRCNN = self.train_instance.model

        self.train_instance.load_dataset(self.dataset_location)
        # Data generators
        train_generator = DataGenerator(
            self.train_instance.dataset_train,
            m.config,
            shuffle=True,
            augmentation=self.iaa
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
        col1, col2 = st.columns(2)
        for i in range(20):
            image_id = np.random.choice(ds.image_ids)
            image, image_meta, gt_class_ids, gt_boxes, gt_masks = load_image_gt(
                dataset=ds,
                config=self.train_instance.config,
                image_id=image_id,
                training=False,
                augmentation=self.iaa
            )

            out = display_box_instances(image.astype(np.uint8), gt_boxes, gt_masks, gt_class_ids, ds.class_names)
            col1.image(pil(out))
            col2.image(pil(image))

    def display_pred(self, pred: np.array) -> None:
        if isinstance(pred, Image.Image):
            st.image(pred)
        else:
            st.write(pred)
