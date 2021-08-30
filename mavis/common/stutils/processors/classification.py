import numpy as np
import tensorflow as tf

from mavis import config
from mavis.pdutils import image_columns
from mavis.stutils.processors.tfmodel import TfModelProcessor
from mavis.tfutils.dataset import ImageToCategoryDataset
from mavis.tfutils.models.resnet_transfer import ResNet_Transfer


class ClassificationModelProcessor(TfModelProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(
            input_labels=[
                "Select Input Images Paths for Classification",
                "Select Labels. Paths in this column will correspond row-wise to inputs."
            ],
            inputs_column_filter=[
                image_columns,
                None
            ],
            output_label="Pred. Classes",
            class_info_required=True,
            color_info_required=False,
            *args, **kwargs
        )
        self.dataset = ImageToCategoryDataset()

    def training_parameter(self):
        config.c.classification_training_args(self)

    def inference_parameter(self):
        pass

    def models(self):
        self.class_names = config.c.CLASS_NAMES
        yield self.model()

    def model(self):
        model = ResNet_Transfer(len(self.class_names))
        model._name = "ResNet_Transfer"
        return model

    def store_preds(self, preds, df):
        for i, pred in enumerate(preds):
            df[self.column_out].loc[i] = config.c.CLASS_NAMES[np.argmax(pred)]
        return df

    def compile_args(self):
        return {
            "optimizer": config.c.OPTIMIZER,
            "loss": 'categorical_crossentropy',
            "metrics": [tf.keras.metrics.CategoricalAccuracy()]
        }