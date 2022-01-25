import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import seaborn as sns

from db import ModelDAO, current_model_dir, ProjectDAO
from v2.presets.augmentation import AugmentationConfig
from v2.presets.base import BaseProperty, BaseConfig


class ClassConfig(BaseProperty):
    CLASS_NAMES = ["Background", "Foreground"]
    CLASS_INDICES = [0, 1]
    CLASS_COLORS = [(0, 0, 0), (255, 255, 255)]

    def parameter_block(self):
        self.st.write("##### Class Information")
        from ui.widgets import rgb_picker

        column_list = ["Class Indices", "Class Names", "Class Colors"]

        if self.CLASS_NAMES is None or not len(self.CLASS_NAMES):
            self.st.error("You need to create Class Information first")
            self.CLASS_NAMES = []
            self.CLASS_COLORS = []
            self.CLASS_INDICES = []

        if self.st.checkbox("Show JSON format of Class Config"):
            self.st.write("Example")
            self.st.json(
                '{"Class Indices": [0, 1], '
                '"Class Names": ["Background", "Foreground"], '
                '"Class Colors": [[0, 0, 0], [255, 255, 255]]}'
            )

            if len(self.CLASS_COLORS):
                self.CLASS_COLORS = np.stack(self.CLASS_COLORS)

            default_json = json.dumps({
                k: v for k, v in
                zip(column_list, [self.CLASS_INDICES, self.CLASS_NAMES, self.CLASS_COLORS])
            })

            text_input = self.st.text_input(
                "JSON Configuration", default_json
            )

            class_infos = list(json.loads(text_input).values())

            self.CLASS_INDICES = class_infos[0]
            self.CLASS_NAMES = class_infos[1]
            self.CLASS_COLORS = [tuple(i) for i in class_infos[2]]

        else:
            default_classes = self.CLASS_NAMES
            n_classes = int(self.st.number_input(
                "Number of Classes", 2,
                value=len(default_classes) if len(default_classes) else 2
            ))

            hex_colors = [
                (
                    sns.color_palette("cubehelix", n_colors=n_classes).as_hex()[i]
                    if not len(self.CLASS_COLORS) or i >= len(self.CLASS_COLORS) else
                    '#%02x%02x%02x' % tuple(self.CLASS_COLORS[i])

                ) for i in range(n_classes)
            ]

            class_info = [(
                i,
                self.st.text_input(
                    f"Class Name for Network Output {i}",
                    default_classes[i] if len(default_classes) and i < len(default_classes) else ""
                ),
                rgb_picker(
                    f"Color for Network Output {i}", hex_colors[i]
                ))
                for i in range(n_classes)
            ]

            self.CLASS_INDICES, self.CLASS_NAMES, self.CLASS_COLORS = list(zip(*class_info))


class ModelConfig(BaseProperty):
    MODEL_PATH: Path = ModelDAO().get_all()[0]

    def parameter_block(self):
        self.st.write("##### Model Path")
        new_path = Path(current_model_dir()) / f"{datetime.now():%y%m%d_%H-%M}_{Path(ProjectDAO().get()).stem}.h5"

        if self.st.checkbox("Custom Model Path"):
            new_path = Path(self.st.text_input(
                "Path", Path(current_model_dir()) / "new_model.h5"
            ))

        if self.st.button(f"Add new model named: `{new_path.name}`"):
            ModelDAO().add(new_path)

        all_model_paths = ModelDAO().get_all()
        self.MODEL_PATH = self.st.selectbox(
            "Select Model",
            all_model_paths,
            all_model_paths.index(self.MODEL_PATH)
            if self.MODEL_PATH in all_model_paths else 0
        )
        if Path(self.MODEL_PATH).is_file() and self.st.button("Get download link"):
            path = Path(self.MODEL_PATH)
            with open(path, "rb") as f:
                self.st.download_button(
                    label="Download Model",
                    data=f,
                    file_name=path.name,
                )


class TrainingConfig(BaseProperty):
    """
    Base Training configuration
    """
    name: str = "Default"

    EPOCHS: int = 10
    BATCH_SIZE: int = 2
    VAL_SPLIT: int = 1
    STEPS_PER_EPOCH: int = 10
    EARLY_STOPPING: int = 5
    CONTINUE_TRAINING: Path = ""

    IMPLICIT_BACKGROUND_CLASS: bool = False
    RED_ON_PLATEAU_PATIENCE: int = 0

    def parameter_block(self):
        self.st.markdown("##### Training Duration")

        self.BATCH_SIZE = int(self.st.number_input(
            "Batch Size", 1, 1024,
            int(self.BATCH_SIZE)
        ))

        self.EPOCHS = int(self.st.number_input(
            "Epochs", 1, 10 * 1000,
            int(self.EPOCHS)
        ))

        self.STEPS_PER_EPOCH = int(self.st.number_input(
            "Steps per epoch", 1, 10 * 1000,
            int(self.STEPS_PER_EPOCH)
        ))

        self.VAL_SPLIT = int(self.st.number_input(
            "Validation Images in batches", 0, 100,
            int(self.VAL_SPLIT)
        ))

        self.EARLY_STOPPING = int(self.st.number_input(
            "Early Stopping - Patience in epochs", 1, 10 * 1000,
            int(self.EARLY_STOPPING)
        ))

        self.RED_ON_PLATEAU_PATIENCE = int(self.st.number_input(
            "Reduce learning rate on plateau",
            0, 10 * 1000,
            int(self.RED_ON_PLATEAU_PATIENCE),
            help=
            "Patience in epochs. "
            "Set to zero to ignore or when you use a learning rate schedule."
        ))

        all_model_paths = ModelDAO().get_all()
        self.CONTINUE_TRAINING = self.st.selectbox(
            """Select a base model to retrain from. Leave empty to train a new model""",
            [""] + all_model_paths,
            all_model_paths.index(self.CONTINUE_TRAINING)
            if self.CONTINUE_TRAINING in all_model_paths else 0
        )


class DatasetConfig(BaseProperty):
    AUGMENTATION = AugmentationConfig()

    BUFFER_SIZE: int = 0
    RESHUFFLE_EACH_ITERATION: bool = False

    def parameter_block(self):
        self.st.markdown("##### Dataset")

        self.BUFFER_SIZE = int(self.st.number_input(
            "Shuffle Buffer Size", 0, 100 * 1000,
            value=int(self.BUFFER_SIZE),
            help="1 - No shuffling, sequential reading.  \n"
                 "0 - Ignore buffer size and shuffle full dataset."
        ))

        self.RESHUFFLE_EACH_ITERATION = self.st.checkbox(
            "Reshuffle between epochs.",
            self.RESHUFFLE_EACH_ITERATION,
            help="If selected, the training dataset will be re-shuffled between epochs. "
                 "This is additionally to shuffling at the beginning. "
                 "Validation data will be left out form shuffling. "
                 "Depending on the buffer size, this may take a while "
                 "For bigger images select a buffer size of 10 or 20."
        )

        self.st.markdown("##### Data Augmentation")
        self.AUGMENTATION.parameter_block()


class MLConfig(BaseConfig):
    CLASSES: ClassConfig = ClassConfig()
    TRAIN: TrainingConfig = TrainingConfig()
    MODEL: ModelConfig = ModelConfig()
    DATASET: DatasetConfig = DatasetConfig()

    def compile_args(self) -> Dict:
        """Return your compile args for tensorflow training for your model"""
        raise NotImplementedError

    def parameter_block(self):
        self.CLASSES.parameter_block()
        self.TRAIN.parameter_block()
        self.MODEL.parameter_block()
        self.DATASET.parameter_block()
