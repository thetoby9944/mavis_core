import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Union, Tuple, List, Optional, Any, Callable

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from mavis.db import ModelDAO, current_model_dir, ProjectDAO
from mavis.pdutils import image_columns
from mavis.presets.augmentation import AugmentationConfig
from mavis.presets.base import BaseProperty, BaseConfig, PresetHandler
from mavis.presets.optimizer import OptimizerConfig


class DefaultSettings(BaseConfig):
    _input_labels: Optional[List[str]] = None
    _output_label: Optional[str] = None
    _inputs_column_filter: Optional[List[Optional[Callable]]] = None
    _drop_na_jointly: bool = True
    _suffix: str = ""

    _df: Optional[pd.DataFrame] = None
    input_columns: List[Optional[str]] = []
    column_out: Optional[str] = None
    use_suffix: bool = False

    def set_df(self, df):
        self._df = df

    def sample_path(self):
        return self._df[self.input_columns[0]][0]

    @property
    def suffix(self):
        return self._suffix

    @property
    def drop_na_jointly(self):
        return self._drop_na_jointly

    def column_out_block(self, label="", suffix="") -> None:
        if label:
            self.column_out = st.text_input(
                "Remember Results in",
                f"{self.input_columns[0]} {label}"
            )
        if suffix:
            self._suffix = suffix
        elif label:
            self.use_suffix = st.checkbox("Save results with suffix", self.use_suffix)
            self._suffix = f"-{label.replace(' ', '_')}.png" if self.use_suffix else ".png"

    def parameter_block(self):
        input_labels = self._input_labels
        output_label = self._output_label
        output_suffix = self._suffix
        inputs_column_filter = self._inputs_column_filter
        df = self._df

        if input_labels is not None:
            # st.markdown("### Data")

            # Column Filter
            if inputs_column_filter is None:
                inputs_column_filter = [image_columns] * len(input_labels)
            assert len(inputs_column_filter) == len(input_labels)

            # Default columns
            input_columns = self.input_columns
            if len(input_columns) != len(input_labels):
                input_columns = [None] * len(input_labels)
            assert len(input_columns) == len(input_labels)

            # Configure inputs
            options = [
                column_filter(df) if column_filter is not None else df.columns
                for column_filter in inputs_column_filter
            ]
            self.input_columns = [
                st.selectbox(
                    label=label,
                    options=columns,
                    index=columns.index(default_column) if default_column in columns else 0
                )
                for label, columns, default_column
                in zip(input_labels, options, input_columns)
            ]

            if not all(self.input_columns):
                st.info("Please upload files first")
                st.stop()

        if output_label is not None:
            self.column_out_block(label=output_label, suffix=output_suffix)

        st.session_state["settings_form_placeholder"].__enter__()
        st.markdown("# Settings")


class ClassConfigProperty(DefaultSettings):
    CLASS_NAMES = ["Background", "Foreground"]
    CLASS_INDICES = [0, 1]
    CLASS_COLORS = [(0, 0, 0), (255, 255, 255)]

    def parameter_block(self):
        self.st.write("## Class Information")
        from mavis.ui.widgets.utils import rgb_picker

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

    def class_subset_block(self, label="", defaults=None):
        defaults = list(self.CLASS_NAMES)[1:] if defaults is None else defaults
        class_names = st.multiselect(
            label
            or
            "Select Classes to Process. "
            "Images must be color encoded with class colors.",
            list(self.CLASS_NAMES), defaults
        )
        class_colors = [
            self.CLASS_COLORS[list(self.CLASS_NAMES).index(name)]
            for name in class_names
        ]
        return class_names, class_colors


class ClassConfig(ClassConfigProperty):
    def parameter_block(self):
        super(ClassConfigProperty, self).parameter_block()
        super(ClassConfig, self).parameter_block()


class ClassConfigWithSubsetProperty(ClassConfigProperty):
    class_names: List[str] = []
    class_colors: List[Union[Tuple[int, int, int], List[int]]] = []

    def parameter_block(self):
        super().parameter_block()
        self.class_names, self.class_colors = self.class_subset_block(
            defaults=self.class_names
        )


class ClassConfigWithSubset(ClassConfigWithSubsetProperty):
    def parameter_block(self):
        super(ClassConfigProperty, self).parameter_block()
        super(ClassConfigWithSubset, self).parameter_block()


class ModelConfig(BaseProperty):
    MODEL_PATH: Path = ModelDAO().get_all()[0]

    def parameter_block(self):
        self.st.write("## Model Path")

        new_path = Path(self.st.text_input(
            "Path", Path(current_model_dir()) / f"{datetime.now():%y%m%d_%H-%M}_{Path(ProjectDAO().get()).stem}.h5"
        ))
        if self.st.form_submit_button("Add Model Path"):
            ModelDAO().add(new_path)

        all_model_paths = ModelDAO().get_all()
        self.MODEL_PATH = st.radio(
            "select model",
            all_model_paths,
            index=all_model_paths.index(Path(self.MODEL_PATH)) if Path(self.MODEL_PATH) in all_model_paths else 0
        )


class ContourFilterConfig(BaseProperty):
    min_circularity = 0.
    min_area = 0
    max_area = 0
    ignore_border_cnt = True

    def parameter_block(self):
        self.min_circularity = st.number_input(
            "Minimum circularity for detected segments", 0., 1., self.min_circularity
        )
        self.min_area = st.number_input(
            "Minimum Area for detected segments. Set to zero to disable.", 0, 100000, self.min_area
        )
        self.max_area = st.number_input(
            "Maximum Area for detected segments. Set to zero to disable.", 0, 100000, self.max_area
        )
        self.ignore_border_cnt = st.checkbox(
            "Ignore Border Contours", self.ignore_border_cnt
        )


class TrainingConfig(BaseProperty):
    """
    Base Training configuration
    """
    name: str = "Default"

    EPOCHS: int = 10
    BATCH_SIZE: int = 2
    STEPS_PER_EPOCH: int = 10
    VAL_SPLIT: int = 1
    EARLY_STOPPING: int = 5
    CONTINUE_TRAINING: Union[Path, str] = ""

    IMPLICIT_BACKGROUND_CLASS: bool = False
    RED_ON_PLATEAU_PATIENCE: int = 0

    def parameter_block(self):
        self.st.markdown("## Training Duration")

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
            "Validation Images", 0, 100000,
            int(self.VAL_SPLIT),
            help=
            "Use this many images for validation. For each image, one batch will be created. "
            "This behaviour is useful when you use data augmentation, e.g. with random crops. "
            "Note that, currently, without data augmentation, there will be duplicates in the validation data"
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

        all_model_paths = [""] + ModelDAO().get_all()
        self.CONTINUE_TRAINING = self.st.selectbox(
            """Select a base model to retrain from. Leave empty to train a new model""",
            all_model_paths,
            index=(
                all_model_paths.index(self.CONTINUE_TRAINING)
                if self.CONTINUE_TRAINING in all_model_paths
                else 0
            )
        )


class DatasetConfig(BaseProperty):
    AUGMENTATION = AugmentationConfig()

    BUFFER_SIZE: int = 0
    RESHUFFLE_EACH_ITERATION: bool = True

    def parameter_block(self):
        self.st.markdown("## Dataset")

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

        self.st.markdown("## Data Augmentation")
        self.AUGMENTATION.parameter_block()


class MLConfig(DefaultSettings):
    _input_labels = [
        "Select Input Images Paths",
        "Select Label Images Paths",
    ]
    _output_label = "Prediction"

    TRAIN: TrainingConfig = TrainingConfig()
    MODEL: ModelConfig = ModelConfig()
    DATASET: DatasetConfig = DatasetConfig()
    OPTIMIZER: OptimizerConfig = OptimizerConfig()
    WEIGHT_DECAY: float = 0.0001

    def compile_args(self) -> Dict:
        """Return your compile args for tensorflow training for your model"""
        raise NotImplementedError

    def parameter_block(self):
        super().parameter_block()
        self.TRAIN.parameter_block()
        self.MODEL.parameter_block()
        self.DATASET.parameter_block()

        self.st.markdown("## Model Optimization")
        self.OPTIMIZER.RECTIFIED_ADAM_PARAMETER._total_steps = int(self.TRAIN.EPOCHS * self.TRAIN.STEPS_PER_EPOCH)
        self.OPTIMIZER.parameter_block()
        self.WEIGHT_DECAY = self.st.number_input(
            "L2 Kernel Regularization",
            0., 1.,
            self.WEIGHT_DECAY,
            format="%0.6f"
        )


class NumericalSettings(DefaultSettings):
    _selection_label: str = ""
    opts: List[str] = []

    def parameter_block(self):
        super(NumericalSettings, self).parameter_block()
        self.opts = st.multiselect(
            self._selection_label,
            list(self.all_options.keys()),
            self.opts
        )
        st.form_submit_button(self._selection_label)

    @property
    def all_options(self) -> Dict[str, Callable]:
        raise NotImplementedError


class MultiProcessorSettings(BaseConfig):
    def parameter_block(self):
        pass

    last_selection: int = 0
    show_name: bool = False
    name: str = ""
