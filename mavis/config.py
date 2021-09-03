import base64
import glob
import json
import os
import traceback
from datetime import datetime
from functools import wraps
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED

import numpy as np
import seaborn as sns
import streamlit as st
import tensorflow_addons as tfa
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import PolynomialDecay, ExponentialDecay

from shelveutils import current_model_dir, ModelDAO, ProjectDAO, PresetListDAO, ActivePresetDAO, ConfigDAO, \
    DOWNLOADS_PATH, DFDAO


class BasePreset:
    def update(self):
        raise NotImplementedError

    @staticmethod
    def access(name):
        def wrapper(fn):
            @wraps(fn)
            def access(self, *args, **kwargs):
                res = None
                try:
                    with st.expander(name):
                        fn(self, *args, **kwargs)
                        self._update(self, name)
                except Exception as e:
                    st.error(f"Preset raised a message in **{name}**.")
                    st.code(traceback.format_exc())
                return res

            return access

        return wrapper

    @staticmethod
    def _set(new_config):
        ActivePresetDAO().set(new_config)
        PresetListDAO().add(new_config)
        # st.experimental_rerun()
        st.success(f"Preset is: {new_config}. Press **`R`** for refresh.")

    @staticmethod
    def select(key_name):
        current = ActivePresetDAO().get()
        st.write("#### Preset")
        col1, col2 = st.columns(2)
        with col1:
            all_preset_names = PresetListDAO().get_all()
            selection = st.selectbox(
                "Select Presets",
                all_preset_names,
                all_preset_names.index(current)
                if current in all_preset_names else 0,
                key="sel_pres" + key_name
            )
        if current != selection:
            BasePreset._set(selection)

        st.write("#### Settings")
        st.write("")

    @staticmethod
    def _update(self, key_name):
        current = ActivePresetDAO().get()
        st.write("---")
        btn = st.empty()

        if st.checkbox("New", key=f"custom_model{key_name}"):
            new_name = f"{datetime.now():%y%m%d_%H-%M}_{Path(ProjectDAO().get()).stem}"
            if st.checkbox("Use custom name"):
                new_name = st.text_input("Preset name", key=f"custom_model_name_{key_name}")
                if not new_name:
                    st.warning("Please specify a name")

            if btn.button(f"🞥 Save as new preset: {new_name}", key="sav_new" + key_name):
                BasePreset._set(new_name)
                self.update()

        elif btn.button(f"⭯ Update preset: {current}", key="upd_nam" + key_name):
            BasePreset._set(current)
            self.update()


class Preset(BasePreset):
    def __init__(self):
        self.name = "Default"

        self.CLASS_NAMES = ConfigDAO([])["CLASS_NAMES"]
        self.CLASS_INDICES = ConfigDAO([])["CLASS_INDICES"]
        self.CLASS_COLORS = ConfigDAO([])["CLASS_COLORS"]
        self.RECONSTRUCTION_CLASS = ConfigDAO("")["RECONSTRUCTION_CLASS"]
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
        self.MODEL_PATH = ConfigDAO("")["MODEL_PATH"]
        self.TRAIN_ROIS_PER_IMAGE = ConfigDAO(512)["TRAIN_ROIS_PER_IMAGE"]
        self.ANCHOR_SCALES = ConfigDAO((1, 4, 8, 16, 32, 64, 128, 256))["ANCHOR_SCALES"]
        self.RPN_NMS_THRESHOLD = ConfigDAO(0.8)["RPN_NMS_THRESHOLD"]
        self.IMAGE_RESIZE_MODE = ConfigDAO("square")["IMAGE_RESIZE_MODE"]
        self.MAX_GT_INSTANCES = ConfigDAO(200)["MAX_GT_INSTANCES"]
        self.MORPH_PIPELINE = ConfigDAO([])["MORPH_PIPELINE"]
        self.MORPH_SETTINGS = ConfigDAO([])["MORPH_SETTINGS"]
        self.INSPECT_CHANNEL = ConfigDAO(None)["INSPECT_CHANNEL"]
        self.BACKBONE = ConfigDAO("resnet34")["BACKBONE"]
        self.ENCODER_FREEZE = ConfigDAO(True)["ENCODER_FREEZE"]
        self.TRAIN_HEADS_ONLY = ConfigDAO(True)["TRAIN_HEADS_ONLY"]
        self.BINARY = ConfigDAO(False)["BINARY"]
        self.ARCHITECTURE = ConfigDAO("U-Net")["ARCHITECTURE"]
        self.LOSS = ConfigDAO(None)["LOSS"]
        self.CLASS_WEIGHT = ConfigDAO({})["CLASS_WEIGHT"]
        self.OVERLAP_PERCENTAGE = ConfigDAO(0.0)["OVERLAP_PERCENTAGE"]
        self.SIZE = ConfigDAO(512)["SIZE"]
        self.BUFFER_SIZE = ConfigDAO(1000)["BUFFER_SIZE"]
        self.EPOCHS = ConfigDAO(10)["EPOCHS"]
        self.STEPS_PER_EPOCH = ConfigDAO(10)["STEPS_PER_EPOCH"]
        self.BATCH_SIZE = ConfigDAO(8)["BATCH_SIZE"]
        self.VAL_SPLIT = ConfigDAO(1)["VAL_SPLIT"]
        self.PATIENCE = ConfigDAO(10)["PATIENCE"]
        self.OPTIMIZER = ConfigDAO(None)["OPTIMIZER"]

    def update(self):
        ConfigDAO()["CLASS_NAMES"] = self.CLASS_NAMES
        ConfigDAO()["CLASS_INDICES"] = self.CLASS_INDICES
        ConfigDAO()["CLASS_COLORS"] = self.CLASS_COLORS
        ConfigDAO()["RECONSTRUCTION_CLASS"] = self.RECONSTRUCTION_CLASS
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
        ConfigDAO()["MODEL_PATH"] = self.MODEL_PATH
        ConfigDAO()["TRAIN_ROIS_PER_IMAGE"] = self.TRAIN_ROIS_PER_IMAGE
        ConfigDAO()["ANCHOR_SCALES"] = self.ANCHOR_SCALES
        ConfigDAO()["RPN_NMS_THRESHOLD"] = self.RPN_NMS_THRESHOLD
        ConfigDAO()["IMAGE_RESIZE_MODE"] = self.IMAGE_RESIZE_MODE
        ConfigDAO()["MAX_GT_INSTANCES"] = self.MAX_GT_INSTANCES
        ConfigDAO()["MORPH_PIPELINE"] = self.MORPH_PIPELINE
        ConfigDAO()["MORPH_SETTINGS"] = self.MORPH_SETTINGS
        ConfigDAO()["INSPECT_CHANNEL"] = self.INSPECT_CHANNEL
        ConfigDAO()["BACKBONE"] = self.BACKBONE
        ConfigDAO()["ENCODER_FREEZE"] = self.ENCODER_FREEZE
        ConfigDAO()["TRAIN_HEADS_ONLY"] = self.TRAIN_HEADS_ONLY
        ConfigDAO()["BINARY"] = self.BINARY
        ConfigDAO()["ARCHITECTURE"] = self.ARCHITECTURE
        ConfigDAO()["LOSS"] = self.LOSS
        ConfigDAO()["CLASS_WEIGHT"] = self.CLASS_WEIGHT
        ConfigDAO()["OVERLAP_PERCENTAGE"] = self.OVERLAP_PERCENTAGE
        ConfigDAO()["SIZE"] = self.SIZE
        ConfigDAO()["BUFFER_SIZE"] = self.BUFFER_SIZE
        ConfigDAO()["EPOCHS"] = self.EPOCHS
        ConfigDAO()["STEPS_PER_EPOCH"] = self.STEPS_PER_EPOCH
        ConfigDAO()["BATCH_SIZE"] = self.BATCH_SIZE
        ConfigDAO()["VAL_SPLIT"] = self.VAL_SPLIT
        ConfigDAO()["PATIENCE"] = self.PATIENCE
        ConfigDAO()["OPTIMIZER"] = self.OPTIMIZER

    def _class_info_parameter_block(self, with_color=False):
        from stutils.widgets import rgb_picker

        column_list = ["Class Indices", "Class Names", "Class Colors"]

        if self.CLASS_NAMES is None or not len(self.CLASS_NAMES):
            st.error("You need to create Class Information first")
            self.CLASS_NAMES = np.array([])
            self.CLASS_COLORS = np.array([])
            self.CLASS_INDICES = np.array([])

        if st.checkbox("Read from JSON"):
            st.write("Example")
            st.json('{"Class Indices": [0, 1], '
                    '"Class Names": ["Background", "Foreground"], '
                    '"Class Colors": [[0, 0, 0], [255, 255, 255]]}')

            if len(self.CLASS_COLORS):
                self.CLASS_COLORS = np.stack(self.CLASS_COLORS)

            default_json = json.dumps({
                k: v.tolist() for k, v in
                zip(column_list, [self.CLASS_INDICES, self.CLASS_NAMES, self.CLASS_COLORS])
            })

            class_infos = list(json.loads(st.text_input("JSON Configuration", default_json)).values())

            self.CLASS_INDICES = np.array(class_infos[0])
            self.CLASS_NAMES = np.array(class_infos[1])
            self.CLASS_COLORS = np.array([tuple(i) for i in class_infos[2]])

        else:
            default_classes = self.CLASS_NAMES
            n_classes = st.slider(
                "Number of Classes", 2, 20,
                len(default_classes) if len(default_classes) else 2
            )

            hex_colors = (
                sns.color_palette("cubehelix", n_colors=n_classes).as_hex()
                if not len(self.CLASS_COLORS) else
                ['#%02x%02x%02x' % tuple(col) for col in self.CLASS_COLORS]
            )

            fns = [
                lambda i: i,
                lambda i: st.text_input(
                    f"Class Name for Network Output {i}",
                    default_classes[i] if len(default_classes) and i < len(default_classes) else ""
                ),
                lambda i: rgb_picker(f"Color for Network Output {i}", hex_colors[i])
            ]

            self.CLASS_INDICES, self.CLASS_NAMES, self.CLASS_COLORS = [np.array(a) for a in list(zip(*[
                [fn(i) for fn in fns]
                for i in range(n_classes)
            ]))]

    def _reconstruction_class(self):
        all_classes = list(self.CLASS_NAMES) + ["None"]
        self.RECONSTRUCTION_CLASS = st.selectbox(
            "Choose Reference Class for Reconstruction."
            "Only Images of this class will be used for training. E.g. 'Good'. "
            "Select None to run on all",
            all_classes,
            all_classes.index(self.RECONSTRUCTION_CLASS)
            if self.RECONSTRUCTION_CLASS in all_classes else 0
        )

    def class_subset_block(self, label="", defaults=None):
        defaults = list(self.CLASS_NAMES)[1:] if defaults is None else defaults
        class_names = st.multiselect(label or "Select Classes to Process. "
                                              "Images must be color encoded with class colors.",
                                     list(self.CLASS_NAMES), defaults)
        class_colors = [self.CLASS_COLORS[list(self.CLASS_NAMES).index(name)] for name in class_names]
        return class_names, class_colors

    def _augmentation_parameter_block(self, model_processor):
        st.markdown("### Data Augmentation")
        all_aug = list(model_processor.dataset.all_augmentations.keys())
        if st.button("Add all"):
            self.AUGMENTATIONS = all_aug
        self.AUGMENTATIONS = st.multiselect(
            "Data Augmentations",
            all_aug,
            [a for a in self.AUGMENTATIONS if a in all_aug]
        )
        self.AUG_ROTATE = st.selectbox(
            "Rotation Border Treatment",
            ["zeros", "mirror"],
            ["zeros", "mirror"].index(self.AUG_ROTATE)
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
            "Zoom Percentage - Maximum Zoom multiplier", 0., 1.,
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

    def _model_path(self):
        btn = st.empty()
        new_path = Path(current_model_dir()) / f"{datetime.now():%y%m%d_%H-%M}_{Path(ProjectDAO().get()).stem}.h5"
        if st.checkbox("Custom Model Path"):
            new_path = Path(st.text_input(
                "Path", Path(current_model_dir()) / "new_model.h5"
            ))
        if btn.button(f"Add new model name: `{new_path.name}`"):
            ModelDAO().add(new_path)

        all_model_paths = ModelDAO().get_all()
        self.MODEL_PATH = st.selectbox(
            "Select Model",
            all_model_paths,
            all_model_paths.index(self.MODEL_PATH)
            if self.MODEL_PATH in all_model_paths else 0
        )
        if Path(self.MODEL_PATH).is_file() and st.button("Get download link"):
            path = Path(self.MODEL_PATH)
            ExportWidget(path.name).model_link(path)

    def overlap_information_block(self):
        if self.OVERLAP_PERCENTAGE is None:
            st.error("Create Overlap Information first")

        self.OVERLAP_PERCENTAGE = st.slider(
            "Overlap Percentage", 0., 1.,
            self.OVERLAP_PERCENTAGE
        )

    def _basic_training_duration_parameter_block(self):
        st.markdown("### Training Duration")

        self.BATCH_SIZE = int(st.number_input(
            "Batch Size", 1, 1024,
            int(self.BATCH_SIZE)
        ))

        self.EPOCHS = int(st.number_input(
            "Epochs", 1, 10 * 1000,
            int(self.EPOCHS)
        ))

        self.PATIENCE = int(st.number_input(
            "Patience in epochs", 1, 10 * 1000,
            int(self.PATIENCE)
        ))

        self.STEPS_PER_EPOCH = int(st.number_input(
            "Steps per epoch", 1, 10 * 1000,
            int(self.STEPS_PER_EPOCH)
        ))

        self.VAL_SPLIT = st.number_input(
            "Validation Images in batches", 0, 100,
            int(self.VAL_SPLIT)
        )

    def _advanced_training_duration_parameter_block(self):
        self.BUFFER_SIZE = st.number_input(
            "Shuffle Buffer Size. 1 - no shuffling", 1, 100 * 1000,
            int(self.BUFFER_SIZE)
        )

    def _image_size_parameter_block(self):
        self.SIZE = int(st.number_input(
            "Image Resize", 1, 4096,
            self.SIZE,
            help="Automatically resizes the images to desiered model input"
        ))

    @staticmethod
    def _learning_rate_block(default_lr):
        st.markdown("Learning Rate Schedules")

        default_name = default_lr["class_name"] if type(
            default_lr) is dict and "class_name" in default_lr else "NoSchedule"

        config = default_lr["config"] if type(default_lr) is dict and "config" in default_lr else {}

        def default_config(val, default):
            return config[val] if val in config else default

        lr = default_config("initial_learning_rate", default_lr)

        lr_cfg = {
            "NoSchedule": lambda: (10 ** st.slider(
                "Learning Rate (10^x)", -6, 0,
                int(np.log10(lr))
            )),
            "PolynomialDecay": lambda: PolynomialDecay(
                initial_learning_rate=10 ** st.slider(
                    "Initial Learning Rate (10^-x)", -6, 0,
                    int(np.log10(lr))
                ),
                end_learning_rate=10 ** st.slider(
                    "End Learning Rate (10^x)", -6, 0,
                    int(np.log10(default_config("end_learning_rate", 0.0001)))
                ),
                decay_steps=10 ** st.slider(
                    "Total earning Rate Decay steps (10^x)", 1, 6,
                    int(np.log10(default_config("decay_steps", 1000)))
                ),
                power=st.slider(
                    "Power of polynomial", 0., 1.,
                    default_config("power", 1.)),
                cycle=st.checkbox(
                    "Cyclic Learning Rate",
                    default_config("cycle", False)
                )
            ),
            "ExponentialDecay": lambda: ExponentialDecay(
                initial_learning_rate=10 ** st.slider(
                    "Initial Learning Rate (10^x)", -6, 0,
                    int(np.log10(lr))
                ),
                decay_rate=st.slider(
                    "Decay Factor", 0., 1.,
                    default_config("decay_rate", 0.25)
                ),
                decay_steps=10 ** st.slider(
                    "Decay every (10^x) steps", 1, 6,
                    int(np.log10(default_config("decay_steps", 1000)))
                ),
                staircase=st.checkbox(
                    "Discrete Decay (Stepwise)",
                    default_config("staircase", False)
                )
            )
        }
        scheduler_name = st.selectbox(
            "Scheduler",
            list(lr_cfg.keys()),
            list(lr_cfg.keys()).index(default_name)
        )
        return lr_cfg[scheduler_name]()

    def _optimizer_parameter_block(self):
        st.markdown("### Learning Parameter")
        st.warning("Some of the optimizer currently cannot be persisted in the databse. "
                   "Make sure the preset is configured properly before training.")
        try:
            config = self.OPTIMIZER.get_config() if self.OPTIMIZER is not None else {}
        except:
            st.warning("Failed to restore optimizer.")
            config = {}

        optimizer_key = "name"
        # Special case nested optimizer for Lookahead
        if "optimizer" in config:
            config = config["optimizer"]
            optimizer_key = "class_name"

        def default_config(val, default):
            return config[val] if val in config else default

        optimizer_cfg = {
            "Adam": lambda: Adam(
                learning_rate=self._learning_rate_block(
                    default_config("learning_rate", 0.01)
                ),
                decay=st.slider(
                    "Optimizer Learning Rate Decay per epoch", 0., 1.,
                    float(default_config("decay", 0.))
                ),
                amsgrad=st.checkbox(
                    "Use AMS Grad",
                    bool(default_config("amsgrad", False))
                )
            ),
            "SGD": lambda: SGD(
                learning_rate=self._learning_rate_block(
                    default_config("learning_rate", 0.001)
                ),
                momentum=st.slider(
                    "Momentum", 0., 1.,
                    default_config("momentum", 0.9)
                ),
                nesterov=st.checkbox(
                    "Use Nesterov Momentum",
                    default_config("nesterov", False)
                ),
                clipnorm=5.0
            ),
            "Addons>RectifiedAdam": lambda: tfa.optimizers.Lookahead(
                optimizer=tfa.optimizers.RectifiedAdam(
                    lr=self._learning_rate_block(
                        default_config("learning_rate", 0.01)
                    ),
                    total_steps=self.EPOCHS * self.STEPS_PER_EPOCH,
                    warmup_proportion=st.slider(
                        "Warmup Proportion", 0., 1.,
                        default_config("warmup_proportion", 0.1)
                    ),
                    min_lr=st.number_input(
                        "Minimum Learning Rate", 0., 1.,
                        default_config("min_lr", 1e-5)
                    )
                ),
                sync_period=6,
                slow_step_size=0.5
            )
        }

        optimizer_name = st.selectbox(
            "Optimizer",
            list(optimizer_cfg.keys()),
            list(optimizer_cfg.keys()).index(default_config(optimizer_key, "Adam"))
        )

        self.OPTIMIZER = optimizer_cfg[optimizer_name]()

    def _architecture_parameter_block(self, model_processor):
        architectures = list(model_processor.all_architectures.keys())

        self.ARCHITECTURE = st.selectbox(
            "Model Architecture", architectures,
            architectures.index(self.ARCHITECTURE)
            if self.ARCHITECTURE in architectures
            else architectures.index(model_processor.default_architecture)
        )

        self.CLASS_WEIGHT = {i: st.number_input(
            f"Treat every occurrence of a {name} value as `X` instances during training loss calculation", 1., 1000.,
            self.CLASS_WEIGHT[i] if i in self.CLASS_WEIGHT else 1.,
            key=f"class_input_{i}"
        ) for i, name in enumerate(self.CLASS_NAMES)}

        losses = list(model_processor.all_losses.keys())

        self.LOSS = st.selectbox(
            "Loss function", losses,
            losses.index(self.LOSS) if self.LOSS in losses
            else losses.index(model_processor.default_loss)
        )

    def _backbone_parameter_block(self, model_processor):
        backbones = model_processor.all_backbones

        self.BACKBONE = st.selectbox(
            "Model Backbone", backbones,
            backbones.index(self.BACKBONE)
            if self.BACKBONE in backbones
            else backbones.index(model_processor.default_backbone)
        )

        self.ENCODER_FREEZE = self.TRAIN_HEADS_ONLY = st.checkbox(
            "Freeze pretrained model",
            self.ENCODER_FREEZE or self.TRAIN_HEADS_ONLY
        )

    def _output_mask_opts(self):
        self.BINARY = st.checkbox(
            "Check to use Sigmoid Output. Default is Softmax",
            self.BINARY
        )

        if not self.BINARY:
            output_opts = ["None"] + list(self.CLASS_NAMES)

            self.INSPECT_CHANNEL = st.selectbox(
                "Get probability masks for a specific class. "
                "Select `None` to get color encoded output for all classes. "
                "Color encoding is then based on maximum probability for each class per pixel",
                output_opts,
                output_opts.index(self.INSPECT_CHANNEL)
                if self.INSPECT_CHANNEL in output_opts else 0
            )

    @BasePreset.access("Morphology Pipeline")
    def morph_options(self, all_processors):
        n_steps = st.number_input("Build new Processing pipeline. Number of processing steps", 0, 20, 0)
        self.MORPH_PIPELINE = [
            st.selectbox(f"Select Processor for step {i}", list(all_processors.keys())) for i in
            range(n_steps)
        ]

        if st.button("Update Pipeline"):
            self.MORPH_SETTINGS = [all_processors[step](i) for i, step in enumerate(self.MORPH_PIPELINE)]

        [loaded_processor.configure_opt() for loaded_processor in self.MORPH_SETTINGS]


    @BasePreset.access("Mask RCNN Settings")
    def mask_rcnn_settings(self):
        self.TRAIN_ROIS_PER_IMAGE = st.number_input("Train ROIs per image", value=self.TRAIN_ROIS_PER_IMAGE)
        self.ANCHOR_SCALES = (4, 8, 16, 32, 64, 128, 256)
        self.RPN_NMS_THRESHOLD = st.number_input("Non max supr. Threshold", 0., 1., self.RPN_NMS_THRESHOLD)
        self.IMAGE_RESIZE_MODE = "square"
        self.MAX_GT_INSTANCES = st.number_input("Max Ground Truth instances", value=self.MAX_GT_INSTANCES)

    ####################################################################################################################
    # access

    @BasePreset.access("Class Info")
    def class_info_block(self, with_color=False):
        self._class_info_parameter_block(with_color)

    @BasePreset.access("Model Settings")
    def common_model_parameter(self):
        st.write("### Class Info")
        self._class_info_parameter_block(True)
        st.write("--- \n ### Model Path")
        self._model_path()
        self._image_size_parameter_block()

    @BasePreset.access("Training Settings")
    def classification_training_args(self, model_processor):
        self.ARCHITECTURE = "Classifcation"
        self._basic_training_duration_parameter_block()
        self._advanced_training_duration_parameter_block()
        self._augmentation_parameter_block(model_processor)
        self._optimizer_parameter_block()

    @BasePreset.access("Training Settings")
    def reconstruction_training_args(self, model_processor):
        self.ARCHITECTURE = "Reconstruction"
        self._basic_training_duration_parameter_block()
        self._augmentation_parameter_block(model_processor)
        self._reconstruction_class()

    @BasePreset.access("Training Settings")
    def instance_segmentation_training_args(self, model_processor):
        self.ARCHITECTURE = "Mask-RCNN"
        self._basic_training_duration_parameter_block()
        self._backbone_parameter_block(model_processor)
        self._augmentation_parameter_block(model_processor)
        self._optimizer_parameter_block()

    @BasePreset.access("Training Settings")
    def semantic_segmentation_training_args(self, model_processor):
        self._output_mask_opts()
        self._basic_training_duration_parameter_block()
        self._backbone_parameter_block(model_processor)
        self._architecture_parameter_block(model_processor)
        self._advanced_training_duration_parameter_block()
        self._augmentation_parameter_block(model_processor)
        self._optimizer_parameter_block()

    @BasePreset.access("Inference Settings")
    def semantic_segmentation_inference_args(self, model_processor):
        pass

    @BasePreset.access("Inference Settings")
    def classification_inference_args(self, model_processor):
        pass

    @BasePreset.access("Inference Settings")
    def instance_segmentation_inference_args(self, model_processor):
        pass

    @BasePreset.access("Inference Settings")
    def reconstruction_inference_args(self, model_processor):
        pass


class ExportWidget:
    def __init__(self, name):
        self.name = name

    def _zip_dir(self, source_dirs, folder_names, pattern="*", verbose=True, recursive=False):
        remove_files = glob.glob(str(DOWNLOADS_PATH / "**"))
        [os.remove(f) for f in remove_files]
        st.info(f"Removed {len(remove_files)} old archvies")

        target_file = str(DOWNLOADS_PATH / self.name)

        with ZipFile(target_file, 'w', ZIP_STORED) as zf:
            for source_dir, folder_name in zip(source_dirs, folder_names):
                src_path = Path(source_dir).expanduser().resolve(strict=True)
                files = list(src_path.rglob(pattern) if recursive else src_path.glob(pattern))
                if verbose:
                    bar = st.progress(0)
                for i, file in enumerate(files):
                    if verbose:
                        bar.progress(i / len(files))
                    zf.write(file, Path(folder_name) / file.relative_to(src_path))

        st.markdown(f"Download [{self.name}](downloads/{self.name})", unsafe_allow_html=True)

    def df_link(self, csv_args):
        csv_args["header"] = True
        csv = DFDAO().get(ProjectDAO().get()).to_csv(index=False, **csv_args)
        b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a download="{self.name}.csv" href="data:file/csv;base64,{b64}">' \
               f'Download {self.name}.csv</a>'
        st.markdown(href, unsafe_allow_html=True)

    def ds_link(self, paths, folder_names, recursive=False):
        self._zip_dir(paths, folder_names, recursive=recursive)

    def model_link(self, path: Path):
        self._zip_dir(path.parent, "model", pattern=path.name)