import json
import pprint
from datetime import datetime
from functools import wraps
from pathlib import Path

import numpy as np
import seaborn as sns
import streamlit as st
import tensorflow_addons as tfa
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import PolynomialDecay, ExponentialDecay

from shelveutils import PresetDAO, current_model_dir, ModelDAO, ProjectDAO, ExportWidget


class EmptyPreset:
    """
    Use this as a base class for presets.
    New presets need to inherit from other than object to resolve inheritence order when
    extending the base preset
    """
    pass


class BasePreset(EmptyPreset):
    @staticmethod
    def add(new_preset_class):
        """
        Use this as a wrapper for a preset to extend the base preset
        Parameters
        ----------
        new_preset_class: The new preset class to be wrapped
        """
        # Extend preset so that config has all the methods of the new preset
        # This is magic TODO research
        if new_preset_class in BasePreset.__bases__:
            BasePreset.__bases__.remove(new_preset_class)

        BasePreset.__bases__ = BasePreset.__bases__ + (new_preset_class,)

    @staticmethod
    def access(name):
        def wrapper(fn):
            @wraps(fn)
            def access(self, *args, **kwargs):
                res = None
                try:
                    with st.expander(name):
                        fn(self, *args, **kwargs)
                        self._update(name)
                except Exception as e:
                    st.error(f"Preset raised a message in **{name}**")
                    st.code(e)
                return res

            return access

        return wrapper

    @staticmethod
    def _set(new_config):
        global c
        PresetDAO().set(new_config)
        PresetDAO().add(new_config)
        c = PresetDAO().get(new_config)
        # print(c.print_preset())
        st.success(f"Preset is: {c.name}. Press **`R`** for refresh.")
        st.experimental_rerun()

    @staticmethod
    def select(key_name):
        st.write("#### Preset")
        current = c
        col1, col2 = st.columns(2)
        with col1:
            all_presets = PresetDAO().get_all(current)
            all_preset_names = list(all_presets.keys())
            selection = all_presets[st.selectbox(
                "Select Presets",
                all_preset_names,
                all_preset_names.index(current.name)
                if current.name in all_preset_names else 0,
                key="sel_pres" + key_name
            )]
        if current.name != selection.name:
            BasePreset._set(selection)

        # if st.checkbox(f"Mark for Reset", key=key_name):
        #    if st.button(f"Reset preset {c.name}", key=key_name):
        #        old_name = c.name
        #        new_config = Preset()
        #        new_config.name = old_name
        #        BasePreset._set(new_config)
        #        st.success("Reset")
        st.write("#### Settings")
        st.write("")

    @staticmethod
    def _update(key_name):
        current = c
        st.write("---")
        btn = st.empty()

        if st.checkbox("Save preset as new", key=f"custom_model{key_name}"):
            new_name = f"{datetime.now():%y%m%d_%H-%M}_{Path(ProjectDAO().get()).stem}"
            if st.checkbox("Use custom name"):
                new_name = st.text_input("Preset name", key=f"custom_model_name_{key_name}")
                if not new_name:
                    st.warning("Please specify a name")

            if btn.button(f"🞥 Save as new preset: {new_name}", key="sav_new" + key_name):
                current.name = new_name
                BasePreset._set(current)

        elif btn.button(f"⭯ Update preset: {current.name}", key="upd_nam" + key_name):
            BasePreset._set(current)

    def print_preset(self):
        return pprint.pformat({k: v.get_config() if k == "OPTIMIZER" and v is not None else v
                               for k, v in vars(self).items() if "__" not in k and k.isupper()})


@BasePreset.add
class Class:
    CLASS_NAMES = []
    CLASS_INDICES = []
    CLASS_COLORS = []
    RECONSTRUCTION_CLASS = ""

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


@BasePreset.add
class Augmentation:
    AUGMENTATIONS = []
    AUG_ROTATE = "mirror"
    AUG_MIN_JPG_QUALITY = 65
    AUG_NOISE = 10
    AUG_CONTRAST_MIN = 0.7
    AUG_CONTRAST_MAX = 1.3
    AUG_SATURATION_MIN = 0.6
    AUG_SATURATION_MAX = 1.6
    AUG_BRIGHTNESS = 0.05
    AUG_ZOOM_PERCENT = 0.1
    AUG_GAUSS_SIGMA = 1.
    AUG_GAUSS_FILTER_RADIUS = 1

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


@BasePreset.add
class ModelPath:
    MODEL_PATH = ""

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


@BasePreset.add
class Overlap:
    OVERLAP_PERCENTAGE = 0.0

    def overlap_information_block(self):
        if self.OVERLAP_PERCENTAGE is None:
            st.error("Create Overlap Information first")

        self.OVERLAP_PERCENTAGE = st.slider(
            "Overlap Percentage", 0., 1.,
            self.OVERLAP_PERCENTAGE
        )


@BasePreset.add
class TrainingBasics:
    SIZE = 512
    BUFFER_SIZE = 1000
    EPOCHS = 10
    STEPS_PER_EPOCH = 10
    BATCH_SIZE = 8
    VAL_SPLIT = 1
    PATIENCE = 10

    def _basic_training_duration_parameter_block(self):
        st.markdown("### Training Duration")

        self.BATCH_SIZE = int(st.number_input(
            "Batch Size", 1, 1024,
            self.BATCH_SIZE
        ))

        self.EPOCHS = int(st.number_input(
            "Epochs", 1, 10 * 1000,
            self.EPOCHS
        ))

        self.PATIENCE = int(st.number_input(
            "Patience in epochs", 1, 10 * 1000,
            self.PATIENCE
        ))

        self.STEPS_PER_EPOCH = int(st.number_input(
            "Steps per epoch", 1, 10 * 1000,
            self.STEPS_PER_EPOCH
        ))

        self.VAL_SPLIT = st.number_input(
            "Validation Images in batches", 0, 100,
            self.VAL_SPLIT
        )

    def _advanced_training_duration_parameter_block(self):
        self.BUFFER_SIZE = st.number_input(
            "Shuffle Buffer Size. 1 - no shuffling", 1, 100 * 1000,
            self.BUFFER_SIZE
        )

    def _image_size_parameter_block(self):
        self.SIZE = int(st.number_input(
            "Image Resize", 1, 4096,
            self.SIZE,
            help="Automatically resizes the images to desiered model input"
        ))


@BasePreset.add
class Optimizer:
    OPTIMIZER = None

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


@BasePreset.add
class Architecture:
    INSPECT_CHANNEL = None
    ARCHITECTURE = "u-net"
    LOSS = None
    CLASS_WEIGHT = {}

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


@BasePreset.add
class Backbone:
    BACKBONE = "resnet34"
    ENCODER_FREEZE = True
    TRAIN_HEADS_ONLY = True
    INSPECT_CHANNEL = "None"
    BINARY = False

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


@BasePreset.add
class Preset:
    def __init__(self):
        self.name = "Default"

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


c = PresetDAO().get(default=BasePreset())
