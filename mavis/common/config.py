"""
Tensorflow configuration,

initializes the devices.
To disable GPU add:


    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"

"""
import json
import pprint
import traceback
from datetime import datetime
from pathlib import Path
import streamlit as st
import numpy as np
import seaborn as sns

import tensorflow_addons as tfa
from functools import wraps
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import PolynomialDecay, ExponentialDecay

from mavis.shelveutils import save_preset, current_model_dir, load_model_paths, save_model_path, current_project, \
    load_presets, last_preset


class Preset:

    def __init__(self):

        self.name = "Default"

        self.CLASS_NAMES = []
        self.CLASS_INDICES = []
        self.CLASS_COLORS = []
        self.CLASS_WEIGHT = {}
        self.PATIENCE = 10
        self.BINARY = False

        self.OVERLAP_PERCENTAGE = 0.0

        # Training
        self.BACKBONE = "resnet34"
        self.ARCHITECTURE = "u-net"
        self.MODEL_PATH = ""
        self.SIZE = 512
        self.BUFFER_SIZE = 1000
        self.EPOCHS = 10
        self.STEPS_PER_EPOCH = 10
        self.BATCH_SIZE = 8
        self.VAL_SPLIT = 1
        self.OPTIMIZER = None
        self.LOSS = None
        self.ENCODER_FREEZE = True
        self.TRAIN_HEADS_ONLY = True

        # Data Augmentation Constants
        self.AUGMENTATIONS = []
        self.AUG_ROTATE = "mirror"
        self.AUG_MIN_JPG_QUALITY = 65
        self.AUG_NOISE = 10
        self.AUG_CONTRAST_MIN = 0.7
        self.AUG_CONTRAST_MAX = 1.3
        self.AUG_SATURATION_MIN = 0.6
        self.AUG_SATURATION_MAX = 1.6
        self.AUG_BRIGHTNESS = 0.05
        self.AUG_ZOOM_PERCENT = 0.1
        self.AUG_GAUSS_SIGMA = 1.
        self.AUG_GAUSS_FILTER_RADIUS = 1

        self.RECONSTRUCTION_CLASS = ""

    # Private Functions
    def _learning_rate_block(self, default_lr):
        st.markdown("Learning Rate Schedules")

        default_name = default_lr["class_name"] if type(
            default_lr) is dict and "class_name" in default_lr else "NoSchedule"

        config = default_lr["config"] if type(default_lr) is dict and "config" in default_lr else {}

        default_config = lambda val, default: config[val] if val in config else default
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

        self.BINARY = st.checkbox(
            "Check to use Sigmoid Output. Default is Softmax",
            self.BINARY
        )
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

        default_config = lambda val, default: config[val] if val in config else default

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

    def _architecture_parameter_block(self, model_processor):
        architectures = list(model_processor.all_architectures.keys())
        losses = list(model_processor.all_losses.keys())

        self.ARCHITECTURE = st.selectbox(
            "Model Architecture", architectures,
            architectures.index(self.ARCHITECTURE)
            if self.ARCHITECTURE in architectures
            else architectures.index(model_processor.default_architecture)
        )

        self.LOSS = st.selectbox(
            "Loss function", losses,
            losses.index(self.LOSS)
            if hasattr(self, "LOSS") and self.LOSS in losses
            else losses.index(model_processor.default_loss)
        )

        self.CLASS_WEIGHT = {i: st.number_input(
            f"Treat every occurrence of a {name} value as `X` instances during training loss calculation", 1., 1000.,
            self.CLASS_WEIGHT[i] if hasattr(self, "CLASS_WEIGHT") and i in self.CLASS_WEIGHT else 1.
        ) for i, name in enumerate(self.CLASS_NAMES)}

    def _augmentation_parameter_block(self, model_processor):
        st.markdown("### Data Augmentation")
        all_aug = list(model_processor.dataset.all_augmentations.keys())
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

    def _image_size_parameter_block(self):
        self.SIZE = int(st.number_input(
            "Image Resize", 1, 2048,
            self.SIZE
        ))

    def _basic_training_duration_parameter_block(self):
        st.markdown("### Training Duration")

        valid = ["1", "2", "4", "8", "16", "32", "64", "128", "256"]
        self.BATCH_SIZE = int(st.selectbox(
            "Batch", valid,
            valid.index(str(self.BATCH_SIZE)) if str(self.BATCH_SIZE) in valid else 0
        ))
        self.EPOCHS = int(st.number_input(
            "Epochs", 1, 10 * 1000,
            self.EPOCHS
        ))

        self.PATIENCE = int(st.number_input(
            "Patience in epochs", 1, 10 * 1000,
            self.PATIENCE if hasattr(self, "PATIENCE") else 10
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
            "Shuffle Buffer Size. 1 - no shuffeling", 1, 100 * 1000,
            self.BUFFER_SIZE
        )

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

    def _access_decorator(name):
        def wrapper(fn):
            @wraps(fn)
            def access(self, *args, **kwargs):
                global c
                res = None
                try:
                    with st.beta_expander(name):
                        fn(self, *args, **kwargs)
                        self._update(name)
                except:
                    st.error("Must specify settings")
                    st.code(traceback.format_exc())
                return res

            return access

        return wrapper

    def _reset(self):
        global c
        name = c.name
        c = Preset()
        c.name = name
        save_preset(c)

    def _update(self, key_name):
        global c
        st.write("")
        new_name = f"{datetime.now():%y%m%d_%H-%M}_{Path(current_project()).stem}"
        create_new = st.button(f"Save as {new_name}", key="sav_new" + key_name)
        update = st.button(f"Update {self.name}", key="upd_nam" + key_name)

        if create_new:
            self.name = new_name

        if create_new or update:
            save_preset(self)
            c = self
            st.code(self.print_preset())
            st.success(f"Saved to preset: {self.name}")

    def _model_path(self):
        new_path = Path(current_model_dir()) / f"{datetime.now():%y%m%d_%H-%M}_{Path(current_project()).stem}.h5"
        if st.checkbox("Custom Model Path"):
            new_path = Path(st.text_input(
                "Path", Path(current_model_dir()) / "new_model.h5"
            ))
        if st.button(f"Add Model {new_path.stem}"):
            save_model_path(new_path)

        all_model_paths = load_model_paths()
        self.MODEL_PATH = st.selectbox(
            "Select Model",
            all_model_paths,
            all_model_paths.index(self.MODEL_PATH)
            if self.MODEL_PATH in all_model_paths else 0
        )

    # Public Functions
    def print_preset(self):
        return pprint.pformat({k: v.get_config() if k == "OPTIMIZER" and v is not None else v
                               for k, v in vars(self).items() if "__" not in k and k.isupper()})

    def select(self, key_name):
        global c
        all_presets = load_presets(self)
        all_preset_names = list(all_presets.keys())
        selection = all_presets[st.selectbox(
            "Select Presets",
            all_preset_names,
            all_preset_names.index(self.name)
            if self.name in all_preset_names else 0,
            key="sel_pres" + key_name
        )]
        if st.button("Select Preset", key="sel_pre" + key_name):
            c = selection
            st.success("Selected")
        if st.checkbox(f"Mark for Reset"):
            if st.button(f"Reset preset {c.name}"):
                c._reset()
                st.success("Reseted")

    @_access_decorator("Model")
    def common_model_parameter(self):
        self._model_path()
        self._image_size_parameter_block()

    @_access_decorator("Training Settings")
    def semantic_segmentation_training_args(self, model_processor):
        self._basic_training_duration_parameter_block()
        self._backbone_parameter_block(model_processor)
        self._architecture_parameter_block(model_processor)
        self._advanced_training_duration_parameter_block()
        self._augmentation_parameter_block(model_processor)
        self._optimizer_parameter_block()

    @_access_decorator("Training Settings")
    def classification_training_args(self, model_processor):
        self.ARCHITECTURE = "Classifcation"
        self._basic_training_duration_parameter_block()
        self._advanced_training_duration_parameter_block()
        self._augmentation_parameter_block(model_processor)
        self._optimizer_parameter_block()

    @_access_decorator("Training Settings")
    def reconstruction_training_args(self, model_processor):
        self.ARCHITECTURE = "Reconstruction"
        self._basic_training_duration_parameter_block()
        self._augmentation_parameter_block(model_processor)
        self._reconstruction_class()

    @_access_decorator("Training Settings")
    def instance_segmentation_training_args(self, model_processor):
        self.ARCHITECTURE = "Mask-RCNN"
        self._basic_training_duration_parameter_block()
        self._backbone_parameter_block(model_processor)
        self._augmentation_parameter_block(model_processor)
        self._optimizer_parameter_block()

    @_access_decorator("Inference Settings")
    def semantic_segmentation_inference_args(self, model_processor):
        pass

    @_access_decorator("Inference Settings")
    def classification_inference_args(self, model_processor):
        pass

    @_access_decorator("Inference Settings")
    def instance_segmentation_inference_args(self, model_processor):
        pass

    @_access_decorator("Inference Settings")
    def reconstruction_inference_args(self, model_processor):
        pass

    def class_subset_block(self, label="", defaults=None):
        defaults = list(self.CLASS_NAMES)[1:] if defaults is None else defaults
        class_names = st.multiselect(label or "Select Classes to Process. "
                                              "Images must be color encoded with class colors.",
                                     list(self.CLASS_NAMES), defaults)
        class_colors = [self.CLASS_COLORS[list(self.CLASS_NAMES).index(name)] for name in class_names]
        return class_names, class_colors

    def overlap_information_block(self):
        if self.OVERLAP_PERCENTAGE is None:
            st.error("Create Overlap Information first")

        self.OVERLAP_PERCENTAGE = st.slider(
            "Overlap Percentage", 0., 1.,
            self.OVERLAP_PERCENTAGE)

    @_access_decorator("Class Info")
    def class_info_block(self, with_color=False):
        from mavis.stutils.widgets import rgb_picker

        column_list = ["Class Indices", "Class Names"]
        if with_color:
            column_list += ["Class Colors"]

        if self.CLASS_NAMES is None or not len(self.CLASS_NAMES) or with_color and self.CLASS_COLORS is None:
            st.error("You need to create Class Information first")
            self.CLASS_NAMES = np.array([])
            self.CLASS_COLORS = np.array([])
            self.CLASS_INDICES = np.array([])

        if st.checkbox("Read from JSON"):
            st.write("Example")
            st.json('{"Class Indices": [0, 1], '
                    '"Class Names": ["Background", "Foreground"], '
                    '"Class Colors": [[0, 0, 0], [255, 255, 255]]}')

            if with_color and len(self.CLASS_COLORS):
                self.CLASS_COLORS = np.stack(self.CLASS_COLORS)

            default_json = json.dumps({
                k: v.tolist() for k, v in
                zip(column_list, [self.CLASS_INDICES, self.CLASS_NAMES, self.CLASS_COLORS])
            })

            class_infos = list(json.loads(st.text_input("JSON Configuration", default_json)).values())
            self.CLASS_INDICES = np.array(class_infos[0])
            self.CLASS_NAMES = np.array(class_infos[1])

            if with_color:
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

            fns = [lambda i: i,
                   lambda i: st.text_input(
                       f"Class Name for Network Output {i}",
                       default_classes[i] if len(default_classes) and i < len(default_classes) else ""
                   )]
            if with_color:
                fns += [lambda i: rgb_picker(f"Color for Network Output {i}", hex_colors[i])]

            class_infos = [np.array(a) for a in list(zip(*[
                [fn(i) for fn in fns]
                for i in range(n_classes)
            ]))]
            if with_color:
                self.CLASS_INDICES, self.CLASS_NAMES, self.CLASS_COLORS = class_infos
            else:
                self.CLASS_INDICES, self.CLASS_NAMES = class_infos


c = last_preset(default=Preset())
