import tensorflow as tf
import tensorflow_addons as tfa
from keras.optimizer_v2.adam import Adam
from keras.optimizer_v2.gradient_descent import SGD
from tensorflow_addons.optimizers import SWA

from mavis.presets.base import BaseProperty, PropertyContainer
from mavis.presets.learning_rates import LearningRateConfig


class OptimizerProperty(BaseProperty):
    LEARNING_RATE: LearningRateConfig = LearningRateConfig()

    def parameter_block(self):
        self.LEARNING_RATE.parameter_block()

    def get(self) -> tf.keras.optimizers.Optimizer:
        raise NotImplementedError


class AdamConfig(OptimizerProperty):
    """
    Adam optimizer kwargs
    """
    _name = "Adam"
    decay: float = 0.
    amsgrad: bool = False

    def parameter_block(self):
        super().parameter_block()
        self.decay = self.st.slider(
            "Optimizer Learning Rate Decay per epoch", 0., 1.,
            self.decay
        )
        self.amsgrad = self.st.checkbox(
            "Use AMS Grad",
            self.amsgrad
        )

    def get(self) -> tf.keras.optimizers.Optimizer:
        return Adam(
            lr=self.LEARNING_RATE.get(),
            decay=self.decay,
            amsgrad=self.amsgrad
        )


class SGDConfig(OptimizerProperty):
    """
    SGD optimizer kwargs
    """
    _name = "SGD"
    momentum: float = 0.9
    nesterov: bool = False
    clipnorm: float = 5.0

    def parameter_block(self):
        super().parameter_block()
        self.momentum = self.st.number_input(
            "Momentum", 0., 1.,
            self.momentum
        )
        self.nesterov = self.st.checkbox(
            "Use Nesterov Momentum",
            self.nesterov
        )
        self.clipnorm = 5.0

    def get(self) -> tf.keras.optimizers.Optimizer:
        return SGD(
            learning_rate=self.LEARNING_RATE.get(),
            momentum=self.momentum,
            nesterov=self.nesterov,
            clipnorm=self.clipnorm,
        )


class RectifiedAdamConfig(OptimizerProperty):
    """
    Rectified Adam optimizer kwargs
    """
    _name = "Rectified Adam"
    _total_steps: int = 0
    warmup_proportion: float = 0.1
    min_lr: float = 0.00001

    def parameter_block(self):
        super().parameter_block()
        self.warmup_proportion = self.st.slider(
            "Warmup Proportion", 0., 1.,
            self.warmup_proportion
        )
        self.min_lr = self.st.number_input(
            "Minimum Learning Rate", 0., 1.,
            self.min_lr
        )

    def get(self):
        return tfa.optimizers.Lookahead(
            optimizer=tfa.optimizers.RectifiedAdam(
                lr=self.LEARNING_RATE.get(),
                total_steps=self._total_steps,
                min_lr=self.min_lr,
                warmup_proportion=self.warmup_proportion,
            ),
            sync_period=6,
            slow_step_size=0.5
        )


class SWA_SGD_Optimizer(OptimizerProperty):
    _name = "Stochastic Weights Averaging SGD"
    momentum: float = 0.9
    nesterov: bool = False
    average_period: int = 20

    def parameter_block(self):
        super().parameter_block()
        self.nesterov = self.st.checkbox(
            "Use Nesterov Momentum",
            self.nesterov
        )
        if self.nesterov:
            self.momentum = self.st.slider(
                "Nesterov Momentum", 0., 1.,
                self.momentum
            )
        self.average_period = int(self.st.number_input(
            "Averaging Period", 1,
            value=self.average_period
        ))

    def get(self):
        return SWA(
            optimizer=SGD(
                learning_rate=self.LEARNING_RATE.get(),
                momentum=self.momentum,
                nesterov=self.nesterov,
                clipnorm=5.0
            ),
            average_period=int(self.average_period)
        )


class OptimizerConfig(PropertyContainer):
    """
    Optimizer config that supplies the currently configured optimizer upon it's .get() function
    """
    _name = "Optimizer"
    ACTIVE = "SGD"

    ADAM_PARAMETER: AdamConfig = AdamConfig()
    SGD_PARAMETER: SGDConfig = SGDConfig()
    RECTIFIED_ADAM_PARAMETER: RectifiedAdamConfig = RectifiedAdamConfig()
    SWA_SGD_PARAMETER: SWA_SGD_Optimizer = SWA_SGD_Optimizer()

