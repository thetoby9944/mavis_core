from abc import ABC
import numpy as np
import tensorflow as tf
from keras.optimizer_v2.learning_rate_schedule import LearningRateSchedule, PolynomialDecay, \
    ExponentialDecay
from tensorflow_addons.optimizers import Triangular2CyclicalLearningRate, TriangularCyclicalLearningRate

from mavis.presets.base import BaseProperty, PropertyContainer, ExtendedEnum


class AvailableLearningRates(str, ExtendedEnum):
    NoSchedule = "No Schedule"
    ExponentialDecay = "Exponential Decay"


class LearningRateBase(BaseProperty, ABC):
    def get(self) -> LearningRateSchedule:
        raise NotImplementedError


class NoScheduleConfig(LearningRateBase):
    """
    Config for a simple fixed learning rate without scheduler
    """
    _name = "No Schedule"
    lr: float = 0.01

    def parameter_block(self):
        self.lr = self.st.number_input(
            "Learning Rate", 0., 1.,
            float(self.lr)
        )

    def get(self):
        return self.lr


class PolynomialDecayConfig(LearningRateBase):
    """
    Config kwargs for a polynomial decay learning rate scheduler
    """
    _name = "Polynomial Decay"
    initial_learning_rate: float = 0.01
    end_learning_rate: float = 0.0001
    decay_steps: int = 1000
    power: float = 1.
    cycle: bool = False

    def parameter_block(self):
        self.initial_learning_rate = self.st.number_input(
            "Initial Learning Rate ", 0., 1.,
            self.initial_learning_rate
        )
        self.end_learning_rate = self.st.number_input(
            "End Learning Rate (10^x)", 0., 1.,
            self.end_learning_rate
        )
        self.decay_steps = self.st.number_input(
            "Total learning Rate Decay steps", 1, 100 * 1000,
            self.decay_steps
        )
        self.power = self.st.slider(
            "Power of polynomial", 0., 1.,
            self.power
        )
        self.cycle = self.st.checkbox(
            "Cyclic Learning Rate",
            self.cycle
        )

    def get(self):
        return PolynomialDecay(
            **self.dict()
        )


class ExponentialDecayConfig(LearningRateBase):
    """
    Config kwargs for a exponential decay learning rate scheduler
    """
    _name = "Exponential Decay"
    initial_learning_rate: float = 0.01
    decay_rate: float = 0.25
    decay_steps: int = 1000
    staircase: bool = False

    def parameter_block(self):
        self.initial_learning_rate = self.st.number_input(
            "Initial Learning Rate", 0., 1.,
            self.initial_learning_rate
        )
        self.decay_rate = self.st.slider(
            "Decay Factor", 0., 1.,
            self.decay_rate
        )
        self.decay_steps = self.st.number_input(
            "Decay every `x` steps", 1, 100 * 1000,
            self.decay_steps
        )
        self.staircase = self.st.checkbox(
            "Discrete Decay (Stepwise)",
            self.staircase
        )

    def get(self):
        return ExponentialDecay(
            **self.dict()
        )


class Triangular2CyclicalLearningRateConfig(LearningRateBase):
    """
    Config kwargs for a exponential decay learning rate scheduler
    """
    _name = "Triangular V2 Cyclical Learning Rate"
    initial_learning_rate: float = 0.001
    maximal_learning_rate: float = 0.1
    step_size: int = 20

    def parameter_block(self):
        self.initial_learning_rate = self.st.number_input(
            "Initial Learning Rate", 0., 1.,
            self.initial_learning_rate,
            format="%0.6f"
        )
        self.maximal_learning_rate = self.st.number_input(
            "Maximal Learning Rate", 0., 1.,
            self.maximal_learning_rate,
            format="%0.6f"
        )
        self.step_size = self.st.number_input(
            "Step Size", 1,
            value=self.step_size
        )

    def get(self):
        return Triangular2CyclicalLearningRate(
            maximal_learning_rate=tf.convert_to_tensor(self.maximal_learning_rate),
            **self.dict(exclude={"maximal_learning_rate"})
        )


class CyclicalLearningRateConfig(LearningRateBase):
    """
    Config kwargs for a exponential decay learning rate scheduler
    """
    _name = "Cyclical Learning Rate (Triangular with Fixed Max)"
    initial_learning_rate: float = 0.001
    maximal_learning_rate: float = 0.1
    step_size: int = 20

    def parameter_block(self):
        self.initial_learning_rate = self.st.number_input(
            "Initial Learning Rate", 0., 1.,
            self.initial_learning_rate,
            format="%0.6f"
        )
        self.maximal_learning_rate = self.st.number_input(
            "Maximal Learning Rate", 0., 1.,
            self.maximal_learning_rate,
            format="%0.6f"
        )
        self.step_size = self.st.number_input(
            "Step Size", 1,
            value=self.step_size
        )

    def get(self):
        return TriangularCyclicalLearningRate(
            maximal_learning_rate=np.float32(self.maximal_learning_rate),
            **self.dict(exclude={"maximal_learning_rate"})
        )


class LearningRateConfig(PropertyContainer):
    """
    Learning rate config that supplies the currently configured learning rate upon it's .get() function
    """
    _name = "Learning Rate Schedule"
    ACTIVE: str = "No Schedule"

    NO_SCHEDULE_PARAMETER: NoScheduleConfig = NoScheduleConfig()
    POLYNOMIAL_DECAY_PARAMETER: PolynomialDecayConfig = PolynomialDecayConfig()
    EXPONENTIAL_DECAY_PARAMETER: ExponentialDecayConfig = ExponentialDecayConfig()
    CYCLICAL_V2_PARAMETER: Triangular2CyclicalLearningRateConfig = Triangular2CyclicalLearningRateConfig()
    CYCLICAL_FIXED_PARAMETER: CyclicalLearningRateConfig = CyclicalLearningRateConfig()
