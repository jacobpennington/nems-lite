'''
Collects commonly-used Modules for easier import.

Ex: `from nems.modules import WeightChannels, FIR, DoubleExponential`

'''

from .nonlinearity import LevelShift, DoubleExponential, RectifiedLinear, ReLU
from .filter import FiniteImpulseResponse, FIR, STRF
from .weight_channels import WeightChannels, GaussianWeightChannels
from .state import StateGain

from .base import Layer, Phi, Parameter
