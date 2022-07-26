'''
Collects commonly-used Modules for easier import.

Ex: `from nems.modules import WeightChannels, FIR, DoubleExponential`

'''

from .nonlinearity import LevelShift, DoubleExponential, RectifiedLinear
from .filter import FIR, STRF
from .weight_channels import WeightChannels, GaussianWeightChannels

from .base import Layer, Phi, Parameter
