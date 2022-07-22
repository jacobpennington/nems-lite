'''
Collects commonly-used Modules for easier import.

Ex: `from nems.modules import WeightChannels, FIR, DoubleExponential`

'''

from .nonlinearity import DoubleExponential
from .filter import FIR, STRF
from .weight_channels import WeightChannels
from .level_shift import LevelShift

from .base import Phi, Parameter
