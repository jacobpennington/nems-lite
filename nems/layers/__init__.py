'''
Collects commonly-used Modules for easier import.

Ex: `from nems.modules import WeightChannels, FIR, DoubleExponential`

'''

from .double_exponential import DoubleExponential
from .filter import FIR, STRF
from .weight_channels import WeightChannels
from .level_shift import LevelShift

from .base import Phi, Parameter
