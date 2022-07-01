'''
Collects commonly-used Modules for easier import.

Ex: `from nems.modules import WeightChannels, FIR, DoubleExponential`

'''

# NOTE: These don't necessarily need to all stay as separate files, just did
# that for the example skeleton. Could make sensible groups like `filters.py`
# for STRF, FIR and `nonlinearities.py` for DoubleExponential, others.
from .double_exponential import DoubleExponential
from .fir_filter import FIR
from .strf import STRF
from .weight_channels import WeightChannels
from .level_shift import LevelShift

from .base import Layer, Phi, Parameter  # Keep these imports last
