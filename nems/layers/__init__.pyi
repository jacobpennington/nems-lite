'''
Collects commonly-used Modules for easier import.

Ex: `from nems.modules import WeightChannels, FIR, DoubleExponential`

'''

# NOTE: These don't necessarily need to all stay as separate files, just did
# that for the example skeleton. Could make sensible groups like `filters.py`
# for STRF, FIR and `nonlinearities.py` for DoubleExponential, others.
from nems.layers.double_exponential import DoubleExponential
from nems.layers.fir_filter import FIR
from nems.layers.strf import STRF
from nems.layers.weight_channels import WeightChannels
from nems.layers.level_shift import LevelShift
