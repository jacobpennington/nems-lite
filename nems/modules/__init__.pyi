'''
Collects commonly-used Modules for easier import.

Ex: `from nems.modules import WeightChannels, FIR, DoubleExponential`

'''

# NOTE: These don't necessarily need to all stay as separate files, just did
# that for the example skeleton. Could make sensible groups like `filters.py`
# for STRF, FIR and `nonlinearities.py` for DoubleExponential, others.
from nems.modules.double_exponential import DoubleExponential
from nems.modules.fir_filter import FIR
from nems.modules.strf import STRF
from nems.modules.weight_channels import WeightChannels
from nems.modules.level_shift import LevelShift
