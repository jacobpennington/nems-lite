"""Demonstrates how to build models using keywords.

NOTE: This script does not actually work yet!

"""

from nems import ModelSpec
from nems.modules import STRF, WeightChannels, FIR, DoubleExponential
from nems.registry import keyword_lib

# View all defined keywords
keyword_lib.keywords


# Build a model from keywords. The following are equivalent.
model = ModelSpec.from_keywords('wc', 'fir', 'dexp')
model = ModelSpec.from_keywords('wc-fir-dexp')

keywords = ['wc', 'fir', 'dexp']
modules = [keyword_lib[k] for k in keywords]
model = ModelSpec(modules=modules)

modules = [WeightChannels, FIR, DoubleExponential]
model = ModelSpec(modules=modules)


# Specify module options using keywords. The following are equivalent.
# TODO: add a str vs Module check in add_module instead of using
#       a separate .from_keywords method?
model.add_module('wc.4x18.g')
model.add_module(WeightChannels(shape=(4,18), parameterization='gaussian'))
