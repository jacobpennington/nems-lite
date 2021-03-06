"""Demonstrates how to build models using keywords.

NOTE: This script does not actually work yet!

"""

from nems import Model
from nems.layers import STRF, WeightChannels, FIR, DoubleExponential
from nems.registry import keyword_lib

# View all defined keywords
keyword_lib.keywords


# Build a model from keywords. The following are equivalent.
model = Model.from_keywords('wc', 'fir', 'dexp')
model = Model.from_keywords('wc-fir-dexp')

keywords = ['wc', 'fir', 'dexp']
layers = [keyword_lib[k] for k in keywords]
model = Model(layers=layers)

layers = [WeightChannels, FIR, DoubleExponential]
model = Model(layers=layers)


# Specify layer options using keywords. The following are equivalent.
# TODO: add a str vs Module check in add_layer instead of using
#       a separate .from_keywords method?
model.add_layer('wc.4x18.g')
model.add_layer(WeightChannels(shape=(4,18), parameterization='gaussian'))
