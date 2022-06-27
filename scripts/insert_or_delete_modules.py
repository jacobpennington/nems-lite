'''
Demonstrates how to change a Model's modules after construction.
NOTE: This script does not actually work yet!

'''

import numpy as np

from nems import Model, load_recording
from nems.layers import WeightChannels, FIR, LevelShift, DoubleExponential


# Sometimes we might want to insert or delete individual modules without
# building and fitting a new Model from scratch
# (for example, to compare prediction accuracy with/out that module).

# Build a simple LN model and fit it to some data.
model = Model()
model.add_modules(
    WeightChannels(shape=(4,18)),
    FIR(shape=(4, 25)),
    LevelShift(name='levelshift'),
    DoubleExponential(),
)

recording = load_recording
model.fit(recording=recording, input_name='stimulus',
          target_name='response')

# Now let's say we want to test whether the LevelShift() module is really
# necessary. Technically it's redundant with DoubleExponential(), but in
# practice it may make our fit routine work better. We can generate a prediction,
# remove LevelShit, and re-fit the model (without resetting the parameters).
prediction_with = model.predict(recording['stimulus'])
model.delete_module(2)  # Remove the third module in list-order
# This would also work, since we specified a name for the module.
# model.delete_module('levelshift')
model.fit(recording=recording, input_name='stimulus',
          target_name='response', reset_parameters=False)
prediction_without = model.predict(recording['stimulus'])
