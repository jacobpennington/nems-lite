'''
Demonstrates how to change a Model's layers after construction.
NOTE: This script does not actually work yet!

'''

import numpy as np

from nems import Model, load_recording
from nems.layers import WeightChannels, FIR, LevelShift, DoubleExponential


# Sometimes we might want to insert or delete individual layers without
# building and fitting a new Model from scratch
# (for example, to compare prediction accuracy with/out that layer).

# Build a simple LN model and fit it to some data.
model = Model()
model.add_layers(
    WeightChannels(shape=(4,18)),
    FIR(shape=(4, 25)),
    LevelShift(name='levelshift'),
    DoubleExponential(),
)

recording = load_recording
model.fit(recording=recording, input_name='stimulus',
          target_name='response')

# Now let's say we want to test whether the LevelShift() layer is really
# necessary. Technically it's redundant with DoubleExponential(), but in
# practice it may make our fit routine work better. We can generate a prediction,
# remove LevelShit, and re-fit the model (without resetting the parameters).
prediction_with = model.predict(recording['stimulus'])
model.delete_layer(2)  # Remove the third layer in list-order
# This would also work, since we specified a name for the layer.
# model.delete_layer('levelshift')
model.fit(recording=recording, input_name='stimulus',
          target_name='response', reset_parameters=False)
prediction_without = model.predict(recording['stimulus'])
