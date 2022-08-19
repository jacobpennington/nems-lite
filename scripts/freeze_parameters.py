"""Shows how to freeze module parameters in a multi-step fitting process.
NOTE: This script does not actually work yet!

"""

import numpy as np

from nems import Model, load_recording
from nems.layers import WeightChannels, FIR, LevelShift, DoubleExponential


# In the past, we've found that our models predict our data best if we split
# the fitting process into linear and nonlinear steps. We can do this with
# NEMS using `model.freeze_parameters()` and repeated calls to `model.fit()`.

# Load the data
recording = load_recording

# Build a simple linear model.
model = Model()
model.add_modules(
    WeightChannels(shape=(4,18), name='wc'),
    FIR(shape=(4, 25), name='fir'),
    LevelShift(name='lvl'),
)

# Fit the linear model using a coarse tolerance.
initialization = {'tolerance': 1e3}
model.fit(recording=recording, stimulus_name='stimulus',
          response_name='response', fit_options=initialization)

# Add a nonlinearity to the end of the model.
model.add_module(DoubleExponential(name='dexp'))

# Freeze the parameters of the linear portion of the model.
# The values of these parameters will not be changed during fitting.
model.freeze_layers('wc', 'fir', 'lvl')
# model.freeze_layers(0, 1, 2)  # integer indexing would also work

# Fit the model again, using the previous fit as a starting point.
model.fit(recording=recording, stimulus_name='stimulus',
          response_name='response', fitter_options=initialization,
          reset_parameters=False)

# Unfreeze the linear portion.
model.unfreeze_layers('wc', 'fir', 'lvl')
# In this case, this is equivalent to unfreezing all modules, which we can
# do by *not* specifying any names or indices:
# model.unfreeze_layers()

# Now perform a final fit of all modules simultaneously, using a finer
# optimization tolerance.
final_fit = {'tolerance': 1e4}
model.fit(recording=recording, stimulus_name='stimulus',
          response_name='response', fitter_options=final_fit,
          reset_parameters=False)

# NOTE: In this example, we always froze all parameters of a given module using
#       the model-level method. If we had wanted to freeze some of a module's
#       parameters but not others, we could use the module-level method.
#       For example:
# Just freeze DoubleExponential's kappa parameter:
model.layers['dexp'].freeze_parameters('kappa')
# model.get_layer('dexp').freeze_parameters('kappa')  # also works

# Idea being that this would copy all previous arguments, unless a new one
# is provided to overwrite them.
