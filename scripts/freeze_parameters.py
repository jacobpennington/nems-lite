'''
Show how to freeze module parameters in a multi-step fitting process.
NOTE: This script does not actually work yet!

'''

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
model.freeze_parameters('wc', 'fir', 'lvl')
# model.freeze_parameters(0, 1, 2)  # integer indexing would also work
# TODO: reasoning behind adding the `name` kwarg was that integer indexing
#       can get confusing / hard to read when coming back to code later on.
#       However, this adds another layer of string references that are separate
#       from keywords. Maybe better to have those be the same thing?
#       I.e. if a keyword is specified, it's also the name (unless overwritten).
#       If a name is specified, it's also a keyword for default options
#       (unless overwritten). This comes with complications though, like dealing
#       with  name clashes if there are multiple 'wc' keywords used in a model.


# Fit the model again, using the previous fit as a starting point.
model.fit(recording=recording, stimulus_name='stimulus',
          response_name='response', fit_options=initialization,
          reset_parameters=False)

# Unfreeze the linear portion.
model.unfreeze_parameters('wc', 'fir', 'lvl')
# In this case, this is equivalent to unfreezing all modules, which we can
# do by *not* specifying any names or indices:
# model.unfreeze_parameters()

# Now perform a final fit of all modules simultaneously, using a finer
# optimization tolerance.
final_fit = {'tolerance': 1e4}
model.fit(recording=recording, stimulus_name='stimulus',
          response_name='response', fit_options=final_fit,
          reset_parameters=False)

# NOTE: In this example, we always froze all parameters of a given module using
#       the model-level method. If we had wanted to freeze some of a module's
#       parameters but not others, we could use the module-level method.
#       For example:
# Just freeze DoubleExponential's kappa parameter:
model.modules('dexp').freeze_parameters('kappa')


# TODO: This script made me realize it gets unnecessarily cumbersome to specify
#       the same recording=recording, stimulus_name='stimulus', etc. over and
#       over for use-cases like this, when the arguments always stay the same.
#       Maybe it would be useful to store pointers to the last fit arguments
#       used in the model? Then we could use something like:
model.refit(fit_options=final_fit)

# Idea being that this would copy all previous arguments, unless a new one
# is provided to overwrite them.
