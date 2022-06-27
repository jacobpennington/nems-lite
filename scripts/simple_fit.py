'''
Demonstrates how to fit a basic LN-STRF model using NEMS.

NOTE: This script does not actually work yet! This is currently intended as a
style guide / planning document, i.e. this is what NEMS *should* do. When this
workflow is actually functional, it should be converted into a jupyter notebook
that uses some sample data from our lab and demonstrates some plotting/analysis.

'''

import numpy as np

from nems import Model, Recording
from nems.layers import STRF, WeightChannels, FIR, DoubleExponential
from nems.layers.base import Module
from nems.models import LN_STRF

# All data should be loaded as 2D numpy arrays. We leave this step up to
# individual users, since data formats are so variable. Built-in model layers
# (see `nems.layers`)
# as a 2D numpy array, with dimensions T (time bins) x N (neurons). Other
# data formatted as needed, but multi-dimensional data should place time
# on the 0-axis (different order from old NEMS).

# NOTE: if we switched to xarrays or some other labeled structure, we wouldn't
# have to bother with this. Just say that the time axis has to be called 'time'
# (or whatever) and then we always know which axis it is. Or we could even use
# dataframes which I think would simplify epochs quite a bit.
def my_data_loader(file_path):
    # Dummy function to demonstrate the data format
    print(f'Loading data from {file_path}, but not really...')
    stimulus = np.random.random(size=(10000, 18))
    response = np.random.choice([0,1], size=(10000, 100), p=[0.99, 0.01])
    pupil_size = np.random.random(size=(10000, 1))

    return stimulus, response, state

stimulus, response, pupil_size = my_data_loader('/path/to/my/data.csv')


# Build the model, which is a NEMS Model that composes the operations of
# several Module instances.
model = Model()
model.add_modules(
    STRF(shape=(25,18)),  # Full-rank STRF, 25 temporal x 18 spectral bins
    DoubleExponential()   # Double-exponential nonlinearity
)

# Fit the model to the data. Any preprocessing should happen separately,
# before this step. No inputs or outputs were specified for the Modules,
# so `input` will be the input to the first module, and the output
# of the first module will be the input to the second module. The fitter will
# try to match the final output to `target`.
# NOTE: this gets around the problem of hard-coded signal names (I think)
model.fit(input=stimulus, target=response, state=pupil_size,
          backend='scipy')

# Predict the response to the stimulus.
prediction = model.predict(stimulus, state=pupil_size, backend='tf')

# NOTE: switched backends here to point out that a model should be able to
# interchange backends. I.e. after the fit is finished, parameters are stored in
# the Modules in a backend-agnostic fashion. This should also make it easier
# to validate backends against each other, e.g.:
model.fit(backend='tf')
prediction = model.predict(backend='tf')
model.fit(backend='scipy', reset_parameters=True)
same_prediction = model.predict(backend='scipy')
prediction == same_prediction

# I don't love having to specify backend=x everywhere, but we could always add
# something like
model.default_backend = 'tf'


# Some models will need more data than just stimulus response (or x and y)
def my_complicated_data_loader(file_path):
    # Dummy function to demonstrate the data format
    print(f'Loading data from {file_path}, but not really...')
    stimulus = np.random.random(size=(10000, 18))
    response = np.random.choice([0,1], size=(10000, 100), p=[0.99, 0.01])
    pupil = np.random.random(size=(10000, 1))
    state = np.random.choice(['state A', 'state B'], size=(10000, 100))

    return stimulus, response, pupil, state

# Dummy module that 'make use of' pupil and state
class LinearWeighting(Module):
    pass

stimulus, response, pupil, state = my_complicated_data_loader('/path/data.csv')


# For a model that uses multiple inputs, we need to package the data into a
# Recording. Each data variable will be converted to RasterizedSignal by default
# (a wrapper around a Numpy array with some utility methods).
recording = Recording(
    {'stimulus': stimulus, 'response': response, 'pupil': pupil, 'state': state}
)
# TODO: is it worth it to keep using recordings & signals, or should we just
#       use a dict of numpy arrays? signals/recordings were nice for dealing
#       with epochs & preprocessing, but depending on what makes it into the
#       final version this may not be necessary. Means there will be a lot of
#       functions where we have to check if an array was passed vs a signal,
#       instead of just working with one data-type.
#       (alternatively, always require that loaded data is packaged into a
#        recording object, but I don't really like that option)


# Now we build the Model as before, but we specify which Module receives
# which input(s) during fitting. We'll also use a factorized, parameterized
# STRF inplace of the full-rank version.
# TODO: shorter keyword name for parameterization?
modules = [
    WeightChannels(shape=(18,4), parameterization='gaussian', input='stimulus'),
    FIR(shape=(4, 25), parameterization='P3Z1'),
    DoubleExponential(output='LN_output'),
    LinearWeighting(input=['LN_output', 'state', 'pupil'],
                    output='weighted_output')
]
model = Model(modules=modules)
# Note that we passed a list of module instances to the constructor instead of
# using the add_modules() method. These approaches are interchangeable.



# We fit as before, but provide the recording in place of individual data
# variables.
model.fit(recording=recording, target_name='response', backend='scipy')

# There's no need to separately generate a prediction since it will already be
# represented as 'weighted_output' in the recording (unless we wanted to predict
# response to a validation stimulus, for example).
# TODO: Going back and forth on this. Modifying the recording in-place when
#       calling a method on a completely separate object is pretty bad form.
#       Running through the model evaluation doesn't take that long, so would
#       be better to just fit on a copy and expect a separate prediction call.
#       But then you have two copies of the data in memory while fitting, which
#       isn't ideal.
#       Another option: make this work on a copy by default, but add an
#       `inplace=True` kwarg for cases where memory is an issue.


# Instead of specifying a custom model, we can also use a pre-built model
# (scipy is the default backend, so we don't actually need to specify it).
# In this case we've also specified an output_name. Now the output of any
# module that doesn't specify an output name will be called 'pred' instead
# of 'output'.
LN_STRF.fit(recording=recording, input_name='stimulus',
            target_name='response', output_name='pred')
prediction = recording['pred']
