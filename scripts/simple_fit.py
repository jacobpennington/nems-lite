'''
Demonstrates how to fit a basic LN-STRF model using NEMS.

NOTE: This script does not actually work yet! This is currently intended as a
style guide / planning document, i.e. this is what NEMS *should* do. When this
workflow is actually functional, it should be converted into a jupyter notebook
that uses some sample data from our lab and demonstrates some plotting/analysis.

NOTE: Things that are currently missing/undecided:
      1) How should input/output signals be specified for more general cases?
         a) If input is specified, look for the recording signal with that name.
            Otherwise, assume output of previous Module is the input. If output
            is not named, call it '{module_name}.output' or something like that.

            ex: `STRF(shape=(x,y), input='stimulus', output='prediction')`
                `DoubleExponential()

            Would cause this transformation:
            Recording({'response': y, 'stimulus': x0})
            -->
            Recording({'response': y, 'stimulus': x0, 'prediction': x1,
                      'DoubleExponential.output': x2})
            (or just overwrite a single generic name like "output" to avoid
             keeping too many signals in memory)

         b) Akin to Tensorflow, specify "X,Y" (stimulus, response?) in fit call,
            and just pass output through Modules in order (similar to above).

            ex: `model.fit(stimulus=data_X, response=data_Y)`
            or: `model.fit(x=Recording('stim'), y=Recording('resp'))

            But still allow specific per-module inputs some other way, for
            state models etc. that depend on multiple inputs.

      2) Sensible naming scheme in place of stimulus & response? (low priority)
         a) Stimulus & Response are probably what most people would be using,
            but what if some one is looking at encoding of pupil, behavior, etc.
            without considering stimulus?
         b) x, y (or similar) would be more general and reminiscent of sklearn,
            Tensorflow, etc. This might be *too* general (i.e. we still want
            to target comp. neuro), but would reinforce the idea that these are
            really just placeholders and users should feel free to name signals
            as whatever fits their data best.
         c) I think general terms that represent "thing that is being encoded"
            and "variable the thing is encoded in" would be ideal. But I'm not
            coming up with any good ideas for that.

'''

import numpy as np

from nems import ModelSpec, Recording
from nems.modules import STRF, WeightChannels, FIR, DoubleExponential
from nems.modules.base import Module
from nems.models import LN_STRF

# Assume user is able to somehow load their data and format responses as
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

    return stimulus, response

stimulus, response = my_data_loader('/path/to/my/data.csv')


# Build the model, which is a NEMS ModelSpec that composes the operations of
# several Module instances.
model = ModelSpec()
model.add_modules(
    STRF(shape=(25,18)),  # Full-rank STRF, 25 temporal x 18 spectral bins
    DoubleExponential()   # Double-exponential nonlinearity
)

# Fit the model to the data. Any preprocessing should happen separately,
# before this step. No inputs or outputs were specified for the Modules,
# so `stimulus` will be the input to the first module, and the output
# of the first module will be the input to the second module.
# NOTE: this gets around the problem of hard-coded signal names (I think)
model.fit(stimulus=stimulus, response=response, backend='scipy')

# Predict the response to the stimulus.
prediction = model.predict(stimulus, backend='tf')

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
model.set_default_backend('tf')


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


# For a model that uses multiple inputs, we need to package all of the data
# in a single NEMS Recording. Each data variable will be converted to a
# NEMS RasterizedSignal by default (a wrapper around a Numpy array
# with some utility methods).
recording = Recording({'stimulus': stimulus, 'response': response,
                       'pupil': pupil, 'state': state})

# Now we build the ModelSpec as before, but we specify which Module receives
# which input(s) during fitting. We'll also use a factorized, parameterized
# STRF inplace of the full-rank version.
modules = [
    WeightChannels(shape=(4,18), parameterization='gaussian', input='stimulus'),
    FIR(shape=(4, 25), parameterization='P3Z1'),
    DoubleExponential(output='LN_output'),
    LinearWeighting(input=['LN_output', 'state', 'pupil'],
                    output='weighted_output')
]
model = ModelSpec(modules=modules)
# Note that we passed a list of module instances to the constructor instead of
# using the add_modules() method. These approaches are interchangeable.


# We fit as before, but provide the recording in place of individual data
# variables.
model.fit(recording=recording, response_name='response', backend='scipy')

# There's no need to separately generate a prediction since it will already be
# represented as 'weighted_output' in the recording (unless we wanted to predict
# response to a validation stimulus, for example).


# Instead of specifying a custom model, we can also use a pre-built model.
# (scipy is the default backend, so we don't actually need to specify it)
LN_STRF.fit(recording=recording, stimulus_name='stimulus',
            response_name='response')
prediction = recording['output']
