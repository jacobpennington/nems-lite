'''
Demonstrates how to fit a basic LN-STRF model using NEMS.

NOTE: This script does not actually work yet! This is currently intended as a
style guide / planning document, i.e. this is what NEMS *should* do. When this
workflow is actually functional, it should be converted into a jupyter notebook
that uses some sample data from our lab and demonstrates some plotting/analysis.

'''

import numpy as np

from nems import Model
from nems.layers import STRF, WeightChannels, FIR, DoubleExponential, StateGain
from nems.layers.base import Layer
from nems.models import LN_STRF


# All data should be loaded as 2D numpy arrays. We leave this step up to
# individual users, since data formats are so variable. Built-in model layers
# (see `nems.layers`) expect time to be represented on the first axis, and
# other data dimensions (neurons, spectral channels, etc.) to be represented on
# the second axis. For example, spiking responses from a population of neurons
# should have shape (T, N) where T is the number of time bins and N is the
# number of neurons. State data (like pupil size) should have shape (T, M) where
# M is the number of state variables.
def my_data_loader(file_path):
    # Dummy function to demonstrate the data format. This resembles LBHB data,
    # which includes a sound stimulus (assumed here to be pre-converted to a
    # spectrogram), spiking responses recorded with implanted electrodes
    # (assumed here to be pre-converted to firing rate / PSTH), and pupil size
    # as a measure of arousal.
    print(f'Loading data from {file_path}, but not really...')
    spectrogram = np.random.random(size=(10000, 18))
    response = np.random.random(size=(10000, 100))
    pupil_size = np.random.random(size=(10000, 1))

    return spectrogram, response, state

spectrogram, response, pupil_size = my_data_loader('/path/to/my/data.csv')


# Build a Model instance, which composes the (typically sequential)
# operations of several Layer instances.
model = Model()
model.add_layers(
    STRF(shape=(25,18)),  # Full-rank STRF, 25 temporal x 18 spectral bins
    DoubleExponential(),  # Double-exponential nonlinearity
    StateGain()
)
# TODO: Not actually sure where the StateGain layer should go, but I think this
#       is appropriate? Ask Stephen to be sure.


# Fit the model to the data. Any data preprocessing should happen separately,
# before this step. No inputs or outputs were specified for the Layers,
# so `input` will be the input to the first layer, the output of the first layer
# will be the input to the second layer and so on. The fitter will try to match
# the final output (of the StateGain layer in this case) to `target`.
#
# Some modules, like StateGain, expect state data to be present in some form
# as an additional model input, so we also specify `pupil_size` as state data.
# 
# By default, `scipy.optimize.minimize` will be used for optimization
# (which can also be specified using the `backend` parameter). This also tells
# the model to use each layer's standard `evaluate` method for transforming
# inputs (whereas `backend='tf'`, for example, would use `Layer.tf_layer`).
# See `nems.models.base.Model.fit` for additional fitting options.
model.fit(input=spectrogram, target=response, state=pupil_size, backend='scipy')

# Predict the response to the stimulus spectrogram using the fitted model.
prediction = model.predict(spectrogram, state=pupil_size, backend='scipy')

# As mentioned, 'scipy' is already the default backend. However, if we wanted to
# set a different backend specified in the `model.fit` documentation, we can
# do the following so that we don't need to keep specifying it.
model.default_backend = 'scipy'


# Some models will need more data than input, output, and state. In that case,
# the fit process requires a couple of extra steps. First, we'll define some
# simple example Layers that require additional inputs and then load data again.

# Dummy layer that can make use of an arbitrary number of inputs.
class Sum(Layer):
    def evaluate(self, *inputs):
        # All inputs are treated the same
        return np.sum(inputs)

# Dummy layer that makes use of its inputs in different ways.
# NOTE: Some implementation details are hidden for simplicity.
class LinearWeighting(Layer):
    def evaluate(self, prediction, pupil, *other_states):
        # `prediction` and `pupil` are treated one way, whereas all non-pupil
        # state variables (`other_state`) are treated the same.
        #
        # Assume fittable parameters 'a' and 'b' have been initialized.
        a, b = self.get_parameter_values('a', 'b')
        return a*prediction + b*pupil + np.sum(other_states)


def my_complicated_data_loader(file_path):
    # Dummy function to demonstrate the data format.
    print(f'Loading data from {file_path}, but not really...')
    spectrogram = np.random.random(size=(10000, 18))
    response = np.random.random(size=(10000, 100))
    pupil = np.random.random(size=(10000, 1))
    other_state = np.random.choice(['on', 'off'], size=(10000, 1))

    return spectrogram, response, pupil, other_state

stim, resp, pupil, state = my_complicated_data_loader('/path/data.csv')


# For a model that uses multiple inputs, we need to package the input data into
# a dictionary. This way, modules can specify which inputs they need using the
# assigned keys.
data = {'stimulus': stim, 'pupil': pupil, 'state': state}


# Now we build the Model as before, but we can specify which Layer receives
# which input(s) during fitting. We'll also use a factorized, parameterized STRF
# in place of the full-rank version to demonstrate usage. For layers that only
# need a single input, we provide the approprtiate dictionary key. In this case,
# we named the spectrogram data 'stimulus' so that will be the input to
# `WeightChannels`. The inputs for `FIR` and `DoubleExponential` should be the
# outputs of their preceding layers, so we don't need to specify an input.
# However, we want to track the output of the LN portion of the model
# (WeightChannels, FIR, and DoubleExponential) separately from the rest, so we
# specify `output='LN_output'`.

# We want to apply the `Sum` module to the output of the LN portion of the
# model and to both of our state variables. The ordering of the data variables
# doesn't matter for this layer, so we provide a list of the dictionary keys.
# However, the ordering does matter for the `LinearWeighting` layer, so we
# provide a dictionary instead to ensure the correct data is mapped to each
# parameter for `LinearWeighting.evaluate`.
# NOTE: We could also have used a list as long as the keys were specified in
#       the same order as in the method definition. However, using a dictionary
#       makes it clear what the data is mapped to without needing to refer back
#       to the Layer implementation.
layers = [
    WeightChannels(shape=(18,4), parameterization='gaussian', input='stimulus'),
    FIR(shape=(4, 25), parameterization='P3Z1'),
    DoubleExponential(output='LN_output'),
    Sum(input=['LN_output', 'state', 'pupil'],
                    output='summed_output'),
    LinearWeighting(input={'pred': 'summed_output', 'pupil': 'pupil',
                           'other_states': 'state'})
]
model = Model(layers=layers)
# Note that we also passed a list of `Layer` instances to the `Model` constructor
# instead of using the `add_layers()` method. These approaches are
# interchangeable.


# We fit as before, but provide the `data` dictionary in place of individual
# variables. The necessary inputs are already specified in the layers, so we
# only need to tell the model what data to match its output to
# (`resp`) in this case.
model.fit(input=data, target=resp)
prediction = model.predict(data)


# Instead of specifying a custom model, we can also use a pre-built model.
# In this case we've also specified `output_name`. Now the output of any
# layer that doesn't specify an output name will be called 'pred' instead
# of the default, 'output'. Our standard LN_STRF model only needs the stimulus
# spectrogram as input and neural response (as firing rates / PSTH) as a target.
LN_STRF.fit(input=stim, target=resp, output_name='pred')
prediction = LN_STRF.predict(stim)
