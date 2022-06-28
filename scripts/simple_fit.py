'''
Demonstrates how to fit a basic LN-STRF model using NEMS.

NOTE: This script does not actually work yet! This is currently intended as a
style guide / planning document, i.e. this is what NEMS *should* do. When this
workflow is actually functional, it should be converted into a jupyter notebook
that uses some sample data from our lab and demonstrates some plotting/analysis.

'''

import numpy as np

from nems import Model, Recording
from nems.layers import STRF, WeightChannels, FIR, DoubleExponential, StateGain
from nems.layers.base import Layer
from nems.models import LN_STRF

# TODO: 2D covers most cases, but we should support higher-dimensional data.
#       I.e. built-in layers wouldn't change, but the Signal class and related
#       should be more general. Update script documentation after investigating
#       support for this.

# TODO: I still don't love the idea of hard-coding the dimension order, but
#       I don't see a good way around it without switching to xarrays. But that
#       would have its own issues, because then we have to hard-code a dimension
#       name (at least for the built-in modules). Probably best to just stick
#       with ndarray for simplicity and enforce the ordering for built-ins.

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
    # which includes a sound stimulus (assumed to be pre-converted to a
    # spectrogram here), spiking responses recorded with implanted electrodes,
    # and pupil size as a measure of arousal.
    print(f'Loading data from {file_path}, but not really...')
    spectrogram = np.random.random(size=(10000, 18))
    response = np.random.choice([0,1], size=(10000, 100), p=[0.99, 0.01])
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
# TODO: Not actually sure where the StateGain layer should go, ask Stephen


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


# Some models will need more data than input, output, and state.
def my_complicated_data_loader(file_path):
    # Dummy function to demonstrate the data format.
    print(f'Loading data from {file_path}, but not really...')
    spectrogram = np.random.random(size=(10000, 18))
    response = np.random.choice([0,1], size=(10000, 100), p=[0.99, 0.01])
    pupil = np.random.random(size=(10000, 1))
    other_state = np.random.choice(['on', 'off'], size=(10000, 1))

    return spectrogram, response, pupil, other_state

stim, resp, pupil, state = my_complicated_data_loader('/path/data.csv')


# For a model that uses multiple inputs, we need to package the data into a
# Recording. Each data variable will be converted to RasterizedSignal by default
# (a wrapper around a Numpy array with some utility methods).
recording = Recording({
    'stimulus': stim, 'response': resp, 'pupil': pupil, 'state': state
    })
# TODO: is it worth it to keep using recordings & signals, or should we just
#       use a dict of numpy arrays? signals/recordings were nice for dealing
#       with epochs & preprocessing, but depending on what makes it into the
#       final version this may not be necessary. Means there will be a lot of
#       functions where we have to check if an array was passed vs a signal,
#       instead of just working with one data-type.
#       (alternatively, always require that loaded data is packaged into a
#        recording object, but I don't really like that option)

# Dummy layer that can make use of an arbitrary number of state variables.
class Sum(Layer):
    def evaluate(self, *inputs):
        # All inputs are treated the same
        return np.sum(inputs)

# Dummy layer that makes use of its inputs in different ways.
class LinearWeighting(Layer):

    def __init__(self, **kwargs):
        super.__init__(**kwargs)
        self.set_parameter_values({'a': 0.2, 'b': 0.5})

    def evaluate(self, prediction, pupil, *other_states):
        # prediction and pupil are treated one way, whereas all non-pupil state
        # variables are treated the same.
        a, b = self.get_parameter_values('a', 'b')
        return a*prediction + b*pupil + np.sum(other_states)


# Now we build the Model as before, but we can specify which Layer receives
# which input(s) during fitting. For layers that only need a single input, we
# provide a string referencing the name of signal in the recording. In this case,
# we named the spectrogram data 'stimulus' so that will be the input to
# `WeightChannels`. The inputs for `FIR` and `DoubleExponential` should be the
# outputs of their preceding layers, so we don't need to specify an input.
# However, we want to track the output of the LN portion of the model separately
# from the rest, so we specify `output='LN_output'`.

# We want to apply the `Sum` module both to the output of the LN portion of the
# model and to both of our state variables. The ordering of the data variables
# doesn't matter in this case, so we provide a list with the name of each signal.
# However, the ordering does matter for the `LinearWeighting` layer, so we
# provide a dictionary instead to ensure the correct signals are mapped to each
# parameter for `LinearWeighting.evaluate`. Note that we could also still use a
# list as long as the names are specified in the correct order.
# 
# We'll also use a factorized, parameterized STRF in place of the full-rank
# version to demonstrate usage.
# TODO: shorter keyword name for parameterization?
layers = [
    WeightChannels(shape=(18,4), parameterization='gaussian', input='stimulus'),
    FIR(shape=(4, 25), parameterization='P3Z1'),
    DoubleExponential(output='LN_output'),
    Sum(input=['LN_output', 'state', 'pupil'],
                    output='weighted_output'),
    LinearWeighting(input={'pred': 'weighted_output', 'pupil': 'pupil',
                           'other_states': 'state'})
]
model = Model(layers=layers)
# Note that we also passed a list of layer instances to the constructor instead
# of using the `add_layers()` method. These approaches are interchangeable.


# We fit as before, but provide the recording in place of individual data
# variables. The input to the first layer is already specified in the layer
# itself, so we only need to tell the model what data to match its output to
# (the 'response' Signal in this case).
model.fit(input=recording, target='response')
prediction = model.predict(recording)


# Instead of specifying a custom model, we can also use a pre-built model
# (scipy is the default backend, so we don't actually need to specify it).
# In this case we've also specified an output_name. Now the output of any
# layer that doesn't specify an output name will be called 'pred' instead
# of 'output'.
LN_STRF.fit(recording=recording, input_name='stimulus',
            target_name='response', output_name='pred')
prediction = LN_STRF.predict(recording)
