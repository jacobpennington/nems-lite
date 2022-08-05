import numpy as np
import tensorflow as tf
from tensorflow.keras import Input


def build_model(nems_model, tf_layers, input, batch_size=None):
    """Docs TODO
    
    Parameters
    ----------
    nems_model : nems.layers.base.Model
    tf_layers : list of tf.keras.layers.Layer
    input : dict of ndarray
        Same as `Model.evaluate`, with some exceptions:
        1) Single `np.ndarray` not allowed.
        2) First dimension of arrays represents samples/trials instead of time.
           Ex: `stim(shape=(10000,18)) -> stim(shape=(10,1000,18))` for an
               18-channel spectrogram consisting of 10 trials of 1000 time bins.
    
    """

    tf_input_dict = []
    for k, v in input.items():
        # Skip trial/sample dimension when determining shape.
        tf_in = Input(shape=v.shape[1:], name=k, batch_size=batch_size,
                      dtype='float32')  # TODO: why hard-code float32?
        tf_inputs.append(tf_in)

    tf_outputs = []
    # TODO: iterate through tf_layers, call tf_layer(previous_output) to get
    #       next output. But need to figure out input/output mapping just like
    #       with `Model.evaluate`.
    #       Probably best to start by re-thinking `Model.evaluate`, write it in
    #       a way that's portable so it can be re-used here.

    #       possible snag: NEMS layers can depend on data names that don't exist
    #       in input. Not seeing a way to do that here.

    # TODO: make this work with state as the first layer
    # outputs = stim_input
    # for layer in self.model_layers:
    #     if layer._STATE_LAYER:
    #         outputs = layer([outputs, state_input])
    #     else:
    #         outputs = layer(outputs)

    tf_inputs = list(tf_input_dict.values())
    model = tf.keras.Model(inputs=tf_inputs, outputs=tf_outputs,
                           name=nems_model.name)

    return model
