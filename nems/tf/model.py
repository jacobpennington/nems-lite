import numpy as np
import tensorflow as tf
from tensorflow.keras import Input


def build_model(data_maps, tf_layers, input, batch_size=None, name=None):
    """Docs TODO
    
    Parameters
    ----------
    data_maps : dict of nems.layers.base.DataMap
    tf_layers : list of tf.keras.layers.Layer
    input : dict of ndarray
        Same as `Model.evaluate`, with some exceptions:
        1) Single `np.ndarray` not allowed.
        2) First dimension of arrays represents samples/trials instead of time.
           Ex: `stim(shape=(10000,18)) -> stim(shape=(10,1000,18))` for an
               18-channel spectrogram consisting of 10 trials of 1000 time bins.
    
    """

    tf_input_dict = {}
    for k, v in input.items():
        # Skip trial/sample dimension when determining shape.
        tf_in = Input(shape=v.shape[1:], name=k, batch_size=batch_size,
                      dtype='float32')  # TODO: why hard-code float32?
        tf_input_dict[k] = tf_in
    unused_inputs = list(tf_input_dict.keys())

    last_output = None
    tf_data = tf_input_dict.copy()  # Need to keep actual Inputs separate
    for layer in tf_layers:
        # Get all `data` keys associated with Layer args and kwargs
        # TODO: how are Layers supposed to know which one is which?
        #       have to check the name?
        layer_map = data_maps[layer.name]
        all_data_keys = layer_map.args + list(layer_map.kwargs.values())
        all_data_keys = np.array(all_data_keys).flatten().tolist()

        layer_inputs = []
        for k in all_data_keys:
            if k is None:
                # Add last output
                layer_inputs.append(last_output)
            else:
                # Add Input with matching key
                layer_inputs.append(tf_data[k])
            if k in unused_inputs:
                unused_inputs.pop(unused_inputs.index(k))

        # TODO: need to do something with tf.keras.layers.concatenate
        #       when there are multiple inputs. Adding the [0] for now because
        #       singleton lists mess up some of the Layers.
        last_output = layer(layer_inputs[0])
        
        if isinstance(last_output, (list, tuple)):
            tf_data.update(
                {k: v for k, v in zip(layer_map.out, last_output)
                if k is not None}  # indicates unsaved intermediate output
                )
        elif layer_map.out[0] is not None:
            tf_data[layer_map.out[0]] = last_output

    # Don't include inputs that were never actually passed to any Layers.
    tf_inputs = [v for k, v in tf_input_dict.items() if k not in unused_inputs]
    # For outputs, get all data entries that aren't inputs
    tf_outputs = [v for k, v in tf_data.items() if k not in tf_input_dict]

    model = tf.keras.Model(inputs=tf_inputs, outputs=tf_outputs,
                           name=name)

    return model
