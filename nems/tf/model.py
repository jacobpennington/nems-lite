import numpy as np
import tensorflow as tf
from tensorflow.keras import Input


def build_model(nems_model, tf_layers, input, batch_size=None, eval_kwargs=None):
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
    tf_input_dict = {}
    for k, v in input.items():
        # Skip trial/sample dimension when determining shape.
        tf_in = Input(shape=v.shape[1:], name=k, batch_size=batch_size,
                      dtype='float32')  # TODO: why hard-code float32?
        tf_input_dict[k] = tf_in

    # Get data mapping of inputs & outputs for each Layer.
    # Remove first dimension of data before passing to `Model.evaluate`.
    # `data_map` should be the same for all batches, so we only need to evaluate
    # one batch.
    input = {k: v[0, ...] for k, v in input.items()}
    eval_kwargs = {} if eval_kwargs is None else eval_kwargs
    data_map = nems_model.evaluate(
        input, save_data_map=True, **eval_kwargs
        )['_data_map']

    last_output = None
    tf_data = tf_input_dict.copy()  # Need to keep actual Inputs separate
    for layer in tf_layers:
        layer_inputs = []
        layer_map = data_map[layer.name]
        # .call method in TF layers only accepts a single `inputs` argument,
        # can't directly specify inputs through keyword arguments and can't
        # use lists of inputs. Instead, we have to combine all inputs into a
        # single flattened list/tuple (which means Layer subclasses have to
        # account for that when defining .call).
        all_data_keys = layer_map['args'] + list(layer_map['kwargs'].values())
        all_data_keys = np.array(all_data_keys).flatten().tolist()
        for k in all_data_keys:
            if k is None:
                # Add last output
                layer_inputs.append(last_output)
            else:
                # Add Input with matching key
                layer_inputs.append(tf_data[k])
        # TODO: does this mean all call methods need to expect a list, even
        #       for just one key? (or add a hook here to unwrap singletons, if
        #       TF doesn't do that automatically).
        last_output = layer(layer_inputs)
        
        # Return of call can be Tensor or list/tuple of tensor.
        # Layers will need to account for this when defining call.
        if not isinstance(last_output, (list, tuple)):
            last_output = [last_output]
        tf_data.update(
            {k: v for k, v in zip(layer_map['out'], last_output)
                if k is not None}  # indicates unsaved intermediate output
            )

    tf_inputs = list(tf_input_dict.values())
    # For outputs, get all data entries that aren't inputs
    tf_outputs = [v for k, v in tf_data.items() if k not in tf_input_dict]

    model = tf.keras.Model(inputs=tf_inputs, outputs=tf_outputs,
                           name=nems_model.name)

    return model
