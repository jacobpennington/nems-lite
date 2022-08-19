import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Input

from ..base import Backend


class TensorFlowBackend(Backend):
    def __init__(self, nems_model, input, eval_kwargs=None, **backend_options):
        super().__init__()

    def build(self, input, eval_kwargs=None):
        # TODO: what backend options to accept?

        batch_size = eval_kwargs.get('batch_size', None)
        data_maps = self.nems_model.get_data_maps()
        tf_kwargs = {}  # TODO
        tf_layers = [layer.as_tensorflow_layer(**tf_kwargs)
                        for layer in self.layers]

        # Convert inputs to TensorFlow format
        tf_input_dict = {}
        for k, v in input.items():
            # Skip trial/sample dimension when determining shape.
            tf_in = Input(shape=v.shape[1:], name=k, batch_size=batch_size,
                        dtype='float32')  # TODO: why hard-code float32?
            tf_input_dict[k] = tf_in
        unused_inputs = list(tf_input_dict.keys())

        # Pass through Keras functional API to map inputs & outputs.
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
                               name=self.nems_model.name)

        print('TF model built...')
        print(model.summary())

        return model

    def fit(self, input, target, eval_kwargs=None, learning_rate=0.001,
            epochs=1):

        # TODO: support more keys in `fitter_options`.

        # NOTE: expects arrays in data to already be formatted as shape
        #       (S,T,C) instead of (T,C), same for target as well.

        # TODO: This assumes a single output (like our usual models).
        #       Need to tweak this to be able to fit outputs from multiple
        #       layers.
        final_layer = self.layers[-1].name
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss={final_layer: keras.losses.MeanSquaredError()}
        )

        self.model.fit(
            input, {final_layer: target}, epochs=epochs
        )

        # Save weights back to NEMS model
        # (first two TF layers are input layers, skip it).
        # TODO: This assumes there aren't any other extra layers added
        #       by TF. That might not be the case for some model types.
        skip = len(self.model.inputs)
        layer_iter = zip(self.nems_model.layers, self.model.layers[skip:])
        for nems_layer, tf_layer in layer_iter:
            nems_layer.set_parameter_values(tf_layer.weights_to_values())

        # TODO: more information here.
        print('TF model fit finished, parameters have been updated.')