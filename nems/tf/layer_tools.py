import tensorflow as tf
from tensorflow.keras.layers import Layer


# TODO: handle frozen parameters
def get_tf_class(nems_layer, **methods):
    class Layer(tf.keras.layers.Layer):
        def __init__(self, regularizer=None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            add_existing_weights(nems_layer, self, regularizer)

        @property
        def parameter_values(self):
            """Returns key value pairs of the weight names and their values."""
            # TF appendss :<int> to the weight names.
            values = {weight.name.split(':')[0]: weight.numpy()
                      for weight in self.weights} 
            return values

        def weights_to_values(self):
            return self.parameter_values

    Layer.__name__ = f'{type(nems_layer).__name__}TF'
    for method_name, method_def in methods.items():
        setattr(Layer, method_name, method_def)

    return Layer


def add_existing_weights(nems_layer, tf_layer, regularizer=None):
    for p in nems_layer.parameters:
        init = tf.constant_initializer(p.values)
        setattr(
            tf_layer, p.name, tf_layer.add_weight(
                shape=p.shape, initializer=init, trainable=True,
                regularizer=regularizer, name=p.name
                )
            )
