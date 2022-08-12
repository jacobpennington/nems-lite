import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.constraints import Constraint


class NemsKerasLayer(tf.keras.layers.Layer):

    @property
    def parameter_values(self):
        """Returns key value pairs of the weight names and their values."""
        # TF appendss :<int> to the weight names.
        values = {weight.name.split(':')[0]: weight.numpy()
                    for weight in self.weights} 
        return values

    def weights_to_values(self):
        return self.parameter_values


# TODO: handle frozen parameters
# NOTE: doing this as a factory instead of putting everything in the above
#       class so that a reference to the nems_layer isn't needed after creating
#       the class (unlike providing it to the initializer). But may switch to
#       that implementation in the future if this becomes impractical.
def get_tf_class(nems_layer, **methods):

    class Layer(NemsKerasLayer):
        def __init__(self, regularizer=None, *args, **kwargs):
            super().__init__(name=nems_layer.name, *args, **kwargs)
            add_existing_weights(nems_layer, self, regularizer)

    Layer.__name__ = f'{type(nems_layer).__name__}TF'
    for method_name, method_def in methods.items():
        setattr(Layer, method_name, method_def)

    return Layer


def add_existing_weights(nems_layer, tf_layer, regularizer=None):
    for p in nems_layer.parameters:
        init = tf.constant_initializer(p.values)
        constraint = Bounds(p.bounds[0], p.bounds[1])
        setattr(
            tf_layer, p.name, tf_layer.add_weight(
                shape=p.shape, initializer=init, trainable=True,
                regularizer=regularizer, name=p.name
                )
            )


class Bounds(Constraint):
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
    def __call__(self, w):
        # TODO: This is not a great way to do constraints, but I'm not sure
        #       where alternatives would be implemented (like modifying the
        #       cost function to grow very large when nearing the bound).
        # TODO: Maybe try this version:
        # https://www.tensorflow.org/probability/api_docs/python/tfp/math/clip_by_value_preserve_gradient
        return tf.clip_by_value(w, self.lower, self.upper)
