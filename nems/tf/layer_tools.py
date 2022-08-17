import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.constraints import Constraint


# TODO: handle frozen parameters
class NemsKerasLayer(Layer):
    
    def __init__(self, nems_layer, regularizer=None, *args, **kwargs):
        """TODO: docs."""
        super().__init__(name=nems_layer.name, *args, **kwargs)
        for p in nems_layer.parameters:
            init = tf.constant_initializer(p.values)
            constraint = Bounds(p.bounds[0], p.bounds[1])
            setattr(
                self, p.name, self.add_weight(
                    shape=p.shape, initializer=init, trainable=True,
                    regularizer=regularizer, name=p.name, constraint=constraint
                    )
                )

    @property
    def parameter_values(self):
        """Returns key value pairs of the weight names and their values."""
        # TF appendss :<int> to the weight names.
        values = {weight.name.split(':')[0]: weight.numpy()
                    for weight in self.weights} 
        return values

    def weights_to_values(self):
        return self.parameter_values


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
