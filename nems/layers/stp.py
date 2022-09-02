
from nems.registry import layer
from nems.layers.base import Layer, Phi, Parameter, require_shape, parse_shape


class ShortTermPlasticity(Layer):

    def __init__(self, **kwargs):
        require_shape(self, kwargs, minimum_ndim=1)
        super().__init__()

    def initial_parameters(self):
        u = Parameter('u', shape=self.shape)
        tau = Parameter('tau', shape=self.shape)
        return Phi(u, tau)

    def evaluate(self, inputs):
        u, tau = self.get_parameter_values()
        raise NotImplementedError  # TODO

    @layer('stp')
    def from_keyword(keyword):
        """TODO: docs"""
        shape = None

        options = keyword.split('.')
        for op in options:
            if ('x' in op) and (op[0].isdigit()):
                dims = op.split('x')
                shape = tuple([int(d) for d in dims])

        return ShortTermPlasticity(shape=shape)


class STP(ShortTermPlasticity):
    pass
