from .base import Layer, Phi, Parameter


class StaticNonlinearity(Layer):
    def __init__(self, **kwargs):
        # TODO
        self.skip_nonlinearity = False

    def evaluate(self, *inputs):
        # TODO: not really this simple
        inputs = [inputs + self.parameters['shift']]  
        if not self.skip_nonlinearity:
            output = self.nonlinearitys(inputs)
        else:
            output = inputs
        return output

    def nonlinearity(self, *inputs):
        pass


class DoubleExponential(StaticNonlinearity):
    def initial_parameters(self):
        pass
    def nonlinearity(self, *inputs):
        pass
