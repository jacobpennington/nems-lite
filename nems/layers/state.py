import numpy as np

from nems.registry import layer
from nems.distributions import Normal
from .base import Layer, Phi, Parameter


class StateGain(Layer):
    def __init__(self, shape, **kwargs):
        """Docs TODO"""
        self.shape = shape
        super().__init__(**kwargs)
        self.state_name = 'state'  # See Layer.__init__

    def initial_parameters(self):
        """Docs TODO
        
        Layer parameters
        ----------------
        gain : TODO
            prior:
            bounds:
        offset : TODO
            prior:
            bounds:
        
        """
        zero = np.zeros(shape=self.shape)
        one = np.ones(shape=self.shape)

        gain_mean = zero.copy()
        gain_mean[:,0] = 1  # Purpose of this?
        gain_sd = one/20
        gain_prior = Normal(gain_mean, gain_sd)
        gain = Parameter('gain', shape=self.shape, prior=gain_prior)

        offset_mean = zero
        offset_sd = one
        offset_prior = Normal(offset_mean, offset_sd)
        offset = Parameter('offset', shape=self.shape, prior=offset_prior)
        
        return Phi(gain, offset)

    def evaluate(self, *inputs, state):
        # TODO: probably need to swap order of data somewhere since we're
        #       defaulting to time_axis=0 here. May also need to transpose
        #       shape of parameters, not sure.
        gain, offset = self.get_parameter_values()
        output = [
            np.matmul(gain, state) * x + np.matmul(offset, state)
            for x in inputs
        ]

        return output

    @layer('stategain')
    def from_keyword(keyword):
        # TODO: other options from old NEMS
        # TODO: document expectation for shape (see comment in .evaluate)
        options = keyword.split('.')
        for op in options[1:]:
            if op[0].isdigit():
                dims = op.split('x')
                shape = tuple([int(d) for d in dims])

        return StateGain(shape=shape)
