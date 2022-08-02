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
        gain_mean[0,:] = 1  # TODO: Explain purpose of this?
        gain_sd = one/20
        gain_prior = Normal(gain_mean, gain_sd)
        gain = Parameter('gain', shape=self.shape, prior=gain_prior)

        offset_mean = zero
        offset_sd = one
        offset_prior = Normal(offset_mean, offset_sd)
        offset = Parameter('offset', shape=self.shape, prior=offset_prior)
        
        return Phi(gain, offset)

    def evaluate(self, *inputs, state):
        # TODO: what about multiple state inputs? E.g. if I want to use
        #       pupil and task-type, this requires merging those into a single
        #       array beforehand. But would be more intuitive to be able to
        #       say StateGain(inputs=['pred', 'pupil', 'task']).
        gain, offset = self.get_parameter_values()
        output = [
            # Output should be same shape as x, * is element-wise mult.
            np.matmul(state, gain) * x + np.matmul(state, offset)
            for x in inputs
        ]

        return output

    @layer('stategain')
    def from_keyword(keyword):
        """Construct StateGain from keyword.
        
        Keyword options
        ---------------
        {digit}x{digit} : specifies shape, (n state channels, n stim channels)
            n stim channels can also be 1, in which case the same weighted
            channel will be broadcast to all stim channels (if there is more
            than 1).
        
        See also
        --------
        Layer.from_keyword

        """
        # TODO: other options from old NEMS
        options = keyword.split('.')
        for op in options[1:]:
            if op[0].isdigit():
                dims = op.split('x')
                shape = tuple([int(d) for d in dims])

        return StateGain(shape=shape)
