import numpy as np

from nems.registry import layer
from nems.distributions import Normal
from .base import Layer, Phi, Parameter


class StateGain(Layer):
    """Docs TODO.
    
    Parameters
    ----------
    shape : N-tuple of int

    Examples
    --------
    TODO
    
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state_name = 'state'  # see Layer.__init__

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
        """Multiply and shift input(s) by weighted sums of state channels.
        
        Parameters
        ----------
        inputs : N-tuple of ndarray
            Data to be modulated by state, typically the output(s) of a previous
            Layer. At least one input is expected.
        state : ndarray or list of ndarray
            State data to modulate input(s) with. If list or dict, entries will
            be concatenated along axis 1 in the order they are provided.

        """
        if isinstance(state, list):
            # Concatenate along channel axis
            state = np.concatenate(state, axis=1)

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
