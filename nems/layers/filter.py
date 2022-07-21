import numpy as np
import scipy.signal

from .base import Layer, Phi, Parameter
from nems.registry import layer

class FIR(Layer):
    def __init__(self, shape, **kwargs):
        """Convolve linear filter(s) with input.

        Parameters
        ----------
        shape : N-tuple
            Determines the shape of `FIR.coefficients`.
            TODO: N-dim might not actually be supported, depends on how
                  evaluate ends up being implemented.
                  Old implementation was (n_inputs, time bins, n_filters)

        parameters : Phi or None, optional
            Specifies the value of each variable used to compute
            `FIR.evaluate()`. If `None`, values will be determined by
            `FIR.initial_parameters`.
        
        Returns
        -------
        FIR

        """
        self.shape = shape
        super().__init__(**kwargs)

    def initial_parameters(self):
        """Get initial values for `FIR.parameters`.
        
        Layer parameters
        ----------------
        coefficients : ndarray
            Shape matches `FIR.shape`.
            Prior:  TODO, currently using defaults
            Bounds: TODO

        """
        coefficients = Parameter(name='coefficients', shape=self.shape)
        return Phi(coefficients)

    @property
    def coefficients(self):
        """Filter that will be convolved with input.
        
        Re-parameterized subclasses should overwrite this so that `evaluate`
        doesn't need to change.

        Returns
        -------
        coefficients : ndarray
            coefficients.shape = WeightChannels.shape
        
        """
        return self.parameters['coefficients']

    def evaluate(self, *inputs):
        """Convolve `FIR.coefficients` with input(s).
        
        TODO: doesn't actually work yet, just passes data through
        TODO: need to pick a convolution implementation (see notes in function)
        
        """
        return inputs
        # TODO: Talk to Stephen about this again?. The FIR code in old NEMS
        #       seems way more complicated than it should be, but it's probably
        #       checking for a lot of special cases/boundary effects and it's
        #       not easy to pick through.
        #output = scipy.signal.lfilter(self.coefficients, [1], x_, zi=zi)

        # numpy option
        single_channel_input = None  # select from input somehow
        single_channel_filter = None # select from coefficients somehow
        output = np.convolve(single_channel_input, single_channel_filter)

        # alternate scipy option
        # would still need to deal with boundaries somehow, otherwise just
        # using mode='same' is going to do... "something" to make the shapes
        # match and I'm not clear yet on what that is exactly.
        scipy.signal.convolve(input, self.coefficients)
    
    @layer('fir')
    def from_keyword(keyword):
        """Construct FIR (or subclass) from keyword.

        Keyword options
        ---------------
        {digit}x{digit}x ... x{digit} : N-dimensional shape.
            TODO: N-dim might not actually be supported, depends on how
                  evaluate ends up being implemented.

        See also
        --------
        Layer.from_keyword
        
        """
        kwargs = {}

        options = keyword.split('.')
        for op in options:
            if ('x' in op) and (op[0].isdigit()):
                dims = op.split('x')
                kwargs['shape'] = tuple([int(d) for d in dims])

        if 'shape' not in kwargs:
            raise ValueError("FIR requires a shape, ex: `fir.4x25`.")

        return FIR(**kwargs)

    def to_json(self):
        """Encode FIR as a dictionary. See Layer.to_json."""
        data = Layer.to_json(self)
        data['kwargs'].update(shape=self.shape)
        return data


class STRF(FIR):
    # TODO: I guess this is really the same as FIR, just with an expectation
    #       that the shape matches the full spectrogram.
    #       (and none of the re-parameterized subclasses would be supported)
    pass