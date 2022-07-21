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
    
        TODO: need to pick a convolution implementation (see notes in function)
        
        """

        # TODO: this just points to the old code for now, but the subroutines
        #       work on array data so should work without further modification
        #       in the meantime.
        output = [
            per_channel(x, self.coefficients, non_causal=False, rate=1)
            for x in inputs
            ]
        return output

        
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

        fir = FIR(**kwargs)
        fir.name = options[0]  # keyword head, 'fir'

        return fir

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



###############################################################################
###########################      OLD FIR CODE     #############################
###############################################################################

# DELETE ME when no longer needed for testing


from itertools import chain

def get_zi(b, x):
    # This is the approach NARF uses. If the initial value of x[0] is 1,
    # this is identical to the NEMS approach. We need to provide zi to
    # lfilter to force it to return the final coefficients of the dummy
    # filter operation.
    n_taps = len(b)
    #null_data = np.full(n_taps*2, x[0])
    null_data = np.full(n_taps*2, 0)
    zi = np.ones(n_taps-1)
    return scipy.signal.lfilter(b, [1], null_data, zi=zi)[1]


def _insert_zeros(coefficients, rate=1):
    if rate<=1:
        return coefficients

    d1 = int(np.ceil((rate-1)/2))
    d0 = int(rate-1-d1)
    s = coefficients.shape
    new_c = np.concatenate((np.zeros((s[0],s[1],d0)),
                            np.expand_dims(coefficients, axis=2),
                            np.zeros((s[0],s[1],d1))), axis=2)
    new_c = np.reshape(new_c, (s[0],s[1]*rate))
    return new_c


def per_channel(x, coefficients, bank_count=1, non_causal=0, rate=1,
                cross_channels=False):
    '''Private function used by fir_filter().
    Parameters
    ----------
    x : array (n_channels, n_times) or (n_channels * bank_count, n_times)
        Input data. Can be sized two different ways:
        option 1: number of input channels is same as total channels in the
          filterbank, allowing a different stimulus into each filter
        option 2: number of input channels is same as number of coefficients
          in each fir filter, so that the same stimulus goes into each
          filter
    coefficients : array (n_channels * bank_count, n_taps)
        Filter coefficients. For ``x`` option 2, input channels are nested in
        output channel, i.e., filter ``filter_i`` of bank ``bank_i`` is at
        ``coefficients[filter_i * n_banks + bank_i]``.
    bank_count : int
        Number of filters in each bank.
    Returns
    -------
    signal : array (bank_count, n_times)
        Filtered signal.
    '''
    # Make sure the number of input channels (x) match the number FIR filters
    # provided (we have a separate filter for each channel). The `zip` function
    # doesn't require the iterables to be the same length.
    n_in = len(x)
    if rate > 1:
        coefficients = _insert_zeros(coefficients, rate)
        print(coefficients)
    n_filters = len(coefficients)
    if bank_count>0:
        n_banks = int(n_filters / bank_count)
    else:
        n_banks = n_filters
    if cross_channels:
        # option 0: user has specified that each filter should be applied to
        # each input channel (requires bank_count==1)
        # TODO : integrate with core loop below instead of pasted hack
        out = np.zeros((n_in*n_filters, x.shape[1]))
        i_out=0
        for i_in in range(n_in):
            x_ = x[i_in]
            for i_bank in range(n_filters):
                c = coefficients[i_bank]
                zi = get_zi(c, x_)
                r, zf = scipy.signal.lfilter(c, [1], x_, zi=zi)
                out[i_out] = r
                i_out+=1
        return out
    elif n_filters == n_in:
        # option 1: number of input channels is same as total channels in the
        # filterbank, allowing a different stimulus into each filter
        all_x = iter(x)
    elif n_filters == n_in * bank_count:
        # option 2: number of input channels is same as number of coefficients
        # in each fir filter, so that the same stimulus goes into each
        # filter
        one_x = tuple(x)
        all_x = chain.from_iterable([one_x for _ in range(bank_count)])
    else:
        if bank_count == 1:
            desc = '%i FIR filters' % n_filters
        else:
            desc = '%i FIR filter banks' % n_banks
        raise ValueError(
            'Dimension mismatch. %s channels provided for %s.' % (n_in, desc))

    c_iter = iter(coefficients)
    out = np.zeros((bank_count, x.shape[1]))
    for i_out in range(bank_count):
        for i_bank in range(n_banks):
            x_ = next(all_x)
            c = next(c_iter)
            if non_causal:
                # reverse model (using future values of input to predict)
                x_ = np.roll(x_, -non_causal)

            # It is slightly more "correct" to use lfilter than convolve at
            # edges, but but also about 25% slower (Measured on Intel Python
            # Dist, using i5-4300M)
            zi = get_zi(c, x_)
            r, zf = scipy.signal.lfilter(c, [1], x_, zi=zi)
            out[i_out] += r
    return out
