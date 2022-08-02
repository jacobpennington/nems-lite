from itertools import chain

import numpy as np
import scipy.signal

from .base import Layer, Phi, Parameter
from nems.registry import layer

class FIR(Layer):
    """Convolve linear filter(s) with input.

    Parameters
    ----------
    shape : N-tuple
        Determines the shape of `FIR.coefficients`. Axes should be:
        (C input channels, T time bins, ..., N filters)
        where only the first two dimensions are required. Aside from the
        time and filter axes (index 1 and -1, respectively), the size of
        each dimension must match the size of the input's dimensions.

        NOTE: The C and T dimensions are actually transposed (i.e. input
        data should still have shape (T,C,...)). This format was chosen to
        make shape specification intuitive and consistent across layers,
        e.g. `WeightChannels(shape=(18,4))` and `FIR(shape=(4,25))` will
        line up correctly because the inner 4's match.

        If only two dimensions are present, a singleton dimension will be
        appended to represent a single filter. For higher-dimensional data
        with single filter, users are responsible for adding this singleton
        dimension.

        TODO: N-dim might not actually be supported, depends on how
                evaluate ends up being implemented.

    See also
    --------
    nems.layers.base.Layer

    Examples
    --------
    >>> fir = FIR(shape=(4,15))
    >>> weighted_input = np.random.rand(10000, 4) # (time, channels)
    >>> out = fir.evaluate(weighted_input)
    >>> out.shape
    (10000, 1)

    >>> fir = FIR(shape=(18,25,4))                # full-rank STRF
    >>> spectrogram = np.random.rand(10000, 18)
    >>> out = fir.evaluate(spectrogram)
    >>> out.shape
    (10000, 4)

    """

    def initial_parameters(self):
        """Get initial values for `FIR.parameters`.
        
        Layer parameters
        ----------------
        coefficients : ndarray
            Shape matches `FIR.shape`.
            Prior:  TODO, currently using defaults
            Bounds: TODO

        Returns
        -------
        nems.layers.base.Phi

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
        # .values is needed in this case since we have to use `np.swapaxes`,
        # which is not supported through ufunc.
        return self.parameters['coefficients'].values

    def evaluate(self, *inputs):
        """Convolve `FIR.coefficients` with input(s).
        
        TODO: need to do more testing for dim > 2 case, but this at least
              doesn't break dim = 2 (i.e. still matches old code output) and
              works with multiple filters.
        
        """
        n_channels, filter_length = self.shape[:2]
        other_dims = self.shape[2:-1]
        coefficients = self.coefficients
        if len(self.shape) == 2:
            # Add a dummy filter axis
            n_filters = 1
            coefficients = coefficients[..., np.newaxis]
        else:
            n_filters = self.shape[-1]

        # Swap channels and time to  match input.
        coefficients = coefficients.swapaxes(0, 1)
        # Coefficients are applied "backwards" (convolution) relative to how
        # they are specified (filter), so have to flip all dimensions except
        # time and number of filters.
        flipped_axes = [1]  # Always flip channels
        for i, d in enumerate(other_dims):
            # Also flip any additional dimensions
            flipped_axes.append(i+2)
        coefficients = np.flip(coefficients, axis=flipped_axes)

        # Pad zeros to handle boundary effects.
        # TODO: this only really works if the data itself has surrounding
        #       silence (or other null stimulus). Is there a better way to
        #       handle this?
        padding = np.zeros(shape=((filter_length-1, n_channels) + other_dims))

        output = []
        for x in inputs:
            filter_outputs = []
            for j in range(n_filters):
                c = coefficients[..., j]
                input_with_padding = np.concatenate([padding, x])
                z = scipy.signal.convolve(input_with_padding, c, mode='valid')
                filter_outputs.append(z)
            # Output of each convolution should be (T,1), concatenate to (T,N).
            out = np.concatenate(filter_outputs, axis=1)
            # For data with dimension > 2, there will also be extra singleton
            # axes, so squeeze those out.
            if len(other_dims) > 0:
                out = out.squeeze()
            output.append(out)

        return output

    def old_evaluate(self, *inputs):
        """Temporary copy of old nems implementation, for testing.
        
        TODO: DELETE ME when no longer needed for testing (along with the
        copied code toward the bottom of the file).
        
        """
        c = self.coefficients
        if len(c.shape) == 3:
            # kludge to make multiple filterbanks work with new coefficient
            # structure, just for testing. Split n filters dimension (2) and
            # concatenate along n channels dimension (0).
            # E.g. shape (d1, d2, d3) -> (d1*d3, d2)
            d1, d2, d3 = c.shape
            c = np.concatenate(np.split(c, d3, axis=2), axis=0).squeeze()
            banks = d3
        else:
            banks = 1

        output = [
            per_channel(x.T, c, non_causal=False, rate=1, bank_count=banks).T
            for x in inputs
        ]

        return output
    
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

        return fir


# Optional alias to use when FIR is applied to the full spectrogram.
STRF = FIR


###############################################################################
###########################      OLD FIR CODE     #############################
###############################################################################

# TODO: DELETE ME when no longer needed for testing




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


##### COPY WITH MY NOTES
def evaluate(self, *inputs):
    """Convolve `FIR.coefficients` with input(s).

    TODO: need to pick a convolution implementation (see notes in function)
    
    """
    # TODO: this just points to the old code for now, but the subroutines
    #       work on array data so should work without much modification
    #       in the meantime.


    # TODO: ask Stephen about this. Still wasn't clear on the purpose so I
    #       tried it out with some random data, and for a wide range of
    #       input lengths and filter sizes it just ends up creating a vector
    #       of zeros with length one less than the filter channel. So why not
    #       just do that directly? Are there special edge cases where something
    #       different happens?
    def _get_zi(b, x):
        # This is the approach NARF uses. If the initial value of x[0] is 1,
        # this is identical to the NEMS approach. We need to provide zi to
        # lfilter to force it to return the final coefficients of the dummy
        # filter operation.

        # b is one channel of the filter, so this is number of time points
        n_taps = len(b)   
        # all zeros, twice the length of the filter
        null_data = np.zeros(shape=(n_taps*2,))
        zi = np.ones(n_taps-1)

        # TODO: I still don't understand what this accomplishes...
        #       looks like docs for `scipy.signal.lfilter_zi` might be relevant?
        return scipy.signal.lfilter(b, [1], null_data, zi=zi)[1]

    # TODO: see notes below regarding rate. When would some one want to use
    #       this? need to either cut it or be able to document.
    #       Also would need refactor since old nems used different shape
    #       for coefficients
    def __insert_zeros(coefficients, rate=1):
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


    # TODO: clarify language on "bank" vs "filter". I'd rather just get rid
    #       of the bank terminology, I don't think it's necessary at this
    #       level (e.g. what this code refers to as "one bank -> many filters"
    #       can just as easily be referred to as "one filter -> many channels",
    #       esp. if we switch from the lfilter method. Bank in that case would
    #       be used for an higher level of organiziation that doesn't exist
    #       (yet) here.
    def _per_channel(input, coefficients, bank_count=1, non_causal=0, rate=1,
                    cross_channels=False):
        """Private function used by fir_filter().

        TODO: this assumes a specific ordering of axes of input, add option
                to make this more general (e.g. input_axis=i).
                Similar issue for WeightChannels.


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
        rate : int
            TODO: What does this do?
            Looks like it puts `rate-1` zeros between coefficients
            (e.g. rate=3 with a shape=(N,15) filter ends up as
                `[0, x1, 0, 0, x2, 0, 0, x3, ... x14, 0, 0, x15, 0])
            But why? What is this for?
        non_causal : int
            TODO: what does this do?
        cross_channels : bool
            TODO: what does this do?

        Returns
        -------
        ndarray (bank_count, n_times)

        """
        n_channels, n_times, n_filters = self.coefficients.shape

        # TODO: part 2. Ah... old nems is actually multiplying the number
        #       of filters by the number of channels to form fir.coefficients,
        #       hence my confusion. E.g. for fir.4x15x70, coefficients.shape[0]
        #       in old nems would be (280,15) instead of (4,15,70).
        n_banks = int(n_channels / bank_count)

        # TODO: this should probably just be its own subclass with a
        #       corresponding option in the keyword. Looks like it's doing
        #       the full operation but in a different way.
        if cross_channels:
            # option 0: user has specified that each filter should be applied to
            # each input channel (requires bank_count==1)
            # TODO : integrate with core loop below instead of pasted hack
            out = np.zeros((n_channels*n_channels, input.shape[1]))
            i_out=0
            for i_in in range(n_channels):
                input_ = input[i_in]
                for i_bank in range(n_channels):
                    c = coefficients[i_bank]
                    zi = _get_zi(c, input_)
                    r, zf = scipy.signal.lfilter(c, [1], input_, zi=zi)
                    out[i_out] = r
                    i_out+=1
            return out


        # NOTE: okay... so I guess n_filters is referring to the number of
        #       channels in what I would refer to as a single filter. E.g.
        #       I think of (4x15) as a single filter, but this is referring
        #       to it as four (1x15) filters. Is there an important reason
        #       this terminology is used or does it just happen to be the
        #       norm for a certain context?
        elif n_channels == n_channels:
            # option 1: number of input channels is same as total channels in the
            # filterbank, allowing a different stimulus (channel?) into each filter
            all_inputs = iter(input)


        # TODO: Might be worth making this a separate subclass similar to
        #       option 0 above, to organize the code a little better. The
        #       The filtering operation is the same, but this is still a
        #       fundamenally different use of the Layer if i'm understanding
        #       correctly.
        elif n_channels == n_channels * bank_count:
            # option 2: number of input channels is same as number of coefficients
            # in each fir filter, so that the same stimulus (channel?) goes into each
            # filter
            one_input = tuple(input)

            # TODO: flattens channels and repeats bank_count times. E.g.
            #       if input has shape (10000,18) and bank_count=2,
            #       then all_inputs would have 36 entries each shape (10000,)
            #       In other words, same stimulus gets used bank_count many times.

            #       Again, what's the point of this vs just iterating over
            #       indices?
            all_inputs = chain.from_iterable([one_input for _ in range(bank_count)])
        else:
            if bank_count == 1:
                desc = '%i FIR filters' % n_channels
            else:
                desc = '%i FIR filter banks' % n_banks
            raise ValueError(
                'Dimension mismatch. %s channels provided for %s.' % (n_channels, desc))


        # TODO: Using iters to reduce memory load I guess? But the coefficients
        # themselves should have minimal memory overhead, creating multiple
        # copies of the input would be the memory-intensive part. So why not
        # just... *not* generate copies of the input? The filter operation
        # shouldn't be doing anything inplace, so why would copies be needed?
        # Doesn't seem like this really has any benefit instead of just
        # indexing into the existing arrays.
        c_iter = iter(coefficients)
        out = np.zeros((bank_count, input.shape[1]))

        for i_out in range(bank_count):  # bank_count: n channels to use for each filtering operation
            
            for i_bank in range(n_banks):  # n_banks: number of filter channels / bank_count

                # TODO: I guess it's done this way b/c lfilter only works
                #       with 1D vectors? How important is it to use the
                #       "slightly more correct" method? just using convolve()
                #       would be more intuitive, and the speed increase
                #       might be bigger as the input dimension increases?
                #
                #       Other advantage would be simplifying the logic here.
                #       E.g. if using fir.4x15x70, will only work if input
                #       has shape (4,T,70)  (or something along those lines),
                #       so would still just be convolving a "single filter",
                #       that happens to have a higher dimensional shape,
                #       along the time axis of the input.
                #
                #       Actually, maybe that's not the case... reading
                #       convolve() again, I think by N-dimensional arrays
                #       they may have meant 1-D with N entries.
                input_ = next(all_inputs)  # one channel of input -- (or not? See comment next to chain.from_iterable)
                c = next(c_iter)           # one channel of filter

                # NOTE: Second argument for roll is
                #       "The number of places by which elements are shifted."
                #       So this is shifting the input "backward" in time?...
                #       Which I guess means the filter would be applied to
                #       future time points. But this would also put the first
                #       X time points at the end of the array instead. I guess
                #       that doesn't matter for our data since there's surrounding
                #       silence, but could be problematic for some data
                #       (esp. if non_causal is large)
                if non_causal > 0:
                    # reverse model (using future values of input to predict)
                    input_ = np.roll(input_, -non_causal)

                # It is slightly more "correct" to use lfilter than convolve at
                # edges, but but also about 25% slower (Measured on Intel Python
                # Dist, using i5-4300M)
                
                # TODO: What's more "correct" about it?
                # "initial conditions for filter delays" -- eh?
                zi = _get_zi(c, input_)
                # maybe should be using this instead? scipy.signal.lfilter_zi
                r, zf = scipy.signal.lfilter(c, [1], input_, zi=zi)

                # sum all channels from one filter
                # (this step not necessary if switch to convolve())
                out[i_out] += r

        return out

    output = [
        _per_channel(x.T, self.coefficients, non_causal=False, rate=1)
        for x in inputs
        ]
    return output