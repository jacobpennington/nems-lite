import copy

import numpy as np
import scipy.signal

from .base import Layer, Phi, Parameter
from nems.registry import layer

class FiniteImpulseResponse(Layer):
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

    def evaluate(self, input):
        """Convolve `FIR.coefficients` with input.
        
        TODO: need to do more testing for dim > 2 case, but this at least
              doesn't break dim = 2 (i.e. still matches old code output) and
              works with multiple filters.
        
        """

        coefficients = self._reshape_coefficients()
        n_filters = coefficients.shape[-1]
        padding = self._get_filter_padding()

        filter_outputs = []
        for j in range(n_filters):
            c = coefficients[..., j]
            input_with_padding = np.concatenate([padding, input])
            z = scipy.signal.convolve(input_with_padding, c, mode='valid')
            filter_outputs.append(z)
        # Output of each convolution should be (T,1), concatenate to (T,N).
        output = np.concatenate(filter_outputs, axis=1)

        other_dims = self.shape[2:-1]
        if len(other_dims) > 0:
            # There will be extra singleton axes, so squeeze those out.
            # TODO: better way to do this? I think this might cause issues for
            #       high-D data with one bank.
            output = output.squeeze()

        return output

    # TODO: Similar to note about possibly just swapping time and channels
    #       permanently, I'm wondering if this entire process should just happen
    #       at initialization so that we don't need to worry about swapping
    #       back and forth. That would probably be more intuitive for anyone not
    #       from our lab, but would require a mindset-switch for people that are
    #       used to specifying coefficients in a particular order (e.g. filter
    #       vs convolution).
    def _reshape_coefficients(self):
        """Get `coefficients` in the format needed for `evaluate`."""
        other_dims = self.shape[2:-1]
        coefficients = self.coefficients
        if len(self.shape) == 2:
            # Add a dummy filter axis
            coefficients = coefficients[..., np.newaxis]

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

        return coefficients

    def _unshape_coefficients(self, coefficients):
        """Revert re-formatted `coefficients` to their original shape."""
        
        # Reverse the axis flips
        flipped_axes = [1]
        other_dims = coefficients.shape[2:-1]
        for i, d in enumerate(other_dims):
            # Also flip any additional dimensions
            flipped_axes.append(i+2)
        coefficients = np.flip(coefficients, axis=flipped_axes)

        # Unswap channels and time.
        coefficients = coefficients.swapaxes(0, 1)
        # Remove dummy filter axis if not originally present
        coefficients = np.reshape(coefficients, self.coefficients.shape)

        return coefficients

    def _get_filter_padding(self):
        n_channels, filter_length = self.shape[:2]
        other_dims = self.shape[2:-1]

        # Pad zeros to handle boundary effects.
        # TODO: this only really works if the data itself has surrounding
        #       silence (or other null stimulus). Is there a better way to
        #       handle this?
        padding = np.zeros(shape=((filter_length-1, n_channels) + other_dims))

        return padding

    def as_tensorflow_layer(self):
        """TODO: docs"""
        import tensorflow as tf
        from tensorflow.keras.layers import Layer
        from nems.tf import Bounds, NemsKerasLayer

        old_c = self.parameters['coefficients']
        new_c = self._reshape_coefficients()
        if len(new_c.shape) > 3:
            raise NotImplementedError(
                "FIR TF implementation currently only works for 2D data."
                )
        pad_t, _ = self._get_filter_padding().shape
        fir = self  # so that this can be referenced inside class

        class FiniteImpulseResponseTF(NemsKerasLayer):
            def __init__(self, regularizer=None, *args, **kwargs):
                super().__init__(name=fir.name, *args, **kwargs)
                init = tf.constant_initializer(new_c)
                constraint = Bounds(old_c.bounds[0], old_c.bounds[1])
                self.coefficients = self.add_weight(
                    shape=new_c.shape, initializer=init, trainable=True,
                    regularizer=regularizer, name='coefficients'
                    )

            def weights_to_values(self):
                c = self.parameter_values['coefficients']
                return fir._unshape_coefficients(c)

            def call(self, inputs):
                """Normal call."""

                # TODO: Why does this pad work for old NEMS? I'm getting an
                #       error b/c input is rank-4 (there's an extra singleton
                #       axis added by WeightChannels). Adding this kludgy
                #       solution for now, but I don't think those extra dims
                #       are supposed to be showing up in the first place.
                if isinstance(inputs, list):
                    inputs = inputs[0]
                padded_input = tf.pad(inputs, [[0, 0], [pad_t, 0], [0, 0]])
                # TODO: old NEMS notes that this doesn't run on CPU under certain
                #       circumstances. Check if that's still the case, or if that
                #       was for one of the special versions that will be implemented
                #       as a subclass instead.
                #transposed = tf.transpose(tf.reverse(self.coefficients, axis=[-1]))
                Y = tf.nn.conv1d(padded_input, self.coefficients, stride=1, padding='VALID')
                return Y

        return FiniteImpulseResponseTF
    
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


# Aliases, STRF specifically for full-rank (but not enforced)
class FIR(FiniteImpulseResponse):
    pass
class STRF(FiniteImpulseResponse):
    pass
