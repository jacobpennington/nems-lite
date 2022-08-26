import copy

import numpy as np
import scipy.signal

from .base import Layer, Phi, Parameter, require_shape
from nems.registry import layer
from nems.distributions import Normal, HalfNormal

class FiniteImpulseResponse(Layer):

    def __init__(self, **kwargs):
        """Convolve linear filter(s) with input.

        Parameters
        ----------
        shape : N-tuple
            Determines the shape of `FIR.coefficients`. Axes should be:
            (T time bins, C input channels (rank),  ..., N output channels)
            where only the first two dimensions are required. Aside from the
            time and filter axes (index 0 and -1, respectively), the size of
            each dimension must match the size of the input's dimensions.

            If only two dimensions are present, a singleton dimension will be
            appended to represent a single output. For higher-dimensional data,
            users are responsible for adding this singleton dimension if needed.

        See also
        --------
        nems.layers.base.Layer

        Examples
        --------
        >>> fir = FiniteImpulseResponse(shape=(15,4))   # (time, input channels)
        >>> weighted_input = np.random.rand(10000, 4)   # (time, channels)
        >>> out = fir.evaluate(weighted_input)
        >>> out.shape
        (10000, 1)

        # strf alias
        >>> fir = STRF(shape=(25, 18))                   # full-rank STRF
        >>> spectrogram = np.random.rand(10000,18)
        >>> out = fir.evaluate(spectrogram)
        >>> out.shape
        (10000, 1)

        # FIR alias                                     
        >>> fir = FIR(shape=(25, 4, 100))               # rank 4, 100 filters
        >>> spectrogram = np.random.rand(10000,4)
        >>> out = fir.evaluate(spectrogram)
        >>> out.shape
        (10000, 1, 100)

        """
        require_shape(self, kwargs, minimum_ndim=2)
        super().__init__(**kwargs)


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
        mean = np.full(shape=self.shape, fill_value=0.0)
        sd = np.full(shape=self.shape, fill_value=1/np.prod(self.shape))
        mean[1, :] = 2/np.prod(self.shape)
        mean[2, :] = -1/np.prod(self.shape)
        prior = Normal(mean, sd)

        coefficients = Parameter(name='coefficients', shape=self.shape,
                                 prior=prior)
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
        return self.parameters['coefficients'].values

    def evaluate(self, input):
        """Convolve `FIR.coefficients` with input."""
        # Add axis for n output channels to input if one doesn't exist.
        # NOTE: This will only catch a missing output dimension for 2D data.
        #       For higher-dimensional data, the output dimension needs to be
        #       specified by users.
        if input.ndim < 3:
            input = input[..., np.newaxis]

        # Flip rank, any other dimensions except time & number of outputs.
        coefficients = self._reshape_coefficients()
        n_filters = coefficients.shape[-1]
        padding = self._get_filter_padding(coefficients, input)
        # Prepend zeros.
        #input_with_padding = np.concatenate([padding, input])
        input_with_padding = np.pad(input, padding)

        # Convolve each filter with the corresponding input channel.
        outputs = []
        for i in range(n_filters):
            y = scipy.signal.convolve(
                input_with_padding[...,i], coefficients[...,i], mode='valid'
                )
            outputs.append(y[..., np.newaxis])
        # Concatenate on n_outputs axis
        output = np.concatenate(outputs, axis=-1)
        # Squeeze out rank dimension
        output = np.squeeze(output, axis=1)
        
        return output

    def _reshape_coefficients(self):
        """Get `coefficients` in the format needed for `evaluate`."""
        coefficients = self.coefficients
        if coefficients.ndim == 2:
            # Add a dummy filter/output axis
            coefficients = coefficients[..., np.newaxis]

        # Coefficients are applied "backwards" (convolution) relative to how
        # they are specified (filter), so have to flip all dimensions except
        # time and number of filters/outputs.
        flipped_axes = [1]  # Always flip rank
        other_dims = coefficients.shape[2:-1]
        for i, d in enumerate(other_dims):
            # Also flip any additional dimensions
            flipped_axes.append(i+2)
        coefficients = np.flip(coefficients, axis=flipped_axes)

        return coefficients

    @staticmethod
    def _unshape_coefficients(coefficients, shape):
        """Revert re-formatted `coefficients` to their original shape."""
        # Reverse the axis flips
        flipped_axes = [1]
        other_dims = coefficients.shape[2:-1]
        for i, d in enumerate(other_dims):
            # Also flip any additional dimensions
            flipped_axes.append(i+2)
        coefficients = np.flip(coefficients, axis=flipped_axes)

        # Remove dummy filter axis if not originally present
        coefficients = np.reshape(coefficients, shape)

        return coefficients

    def _get_filter_padding(self, coefficients, input):
        """Get zeros of correct shape to prepend to input on time axis."""
        # filter_length, n_channels = coefficients.shape[:2]
        # other_dims = coefficients.shape[2:-1]
        # n_outputs = coefficients.shape[-1]
        # Pad zeros to handle boundary effects.
        # TODO: Is there a better way to handle this?
        # padding = np.zeros(
        #     shape=((filter_length-1, n_channels) + other_dims + (n_outputs,))
        #     )
        filter_length = coefficients.shape[0]
        # Prepend 0s on time axis, no padding on other axes
        padding = [[filter_length-1, 0]] + [[0, 0]]*(input.ndim-1)

        return padding

    @layer('fir')
    def from_keyword(keyword):
        """Construct FIR (or subclass) from keyword.

        Keyword options
        ---------------
        {digit}x{digit}x ... x{digit} : N-dimensional shape.
            (time, input channels a.k.a. rank, ..., output channels) 

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
    
    def as_tensorflow_layer(self, **kwargs):
        """TODO: docs"""
        import tensorflow as tf
        from nems.backends.tf import Bounds, NemsKerasLayer

        old_c = self.parameters['coefficients']
        # TODO: Not clear why flipping time axis is needed, something funky with
        #       tensorflow's conv1d implementation.
        new_c = np.flip(self._reshape_coefficients(), axis=0)
        filter_width, rank, n_outputs = new_c.shape
        if new_c.ndim > 3:
            raise NotImplementedError(
                "FIR TF implementation currently only works for 2D data."
                )
        fir = self  # so that this can be referenced inside class definition
        new_values = {'coefficients': new_c}  # override Parameter.values

        num_gpus = len(tf.config.list_physical_devices('GPU'))
        if num_gpus == 0:
            # Use CPU-compatible (but slower) version.
            @tf.function
            def call_fn(self, inputs):
                # Reorder coefficients to shape (n outputs, time, rank, 1)
                new_coefs = tf.expand_dims(
                    tf.transpose(self.coefficients, [2, 0, 1]), -1
                    )  
                # This will add an extra dim if there is no output dimension.
                input_width = tf.shape(inputs)[1]
                rank_4 = tf.reshape(inputs, [-1, input_width, rank, n_outputs])
                padded_input = tf.pad(
                    rank_4, [[0,0], [filter_width-1,0], [0,0], [0,0]]
                    )
                # Reorder input to shape (n outputs, batch, time, rank)
                x = tf.transpose(padded_input, [3, 0, 1, 2])
                fn = lambda t: tf.cast(tf.nn.conv1d(  # TODO: don't like forcing dtype here
                    t[0], t[1], stride=1, padding='VALID'
                    ), tf.float64)
                # Apply convolution for each output
                y = tf.map_fn(
                    fn=fn,
                    elems=(x, new_coefs),
                    fn_output_signature=tf.float64
                    )
                # Reorder output back to (batch, time, n outputs)
                z = tf.transpose(tf.squeeze(y, axis=3), [1, 2, 0])
                return z
        else:
            # Use GPU-only version (grouped convolutions), much faster.
            @tf.function
            def call_fn(self, inputs):
                # filter_width, rank, n_outputs are constants defined above.
                input_width = tf.shape(inputs)[1]
                # This will add an extra dim if there is no output dimension.
                rank_4 = tf.reshape(inputs, [-1, input_width, rank, n_outputs])
                # Reshape will group by output before rank w/o transpose.
                transposed = tf.transpose(rank_4, [0, 1, 3, 2])
                # Collapse rank and n_outputs to one dimension.
                # -1 for batch size b/c it can be None.
                reshaped = tf.reshape(
                    transposed, [-1, input_width, rank*n_outputs]
                    )
                # Prepend 0's on time axis as initial conditions for filter.
                padded_input = tf.pad(
                    reshaped, [[0, 0], [filter_width-1, 0], [0, 0]]
                    )
                # Convolve filters with input slices in groups of size `rank`.
                y = tf.nn.conv1d(
                    padded_input, self.coefficients, stride=1, padding='VALID'
                    )
                return y  


        class FiniteImpulseResponseTF(NemsKerasLayer):
            def weights_to_values(self):
                c = self.parameter_values['coefficients']
                unflipped = np.flip(c, axis=0)  # Undo flip time
                unshaped = fir._unshape_coefficients(unflipped, old_c.shape)
                return {'coefficients': unshaped}

            def call(self, inputs):
                return call_fn(self, inputs)

        return FiniteImpulseResponseTF(self, new_values=new_values, **kwargs)


# Aliases, STRF specifically for full-rank (but not enforced)
class FIR(FiniteImpulseResponse):
    pass
class STRF(FiniteImpulseResponse):
    pass
