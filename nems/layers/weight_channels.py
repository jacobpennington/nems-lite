import numpy as np

from nems.registry import layer
from nems.distributions import Normal, HalfNormal
from .base import Layer, Phi, Parameter, ShapeError


# TODO: double check all shape references after dealing w/ data order etc,
#       make sure correct dims are lined up.
class WeightChannels(Layer):
    """Compute linear weighting of input channels, akin to a dense layer.

    Parameters
    ----------
    shape : N-tuple (usually N=2)
        Determines the shape of `WeightChannels.coefficients`.
        First dimension should match the spectral dimension of the input,
        second dimension should match the spectral dimension of the output.
        Note that higher-dimesional shapes are also allowed and should work
        as-is for this base class, but overall Layer design is intended for
        2-dimensional data so subclasses might not support other shapes.

    See also
    --------
    nems.layers.base.Layer

    Examples
    --------
    >>> wc = WeightChannels(shape=(18,4))
    >>> spectrogram = np.random.rand(10000, 18)  # (time, channels)
    >>> out = spectrogram @ wc.coefficients      # wc.evaluate(spectrogram)
    >>> out.shape
    (10000, 1)

    """

    def initial_parameters(self):
        """Get initial values for `WeightChannels.parameters`.
        
        Layer parameters
        ----------------
        coefficients : ndarray
            Shape matches `WeightChannels.shape`.
            Prior:  TODO, currently using defaults
            Bounds: TODO

        Returns
        -------
        nems.layers.base.Phi

        """
        # Mean and sd for priors were chosen mostly arbitrarily, to start
        # with most weights near zero (but not exactly at 0).
        mean = np.full(shape=self.shape, fill_value=0.01)
        sd = np.full(shape=self.shape, fill_value=0.05)
        prior = Normal(mean, sd)

        coefficients = Parameter(
            name='coefficients', shape=self.shape, prior=prior
            )
        return Phi(coefficients)

    @property
    def coefficients(self):
        """Weighting matrix that will be applied to input.
        
        Re-parameterized subclasses should overwrite this so that `evaluate`
        doesn't need to change.

        Returns
        -------
        coefficients : ndarray
            coefficients.shape = WeightChannels.shape
        
        """
        return self.parameters['coefficients']

    def evaluate(self, input):
        """Multiply input(s) by WeightChannels.coefficients.

        Computes $y = XA$ for each input $X$,
        where $A$ is `WeightChannels.coefficients` and $y$ is one output.
        
        Parameters
        ----------
        inputs : N-tuple of ndarray

        Returns
        -------
        list of ndarray
            Length of list matches number of inputs.
        
        """
        # TODO: this won't use DxDxD shape as intended. Could do a for loop
        #       for last dimension (if dim >2) similar to FIR, but that would
        #       probable be slow. Could also take product of last N dimensions
        #       (similar to old FIR)
        #       - from numpy matmul docs:
        #         "If either argument is N-D, N > 2, it is treated as a stack
        #          of matrices residing in the last two indexes and broadcast
        #          accordingly."
        #       - Maybe this could solve the problem if we reshape coefficients
        #         accordingly? Ex: (not working so far)
        # if len(self.shape) > 2:
        #     # (time, chans, ...) -> (..., time, chans)
        #     c = np.moveaxis(self.coefficients.values, [0, 1], [-2, -1])
        # else:
        #     c = self.coefficients
        
        try:
            output = input @ self.coefficients
        except ValueError as e:
            # Check for dimension swap, to give more informative error message.
            if 'mismatch in its core dimension' in str(e):
                raise ShapeError(self, input=input.shape,
                                 coefficients=self.coefficients.shape)
            else:
                # Otherwise let the original error through.
                raise e
        return output

    def as_tensorflow_layer(self):
        """TODO: docs"""
        import tensorflow as tf
        from nems.tf import get_tf_class

        def call(self, inputs):
            # TODO: docs, explain (at least briefly) why this is the same thing.
            return tf.nn.conv1d(
                inputs, tf.expand_dims(self.coefficients, 0), stride=1,
                padding='SAME'
                )
        
        return get_tf_class(self, call=call)

    @layer('wc')
    def from_keyword(keyword):
        """Construct WeightChannels (or subclass) from keyword.

        Keyword options
        ---------------
        {digit}x{digit}x ... x{digit} : N-dimensional shape.
        g : Use gaussian function(s) to determine coefficients.

        See also
        --------
        Layer.from_keyword
        
        """
        wc_class = WeightChannels
        kwargs = {}

        options = keyword.split('.')
        for op in options:
            if ('x' in op) and (op[0].isdigit()):
                dims = op.split('x')
                kwargs['shape'] = tuple([int(d) for d in dims])
            elif op == 'g':
                wc_class = GaussianWeightChannels

        if 'shape' not in kwargs:
            raise ValueError("WeightChannels requires a shape, ex: `wc.18x4`")

        wc = wc_class(**kwargs)

        return wc

    @property
    def plot_kwargs(self):
        """Add incremented labels to each output channel for plot legend.
        
        See also
        --------
        Layer.plot
        
        """
        kwargs = {
            'label': [f'Channel {i}' for i in range(self.shape[1])]
        }
        return kwargs

    @property
    def plot_options(self):
        """Add legend at right of plot, with default formatting.

        Notes
        -----
        The legend will grow quite large if there are many output channels,
        but for common use cases (< 10) this should not be an issue. If needed,
        increase figsize to accomodate the labels.

        See also
        --------
        Layer.plot
        
        """
        return {'legend': True}


class GaussianWeightChannels(WeightChannels):
    """As WeightChannels, but sample coefficients from gaussian functions."""

    def initial_parameters(self):
        """Get initial values for `GaussianWeightChannels.parameters`.
        
        # TODO: Currently this assumes 2D shape, we should refactor to support
        # higher-dimensional weights.

        Layer parameters
        ----------------
        mean : scalar or ndarray
            Mean of gaussian, shape is (N_outputs,).
            Prior:  TODO  # Currently using defaults
            Bounds: (0, 1)
        sd : scalar or ndarray
            Standard deviation of gaussian, shape is (N_outputs,).
            Prior:  TODO  # Currently using defaults
            Bounds: (0*, np.inf)
            * Actually set to machine epsilon to avoid division by zero.

        Returns
        -------
        nems.layers.base.Phi

        """

        mean_bounds = (0, 1)
        sd_bounds = (0, np.inf)
        
        _, n_output_channels = self.shape
        shape = (n_output_channels,)
        # Pick means so that the centers of the gaussians are spread across the 
        # available frequencies.
        channels = np.arange(n_output_channels + 1)[1:]
        tiled_means = channels / (n_output_channels*2 + 2) + 0.25
        mean_prior = Normal(tiled_means, np.full_like(tiled_means, 0.2))
        sd_prior = HalfNormal(np.full_like(tiled_means, 0.4))
            
        parameters = Phi(
            Parameter(name='mean', shape=shape, bounds=mean_bounds,
                      prior=mean_prior),
            Parameter(name='sd', shape=shape, bounds=sd_bounds,
                      prior=sd_prior, zero_to_epsilon=True)
            )

        return parameters

    @property
    def coefficients(self):
        """Return N discrete gaussians with T bins, where `shape=(T,N)`."""
        mean = self.parameters['mean'].values
        sd = self.parameters['sd'].values
        n_input_channels, _ = self.shape

        x = np.arange(n_input_channels)/n_input_channels
        mean = np.asanyarray(mean)[..., np.newaxis]
        sd = np.asanyarray(sd)[..., np.newaxis]
        coefficients = np.exp(-0.5*((x-mean)/sd)**2).T

        # Normalize by the cumulative sum for each channel
        cumulative_sum = np.sum(coefficients, axis=1, keepdims=True)
        # If all coefficients for a channel are 0, skip normalization
        cumulative_sum[cumulative_sum == 0] = 1
        coefficients /= cumulative_sum

        return coefficients

    def as_tensorflow_layer(self):
        """TODO: docs"""
        import tensorflow as tf
        from nems.tf import get_tf_class

        def call(self, inputs):
            # TODO: docs. Explain (at least briefly) why this is the same thing.
            input_features = tf.cast(tf.shape(inputs)[-1], dtype='float32')
            temp = tf.range(input_features) / input_features
            temp = (tf.reshape(temp, [1, input_features, 1]) - self.mean) / (self.sd/10)
            temp = tf.math.exp(-0.5 * tf.math.square(temp))
            kernel = temp / tf.math.reduce_sum(temp, axis=1)

            return tf.nn.conv1d(inputs, kernel, stride=1, padding='SAME')
        
        def weights_to_values(self):
            values = self.parameter_values
            values['sd'] = values['sd']/10  # undo kludge mentioned below
            return values

        # TODO: Ask SVD about this kludge in old NEMS code. Is this still needed?
        # If so, explain: I think this was to keep gradient from "blowing up"?
        # Scale up sd bound
        sd_lower, sd_upper = self.parameters['sd'].bounds
        self.parameters['sd'].bounds = (sd_lower, sd_upper*10)
        tf_class = get_tf_class(
            self, call=call, weights_to_values=weights_to_values
            )
        # Reset sd bound
        self.parameters['sd'].bounds = (sd_lower, sd_upper)
        # TODO: this also means the parameter value for sd will need to be
        #       scaled down when retrieving from tf

        return tf_class
