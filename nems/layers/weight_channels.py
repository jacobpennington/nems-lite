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
        return self.parameters['coefficients'].values

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

        try:
            output = np.tensordot(input, self.coefficients, axes=(1, 0))
        except ValueError as e:
            # Check for dimension swap, to give more informative error message.
            #if 'mismatch in its core dimension' in str(e):  # for @
            if 'shape-mismatch for sum' in str(e):
                raise ShapeError(self, input=input.shape,
                                 coefficients=self.coefficients.shape)
            else:
                # Otherwise let the original error through.
                raise e

        return output

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

    def as_tensorflow_layer(self, **kwargs):
        """TODO: docs"""
        import tensorflow as tf
        from nems.tf import NemsKerasLayer

        c = self.coefficients
        c_shape = c.shape
        # Force at least 3 dimensions for compatibility with TF version of FIR.
        if c.ndim < 3:
            new_values = {'coefficients': c[...,np.newaxis]}
        else:
            new_values = {}

        class WeightChannelsTF(NemsKerasLayer):
            @tf.function
            def call(self, inputs):
                # Add an n_outputs dimension to input if one is not present.
                out = tf.tensordot(inputs, self.coefficients, axes=[[2], [0]])
                return out

            def weights_to_values(self):
                values = self.parameter_values
                # Remove extra dummy axis if one was added.
                values['coefficients'] = values['coefficients'].reshape(c_shape)
                return values
        
        return WeightChannelsTF(self, new_values=new_values, **kwargs)

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

        Layer parameters
        ----------------
        mean : scalar or ndarray
            Mean of gaussian, shape is `self.shape[1:]`.
            Prior:  TODO  # Currently using defaults
            Bounds: (0, 1)
        sd : scalar or ndarray
            Standard deviation of gaussian, shape is `self.shape[1:]`.
            Prior:  TODO  # Currently using defaults
            Bounds: (0*, np.inf)
            * Actually set to machine epsilon to avoid division by zero.

        Returns
        -------
        nems.layers.base.Phi

        """

        mean_bounds = (0, 1)
        sd_bounds = (0, np.inf)

        rank = self.shape[1]
        other_dims = self.shape[2:]
        shape = (rank,) + other_dims
        # Pick means so that the centers of the gaussians are spread across the 
        # available frequencies.
        channels = np.arange(rank + 1)[1:]
        tiled_means = channels / (rank*2 + 2) + 0.25
        for dim in other_dims:
            # Repeat tiled gaussian structure for other dimensions.
            tiled_means = tiled_means[...,np.newaxis].repeat(dim, axis=-1)

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
        n_input_channels = self.shape[0]

        x = np.arange(n_input_channels)/n_input_channels
        mean = np.asanyarray(mean)[..., np.newaxis]
        sd = np.asanyarray(sd)[..., np.newaxis]
        coefficients = np.exp(-0.5*((x-mean)/sd)**2)  # (rank, ..., outputs, T)
        reordered = np.moveaxis(coefficients, -1, 0)  # (T, rank, ..., outputs)
        # Normalize by the cumulative sum for each channel
        cumulative_sum = np.sum(reordered, axis=-1, keepdims=True)
        # If all coefficients for a channel are 0, skip normalization
        cumulative_sum[cumulative_sum == 0] = 1
        normalized = reordered/cumulative_sum

        return normalized

    def as_tensorflow_layer(self, **kwargs):
        """TODO: docs"""
        import tensorflow as tf
        from nems.tf import NemsKerasLayer

        # TODO: Ask SVD about this kludge in old NEMS code. Is this still needed?
        # If so, explain: I think this was to keep gradient from "blowing up"?
        # Scale up sd bound
        sd, mean = self.get_parameter_values('sd', 'mean')
        sd_shape = sd.shape
        if sd.ndim < 2:
            # Always use at least 2 dims for compatibility with TF FIR.
            sd = sd[..., np.newaxis]
            mean = mean[..., np.newaxis]
        sd_lower, sd_upper = self.parameters['sd'].bounds
        new_values = {'sd': sd*10, 'mean': mean}
        new_bounds = {'sd': (sd_lower, sd_upper*10)}

        class GaussianWeightChannelsTF(NemsKerasLayer):
            def call(self, inputs):
                # TODO: docs. Explain (at least briefly) why this is the same thing.

                # TODO: convert to tensordot

                # TODO: the shapes aren't lining up right for new syntax,
                #       WIP
                mean = tf.expand_dims(self.mean, -1)
                sd = tf.expand_dims(self.sd/10, -1)
                input_features = tf.cast(tf.shape(inputs)[-1], dtype='float32')
                temp = tf.range(input_features) / input_features
                temp = (tf.reshape(temp, [1, input_features, 1]) - mean) / sd
                temp = tf.math.exp(-0.5 * tf.math.square(temp))
                norm = tf.math.reduce_sum(temp, axis=1)
                kernel = temp / norm

                return tf.tensordot(inputs, kernel, axes=[[2], [0]])
                #return tf.nn.conv1d(inputs, kernel, stride=1, padding='SAME')
            
            def weights_to_values(self):
                values = self.parameter_values
                # Remove extra dummy axis if one was added, and undo scaling.
                values['sd'] = (values['sd'].reshape(sd_shape)) / 10
                values['mean'] = values['mean'].reshape(sd_shape)
                return values

        # TODO
        raise NotImplementedError("Tensorflow wc.g is still a WIP")
        return GaussianWeightChannelsTF(self, new_values=new_values,
                                        new_bounds=new_bounds, **kwargs)
