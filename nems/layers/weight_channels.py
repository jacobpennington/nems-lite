from re import S
import numpy as np

from nems.registry import layer
from nems.distributions import Normal, HalfNormal
from .base import Layer, Phi, Parameter


# TODO: double check all shape references after dealing w/ data order etc,
#       make sure correct dims are lined up.
class WeightChannels(Layer):

    def __init__(self, shape, **kwargs):
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
        
        Returns
        -------
        WeightChannels

        Examples
        --------
        >>> wc = WeightChannels(shape=18,4)
        >>> spectrogram = np.random.rand(10000, 18)  # (time, channels)
        >>> out = spectrogram @ wc.coefficients      # wc.evaluate(spectrogram)
        >>> out.shape
        (10000, 4)

        """
        self.shape = shape
        super().__init__(**kwargs)

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

    def evaluate(self, *inputs):
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
        return [x @ self.coefficients for x in inputs]

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

    def to_json(self):
        """Encode WeightChannels as a dictionary. See Layer.to_json."""
        data = Layer.to_json(self)
        data['kwargs'].update(shape=self.shape)
        return data


class GaussianWeightChannels(WeightChannels):

    def initial_parameters(self):
        """Get initial values for `GaussianWeightChannels.parameters`.
        
        # TODO: Currently this assumes 2D shape, with time on the first axis
        #       We should refactor to support higher-dimensional weights
        #       and different ordering of dimensions.

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
        """Return N discrete gaussians with T bins, where `self.shape=(T,N)`.
        
        # TODO: add axis_X = Y options to make this not assume specific order
        
        """
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
