import numpy as np

from .base import Layer, Phi, Parameter
from nems.registry import layer


# TODO: double check all shape references after dealing w/ data order etc,
#       make sure correct dims are lined up.
class WeightChannels(Layer):

    def __init__(self, shape, **kwargs):
        """Compute linear weighting of input channels, akin to a dense layer.

        Parameters
        ----------
        shape : 2-tuple
            Determines the shape of `WeightChannels.coefficients`.
            First dimension should match the spectral dimension of the input,
            second dimension should match the spectral dimension of the output.
            Note that higher-dimesional shapes are also allowed and should work
            as-is for this base class, but overall Layer design is intended for
            2-dimensional data so subclasses might not support other shapes.
        parameters : Phi or None, optional
            Specifies the value of each variable used to compute
            `WeightChannels.evaluate()`. If `None`, values will be determined by
            `WeightChannels.initial_parameters`.
        
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

        """
        coefficients = Parameter(name='coefficients', shape=self.shape)
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

        return wc_class(**kwargs)

    def to_json(self):
        """Encode WeightChannels as a dictionary. See Layer.to_json."""
        data = Layer.to_json(self)
        data['kwargs'].update(shape=self.shape)
        return data


class GaussianWeightChannels(WeightChannels):

    def initial_parameters(self):
        """Get initial values for `GaussianWeightChannels.parameters`.
        
        Layer parameters
        ----------------
        mean : scalar or ndarray
            Mean of gaussian, shape is (N_outputs,).
            Prior:  TODO  # Currently using defaults
            Bounds: (0, 1)

        std : scalar or ndarray
            Standard deviation of gaussian, shape is (N_outputs,).
            Prior:  TODO  # Currently using defaults
            Bounds: (0, np.inf)

        """
        n_output_channels, _ = self.shape
        shape = (n_output_channels,)
        parameters = Phi(
            Parameter(name='mean', shape=shape, bounds=(0, 1)),
            Parameter(name='std', shape=shape, bounds=(0, np.inf))
            )

        return parameters

    # TODO: question for Stephen. Have we tried this with an added "vertical"
    #       shift to allow negative coefficients at the edges? I.e. if peak of
    #       gaussian is at BF, a narrowband sound would get big excitation but
    #       broadband would get a reduced response.
    @property
    def coefficients(self):
        """Return N discrete gaussians with T bins, where `self.shape=(N,T)`."""
        mean = self.parameters['mean'].values
        std = self.parameters['std'].values
        _, n_input_channels = self.shape

        x = np.arange(n_input_channels)/n_input_channels
        mean = np.asanyarray(mean)[..., np.newaxis]
        sd = np.asanyarray(sd)[..., np.newaxis]
        coefficients = np.exp(-0.5*((x-mean)/sd)**2)

        # Normalize by the cumulative sum for each channel
        cumulative_sum = np.sum(coefficients, axis=1, keepdims=True)
        # If all coefficients for a channel are 0, skip normalization
        cumulative_sum[cumulative_sum == 0] = 1
        coefficients /= cumulative_sum

        return coefficients
