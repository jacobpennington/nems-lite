import re

import numpy as np

from .base import Layer, Phi, Parameter
from nems.registry import layer


# TODO: double check all shape references after dealing w/ data order etc,
#       make sure correct dims are lined up.
class WeightChannels(Layer):

    def __init__(self, shape, **kwargs):
        """TODO: docstring

        Parameters
        ----------
        shape : 2-tuple
            First entry specifies the expected spectral dimension of the input,
            second entry is the spectral dimension of the output. Shape will be
            used to initialize the `'coefficients'` Parameter
            if parameters=None.
        parameters : Phi or None, optional
            Specifies the value of each variable used to compute
            `WeightChannels.evaluate()`. If `None`, values will be determined by
            `WeightChannels.initial_parameters`.
        
        Returns
        -------
        WeightChannels
        
        """
        self.shape = shape
        super().__init__(**kwargs)

    def initial_parameters(self):
        coefficients = Parameter(name='coefficients', shape=self.shape)
        return Phi(coefficients)

    @property
    def coefficients(self):
        """Subclasses overwrite this, so `evaluate` doesn't need to change."""
        return self.parameters['coefficients']

    def evaluate(self, *inputs):
        """TODO: docstring, and check ordering on matrix multiplication."""
        y = [x @ self.coefficients for x in inputs]
        return y

    @layer('wc')
    def from_keyword(keyword):
        wc_class = WeightChannels
        kwargs = {}

        options = keyword.split('.')
        in_out_pattern = re.compile(r'^(\d{1,})x(\d{1,})$')
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
        data = Layer.to_json(self)
        data['kwargs'].update(shape=self.shape)
        return data


class GaussianWeightChannels(WeightChannels):

    def __init__(**kwargs):
        """TODO: overwrite docstring for expected parameters but same init."""
        super().__init__(**kwargs)

    def initial_parameters(self):
        n_output_channels, _ = self.shape
        shape = (n_output_channels,)
        parameters = Phi(
            Parameter(name='mean', shape=shape, bounds=(0, 1)),
            Parameter(name='std', shape=shape, bounds=(0, None))
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
