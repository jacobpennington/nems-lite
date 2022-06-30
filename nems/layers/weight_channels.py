import re

import numpy as np

from .base import Layer, Phi, Variable
from nems.registry import module

# TODO: double check all shape references after dealing w/ data order etc,
#       make sure correct dims are lined up.


class WeightChannels(Layer):
    """TODO: docstring"""

    # TODO: would it make sense to subclass this for parameterization instead?
    #       (and same for multiple FIR)
    #       no more work to call GaussianWeightChannels instead of
    #       WeightChannels(parameterization='gaussian'), and they would still
    #       use the same .evaluate etc. But that would eliminate the need for
    #       potentially complicated if/else or case/switch chains, which would
    #       make the code a lot neater. And they could still use the same
    #       keyword function, the base cass method would just return whichever
    #       subclass is appropriate.
    #
    #       Already started writing it as the `parameterization=` version,
    #       and it got complicated fast. So, I'm going to write out the subclass
    #       version instead and see how it compares.

    def __init__(self, shape, parameterization=None, parameters=None, **kwargs):
        """TODO: docstring

        Parameters
        ----------
        shape : 2-tuple
            First entry specifies the expected spectral dimension of the input,
            second entry is the spectral dimension of the output.
        parameterization : str or function, optional
        parameters : dict or None, optional
            Specifies the value of each parameter used to compute
            `WeightChannels.evaluate()`. Specified entries will be different
            depending on which `parameterization` is used, if any. If `None`, 
            values will be determined by `WeightChannels.initialize_parameters`.
        
        Returns
        -------
        WeightChannels
        
        """
        self.shape = shape
        self.parameterization
        super.__init__(**kwargs)

    # TODO: this requires python 3.10. If we don't want to require that
    #       change back to if/else chain. But if we're re-doing everything
    #       anyway, would be nice to take advantage of newer syntax for stuff
    #       like this.
    def initial_parameters(self):
        parameters = {}

        match self.parameterization:
            case None:
                # TODO: better default?
                default = np.zeros(shape=self.shape)
                def get_coefficients(self):
                    return self.parameters.get('coefficients', default)
                parameters = {'coefficients': get_coefficients}

            case 'gaussian':
                raise NotImplementedError(
                    'Gaussian WeightChannels is not yet implemented'
                    )
                # TODO
                parameters = {'mean': 0, 'std': 0}
                def get_coefficients(self):
                    pass

        self.get_coefficients = get_coefficients
        return parameters


    def initial_parameters(self):
        self.parameters = {}
        self.parameters['coefficients'] = self.get_coefficients

    def evaluate(self, *inputs):
        """TODO: docstring, and check ordering on matrix multiplication."""
        coefficients = self.get_coefficients()
        y = [coefficients @ x for x in inputs]
        return y

    def tensorflow_layer(self):
        # TODO
        pass


# TODO: only keep one version, this is just for comparison when discussing
#       with Stephen.
class WeightChannelsV2(Layer):
    """TODO: docstring"""

    def __init__(self, shape, **kwargs):
        """TODO: docstring

        Parameters
        ----------
        shape : 2-tuple
            First entry specifies the expected spectral dimension of the input,
            second entry is the spectral dimension of the output. Shape will be
            used to initialize the `'coefficients'` parameter if parameters=None.
        parameters : dict or None, optional
            Specifies the value of each parameter used to compute
            `WeightChannels.evaluate()`. If `None`, values will be determined by
            `WeightChannels.initial_parameters`.
            Expected format: {'coefficients': numpy.ndarray(shape=shape)}
        
        Returns
        -------
        WeightChannels
        
        """
        self.shape = shape
        super.__init__(**kwargs)


    # TODO: don't really *need* a separate method in this case since it's so
    #       simple, but idea is that other WeightChannels might have much more
    #       complicated defaults. Also returning instead of setting  makes it
    #       easy to reset to defaults.
    def initial_parameters(self):
        # TODO: better default?
        parameters = {'coefficients': np.zeros(shape=self.shape)}
        return parameters

    # NOTE: yes, this is super simple. But the point is that subclasses
    #       only have to overwrite this and `initial_parameters`, everything
    #       else (including .evaluate) can stay the same.
    def _get_coefficients(self):
        return self.parameters['coefficients']

    def evaluate(self, *inputs):
        """TODO: docstring, and check ordering on matrix multiplication."""
        coefficients = self._get_coefficients()
        y = [coefficients @ x for x in inputs]
        return y

    def tensorflow_layer(self):
        # TODO
        pass


class GaussianWeightChannelsV2(WeightChannelsV2):
    """Same __init__, evaluate, etc"""

    def __init__(**kwargs):
        """TODO: overwrite docstring for expected parameters but same init."""
        super.__init__(**kwargs)

    def initial_parameters(self):
        # TODO: better defaults? (see old NEMS)
        parameters = {'mean': 0, 'std': 1}
        return parameters

    def _get_coefficients(self):
        return NotImplementedError('GaussianWeightChannels is not set up yet.')
        coefficients = '# TODO (see old NEMS for formula)'

    def tensorflow_layer(self):
        # TODO: tensorflow_layer would probably be different as well.
        pass


# TODO: only keep one version, this is just for comparison when discussing
#       with Stephen.
class WeightChannelsV3(Layer):
    # Same as V2, but uses the Phi & Variable classes

    def __init__(self, shape, **kwargs):
        """TODO: docstring

        Parameters
        ----------
        shape : 2-tuple
            First entry specifies the expected spectral dimension of the input,
            second entry is the spectral dimension of the output. Shape will be
            used to initialize the `'coefficients'` Variable if parameters=None.
        parameters : Phi or None, optional
            Specifies the value of each variable used to compute
            `WeightChannels.evaluate()`. If `None`, values will be determined by
            `WeightChannels.initial_parameters`.
            Expected format: Phi(Variable(name='coefficients', shape=shape))
        
        Returns
        -------
        WeightChannels
        
        """
        self.shape = shape
        super.__init__(**kwargs)

    def initial_parameters(self):
        coefficients = Variable(name='coefficients', shape=self.shape)
        return Phi(coefficients)

    @property
    def coefficients(self):
        return self.parameters['coefficients'].values

    def evaluate(self, *inputs):
        """TODO: docstring, and check ordering on matrix multiplication."""
        y = [self.coefficients @ x for x in inputs]
        return y

    @module('wc')
    def from_keyword(keyword):
        wc_class = WeightChannelsV3
        kwargs = {}

        options = keyword.split('.')
        in_out_pattern = re.compile(r'^(\d{1,})x(\d{1,})$')
        for op in options:
            if 'x' in op:
                parsed = re.fullmatch(in_out_pattern, op)
                if parsed is not None:
                    n_inputs = int(parsed.group(1))
                    n_outputs = int(parsed.group(2))
                    kwargs['shape'] = (n_inputs, n_outputs)

            elif op == 'g':
                wc_class = GaussianWeightChannelsV3

        if 'shape' not in kwargs:
            return ValueError("WeightChannels requires a shape, ex: `wc.18x4`")

        return wc_class(**kwargs)


class GaussianWeightChannelsV3(WeightChannelsV3):
    # Same as V2, but uses the Phi & Variable classes

    def __init__(**kwargs):
        """TODO: overwrite docstring for expected parameters but same init."""
        super.__init__(**kwargs)

    def initial_parameters(self):
        n_output_channels, _ = self.shape
        shape = (n_output_channels,)
        parameters = Phi(
            Variable(name='mean', shape=shape, bounds=(0, 1)),
            Variable(name='std', shape=shape, bounds=(0, None))
            # NOTE: bounds aren't actually set up for Variables yet, but this
            #       is an example of why I think it would be an intuitive place
            #       for them (can still pass through Modules, but they would
            #       ultimately be implemented in the Variable class).
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
