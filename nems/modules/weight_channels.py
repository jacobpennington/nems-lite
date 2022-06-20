import numpy as np

from nems.modules.base import Module


class WeightChannels(Module):
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
        if parameters is None:
            parameters = self.initial_parameters(parameterization)
        super.__init__(parameters=parameters, **kwargs)

    # TODO: this requires python 3.10. If we don't want to require that
    #       change back to if/else chain. But if we're re-doing everything
    #       anyway, would be nice to take advantage of newer syntax for stuff
    #       like this.
    def initial_parameters(self, parameterization):
        parameters = {}

        match parameterization:
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
        self.parameterization = parameterization
        return parameters


    def initial_parameters(self):
        self.parameters = {}
        self.parameters['coefficients'] = self.get_coefficients

    def evaluate(self, *inputs):
        """TODO: docstring, and check ordering on matrix multiplication."""
        coefficients = self.get_coefficients()
        y = [x @ coefficients for x in inputs]
        return y

    def tensorflow_layer(self):
        # TODO
        pass


# TODO: only keep one version, this is just for comparison when discussing
#       with Stephen.
class WeightChannelsV2(Module):
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
        super.__init__(**kwargs)
        self.shape = shape
        if self.parameters is None:
            parameters = self.initial_parameters()
        # TODO: should this be done automatically? maybe expose an option, like
        #       `initialize_to_mean=True`?
        if self.priors is not None:
            parameters = self.mean_of_priors()

    # TODO: don't really *need* a separate method in this case since it's so
    #       simple, but idea is that other Modules might have much more
    #       complicated defaults. Also this makes it easy to reset to defaults.
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
        y = [x @ coefficients for x in inputs]
        return y

    def tensorflow_layer(self):
        # TODO
        pass

class GaussianWeightChannels(WeightChannelsV2):
    """Same __init__, evaluate, etc"""
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
