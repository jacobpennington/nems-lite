class Module:

    def __init__(self, input, output, name=None, parameters=None, bounds=None,
                 priors=None):
        self.input = input
        self.output = output
        self.name = name if name is not None else 'Module'
        self.bounds = bounds
        self.priors = priors


        # TODO: initialize phi from priors if none

    def __getitem__(self, key=None, default=None):
        return self.parameters.get(key, default)

    def __setitem__(self, key, val):
        self.parameters[key] = val

    def sample_from_priors(self):
        pass

    def parameters_to_vector(self):
        pass

    def vector_to_parameters(self):
        pass

    @property
    def phi(self):
        '''Alias for `self.parameters`.'''
        return self.parameters

    def evaluate(self, *args):  
        """Applies some mathematical operation to the argument(s).
        
        Each Module subclass is expected to redefine this method. Any number of
        parameters is acceptable, but each should correspond to one name in
        `self.input`. An arbitrary number of return values is also allowed, and
        each should correspond to one name in `self.output`. Input and output
        names will be associated with arguments and return values, respectively,
        in list-order.

        Parameters
        ----------
        *args : N-tuple of numpy.ndarray
                Arbitrary number of arrays to which this method will be applied.
        
        Returns
        -------
        N-tuple of numpy.ndarray

        Examples
        --------
        ```
        def evaluate(self, x, y, z):
            a = x + y
            b = 2*z

            return a, b
        ```
        For this example method, assume `self.input = ['one', 'two', 'three']`
        and `self.output = ['two', 'new']` and we have a Recording with
        `Recording.signals = {'one': x, 'two': y, 'three': z}`. After evaluation,
        the result would be:
            ```
            *args = (x, y, z)
            Module.evaluate(*args) = (x+y, 2*z)
            Recording.signals = {'one': x, 'two': x+y, 'three': z, 'new': 2*z}
            ```

        """

        raise NotImplementedError(f'{self.__class__} has not defined evaluate.')


# options['fn_coefficients'] = options.get('fn_coefficients', None)
# options['plot_fns'] = options.get('plot_fns',
#                                     ['nems.plots.api.mod_output',
#                                     'nems.plots.api.spectrogram_output',
#                                     'nems.plots.api.pred_resp'])
# options['plot_fn_idx'] = options.get('plot_fn_idx', 2)
# options['prior'] = options.get('prior', {})
# options['phi'] = options.get('phi', None)
