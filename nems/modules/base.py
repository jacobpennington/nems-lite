import numpy as np

class Module:
    """
    Encapsulates one data-transformation step of a NEMS ModelSpec.

    Base class for NEMS Modules.
    TODO / NOTE:
    1) Currently keeping phi the same (but renamed self.parameters for clarity)
       This means users need to stick to the dictionary format, which I think
       is reasonable and means we can re-use the old phi-to-vector etc for
       scipy fitting.

       Thinking of going one layer deeper on classing as an alternative.
       I.e. make a Phi or Parameters class that is 95% a wrapper on a dictionary,
       but would be a sensible place to put parameterization functions. Not a
       big difference either way, just enforces the phi format a little more
       strongly. And if they mimic dicts properly, utility functions mentioned
       above can still be the same.

       Similar question for parameterization functions. Make them a tiny class
       instead, to enforce structure? E.g. `GaussianCoefficients` instead of
       `gaussian_coefficients`. Either way, I'd like to integrate them more
       seemlessly into the phi dict. I.e. calling `Module.phi` would return
       `{'coefficients': GaussianCoefficients(mean=x, std=y, shape=(3,2))}` but
       `Module.parameters_to_vector()` returns `[[a,b],[c,d],[e,f]]`.

    2) Simple cache scheme for repeated evals? If the parameters don't change
       (like for frozen modules), we can skip a call to evaluate() and 
       vector_to_parameters() on each fit iteration.

    """


    def __init__(self, input=None, output=None, name=None, parameters=None,
                 bounds=None, priors=None):
        """
        TODO

        Note
        ----
        Subclasses that need to overwrite `__init__()` should only specify new
        parameters in the method definition, followed by **kwargs, and invoke
        super().__init__(**kwargs) to ensure all required attributes are set
        correctly. While not strictly required, this is the easiest way to
        ensure Modules are functional.

        For example:
        ```
        def __init__(self, new_parameter1, new_parameter2=None, **kwargs):
            self.something_new = new_parameter1
            super().__init__(**kwargs)
            self.do_something_to_priors(new_parameter2)

        ```

        """

        if isinstance(input, str):
            self.input = [input]
        else:
            # Should be a list of strings or None
            self.input = input

        if isinstance(output, str):
            self.output = [output]
        else:
            self.output = output

        self.name = name if name is not None else 'Module'
        self.bounds = bounds
        self.priors = priors
        self.model = None  # pointer to parent ModelSpec

        if parameters is not None:
            self.parameters = parameters
        else:
            self.parameters = self.sample_from_priors()


    def __getitem__(self, key=None, default=None):
        return self.parameters.get(key, default)

    def __setitem__(self, key, val):
        self.parameters[key] = val

    def sample_from_priors(self):
        pass

    def parameters_to_vector(self):
        """
        Return `self.parameters` formatted as list.
        
        This is a helper method for fitters that use scipy.optimize. SciPy
        optimizers require parameters to be a single vector, but it's more
        intuitive/user-friendly for Modules to keep a dictionary representation.

        Returns
        -------
        list

        Examples
        --------
        >>> self.parameters = {'baseline': 0, 'coefs': [32, 41.8]}
        >>> parameters_to_vector(self)
        [0, 32, 41.8]
        >>> self.parameters = {'coefs': [[1, 2], [3, 4], [5, 6]]}
        >>> parameters_to_vector(phi)
        [1, 2, 3, 4, 5, 6]

        """
        vector = []  # list append is faster than np array append
        for k in sorted(self.parameters.keys()):
            value = self.parameters[k]
            flattened_value = np.asanyarray(value).ravel()
            vector.extend(flattened_value)

        return vector

    def vector_to_parameters(self, vector):
        '''
        Convert vector back to a dictionary and update `self.parameters`.

        Parameters
        ----------
        vector : list

        Example
        -------
        >>> self.parameters = {'baseline': 0, 'coefs': [0, 0]}
        >>> vector = [0, 32, 41]
        >>> vector_to_phi(self, vector)
        >>> self.parameters
        {'baseline': 0, 'coefs': array([32, 41])}

        '''
        index = 0
        parameter_template = self.parameters.copy()
        
        for k in sorted(parameter_template.keys()):
            value_template = parameter_template[k]
            if np.isscalar(value_template):
                value = vector[index]
                index += 1
            else:
                value_template = np.asarray(value_template)
                size = value_template.size
                value = np.asarray(vector[index:index+size])
                value.shape = value_template.shape
                index += size
            self.parameters[k] = value

    def bounds_to_vector(self):
        lb = {}
        ub = {}
        for name, phi in self.parameters.items():
            bounds = self.bounds.get(name, (None, None))
            lb[name] = Module._to_bounds_array(bounds, phi, 'lower')
            ub[name] = Module._to_bounds_array(bounds, phi, 'upper')

    def _to_bounds_array(value, phi, which):
        if which is 'lower':
            default_value = -np.inf
            i = 0

        elif which is 'upper':
            default_value = np.inf
            i = 1

        value = value[i]

        if isinstance(value, list):
            value = np.array(value)

        if value is None:
            return np.full_like(phi, default_value, dtype=np.float)

        if isinstance(value, np.ndarray):
            if value.shape != phi.shape:
                raise ValueError('Bounds wrong shape')
            return value

        return np.full_like(phi, value, dtype=np.float)


    # TODO: Alternatively, break old terminology a bit and use phi to refer to
    # the vectorized parameters only? Then this can look something like:
    #     ```
    #     if self.phi_vector is None:
    #         return parameters_to_vector()
    #     else:
    #         return self.phi_vector
    #     ```
    # This would make it easier to keep the difference clear when setting
    # up methods/variables for mid-fit representation (vectorized) vs user
    # friendly representation (dictionary).
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
