import numpy as np
from nems.registry import module

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
       I.e. make a Phi or Parameters class that looks like a dict for users,
       but uses a persistent vector representation under the hood. This would
       speed up scipy fits (how much?) and provide a nice place for utility
       functions that manage parameters but otherwise have nothing to do with
       the module (like forming bounds arrays).

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

    def __repr__(self):
        return str(self.__class__)

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
        """Alias for `self.parameters`."""
        return self.parameters

    def sample_from_priors(self):
        pass

    def freeze_parameters(self):
        # TODO: copy to something like fn_kwargs as before? could even automate
        #       the dict updates somewhere to keep .evaluate() simple.
        pass


    # TODO: after working through this a bit (the phi & bounds -> vector stuff),
    #       starting to think it makes more sense to keep this separate from
    #       Module classes. Hard to think of a use-case where some one would
    #       need to break the phi dict format, in which case the existing
    #       functions would still work fine (just have model collect all the
    #       phi dicts and pass them off). And these really clutter up the base
    #       class with stuff that isn't directly related to Module evaluation.
    #
    #       Leaving in for now to illustrate the idea, but probably not needed
    #       since the generic fn should work for all Modules. Would only be a
    #       benefit if we want people to be able to define phi in other ways,
    #       in which case they could also implement the vector mapping.
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
        """
        Convert parameter vector to a dictionary and update `self.parameters`.

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

        """
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
        """Return `self.bounds` formatted as a list."""
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

    def to_json(self):
        # TODO
        pass

    def from_json(json):
        # TODO
        pass

    @module('base')
    def from_keyword(keyword):
        return Module()

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

    # TODO: is this really needed?
    def description(self):
        """Optional short description of `Module`'s function.
        
        Defaults to the docstring for self.evaluate if not overwritten.

        Example
        -------
        def evaluate(self):
            '''A really long docstring with citations and notes other stuff.'''
            return a + np.exp(b-c)

        def description(self):
            return '''Implements a simple exponential: $a + e^{(b-c)}$'''

        """

        return help(self.evaluate)

    def tensorflow_layer(self):
        """Builds a `Tensorflow.keras.layers.Layer` equivalent to this Module.

        TODO: How would this fit into the arbitrary input/output scheme that
              .evaluate() uses for scipy optimization? Maybe it can't, since
              Tensorflow already has its own way of managing the data flow.

        TODO: should layers be instantiated at this point, or just return
              the layer class? (probably with some frozen kwargs). Depends
              on how the higher-level model builder is formated.

        """
        raise NotImplementedError(
            f'{self.__class__} has not defined a Tensorflow implementation.'
            )


# TODO: What does this do? Looks like it imports all module modules (hah) in
#       the same directory. But why? Ask Stephen. If it needs to stay, add
#       documentation.

# mods = glob.glob(join(dirname(__file__), "*.py"))
# __all__ = [ basename(f)[:-3] for f in mods if isfile(f) and not f.endswith('__init__.py')]
# for a in __all__:
#     importlib.import_module(__name__ + "." + a)
# del mods
# del a