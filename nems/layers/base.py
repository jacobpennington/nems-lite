import json

import numpy as np

from nems.registry import module
from nems.distributions import Normal


class Layer:
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

    1b) This is related to another problem I'm coming across: how to properly
        document/enforce the parameter structure for each Module. Currently,
        users are expected to either know the dict format (i.e. which parameters
        are expected, whether they should be scalar or array, etc) or always
        start with the defaults we provide. Documenting the parameters in
        `Module.__init__()` is better than old NEMS, but I think using something
        like a Phi class to explicitly require specific parameters with a given
        shape (similar to declaring model variables in Tensorflow or other
        packages) would be clearer.

    """


    def __init__(self, input=None, output=None, parameters=None,
                 bounds=None, priors=None, name=None):
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

        self.name = name if name is not None else 'unnamed module'
        self.bounds = bounds
        self.priors = priors
        self.model = None  # pointer to parent ModelSpec

        if parameters is None:
            parameters = self.initial_parameters()
        # TODO: should this be done automatically? maybe expose an option, like
        #       `initialize_to_mean=True`? Either way needs more thought, this
        #       would just overwrite initial.
        # if self.priors is not None:
        #     parameters = self.mean_of_priors()
        self.parameters = parameters


    def __getitem__(self, key):
        # TODO: If we want to expose priors, bounds, etc. as well this can be
        #       `return getattr(self, key)` instead, but I think doing this for
        #       just parameters makes more sense. That's the dict that gets
        #       accessed the most, and this makes that require less typing:
        #       `Module['a']` instead of `Module['parameters']['a'].`
        #       The getattr version could also be confusing since not all of
        #       the attributes are dicts (so further indexing won't always work)
        return self.parameters[key]
    
    def get(self, key, default=None):
        if key in self.parameters:
            val = self.parameters[key]
        else:
            val = default
        return val

    def __setitem__(self, key, val):
        self.parameters[key] = val

    def __repr__(self):
        return str(self.__class__)

    def set_parameter_values(self, parameter_dict):
        for k, v in parameter_dict.items():
            self.parameters[k] = v

    def get_parameter_values(self, *parameter_keys):
        if len(parameter_keys) == 0:
            values = self.parameters.get_values
        else:
            values = [self.parameters[k].values for k in parameter_keys]
        return values

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
        # TODO: Don't remember if Prior.sample() is actually correct, but
        #       this should be pretty close. Come back to this.
        parameters = {}
        for key in self.parameters.keys():
            parameters[key] = self.priors[key].sample()
        return parameters

    def mean_of_priors(self):
        # TODO: Don't remember if Prior.mean() is actually correct, but
        #       this should be pretty close. Come back to this.
        parameters = {}
        for key in self.parameters.keys():
            parameters[key] = self.priors[key].mean()
        return parameters

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
        # TODO: package parameters, bounds, etc, into dict w/ same format as old
        #       module dicts. Won't be fully backwards compatible but should
        #       make it easier to write an old -> new model conversion utility.
        pass

    def from_json(json):
        # TODO: Reverse of above, return Module instance using dict for kwargs.
        #       Some attributes may need to be set separately.
        pass

    @module('baseclass')
    def from_keyword(keyword):
        """TODO: doctring explaining how to use this in subclassed modules."""
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

    def initial_parameters(self):
        """TODO: docstring explaining idea, most subclasses will need to write
        their own."""
        return None



# TODO: What does this do? Looks like it imports all module modules (hah) in
#       the same directory. But why? Ask Stephen. If it needs to stay, add
#       documentation.

# mods = glob.glob(join(dirname(__file__), "*.py"))
# __all__ = [ basename(f)[:-3] for f in mods if isfile(f) and not f.endswith('__init__.py')]
# for a in __all__:
#     importlib.import_module(__name__ + "." + a)
# del mods
# del a



class Phi:
    # TODO: Sketch of possible Phi and Variable classes. General idea:
    #       Variables make the structure of the Phi dictionary explicit and easy
    #       to document, Phi provides a dict-like interface for users but keeps
    #       a persistent vector (list) representation under the hood.
    #
    #       In each module, this would be set up in .initial_parameters or
    #       somewhere similar, along the lines of:
    #       ```
    #       self.parameters = Phi(
    #           Variable(name='alpha'),
    #           Variable(name='beta', shape=(3,2)),
    #           Variable(name='gamma', dtype=np.float16),
    #       )
    #       ```
    #       Would also make sense to include bounds here, probably as a property
    #       of the Variables (and then the Phi class can easily collect them as
    #       a bounds array for the start of optimization).
    #
    #       Another nice result of this (in my opinion) is that the string rep.
    #       of phi would now look something like:
    #       ```
    #       print(self.parameters)
    #       >>> {
    #           'alpha': Variable(shape=(1,), dtype=float64)
    #                    .values = 4.9029380298
    #           'beta':  Variable(shape=(3,2), dtype=float64)
    #                    .values = [[3.4445, 7.001],
    #                               [0.44,   133.0],
    #                               [0.858,  11.11]]
    #           ...etc
    #       }
    #       ```
    #       In other words, anyone can immediately see what format is expected
    #       and we can put a little effort into a pretty-print __repr__ method.

    def __init__(self, *variables):
        self._array = [[]]
        self.index = 0
        self._dict = {}
        self.size = 0
        for v in variables:
            self.add_variable(v)

    @property
    def _vector(self):
        return self._array[self.index]

    def add_variable(self, variable):
        variable.first_index = self.size
        self.size += variable.size
        variable.last_index = self.size-1
        self._vector.extend([variable.initial_value]*variable.size)
        self._dict[variable.name] = variable
        variable.phi = self

    def __str__(self):
        return str(self._dict)

    # TODO: would need to propagate to/from_json calls from Module, and collect
    #       json representations from Variables, and store _vector.
    def to_json(self):
        pass

    def from_json(json):
        pass

    # Provide dict-like interface
    def __getitem__(self, key):
        return self._dict[key]
    
    def get(self, key, default=None):
        if key in self._dict:
            val = self._dict[key]
        else:
            val = default
        return val

    def get_values(self, *keys):
        return [self.get(k).values for k in keys]

    def update(self):
        # TODO
        pass

    def __setitem__(self, key, val):
        self._dict[key].update(val)

    def __iter__(self):
        return iter(self._dict)

    def keys(self):
        return self._dict.keys()

    def items(self):
        return self._dict.items()

    def values(self):
        return self._dict.values()

    def __repr__(self):
        return str(self._dict)


class Parameter:
    # TODO: rename to Parameter and refactor rest of code

    # TODO: incorporate bounds and priors here

    # TODO: is there a straightforward way to mimic a numpy array here?
    #       ex: would be nice to be able to use @ operator directly on a
    #       coefficients variable instead of variable.values.

    def __init__(self, name, shape=(1,), dtype=np.float64, 
                 bounds=None, prior=None):
        self.name = name
        self.shape = shape
        self.size = 1
        for axis in shape:
            self.size *= axis
        self.dtype = dtype

        if prior is None:
            self.prior = Normal(mean=0, std=1)  # default to standard normal
        if bounds is None:
            self.bounds = (self.prior.ppf(0.0001), self.prior.ppf(0.9999))
        self.initial_value = self.initialize()

        # Must be set by Phi for .values to work.
        self.phi = None  # Pointer to parent Phi instance.
        self.first_index = None  # Location of data within Phi._vector
        self.last_index = None

    @property
    def values(self):
        values = self.phi._vector[self.first_index:self.last_index+1]
        return np.reshape(values, self.shape)

    def initialize(self):


        return value

    def update(self, value):
        if self.shape == (1,) and np.isscalar(value):
            self.phi._vector[self.first_index] = value
        elif np.shape(value) != self.shape:
            raise ValueError(
                f"Variable {self.name} requires shape {self.shape}, but"
                f"{value} has shape {np.shape(value)}"
            )
        else:
            flat_value = np.ravel(value)
            self.phi._vector[self.first_index:self.last_index+1] = flat_value

    def to_json(self):
        data = {'name': self.name, 'shape': self.shape, 'dtype': self.dtype,
                'values': self.values}
        return json.dumps(data)

    def from_json(json):
        data = json.loads(json)
        return Variable(**data)

    def __repr__(self):
        # TODO: how to fix format for printing? Apparently __repr__ always
        #       ignores linebreak characters.
        string = (f"Variable(shape={self.shape}, dtype={self.dtype})"
                  f".values = {self.values}")
        return string

    def __str__(self):
        string = (f"Variable(shape={self.shape}, dtype={self.dtype})"
                  f".values = {self.values}")
        return string
