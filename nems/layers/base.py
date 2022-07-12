import json

import numpy as np

from nems.registry import layer
from nems.distributions import Normal


# TODO: add examples and tests
class Layer:
    """Encapsulates one data-transformation step of a NEMS ModelSpec.

    Base class for NEMS Layers.

    """

    def __init__(self, input=None, output=None, parameters=None,
                 bounds=None, priors=None, name=None):
        """Encapsulates one data-transformation step of a NEMS ModelSpec.

        Note
        ----
        Subclasses that need to overwrite `__init__()` should only specify new
        parameters in the method definition, followed by **kwargs, and invoke
        super().__init__(**kwargs) to ensure all required attributes are set
        correctly. While not strictly required, this is the easiest way to
        ensure Layers function properly within a Model instance.

        For example:
        >>> def __init__(self, new_parameter1, new_parameter2=None, **kwargs):
        ...     super().__init__(**kwargs)
        ...     self.something_new = new_parameter1
        ...     self.do_something_to_priors(new_parameter2)

        See also
        --------
        nems.models.base.Model

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
        #       `Layer['a']` instead of `Layer['parameters']['a'].`
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
    #       Layer classes. Hard to think of a use-case where some one would
    #       need to break the phi dict format, in which case the existing
    #       functions would still work fine (just have model collect all the
    #       phi dicts and pass them off). And these really clutter up the base
    #       class with stuff that isn't directly related to Layer evaluation.
    #
    #       Leaving in for now to illustrate the idea, but probably not needed
    #       since the generic fn should work for all Layers. Would only be a
    #       benefit if we want people to be able to define phi in other ways,
    #       in which case they could also implement the vector mapping.
    def parameters_to_vector(self):
        """
        Return `self.parameters` formatted as list.
        
        This is a helper method for fitters that use scipy.optimize. SciPy
        optimizers require parameters to be a single vector, but it's more
        intuitive/user-friendly for Layers to keep a dictionary representation.

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
            lb[name] = Layer._to_bounds_array(bounds, phi, 'lower')
            ub[name] = Layer._to_bounds_array(bounds, phi, 'upper')

    def _to_bounds_array(value, phi, which):
        if which == 'lower':
            default_value = -np.inf
            i = 0

        elif which == 'upper':
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
        #       layer dicts. Won't be fully backwards compatible but should
        #       make it easier to write an old -> new model conversion utility.
        pass

    def from_json(json):
        # TODO: Reverse of above, return Layer instance using dict for kwargs.
        #       Some attributes may need to be set separately.
        pass

    @layer('baseclass')
    def from_keyword(keyword):
        """TODO: doctring explaining how to use this in subclassed layers."""
        return Layer()

    def evaluate(self, *args):  
        """Applies some mathematical operation to the argument(s).
        
        Each Layer subclass is expected to redefine this method. Any number of
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
            Layer.evaluate(*args) = (x+y, 2*z)
            Recording.signals = {'one': x, 'two': x+y, 'three': z, 'new': 2*z}
            ```

        """

        raise NotImplementedError(f'{self.__class__} has not defined evaluate.')

    # TODO: is this really needed?
    def description(self):
        """Optional short description of `Layer`'s function.
        
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
        """Builds a `Tensorflow.keras.layers.Layer` equivalent to this Layer.

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


# TODO: add examples and tests
#       (for both Phi and Parameter)
class Phi:
    """Stores and manages updates to parameters for one Layer."""

    def __init__(self, *parameters):
        """Stores and manages updates to parameters for one Layer.

        In general, Phi instances should not need to be interacted with directly
        unless implementing a new Layer subclass or a related function. Instead,
        parameters should be accessed through Model- or Layer-level methods.

        Parameters
        ----------
        parameters : N-tuple of Parameter instances

        See also
        --------
        nems.models.base.Model
        nems.layers.base.Layer
        nems.layers.base.Parameter
        
        """
        self._array = [[]]
        self._index = 0
        self._dict = {}
        self.size = 0
        for p in parameters:
            self.add_parameter(p)

    @property
    def _vector(self):
        """Return a slice of `Phi._array` at `Phi._index`."""
        return self._array[self._index]

    def add_parameter(self, parameter):
        """Add a new parameter to `Phi._dict` and update `Phi._array`.
        
        Sets `parameter.phi`, `parameter.first_index`, and
        `parameter.last_index` to comply with `Phi._vector` formatting.
        
        """
        parameter.first_index = self.size
        self.size += parameter.size
        parameter.last_index = self.size-1
        self._vector.extend(parameter.initial_value*parameter.size)
        self._dict[parameter.name] = parameter
        parameter.phi = self

    def sample(self, inplace=False):
        """Get or set new parameter values by sampling from priors.
        
        Parameters
        ----------
        inplace : bool
            If True, sampled values will be used to update each Parameter
            instance (and, in turn, `Phi._array`) inplace. Otherwise, the
            sampled values will be returned without changing current values.

        Return
        ------
        vector : list
            Parameter values formatted as a flattened vector.

        """
        samples = []
        for p in self._dict.values():
            samples.append(p.sample(inplace=inplace))

        vector = []
        for s in samples:
            vector.extend(np.ravel(s))

        return vector

    def set_index(self, i, new_vector='resample'):
        """Change which vector to reference within `Phi._array`.

        Parameters
        ----------
        i : int
            New index for `Phi._array`. If `i >= len(Phi._array)`, then new
            vectors will be appended until `Phi._array` is sufficiently large.
        new_vector : str or None, default='resample'
            Determines how new vectors are generated if `i` is out of range.
            If `'resample'`: invoke `Phi.sample(inplace=False)`.
            Elif `'copy'`  : copy current `Phi._vector`.
            Elif `None`    : raise IndexError instead of adding new vectors.

        """
        array_length = len(self._array)
        if i >= array_length:
            # Array isn't that big yet, so add new vector(s).
            new_indices = range(array_length, i+1)
            match new_vector:
                case 'resample':
                    new_vectors = [self.sample() for j in new_indices]
                case 'copy':
                    new_vectors = [self._vector.copy() for j in new_indices]
                case None:
                    # Don't add new vectors, raise an error instead. May be
                    # useful for testing.
                    raise IndexError(f'list index {i} out of range for Phi.')

        self._array.extend(new_vectors)
        self._index = i

    # Provide dict-like interface into Phi._dict
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

    def update(self, dct):
        for k, v in dct.items():
            if k in self._dict:
                self._dict[k].update(v)
            else:
                self.add_parameter(
                    Parameter(name=k, shape=v.shape)
                    )
                self._dict[k].update(v)

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
        return str(list(self._dict.values()))

    # Add compatibility for saving to .json
    def to_json(self):
        """Encode Phi object as json. See `nems.tools.json`."""
        p = list(self._dict.values())
        return {'_array': self._array, '_index': self._index, 'parameters': p}

    def from_json(json):
        """Decode Phi object from json. See `nems.tools.json`."""
        phi = Phi(*json['parameters'])
        phi._array = json['_array']
        phi._index = json['_index']

        return phi


class Parameter:
    """Stores and manages updates to values for one parameter of one Layer."""

    def __init__(self, name, shape=(), prior=None, bounds=None,
                 default_bounds='infinite'):
        """Stores and manages updates to values for one parameter of one Layer.

        `Parameter` instances are intended to exist as components of a parent
        `Phi` instance, by invoking `Phi.add_parameter`. Without establishing
        this relationship, most `Parameter` methods will not work.

        As with `Phi` instances, `Parameter` instances should generally not be
        interacted with directly unless implementing new Layer subclasses or
        other core functionality. Wherever possible, users should interact with
        fittable parameters using Model- or Layer-level methods.

        Parameters
        ----------
        name : str
            This will also be the Parameter's key in `Phi._dict`.
        shape : N-tuple, default=()
            Shape of `Parameter.values`.
        prior : nems.distributions.Distribution or None
            Prior distribution for this parameter, with matching shape.
        bounds : 2-tuple or None
            (minimum, maximum) values for the entries of `Parameter.values`.
        default_bounds : str, default='infinite'
            Determines behavior when `bounds=None`.
            If `'infinite'`  : set bounds to (-np.inf, np.inf)
            If `'percentile'`: set bounds to tails of `Parameter.prior`.
                (prior.percentile(0.0001), prior.percentile(0.9999))

        See also
        --------
        nems.models.base.Model
        nems.layers.base.Layer
        nems.layers.base.Phi

        """
        self.name = name
        self.shape = shape
        self.size = 1
        for axis in shape:
            self.size *= axis

        # default to multivariate normal
        if prior is None:
            zero = np.zeros(shape=self.shape)
            one = np.ones(shape=self.shape)
            prior = Normal(mean=zero, sd=one)  
        self.prior = prior
        if prior.shape != self.shape:
            raise ValueError("Parameter.shape != Parameter.prior.shape")

        # set default based on `default_bounds`
        if bounds is None:
            if default_bounds == 'percentile':
                bounds = (prior.percentile(0.0001), prior.percentile(0.9999))
            elif default_bounds == 'infinite':
                bounds = (-np.inf, np.inf)
            else:
                raise ValueError(
                    "default_bounds can be 'percentile' or 'infinite'"
                    )
        self.bounds = bounds

        sample = prior.sample(bounds=bounds)
        self.initial_value = np.ravel(self.sample()).tolist()

        # Must be set by `Phi.add_parameter` for `Parameter.values` to work.
        self.phi = None  # Pointer to parent Phi instance.
        self.first_index = None  # Location of data within Phi._vector
        self.last_index = None

    @property
    def values(self):
        """Get corresponding parameter values stored by parent Phi instance."""
        values = self.phi._vector[self.first_index:self.last_index+1]
        return np.reshape(values, self.shape)

    def sample(self, n=1, inplace=False):
        """Get or set new values by sampling from `Parameter.prior`.

        Re-sampling is used to ensure that the final sample falls within
        `Parameter.bounds`.
        
        Parameters
        ----------
        n : int
            Number of samples to get.
        inplace : bool
            If True, sampled values will be used to update `Parameter.values`
            inplace. Otherwise, the sampled values will be returned without
            changing the current values.

        See also
        --------
        nems.layers.base.Phi.sample
        nems.distributions.base.Distribution.sample

        Return
        ------
        sample : ndarray

        """

        sample = self.prior.sample(n=n, bounds=self.bounds)
        if inplace:
            self.update(sample)
        return sample

    def update(self, value, ignore_bounds=False):
        """Set `Parameters.values` to `value` by updating `Phi._vector`.
        
        Parameters
        ----------
        value : scalar or array-like
            New value for `Parameter.values`. Must match `Parameter.shape`.
        ignore_bounds : bool
            If True, ignore `Parameter.bounds` when updating. Otherwise,
            new values will be rejected if they are less than `bounds[0]` or
            greater than `bounds[1]`.
        
        """
        value = np.asarray(value)
        if not ignore_bounds:
            lower, upper = self.bounds
            if np.any(value < lower) or np.any(value > upper):
                raise ValueError(
                    f"value out-of-bounds for...\n"
                    f"Parameter: {self.name}\n"
                    f"Bounds: {self.bounds}\n"
                    f"Value: {value}"
                )

        if np.shape(value) != self.shape:
            raise ValueError(
                f"Parameter {self.name} requires shape {self.shape}, but "
                f"{value} has shape {np.shape(value)}"
            )
        else:
            flat_value = np.ravel(value)
            self.phi._vector[self.first_index:self.last_index+1] = flat_value

    def __repr__(self):
        data = [
            f'Parameter(name={self.name}, shape={self.shape})',
            f'.values: {self.values.__repr__()}'
            ]
        return json.dumps(data, indent=2)

    # Add compatibility for saving to .json    
    def to_json(self):
        """Encode Parameter object as json. See `nems.tools.json`."""
        data = {
            'name': self.name,
            'shape': self.shape,
            'prior': self.prior,
            'bounds': self.bounds
            }
        # Encoder adds an extra key, so nest the dict to keep kwargs separate.
        return {'data': data}

    def from_json(json):
        """Decode Parameter object from json. See `nems.tools.json`."""
        return Parameter(**json['data'])


    # TODO: is there a straightforward way to mimic a numpy array here?
    #       ex: would be nice to be able to use @ operator directly on a
    #       coefficients parameter instead of parameter.values.
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Propagate numpy ufunc operations to `Parameter.values` (WIP)."""
        raise NotImplementedError("Parameter.__array_ufunc__ doesn't work yet.")
        f = getattr(ufunc, method)
        subbed_inputs = [
            x.values if isinstance(x, Parameter) else x
            for x in inputs
            ]
        output = f(*subbed_inputs, **kwargs)

        if output is None:
            raise NotImplementedError("ufunc cannot modify Parameter in-place.")
        else:
            if not isinstance(output, np.ndarray):
                try:
                    output = np.asarray(output)
                    self.update(output)
                except Exception as e:
                    raise e("Something went wrong in Parameter.__array_ufunc__")
