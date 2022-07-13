import json

import numpy as np

from nems.registry import layer
from nems.distributions import Normal


# TODO: add examples and tests
class Layer:
    """Encapsulates one data-transformation step of a NEMS ModelSpec.

    Base class for NEMS Layers.

    """

    def __init__(self, input=None, output=None, parameters=None, priors=None,
                 bounds=None, default_bounds='infinite', name=None):
        """Encapsulates one data-transformation step of a NEMS ModelSpec.

        Layers are intended to exist as components of a parent Model instance
        by invoking `Model.add_layer` or `Model.__init__(layers=...)`.

        Parameters
        ----------
        input : str, list, dict, or None; optional
            Specifies which data streams should be provided as inputs by
            parent Model during fitting, where strings refer to keys for a
            dict of arrays provided to `Model.fit`.
            If None : output of previous Layer.
            If str  : a single input array.
            If list : many input arrays.
            If dict : many input arrays, with keys specifying which parameter
                      of `Layer.evaluate` each array is associated with.

            # TODO: examples from simple_fit.py to illustrate usage
            
        output : str, list, or None; optional
            Specifies name(s) for array output(s) of `Layer.evaluate`.
            If None : use default name specified by parent Model.
            If str  : same name for every output (incremented if multiple).
            If list : one name per output (length must match).

        parameters : nems.layers.base.Phi or None; optional
            Specifies values for fittable parameters used by `Layer.evaluate`.
            If None : Phi instance returned by `Layer.initial_parameters`.
        bounds : dict of 2-tuples or None; optional
            Determines minimum and maximum values for fittable parameters. Keys
            must correspond to names of parameters, such that each Parameter
            instance utilizes `Parameter(name, bounds=bounds[name])`.
            If None : use defaults defined in `Layer.initial_parameters`.
        default_bounds : str, default='infinite'
            Determines behavior when `bounds=None` for individual parameters.
            If `'infinite'`  : set bounds to (-np.inf, np.inf)
            If `'percentile'`: set bounds to tails of `Parameter.prior`.
                (prior.percentile(0.0001), prior.percentile(0.9999))
        name : str or None; optional
            A name for the Layer so that it can be referenced through the
            parent Model, in addition to integer indexing.

        Notes
        -----
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
        nems.layers.base.Phi
        nems.layers.base.Parameter

        """

        # input/output should be a list of strings, a dict of strings, or None
        if isinstance(input, str):
            self.input = [input]
        else:
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
        self.parameters = parameters

    @layer('baseclass')
    def from_keyword(keyword):
        """TODO: doctring explaining how to use this in subclassed layers."""
        return Layer()

    def initial_parameters(self):
        """TODO: docstring explaining idea, most subclasses will need to write
        their own."""
        return None

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

    def set_parameter_values(self, parameter_dict):
        """Set new parameter values from key, value pairs."""
        self.parameters.update(parameter_dict)

    def get_parameter_values(self, *parameter_keys):
        """Return all parameter values formatted as a list of arrays."""
        return self.parameters.get_values(*parameter_keys)

    def sample_from_priors(self, inplace=True, as_vector=False):
        """Get or set new parameter values by sampling from priors.
        
        Parameters
        ----------
        inplace : bool, default=True
            If True, sampled values will be used to update each Parameter
            instance (and, in turn, `Phi._array`) inplace. Otherwise, the
            sampled values will be returned without changing current values.
        as_vector : bool, default=False
            If True, return sampled values as a flattened list instead of a
            list of arrays.

        Return
        ------
        samples

        See also
        --------
        nems.layers.base.Phi.sample

        """
        return self.parameters.sample(inplace=inplace, as_vector=as_vector)

    def mean_of_priors(self, inplace=True, as_vector=False):
        """Get, or set parameter values to, mean of priors.
        
        Parameters
        ----------
        inplace : bool, default=True
            If True, mean values will be used to update each Parameter
            instance (and, in turn, `Phi._array`) inplace. Otherwise, means
            will be returned without changing current values.
        as_vector : bool, default=False
            If True, return means as a flattened list instead of a
            list of arrays.

        Return
        ------
        means : list

        See also
        --------
        nems.layers.base.Phi.mean

        """
        return self.parameters.mean(inplace=inplace, as_vector=as_vector)

    def freeze_parameters(self):
        # TODO: copy to something like fn_kwargs as before? could even automate
        #       the dict updates somewhere to keep .evaluate() simple.
        pass

    # Passthrough dict-like interface for Layer.parameters
    # NOTE: 'get' operations through this interface return Parameter instances,
    #       not arrays. For array format, reference Parameter.values or use
    #       `Layer.get_parameter_values`.
    def __getitem__(self, key):
        return self.parameters[key]
    
    def get(self, key, default=None):
        return self.parameters.get(key, default=default)

    def __setitem__(self, key, val):
        self.parameters[key] = val

    def __iter__(self):
        return self.parameters.__iter__()

    def keys(self):
        return self.parameters.keys()

    def items(self):
        return self.parameters.items()

    def values(self):
        return self.parameters.values()

    # TODO: make this more informative
    def __repr__(self):
        return str(self.__class__)

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

    # Add compatibility for saving to .json
    def to_json(self):
        # TODO: package parameters, bounds, etc, into dict w/ same format as old
        #       layer dicts. Won't be fully backwards compatible but should
        #       make it easier to write an old -> new model conversion utility.
        pass

    def from_json(json):
        # TODO: Reverse of above, return Layer instance using dict for kwargs.
        #       Some attributes may need to be set separately.
        pass


# TODO: method for removing parameter(s), e.g. when freezing
#       probably return a new Phi with some dropped, make fn_kwargs or equivalent
#       a separate Phi instance.

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
        parameters : N-tuple of Parameter instances; optional

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

    def bounds_vector(self, none_for_inf=True):
        """Return a list of bounds from each parameter in `Phi._dict`.
        
        Parameters
        ----------
        none_for_inf : bool, default=True
            If True, replace (+/-)`np.inf` with None for compatibility with
            `scipy.optimize.minimize`.
        
        Returns
        -------
        bounds : list of 2-tuples
        
        """
        bounds = [p.bounds for p in self._dict.values()]
        if none_for_inf:
            subbed_bounds = []
            for b in bounds:
                lower, upper = b
                if np.isinf(lower):
                    lower = None
                if np.isinf(upper):
                    upper = None
                subbed_bounds.append((lower, upper))
            bounds = subbed_bounds
        
        return bounds


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

    @staticmethod
    def values_to_vector(values):
        """Flatten a list of Parameter.values."""
        vector = []
        for v in values:
            vector.extend(np.ravel(v))
        return vector

    def sample(self, inplace=False, as_vector=True):
        """Get or set new parameter values by sampling from priors.
        
        Parameters
        ----------
        inplace : bool, default=False
            If True, sampled values will be used to update each Parameter
            instance (and, in turn, `Phi._array`) inplace. Otherwise, the
            sampled values will be returned without changing current values.
        as_vector : bool, default=True
            If True, return sampled values as a flattened list instead of a
            list of arrays.

        Return
        ------
        samples : list

        """
        samples = []
        for p in self._dict.values():
            samples.append(p.sample(inplace=inplace))
        if as_vector:
            samples = Phi.values_to_vector(samples)

        return samples

    def mean(self, inplace=False, as_vector=True):
        """Get, or set parameter values to, mean of priors.
        
        Parameters
        ----------
        inplace : bool, default=False
            If True, mean values will be used to update each Parameter
            instance (and, in turn, `Phi._array`) inplace. Otherwise, means
            will be returned without changing current values.
        as_vector : bool, default=True
            If True, return means as a flattened list instead of a
            list of arrays.

        Return
        ------
        means : list

        """
        means = []
        for p in self._dict.values():
            means.append(p.mean(inplace=inplace))
        if as_vector:
            means = Phi.values_to_vector(means)
        
        return means


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
        return [self._dict[k].values for k in keys]

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
        prior : nems.distributions.Distribution or None; optional
            Prior distribution for this parameter, with matching shape.
        bounds : 2-tuple or None; optional
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
            changing current values.

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

    def mean(self, inplace=False):
        """Get, or set parameter values to, mean of priors.
        
        Note that `Parameter.mean()` may return a value outside of
        `Parameter.bounds`. In that event, either priors or bounds should be
        changed.

        Parameters
        ----------
        inplace : bool, default=False
            If True, mean value will be used to update `Parameter.values`
            inplace Otherwise, mean will be returned without changing current
            values.

        Return
        ------
        mean : ndarray

        """
        mean = self.prior.mean()
        if inplace:
            self.update(mean)

        return mean

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
